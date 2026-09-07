#!/usr/bin/env python3
"""Stage 1c -- URDF/kinematic model vs. physically-installed hardware.

What this proves, per the Definition of Done: joint count, joint order, and
per-joint limits in the software's robot description actually match the
physical unit's documented range of motion -- not just that the file is
internally consistent with itself.

This is fully offline -- no ROS, no live hardware, no CAN. It processes the
same xacro the real launch does and diffs the result against a small,
explicitly-maintained reference file describing the physical spec (from the
hardware documentation / datasheet, not from another software file --
diffing the URDF against itself proves nothing). That reference file is the
one piece of this script that's genuinely your data to maintain, the same
way you'd maintain any other config.

Usage:
    python3 stage1_urdf_hardware_consistency.py \\
        --xacro-path /path/to/openarm_v10.urdf.xacro \\
        --xacro-args arm_type:=openarm_v1.0 bimanual:=true ros2_control:=true \\
        --physical-spec-file physical_spec.json

`physical_spec.json`:
    {"openarm_left_joint4": {"lower_rad": 0.0, "upper_rad": 2.443461, "type": "revolute"},
     ...}
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xacro-path", required=True)
    p.add_argument("--xacro-args", nargs="*", default=[], help="key:=value pairs passed straight to xacro")
    p.add_argument("--physical-spec-file", required=True)
    p.add_argument("--expected-joint-names", nargs="*", default=None,
                   help="optional explicit expected joint name list/order; if omitted, only checks against "
                        "physical-spec-file's keys and the openarm_{side}joint{1..7} naming convention")
    p.add_argument("--tolerance-rad", type=float, default=1e-4)
    return p.parse_args()


def process_xacro(xacro_path: str, xacro_args: list[str]) -> ET.Element:
    try:
        result = subprocess.run(
            ["xacro", xacro_path] + xacro_args, capture_output=True, text=True, timeout=30, check=True
        )
    except FileNotFoundError:
        print("`xacro` CLI not found. Falls back to `ros2 run xacro xacro` if you have ROS sourced; "
              "otherwise install/source the xacro package.", file=sys.stderr)
        raise
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"xacro processing failed: {e.stderr}")
    return ET.fromstring(result.stdout)


def extract_joints(urdf_root: ET.Element) -> dict[str, dict]:
    joints = {}
    for joint_el in urdf_root.findall("joint"):
        name = joint_el.get("name")
        jtype = joint_el.get("type")
        limit_el = joint_el.find("limit")
        entry: dict[str, str | float | None] = {"type": jtype}
        if limit_el is not None:
            entry["lower_rad"] = float(limit_el.get("lower", "nan"))
            entry["upper_rad"] = float(limit_el.get("upper", "nan"))
            entry["velocity"] = float(limit_el.get("velocity", "nan"))
            entry["effort"] = float(limit_el.get("effort", "nan"))
        joints[name] = entry
    return joints


def check_naming_convention(joint_names: list[str]) -> list[str]:
    """Flags any joint whose name doesn't follow openarm_{left,right}_joint{1-7}
    or openarm_{left,right}_finger_joint1 -- catches a joint silently added,
    renamed, or dropped without anyone updating the physical-spec reference."""
    import re

    problems = []
    pattern = re.compile(r"^openarm_(left|right)_(joint[1-7]|finger_joint1)$")
    for name in joint_names:
        if not pattern.match(name):
            problems.append(f"'{name}' does not match the expected openarm_{{side}}_{{joint}} naming convention")
    return problems


def main():
    args = parse_args()

    with open(args.physical_spec_file) as f:
        physical_spec = json.load(f)

    print(f"Processing {args.xacro_path} ...")
    root = process_xacro(args.xacro_path, args.xacro_args)
    urdf_joints = extract_joints(root)
    # Only compare against joints that actually have a <limit> (skips fixed joints, etc.)
    urdf_moving_joints = {n: j for n, j in urdf_joints.items() if "lower_rad" in j}

    problems = []

    naming_problems = check_naming_convention(list(urdf_moving_joints.keys()))
    problems.extend(naming_problems)

    spec_names = set(physical_spec.keys())
    urdf_names = set(urdf_moving_joints.keys())

    missing_from_urdf = spec_names - urdf_names
    if missing_from_urdf:
        problems.append(
            f"joints in the physical spec but NOT in the processed URDF: {sorted(missing_from_urdf)} "
            f"-- either the spec file is stale, or the URDF is missing a joint that physically exists"
        )
    extra_in_urdf = urdf_names - spec_names
    if extra_in_urdf:
        problems.append(
            f"joints in the URDF but NOT in the physical spec file: {sorted(extra_in_urdf)} "
            f"-- add them to the spec file (from the hardware documentation) or explain why they're new"
        )

    if args.expected_joint_names is not None:
        if list(urdf_moving_joints.keys()) != list(args.expected_joint_names):
            problems.append(
                f"joint ORDER mismatch: URDF gives {list(urdf_moving_joints.keys())}, expected "
                f"{list(args.expected_joint_names)} -- order matters for anything indexing joints "
                f"positionally rather than by name"
            )

    for name in sorted(spec_names & urdf_names):
        spec = physical_spec[name]
        urdf = urdf_moving_joints[name]
        if spec.get("type") and spec["type"] != urdf.get("type"):
            problems.append(f"'{name}': type mismatch -- URDF={urdf.get('type')}, physical spec={spec['type']}")
        for bound in ("lower_rad", "upper_rad"):
            if bound in spec:
                diff = abs(urdf.get(bound, float("nan")) - spec[bound])
                if not (diff <= args.tolerance_rad):
                    problems.append(
                        f"'{name}': {bound} mismatch -- URDF={urdf.get(bound)}, "
                        f"documented physical spec={spec[bound]} (diff {diff:.6f} rad, "
                        f"tolerance {args.tolerance_rad:.6f})"
                    )

    print(f"\nChecked {len(spec_names & urdf_names)} joints against the physical spec file.")
    if problems:
        print(f"\n{len(problems)} PROBLEM(S) FOUND:")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)

    print("\nSTAGE 1c RESULT: PASS -- URDF matches the documented physical spec for every checked joint.")
    print(
        "\nReminder: this only proves the FILE matches your documented spec. It does not prove your "
        "documented spec is itself correct, or that a specific physical unit hasn't drifted from spec "
        "(a bent joint, a re-geared motor, etc.) -- that's what Stage 1b's physical spot-check is for."
    )


if __name__ == "__main__":
    main()