#!/usr/bin/env bash
# Generate a self-signed cert/key pair for the WebXR teleop bridge's HTTPS
# server (openarm/src/openarm/webxr_teleop/bridge.py).
#
# Certs are per-machine, not per-checkout: the WebXR headset connects over
# your LAN IP, not localhost, so the cert's SAN list must include whatever
# IP this machine currently has -- regenerate if that IP changes (new
# network, new machine).
#
# Output: <repo_root>/certs/{cert,key}.pem -- matches
# teleop_config.py's _default_cert_file()/_default_key_file(), which
# resolve relative to that file's own location, not CWD. This script
# mirrors that by resolving repo root relative to *its own* location
# (openarm/scripts/ -> openarm/), not CWD either.
#
# Usage:
#   scripts/generate_certs.sh            # skip if certs already exist
#   scripts/generate_certs.sh --force    # regenerate even if present
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CERT_DIR="$REPO_ROOT/certs"
CERT_FILE="$CERT_DIR/cert.pem"
KEY_FILE="$CERT_DIR/key.pem"

FORCE=false
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=true ;;
        -h|--help)
            sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done


if [ -f "$CERT_FILE" ] && [ -f "$KEY_FILE" ] && [ "$FORCE" = false ]; then
    echo "Certs already present at $CERT_DIR (use --force to regenerate)."
    exit 0
fi


if ! command -v openssl >/dev/null 2>&1; then
    echo "error: openssl not found. Install it (apt-get install -y openssl) and re-run." >&2
    exit 1
fi


# Best-effort LAN IP 
LAN_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
if [ -z "${LAN_IP:-}" ]; then
    echo "warning: could not auto-detect a LAN IP (hostname -I found nothing)." >&2
    echo "         The cert will only cover localhost/127.0.0.1 -- pass your" >&2
    echo "         machine's IP explicitly if the headset can't connect:" >&2
    echo "           OPENARM_CERT_IP=192.168.1.42 scripts/generate_certs.sh --force" >&2
fi
LAN_IP="${OPENARM_CERT_IP:-$LAN_IP}"


SAN="DNS:localhost,IP:127.0.0.1"
if [ -n "${LAN_IP:-}" ]; then
    SAN="$SAN,IP:$LAN_IP"
fi


mkdir -p "$CERT_DIR"


echo "Generating self-signed cert (SAN: $SAN)..."
openssl req -x509 -newkey rsa:2048 -nodes \
    -keyout "$KEY_FILE" \
    -out "$CERT_FILE" \
    -days 365 \
    -subj "/CN=openarm-teleop" \
    -addext "subjectAltName=$SAN"


chmod 600 "$KEY_FILE"


echo ""
echo "Wrote:"
echo "  $CERT_FILE"
echo "  $KEY_FILE"
if [ -n "${LAN_IP:-}" ]; then
    echo ""
    echo "Headset should connect to: https://$LAN_IP:8443"
fi



