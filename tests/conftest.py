from __future__ import annotations

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: perf smoke tests, skip with -m 'not slow'"
    )

def pytest_sessionfinish(session, exitstatus):
    """Fail the session if every collected test was skipped.
 
    This does not fire on a normal mixed run (some pass, some skip for
    legitimate per-case reasons like an unreachable random fuzz goal) -- only
    on the specific case of zero passes, zero failures, and at least one
    skip, which is what a missing-model-asset environment produces across
    every GATE file at once.
    """
    terminal_reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if terminal_reporter is None:
        return
    stats = terminal_reporter.stats
    n_passed = len(stats.get("passed", []))
    n_failed = len(stats.get("failed", [])) + len(stats.get("error", []))
    n_skipped = len(stats.get("skipped", []))
 
    if n_passed == 0 and n_failed == 0 and n_skipped > 0:
        terminal_reporter.write_sep(
            "!",
            "EVERY TEST SKIPPED (likely missing model assets -- run "
            "`python -m openarm_assets.assembly --sides bimanual` first). "
            "Treating an all-skip run as a FAILURE, not a pass.",
            red=True,
        )
        session.exitstatus = 1