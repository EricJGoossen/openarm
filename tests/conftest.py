def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: perf smoke tests, skip with -m 'not slow'"
    )