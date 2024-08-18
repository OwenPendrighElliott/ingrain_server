import sys
import os
import pytest

# Append the root directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def pytest_addoption(parser):
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: mark a test as an integration test"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--integration"):
        # If --integration is passed, do not skip integration tests
        return
    skip_integration = pytest.mark.skip(reason="need --integration option to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
