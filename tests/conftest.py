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
    parser.addoption(
        "--custom_model",
        action="store_true",
        default=False,
        help="Run custom model tests",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: mark a test as an integration test"
    )
    config.addinivalue_line(
        "markers", "custom_model: mark a test as a custom model test"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--integration") and not config.getoption("--custom_model"):
        # Skip custom_model tests if --integration is passed and --custom_model is not passed
        skip_custom_model = pytest.mark.skip(reason="need --custom_model option to run")
        for item in items:
            if "custom_model" in item.keywords:
                item.add_marker(skip_custom_model)
        return

    if config.getoption("--custom_model") and not config.getoption("--integration"):
        # Skip integration tests if --custom_model is passed and --integration is not passed
        skip_integration = pytest.mark.skip(reason="need --integration option to run")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)
        return

    # if both options are passed, do not skip any tests
    if config.getoption("--custom_model") and config.getoption("--integration"):
        return

    # If neither option is passed, skip both types of tests
    skip_integration = pytest.mark.skip(reason="need --integration option to run")
    skip_custom_model = pytest.mark.skip(reason="need --custom_model option to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
        if "custom_model" in item.keywords:
            item.add_marker(skip_custom_model)
