"""Pytest configuration for sgn-ts tests."""


def pytest_configure(config):
    """Set matplotlib to non-interactive backend before any tests run."""
    # Must be done before matplotlib.pyplot is imported anywhere
    try:
        import matplotlib

        matplotlib.use("Agg")
    except ImportError:
        pass  # matplotlib not installed
