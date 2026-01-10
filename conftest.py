"""Pytest configuration for sgn-ligo tests."""

import shutil


def pytest_configure(config):
    """Set matplotlib to non-interactive backend before any tests run."""
    # Must be done before matplotlib.pyplot is imported anywhere
    try:
        import matplotlib

        matplotlib.use("Agg")
        # Only disable LaTeX if it's not available (e.g., in CI)
        if shutil.which("latex") is None:
            matplotlib.rcParams["text.usetex"] = False
    except ImportError:
        pass  # matplotlib not installed
