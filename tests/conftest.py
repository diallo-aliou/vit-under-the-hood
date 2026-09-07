"""Pytest configuration ensuring headless Agg backend for tests."""

import matplotlib

matplotlib.use("Agg")
