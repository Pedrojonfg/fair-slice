import os
import sys


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: slow tests (run with -m slow)")


# Ensure `src/fair-slice/*.py` modules are importable as plain modules:
#   from partition import compute_partition
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_MOD_DIR = os.path.join(_ROOT, "src", "fair-slice")
if _SRC_MOD_DIR not in sys.path:
    sys.path.insert(0, _SRC_MOD_DIR)

