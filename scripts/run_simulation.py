"""Command-line entrypoint for Orieux 2012 Bayesian SIM."""

from __future__ import annotations

from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from bayesim2012.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
