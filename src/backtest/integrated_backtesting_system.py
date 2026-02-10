"""Deprecated monolithic Streamlit runner.

The functionality formerly hosted here now lives in `app.py` and `backtest_engine.py`.
This thin stub prevents accidental use of the legacy entrypoint while keeping
historical references resolvable.
"""

from __future__ import annotations

import warnings

_DEPRECATION_MESSAGE = (
    "`integrated_backtesting_system.py` has been archived. "
    "Please launch the Streamlit UI via `streamlit run app.py` or rely on "
    "the CLI entrypoint `python main.py` instead."
)

warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)

raise RuntimeError(_DEPRECATION_MESSAGE)