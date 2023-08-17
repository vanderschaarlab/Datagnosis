# stdlib
import sys
import warnings

# third party
import pandas as pd

# datagnosis relative
from . import logger  # noqa: F401

pd.options.mode.chained_assignment = None  # pyright:ignore

warnings.simplefilter(action="ignore")

logger.add(sink=sys.stderr, level="CRITICAL")
