# stdlib
import sys
import warnings

# third party
import pandas as pd

# dc-check relative
# dc_check relative
from . import logger  # noqa: F401

pd.options.mode.chained_assignment = None

warnings.simplefilter(action="ignore")

logger.add(sink=sys.stderr, level="CRITICAL")
