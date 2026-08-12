"""Patch-matcher plug point for the geocoreg refine loop.

``geocoreg/refine.py`` consumes any object implementing the ``FeatureMatcher``
interface (``base.py``) — nothing in the refine loop, the filtering chain or
the phase-correlation stages depends on how the correspondences were produced.
``RootSIFTMatcher`` is the implementation shipped here, and the one the
pipeline builds; users who want a different matcher supply their own object
implementing that interface.
"""

import logging

from geocoreg.matchers.base import FeatureMatcher
from geocoreg.matchers.sift import RootSIFTMatcher
from geocoreg.matchers.utils import robust_8bit_normalize

log = logging.getLogger("geocoreg.matchers")

__all__ = ["FeatureMatcher", "RootSIFTMatcher", "robust_8bit_normalize"]
