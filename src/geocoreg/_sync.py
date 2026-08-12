"""Shared synchronization primitives for multi-threaded refinement stages.

GDAL Dataset handles are not thread-safe: concurrent ReadAsArray calls on the
same handle corrupt state. Every threaded reader of ``ctx.ref.ds`` /
``ctx.mov.ds`` must hold this lock for the duration of the read.
"""

import threading

REF_READ_LOCK = threading.Lock()
