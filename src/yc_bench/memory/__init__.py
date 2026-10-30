"""Structured memory for YC-Bench (project plan variant (ii)).

Public surface::

    from yc_bench.memory import MemorySettings, StructuredMemory, MemoryStore

The package is deliberately dependency-free (stdlib only) so it can be unit
tested and replayed offline without spinning up the simulator or the model.
"""

from .extractor import StructuredExtractor
from .pipeline import MODE_OFF, MODE_STRUCTURED, MemorySettings, StructuredMemory
from .schema import (
    EVENT_TASK_ACCEPTED,
    EVENT_TASK_ASSIGNED,
    EVENT_TASK_CANCELLED,
    EVENT_TASK_COMPLETED,
    EVENT_TASK_INSPECTED,
    MemoryEvent,
    make_task_key,
)
from .store import MemoryStore
from .view import (
    build_records,
    client_stats,
    render_client_view,
    render_staff_view,
    staff_stats,
)

__all__ = [
    "EVENT_TASK_ACCEPTED",
    "EVENT_TASK_ASSIGNED",
    "EVENT_TASK_CANCELLED",
    "EVENT_TASK_COMPLETED",
    "EVENT_TASK_INSPECTED",
    "MODE_OFF",
    "MODE_STRUCTURED",
    "MemoryEvent",
    "MemorySettings",
    "MemoryStore",
    "StructuredExtractor",
    "StructuredMemory",
    "build_records",
    "client_stats",
    "make_task_key",
    "render_client_view",
    "render_staff_view",
    "staff_stats",
]
