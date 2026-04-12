"""
helpers/heat.py

Heat bucket labeling utility.
"""

from server.constants import (
    HEAT_BUCKET_LOW,
    HEAT_BUCKET_MEDIUM,
)


def _heat_bucket(heat: int) -> str:
    if heat <= HEAT_BUCKET_LOW:
        return "low"
    if heat <= HEAT_BUCKET_MEDIUM:
        return "medium"
    return "high"