"""Behavioural responses of the economy to policy differences between scenario and baseline.

`run` fills `state.adjust` (supply-side responses, see `responses.py`); `demand_gap` returns
the short-run output gap caused by the fiscal impulse (see `demand.py`). Elasticities are in
`params/behaviour.yaml` (prefix `beh.`); `beh.enabled` = 0 gives static costing and
`beh.scale` scales every response for sensitivity bands.
"""
from .demand import demand_gap
from .responses import run

__all__ = ["run", "demand_gap"]
