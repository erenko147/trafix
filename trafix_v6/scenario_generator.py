# Backward-compat shim — canonical location is training/scenario_generator.py
from training.scenario_generator import *  # noqa: F401, F403
from training.scenario_generator import (
    ScenarioGenerator, ScenarioEnvironment, ScenarioType,
    _MAIN_INBOUND_OD, _MAIN_OUTBOUND_OD, _LOCAL_OD, _ALL_OD,
    _JUNCTION_FRINGE_IN,
)
