# Backward-compat shim — canonical location is model/rule_governor.py
from model.rule_governor import *  # noqa: F401, F403
from model.rule_governor import RuleGovernor, sample_governed, evaluate_governed
