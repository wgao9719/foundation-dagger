from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Literal, Optional, Sequence, Dict, Any

@dataclass
class VPTConfig:
    version: int
    model: dict
    extra_args: dict

def parse_policy_config(cfg_dict: dict) -> VPTConfig:
    """
    Instantiate the appropriate policy config dataclass from a config dictionary.
    """
    return VPTConfig(**cfg_dict)
