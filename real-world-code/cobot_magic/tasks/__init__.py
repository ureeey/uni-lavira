"""Task package: Factory + Registry for the Cobot Magic navigation tasks."""
from typing import Dict, Type

TASK_FACTORY: Dict[str, Type] = {}


def register_task(name: str):
    def deco(cls):
        TASK_FACTORY[name] = cls
        return cls
    return deco


def TaskFactory(name: str):
    if name not in TASK_FACTORY:
        raise ValueError(f"Unknown task '{name}'. Available: {list(TASK_FACTORY)}")
    return TASK_FACTORY[name]


from .object_nav import ObjectNavTask
from .vln import VLNTask
from .eqa import EQATask

__all__ = [
    "register_task",
    "TaskFactory",
    "TASK_FACTORY",
    "ObjectNavTask",
    "VLNTask",
    "EQATask",
]
