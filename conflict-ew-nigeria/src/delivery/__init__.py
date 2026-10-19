"""
src/delivery package
────────────────────
Delivery interfaces: automated alert system, monthly brief generator, and dashboard integrations.
"""

from src.delivery.alerts import AlertSystem, ConflictAlert
from src.delivery.brief_generator import MonthlyBriefGenerator

__all__ = [
    "AlertSystem",
    "ConflictAlert",
    "MonthlyBriefGenerator",
]
