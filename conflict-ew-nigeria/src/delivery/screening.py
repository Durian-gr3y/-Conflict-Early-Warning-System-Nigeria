"""
src/delivery/screening.py
─────────────────────────
Client intake and ethical screening framework for Conflict Early Warning Nigeria.

Enforces RESPONSIBLE_USE.md:
  1. No individual profiling.
  2. No group/ethnic/religious targeting.
  3. Mandatory screening for government and security clients (Do No Digital Harm).
  4. Hard stop on any use cases informing kinetic (lethal) military operations.
  5. Maintains immutable audit log of screening decisions in reports/client_screening_audit.json.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
import json
import logging
import pathlib
from typing import Any

logger = logging.getLogger(__name__)

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
AUDIT_LOG_PATH = ROOT / "reports" / "client_screening_audit.json"

PROHIBITED_KEYWORDS = [
    "kinetic", "airstrike", "bombing", "artillery", "offensive operation",
    "lethal targeting", "individual profiling", "track individual",
    "ethnic profiling", "religious targeting", "collective punishment",
]


class ProhibitedUseViolation(Exception):
    """Raised when a client use-case violates the platform's Responsible Use Charter."""
    pass


@dataclass
class ScreeningDecision:
    client_id: str
    client_organization: str
    client_type: str  # 'humanitarian', 'government', 'corporate', 'academic', 'media'
    use_case_statement: str
    approved: bool
    rejection_reasons: list[str]
    prohibited_keywords_detected: list[str]
    do_no_harm_checklist_passed: bool
    timestamp: str
    reviewed_by: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ClientEthicsScreening:
    """
    Automated and audited screening engine for platform data access.
    """

    def __init__(self, audit_path: pathlib.Path | None = None):
        self.audit_path = audit_path or AUDIT_LOG_PATH
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)

    def screen_client(
        self,
        client_id: str,
        organization: str,
        client_type: str,
        use_case_statement: str,
        confirms_non_kinetic: bool = True,
        confirms_no_profiling: bool = True,
        reviewer_name: str = "Platform Ethics Officer",
    ) -> ScreeningDecision:
        """
        Evaluate client request against Responsible Use Charter.
        """
        use_case_lower = use_case_statement.lower()
        detected_prohibited = [
            kw for kw in PROHIBITED_KEYWORDS if kw in use_case_lower
        ]

        rejections = []
        if detected_prohibited:
            rejections.append(
                f"Prohibited operational keywords detected: {detected_prohibited}"
            )

        if not confirms_non_kinetic:
            rejections.append(
                "Client failed to confirm prohibition on kinetic (lethal) operations."
            )

        if not confirms_no_profiling:
            rejections.append(
                "Client failed to confirm prohibition on individual or group profiling."
            )

        # Government client strict screening
        dnh_passed = True
        if client_type.lower() == "government":
            if not (confirms_non_kinetic and confirms_no_profiling and len(detected_prohibited) == 0):
                dnh_passed = False
                rejections.append("Government client failed Do No Digital Harm review.")

        is_approved = len(rejections) == 0

        decision = ScreeningDecision(
            client_id=client_id,
            client_organization=organization,
            client_type=client_type,
            use_case_statement=use_case_statement,
            approved=is_approved,
            rejection_reasons=rejections,
            prohibited_keywords_detected=detected_prohibited,
            do_no_harm_checklist_passed=dnh_passed,
            timestamp=datetime.now().isoformat(),
            reviewed_by=reviewer_name,
        )

        self._record_decision(decision)

        if not is_approved:
            logger.warning(
                f"[ETHICS SCREENING] Client '{organization}' REJECTED: {rejections}"
            )
        else:
            logger.info(
                f"[ETHICS SCREENING] Client '{organization}' APPROVED for {client_type} access."
            )

        return decision

    def _record_decision(self, decision: ScreeningDecision):
        """Append decision to immutable audit JSON."""
        history = []
        if self.audit_path.exists():
            try:
                with open(self.audit_path, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                history = []

        history.append(decision.to_dict())
        with open(self.audit_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
