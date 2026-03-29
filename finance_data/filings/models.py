"""Shared data models for SEC filings."""

from dataclasses import dataclass
from enum import Enum


class SecFilingType(str, Enum):
    """Known SEC EDGAR filing form types.

    Each member's string value is the exact form name used by EDGAR.
    The ``.description`` attribute provides a human-readable summary.

    Because ``SecFilingType`` inherits from ``str``, any member can be
    passed directly wherever a plain ``str`` is expected.

    Usage::

        filing_type = SecFilingType.FORM_10_K          # == "10-K"
        filing_type = SecFilingType.FORM_10_K.description  # "Annual report"
    """

    def __new__(cls, value: str, description: str = "") -> "SecFilingType":
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.description = description  # type: ignore[attr-defined]
        return obj

    # ── Prospectus / Offering ────────────────────────────────────────────────
    FORM_424B2 = ("424B2", "Prospectus supplement filed under Rule 424(b)(2)")
    FORM_424B5 = ("424B5", "Prospectus supplement filed under Rule 424(b)(5)")
    FWP = ("FWP", "Free writing prospectus submitted by an issuer")
    S_3ASR = ("S-3ASR", "Automatic shelf registration for well-known seasoned issuers")
    S_8 = ("S-8", "Registration of securities offered to employees under a benefit plan")

    # ── Annual / Quarterly Reports ───────────────────────────────────────────
    FORM_10_K = ("10-K", "Annual report filed by domestic public companies")
    FORM_10_Q = ("10-Q", "Quarterly report filed for the first three fiscal quarters")
    ARS = ("ARS", "Annual report sent directly to shareholders")

    # ── Current / Event Reports ──────────────────────────────────────────────
    FORM_8_K = ("8-K", "Current report disclosing material corporate events")
    FORM_8_A12B = (
        "8-A12B",
        "Registration of a class of securities under Exchange Act Section 12(b)",
    )

    # ── Proxy / Solicitation ─────────────────────────────────────────────────
    DEF_14A = ("DEF 14A", "Definitive proxy statement sent to shareholders before a vote")
    PRE_14A = ("PRE 14A", "Preliminary proxy statement filed before the definitive version")
    DEFA14A = ("DEFA14A", "Definitive additional proxy solicitation materials")
    PX14A6G = (
        "PX14A6G",
        "Notice of exempt solicitation filed by non-management proxy solicitors",
    )

    # ── Beneficial Ownership ─────────────────────────────────────────────────
    FORM_3 = ("3", "Initial statement of beneficial ownership of securities")
    FORM_3_A = ("3/A", "Amendment to the initial statement of beneficial ownership")
    FORM_4 = ("4", "Statement of changes in beneficial ownership (insiders)")
    FORM_144 = ("144", "Notice of proposed sale of restricted securities under Rule 144")
    FORM_144_A = ("144/A", "Amendment to the Rule 144 notice of proposed sale")
    SC_13G = ("SC 13G", "Beneficial ownership report for passive investors holding ≥5%")
    SC_13G_A = ("SC 13G/A", "Amendment to Schedule 13G beneficial ownership report")
    SCHEDULE_13G = ("SCHEDULE 13G", "Schedule 13G beneficial ownership report (long form)")
    SCHEDULE_13G_A = (
        "SCHEDULE 13G/A",
        "Amendment to the long-form Schedule 13G beneficial ownership report",
    )

    # ── Institutional Holdings ───────────────────────────────────────────────
    FORM_13F_HR = (
        "13F-HR",
        "Quarterly holdings report filed by institutional investment managers",
    )
    N_PX = (
        "N-PX",
        "Annual report of proxy voting record required of registered funds",
    )

    # ── Regulatory / Miscellaneous ───────────────────────────────────────────
    CERT = ("CERT", "Certification filed by an officer (e.g. Sarbanes-Oxley)")
    CORRESP = (
        "CORRESP",
        "Correspondence between the SEC staff and the filer (comment letters)",
    )
    IRANNOTICE = (
        "IRANNOTICE",
        "Iran sanctions disclosure notice required under Section 13(r) of the Exchange Act",
    )
    SD = (
        "SD",
        "Specialized disclosure report (e.g. conflict-minerals under Dodd-Frank Section 1502)",
    )
    UPLOAD = ("UPLOAD", "Document uploaded directly to the SEC EDGAR system")


@dataclass(frozen=True)
class SecResults:
    dashes_acc_num: str
    form_name: str
    filing_date: str
    report_date: str
    primary_document: str
