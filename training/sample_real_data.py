#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["datasets", "tqdm", "polars", "huggingface-hub"]
# ///
"""Sample real-world text from HuggingFace datasets for underrepresented categories.

Downloads and samples from publicly available HuggingFace datasets for
artifact, prose, and structured categories. Uses streaming to avoid full
downloads.

Usage:
    python training/sample_real_data.py --output data/real_samples_v2.jsonl
    python training/sample_real_data.py --dry-run
"""

import argparse
import json
import os
import random
import string
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Core helper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourcePlan:
    """Describes one sampling source in the v2 real-data pipeline."""

    name: str
    target_count: int
    category: str
    sub_type: str
    synthetic: bool
    notes: str = ""


def build_source_plans() -> list[SourcePlan]:
    """Return the default real-data sourcing plan."""
    return [
        # Artifact sources
        SourcePlan(
            "sample_finepdfs",
            10000,
            "artifact",
            "pdf_dump",
            False,
            "Primary semi-real PDF extraction source",
        ),
        SourcePlan(
            "sample_scientific_papers",
            5000,
            "artifact",
            "pdf_dump",
            False,
            "Fallback full-article PDF-like extraction text",
        ),
        SourcePlan(
            "sample_boilerplate",
            6000,
            "artifact",
            "boilerplate",
            False,
            "LICENSE/NOTICE/COPYING corpora from The Stack dedup",
        ),
        SourcePlan(
            "generate_hybrid_document_samples",
            4000,
            "artifact",
            "pdf_dump",
            True,
            "Forms, tables, invoices, and copied report snippets",
        ),
        SourcePlan(
            "generate_skip_samples",
            6000,
            "artifact",
            "skip",
            True,
            "Whitespace, empty, and ultra-short fragments",
        ),
        SourcePlan(
            "generate_ocr_garbage",
            8000,
            "artifact",
            "ocr_garbage",
            True,
            "Programmatic OCR-style corruption patterns",
        ),
        # Prose sources
        SourcePlan(
            "sample_wikipedia_paragraphs",
            12000,
            "prose",
            "plain",
            False,
            "Paragraph-level encyclopedia prose",
        ),
        SourcePlan(
            "sample_wikipedia_full",
            8000,
            "prose",
            "plain",
            False,
            "Section-level encyclopedia prose",
        ),
        SourcePlan(
            "sample_arxiv_abstracts",
            8000,
            "prose",
            "plain",
            False,
            "Scientific abstracts for clean prose",
        ),
        SourcePlan(
            "generate_prose_variants",
            6000,
            "prose",
            "plain",
            True,
            "Template-generated letters and dialogues",
        ),
        # Structured sources — balance real and synthetic to avoid
        # code↔structured confusion from too many config files.
        # 10K real configs + 21K synthetic = 31K total structured.
        SourcePlan(
            "sample_stack_configs",
            10000,
            "structured",
            "config",
            False,
            "Real JSON/YAML/TOML/INI config files from The Stack dedup",
        ),
        SourcePlan(
            "generate_csv_samples",
            10000,
            "structured",
            "csv_tsv",
            True,
            "Programmatic CSV and TSV tables",
        ),
        SourcePlan(
            "generate_log_samples",
            5000,
            "structured",
            "log_lines",
            True,
            "Synthetic but realistic log lines",
        ),
        SourcePlan(
            "generate_kv_samples",
            6000,
            "structured",
            "kv_xml",
            True,
            "INI, key/value, and XML config snippets",
        ),
    ]


SOURCE_PLANS = build_source_plans()


def summarize_source_plans(plans: list[SourcePlan]) -> dict[str, int]:
    """Summarize source quotas for dry-run output and tests."""
    summary = {
        "total_samples": 0,
        "artifact_real": 0,
        "artifact_synthetic": 0,
        "hybrid_pdf_dump": 0,
        "structured_real": 0,
    }

    for plan in plans:
        summary["total_samples"] += plan.target_count
        if plan.category == "artifact":
            bucket = "artifact_synthetic" if plan.synthetic else "artifact_real"
            summary[bucket] += plan.target_count
            if plan.name == "generate_hybrid_document_samples":
                summary["hybrid_pdf_dump"] += plan.target_count
        if plan.category == "structured" and not plan.synthetic:
            summary["structured_real"] += plan.target_count

    return summary


def format_source_plan(plans: list[SourcePlan]) -> str:
    """Format the dry-run plan output."""
    summary = summarize_source_plans(plans)
    lines = [f"Data sourcing plan ({summary['total_samples']:,} total samples):", ""]
    lines.append(
        "Quota summary: "
        f"artifact real/semi-real={summary['artifact_real']:,}, "
        f"artifact synthetic={summary['artifact_synthetic']:,}, "
        f"hybrid_pdf_dump={summary['hybrid_pdf_dump']:,}, "
        f"structured real={summary['structured_real']:,}"
    )
    lines.append("")
    for plan in plans:
        source_kind = "synthetic" if plan.synthetic else "real/semi-real"
        lines.append(
            f"  {plan.name}: {plan.target_count:,} [{plan.category}/{plan.sub_type}] "
            f"{source_kind}"
        )
        if plan.notes:
            lines.append(f"    note: {plan.notes}")
    return "\n".join(lines)


def emit_sample(
    text: str, category: str, sub_type: str, source: str
) -> dict | None:
    """Create a sample dict, applying length filter. Returns None if filtered."""
    text = text.strip()
    if len(text) < 50 or len(text) > 10000:
        return None
    return {
        "text": text,
        "expected_category": category,
        "sub_type": sub_type,
        "source": f"real/{source}",
        "model": f"real/{source}",
    }


# ---------------------------------------------------------------------------
# Artifact generators (programmatic)
# ---------------------------------------------------------------------------


def generate_skip_samples(
    n: int = 8000, seed: int = 42
) -> Generator[dict, None, None]:
    """Generate empty/whitespace/single-word fragments. No HF needed."""
    rng = random.Random(seed)

    fragments = [
        "",
        " ",
        "  ",
        "\n",
        "\n\n",
        "\t",
        "\t\t",
        "   \n   ",
        ".",
        "-",
        "N/A",
        "null",
        "None",
        "undefined",
        "nan",
        "NA",
        "n/a",
        "—",
        "...",
        "???",
        "TBD",
        "TODO",
        "FIXME",
        "test",
        "hello",
        "ok",
        "yes",
        "no",
        "true",
        "false",
        "0",
        "1",
        "-1",
        "abc",
        "xxx",
        "asdf",
        "foo",
        "bar",
        "baz",
    ]

    for i in range(n):
        if i < len(fragments):
            text = fragments[i]
        else:
            choice = rng.randint(0, 4)
            if choice == 0:
                # Random whitespace
                text = " " * rng.randint(0, 20) + "\n" * rng.randint(0, 5)
            elif choice == 1:
                # Single random word
                length = rng.randint(1, 10)
                text = "".join(rng.choices(string.ascii_lowercase, k=length))
            elif choice == 2:
                # A few random characters
                text = "".join(
                    rng.choices(string.printable[:62], k=rng.randint(0, 15))
                )
            elif choice == 3:
                # Empty or near-empty with punctuation
                text = rng.choice(["", ".", "..", "...", "---", "***", "___"])
            else:
                # Pick from fragments
                text = rng.choice(fragments)

        yield {
            "text": text,
            "expected_category": "artifact",
            "sub_type": "skip",
            "source": "real/generated_skip",
            "model": "real/generated_skip",
        }


def generate_ocr_garbage(
    n: int = 15000, seed: int = 42
) -> Generator[dict, None, None]:
    """Generate OCR-like garbled text programmatically."""
    rng = random.Random(seed)

    # Unicode ranges for mixed-script characters
    latin_extras = list(range(0x00C0, 0x0180))  # Latin Extended
    cyrillic = list(range(0x0400, 0x0500))
    greek = list(range(0x0370, 0x03FF))
    cjk_sample = list(range(0x4E00, 0x4E50))  # Small CJK subset
    arabic = list(range(0x0600, 0x0650))

    all_ranges = latin_extras + cyrillic + greek + cjk_sample + arabic

    # Common OCR error patterns
    ocr_patterns = [
        "l1I|",
        "0Oo",
        "rn→m",
        "cl→d",
        "vv→w",
        "li→h",
        "cj→g",
    ]

    broken_encodings = [
        "\ufffd",  # replacement character
        "\u00e2\u0080\u0099",  # mojibake for apostrophe
        "\u00c3\u00a9",  # mojibake for e-acute
        "\u00e2\u0080\u009c",  # mojibake for left double quote
        "\u00e2\u0080\u009d",  # mojibake for right double quote
        "â€™",
        "Ã©",
        "â€œ",
        "â€\x9d",
        "Â®",
        "Â©",
    ]

    count = 0
    while count < n:
        length = rng.randint(60, 500)
        method = rng.randint(0, 5)

        if method == 0:
            # Mixed script garbage
            chars = [chr(rng.choice(all_ranges)) for _ in range(length)]
            # Insert random spaces and newlines
            for idx in rng.sample(range(len(chars)), k=min(length // 5, len(chars))):
                chars[idx] = rng.choice([" ", "\n", "\t"])
            text = "".join(chars)

        elif method == 1:
            # Broken encoding patterns
            parts = []
            while len("".join(parts)) < length:
                if rng.random() < 0.3:
                    parts.append(rng.choice(broken_encodings))
                elif rng.random() < 0.5:
                    word_len = rng.randint(1, 8)
                    parts.append(
                        "".join(rng.choices(string.ascii_letters, k=word_len))
                    )
                else:
                    parts.append(rng.choice([" ", "  ", "\n", ".", ","]))
            text = "".join(parts)[:length]

        elif method == 2:
            # OCR confusion patterns mixed with real-ish words
            words = []
            for _ in range(length // 5):
                if rng.random() < 0.4:
                    words.append(rng.choice(ocr_patterns))
                else:
                    word_len = rng.randint(2, 10)
                    w = list("".join(rng.choices(string.ascii_lowercase, k=word_len)))
                    # Introduce OCR-like substitutions
                    for j in range(len(w)):
                        if rng.random() < 0.3:
                            w[j] = rng.choice("l1I|0OoS5B8")
                    words.append("".join(w))
            text = " ".join(words)

        elif method == 3:
            # Garbled numbers with random separators
            parts = []
            for _ in range(length // 4):
                if rng.random() < 0.6:
                    parts.append(str(rng.randint(0, 99999)))
                else:
                    parts.append(
                        rng.choice(
                            [" ", ".", ",", "|", "/", "\\", "-", "_", ":", ";"]
                        )
                    )
            text = "".join(parts)

        elif method == 4:
            # Repeated fragments (scanner artifacts)
            fragment_len = rng.randint(5, 30)
            fragment = "".join(rng.choices(string.ascii_letters + " .", k=fragment_len))
            repeats = length // fragment_len + 1
            text = (fragment * repeats)[:length]
            # Add some line noise
            chars = list(text)
            for idx in rng.sample(range(len(chars)), k=min(length // 10, len(chars))):
                chars[idx] = rng.choice(["\n", chr(rng.choice(all_ranges))])
            text = "".join(chars)

        else:
            # Dense Unicode soup
            chars = []
            for _ in range(length):
                r = rng.random()
                if r < 0.2:
                    chars.append(chr(rng.choice(all_ranges)))
                elif r < 0.4:
                    chars.append(rng.choice(string.punctuation))
                elif r < 0.6:
                    chars.append(rng.choice(string.digits))
                elif r < 0.8:
                    chars.append(rng.choice(string.ascii_letters))
                else:
                    chars.append(rng.choice([" ", "\n", "\t"]))
            text = "".join(chars)

        text = text.strip()
        if len(text) >= 50:
            yield {
                "text": text,
                "expected_category": "artifact",
                "sub_type": "ocr_garbage",
                "source": "real/generated_ocr",
                "model": "real/generated_ocr",
            }
            count += 1


_HYBRID_DOC_TITLES = [
    "Invoice Summary",
    "Quarterly Inspection Report",
    "Claims Intake Form",
    "Procurement Review",
    "Service Ticket Export",
    "Compliance Checklist",
]

_HYBRID_DOC_FIELDS = [
    "Account ID",
    "Invoice No",
    "Prepared By",
    "Region",
    "Case Ref",
    "Approved",
    "Escalation",
    "Amount",
]

_HYBRID_DOC_NOTES = [
    "scanned from duplex printout",
    "copied from OCR export",
    "table borders dropped during extraction",
    "header repeated on every page",
    "glyph fallback active for accented names",
]


def generate_hybrid_document_samples(
    n: int = 4000, seed: int = 42
) -> Generator[dict, None, None]:
    """Generate artifact/pdf_dump samples with structural remnants."""
    rng = random.Random(seed)
    count = 0
    while count < n:
        title = rng.choice(_HYBRID_DOC_TITLES)
        lines = [
            f"{title}  Page {rng.randint(1, 18)}",
            "CONFIDENTIAL - INTERNAL USE ONLY",
            "",
        ]

        for _ in range(rng.randint(4, 8)):
            field = rng.choice(_HYBRID_DOC_FIELDS)
            value = rng.choice(
                [
                    f"{rng.randint(1000, 9999)}-{rng.randint(10, 99)}",
                    f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/2025",
                    rng.choice(["YES", "NO", "PENDING", "REVIEW"]),
                    f"${rng.uniform(25, 9999):.2f}",
                ]
            )
            if rng.random() < 0.5:
                lines.append(f"{field}: {value}")
            else:
                lines.append(f"{field:<16} .... {value}")

        lines.extend(
            [
                "",
                "Line Item   Qty   Amount   Notes",
                f"Scanner Fee   {rng.randint(1,4)}   ${rng.uniform(15,150):.2f}   {rng.choice(_HYBRID_DOC_NOTES)}",
                f"Transit Adj   {rng.randint(1,4)}   ${rng.uniform(15,150):.2f}   copied from paper form",
                "",
                f"Footer copy: {rng.choice(_HYBRID_DOC_NOTES)}",
            ]
        )

        if rng.random() < 0.5:
            lines.append("A1) sanitize leading nulls; A2) rebuild hy-")
            lines.append("phenated lines across pages; A3) ignore repeated footer")
        else:
            lines.append("Section 4.2  Terms / Conditions / Return authorization")
            lines.append("Exhibit B  Attachment register  Copy received via OCR")

        text = "\n".join(lines)
        sample = emit_sample(
            text,
            "artifact",
            "pdf_dump",
            "generated_hybrid_pdf_dump",
        )
        if sample is not None:
            yield sample
            count += 1


# ---------------------------------------------------------------------------
# Prose generators (programmatic)
# ---------------------------------------------------------------------------

_LETTER_TEMPLATES = [
    "Dear {name},\n\nI am writing to inform you about {topic}. "
    "As you may know, {detail}. We believe this is an important matter "
    "that requires your attention.\n\nPlease do not hesitate to reach out "
    "if you have any questions or concerns regarding this matter.\n\n"
    "Sincerely,\n{sender}",
    "To Whom It May Concern,\n\n{topic} has been a subject of discussion "
    "for some time now. {detail}. We would like to take this opportunity "
    "to share our thoughts on the matter.\n\nThank you for your time and "
    "consideration.\n\nBest regards,\n{sender}",
]

_EMAIL_TEMPLATES = [
    "Subject: {topic}\n\nHi {name},\n\nI wanted to follow up on {topic}. "
    "{detail}. Can we schedule a meeting to discuss this further?\n\n"
    "Let me know what works for your schedule.\n\nThanks,\n{sender}",
    "Subject: Re: {topic}\n\nHey {name},\n\nThanks for getting back to me. "
    "{detail}. I think we should move forward with the proposed plan.\n\n"
    "Looking forward to hearing from you.\n\nCheers,\n{sender}",
]

_DIALOGUE_TEMPLATES = [
    '"{name}: Have you heard about {topic}?"\n'
    '"{sender}: No, what about it?"\n'
    '"{name}: Well, {detail}."\n'
    '"{sender}: That\'s really interesting. Tell me more about it."\n'
    '"{name}: Sure. The key thing is that this changes everything "',
    "Interviewer: Could you tell us about {topic}?\n"
    "{name}: Certainly. {detail}. It has been a fascinating journey.\n"
    "Interviewer: And what are the implications?\n"
    "{name}: The implications are far-reaching. We expect significant "
    "changes in the coming years.",
]

_NAMES = [
    "Alice", "Bob", "Carol", "David", "Emily", "Frank", "Grace", "Henry",
    "Irene", "Jack", "Karen", "Leo", "Maria", "Nathan", "Olivia", "Peter",
    "Quinn", "Rachel", "Samuel", "Tina", "Uma", "Victor", "Wendy", "Xavier",
]

_TOPICS = [
    "the quarterly budget review", "the new product launch",
    "climate change mitigation strategies", "advances in renewable energy",
    "the upcoming conference on artificial intelligence",
    "healthcare policy reform", "the evolution of programming languages",
    "space exploration missions", "the impact of social media on society",
    "urban planning and sustainable development",
    "the future of transportation", "cybersecurity best practices",
    "advances in genomic research", "the global supply chain",
    "educational technology trends", "marine conservation efforts",
]

_DETAILS = [
    "recent studies have shown significant progress in this area",
    "the team has been working tirelessly to meet the deadline",
    "several stakeholders have expressed their support",
    "the data indicates a positive trend over the past quarter",
    "new regulations are expected to take effect next month",
    "the research findings have been published in a peer-reviewed journal",
    "community feedback has been overwhelmingly positive",
    "the preliminary results exceeded our expectations",
    "the committee has recommended a phased approach",
    "experts from around the world have contributed their insights",
]


def generate_prose_variants(
    n: int = 9000, seed: int = 42
) -> Generator[dict, None, None]:
    """Generate letter/email/dialogue-style text from templates."""
    rng = random.Random(seed)
    templates = _LETTER_TEMPLATES + _EMAIL_TEMPLATES + _DIALOGUE_TEMPLATES

    for _ in range(n):
        template = rng.choice(templates)
        name = rng.choice(_NAMES)
        sender = rng.choice([x for x in _NAMES if x != name])
        topic = rng.choice(_TOPICS)
        detail = rng.choice(_DETAILS)

        text = template.format(
            name=name, sender=sender, topic=topic, detail=detail
        )

        # Sometimes repeat or extend to add variety
        if rng.random() < 0.3:
            extra_detail = rng.choice(_DETAILS)
            text += f" Furthermore, {extra_detail}."

        yield {
            "text": text,
            "expected_category": "prose",
            "sub_type": "plain",
            "source": "real/generated_prose",
            "model": "real/generated_prose",
        }


# ---------------------------------------------------------------------------
# Structured generators (programmatic)
# ---------------------------------------------------------------------------

_CSV_HEADERS = [
    ["id", "name", "email", "age", "city"],
    ["timestamp", "level", "message", "source", "line"],
    ["product_id", "product_name", "price", "quantity", "category"],
    ["date", "open", "high", "low", "close", "volume"],
    ["student_id", "first_name", "last_name", "grade", "gpa"],
    ["ip_address", "request_path", "status_code", "response_time_ms"],
    ["sensor_id", "temperature", "humidity", "pressure", "timestamp"],
    ["order_id", "customer", "total", "status", "created_at"],
]

_CSV_FIRST_NAMES = ["John", "Jane", "Bob", "Alice", "Charlie", "Diana", "Eve"]
_CSV_LAST_NAMES = ["Smith", "Doe", "Johnson", "Williams", "Brown", "Jones"]
_CSV_CITIES = ["New York", "London", "Tokyo", "Paris", "Berlin", "Sydney"]
_CSV_PRODUCTS = ["Widget", "Gadget", "Gizmo", "Thingamajig", "Doohickey"]
_CSV_STATUSES = ["pending", "completed", "shipped", "cancelled", "processing"]
_CSV_LOG_LEVELS = ["INFO", "WARN", "ERROR", "DEBUG", "TRACE"]


def _generate_csv_row(headers: list[str], rng: random.Random) -> list[str]:
    """Generate a single CSV data row matching the given headers."""
    row = []
    for h in headers:
        h_lower = h.lower()
        if "id" in h_lower:
            row.append(str(rng.randint(1000, 99999)))
        elif "name" in h_lower or "first" in h_lower:
            row.append(rng.choice(_CSV_FIRST_NAMES))
        elif "last" in h_lower:
            row.append(rng.choice(_CSV_LAST_NAMES))
        elif "email" in h_lower:
            name = rng.choice(_CSV_FIRST_NAMES).lower()
            row.append(f"{name}{rng.randint(1, 999)}@example.com")
        elif "age" in h_lower or "grade" in h_lower:
            row.append(str(rng.randint(18, 95)))
        elif "city" in h_lower:
            row.append(rng.choice(_CSV_CITIES))
        elif "price" in h_lower or "total" in h_lower:
            row.append(f"{rng.uniform(1.0, 999.99):.2f}")
        elif "quantity" in h_lower or "volume" in h_lower:
            row.append(str(rng.randint(1, 100000)))
        elif "category" in h_lower:
            row.append(rng.choice(["electronics", "clothing", "food", "books"]))
        elif "status" in h_lower:
            row.append(rng.choice(_CSV_STATUSES))
        elif "timestamp" in h_lower or "date" in h_lower or "created" in h_lower:
            row.append(
                f"2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d} "
                f"{rng.randint(0,23):02d}:{rng.randint(0,59):02d}:{rng.randint(0,59):02d}"
            )
        elif "level" in h_lower:
            row.append(rng.choice(_CSV_LOG_LEVELS))
        elif "message" in h_lower:
            row.append(f"Event occurred at component {rng.randint(1,50)}")
        elif "source" in h_lower or "path" in h_lower:
            row.append(f"/api/v{rng.randint(1,3)}/resource/{rng.randint(1,100)}")
        elif "line" in h_lower:
            row.append(str(rng.randint(1, 5000)))
        elif any(x in h_lower for x in ["open", "high", "low", "close"]):
            row.append(f"{rng.uniform(10.0, 500.0):.2f}")
        elif "gpa" in h_lower:
            row.append(f"{rng.uniform(1.0, 4.0):.2f}")
        elif "ip" in h_lower:
            row.append(
                f"{rng.randint(1,255)}.{rng.randint(0,255)}."
                f"{rng.randint(0,255)}.{rng.randint(0,255)}"
            )
        elif "response" in h_lower or "time" in h_lower:
            row.append(str(rng.randint(1, 5000)))
        elif "temperature" in h_lower or "temp" in h_lower:
            row.append(f"{rng.uniform(-20.0, 45.0):.1f}")
        elif "humidity" in h_lower:
            row.append(f"{rng.uniform(10.0, 100.0):.1f}")
        elif "pressure" in h_lower:
            row.append(f"{rng.uniform(950.0, 1050.0):.1f}")
        elif "customer" in h_lower:
            row.append(
                f"{rng.choice(_CSV_FIRST_NAMES)} {rng.choice(_CSV_LAST_NAMES)}"
            )
        elif "product" in h_lower:
            row.append(rng.choice(_CSV_PRODUCTS))
        else:
            row.append(f"val_{rng.randint(0, 999)}")
    return row


def generate_csv_samples(
    n: int = 15000, seed: int = 42
) -> Generator[dict, None, None]:
    """Generate realistic CSV/TSV data with varied schemas."""
    rng = random.Random(seed)

    count = 0
    while count < n:
        headers = rng.choice(_CSV_HEADERS)
        num_rows = rng.randint(5, 80)
        is_tsv = rng.random() < 0.3
        delimiter = "\t" if is_tsv else ","

        lines = [delimiter.join(headers)]
        for _ in range(num_rows):
            row = _generate_csv_row(headers, rng)
            lines.append(delimiter.join(row))

        text = "\n".join(lines)
        sample = emit_sample(
            text,
            "structured",
            "tsv" if is_tsv else "csv",
            "generated_csv",
        )
        if sample is not None:
            yield sample
            count += 1


_LOG_FORMATS = [
    # syslog
    "{month} {day:2d} {hour:02d}:{minute:02d}:{second:02d} {host} {process}[{pid}]: {message}",
    # Apache access log
    '{ip} - - [{day:02d}/{month}/{year} {hour:02d}:{minute:02d}:{second:02d} +0000] "GET {path} HTTP/1.1" {status} {size}',
    # JSON-ish structured log
    '{{"timestamp":"{year}-{month_num:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}Z","level":"{level}","msg":"{message}","service":"{service}"}}',
    # Nginx error log
    "{year}/{month_num:02d}/{day:02d} {hour:02d}:{minute:02d}:{second:02d} [{level}] {pid}#{tid}: *{conn} {message}, client: {ip}, server: {host}",
]

_LOG_HOSTS = ["web01", "web02", "db01", "app-server", "proxy", "worker-3", "cache01"]
_LOG_PROCESSES = ["sshd", "nginx", "apache2", "systemd", "cron", "kernel", "postfix"]
_LOG_MESSAGES = [
    "Connection established from remote host",
    "Failed password for invalid user admin",
    "Request completed successfully",
    "Timeout waiting for upstream response",
    "Worker process exited with code 0",
    "Starting service maintenance window",
    "Disk usage exceeded threshold at 85 percent",
    "Rate limit exceeded for client",
    "SSL handshake completed",
    "Database connection pool exhausted",
    "Cache miss for key session_data",
    "Health check passed",
    "Configuration reloaded successfully",
    "Memory usage at 72 percent",
    "New connection accepted on port 443",
]
_LOG_SERVICES = ["auth", "api-gateway", "user-service", "payment", "notification"]
_LOG_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_LOG_LEVELS_LOWER = ["info", "warn", "error", "debug", "notice", "crit"]
_LOG_PATHS = ["/api/users", "/api/health", "/api/orders", "/static/app.js", "/login", "/dashboard"]


def generate_log_samples(
    n: int = 6000, seed: int = 42
) -> Generator[dict, None, None]:
    """Generate syslog/access log format lines."""
    rng = random.Random(seed)

    count = 0
    while count < n:
        fmt = rng.choice(_LOG_FORMATS)
        num_lines = rng.randint(10, 60)

        lines = []
        for _ in range(num_lines):
            month_idx = rng.randint(0, 11)
            line = fmt.format(
                month=_LOG_MONTHS[month_idx],
                month_num=month_idx + 1,
                day=rng.randint(1, 28),
                year=rng.choice([2023, 2024, 2025]),
                hour=rng.randint(0, 23),
                minute=rng.randint(0, 59),
                second=rng.randint(0, 59),
                host=rng.choice(_LOG_HOSTS),
                process=rng.choice(_LOG_PROCESSES),
                pid=rng.randint(100, 65535),
                tid=rng.randint(0, 32),
                conn=rng.randint(1, 99999),
                message=rng.choice(_LOG_MESSAGES),
                ip=f"{rng.randint(1,255)}.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(0,255)}",
                path=rng.choice(_LOG_PATHS),
                status=rng.choice([200, 201, 301, 400, 401, 403, 404, 500, 502, 503]),
                size=rng.randint(100, 50000),
                level=rng.choice(_LOG_LEVELS_LOWER),
                service=rng.choice(_LOG_SERVICES),
            )
            lines.append(line)

        text = "\n".join(lines)
        sample = emit_sample(text, "structured", "log_lines", "generated_logs")
        if sample is not None:
            yield sample
            count += 1


_INI_SECTIONS = [
    "database", "server", "logging", "cache", "auth", "email", "api",
    "storage", "monitoring", "queue", "security", "network",
]

_INI_KEYS = {
    "database": ["host", "port", "name", "user", "password", "pool_size", "timeout", "ssl"],
    "server": ["host", "port", "workers", "debug", "log_level", "timeout", "max_connections"],
    "logging": ["level", "file", "format", "max_size", "backup_count", "console"],
    "cache": ["backend", "host", "port", "ttl", "max_entries", "prefix"],
    "auth": ["provider", "secret_key", "token_expiry", "allow_registration", "max_attempts"],
    "email": ["smtp_host", "smtp_port", "use_tls", "from_address", "username"],
    "api": ["base_url", "version", "rate_limit", "timeout", "retry_count"],
    "storage": ["backend", "bucket", "region", "path", "max_file_size"],
    "monitoring": ["enabled", "interval", "endpoint", "api_key"],
    "queue": ["broker", "backend", "concurrency", "prefetch_count"],
    "security": ["cors_origins", "csrf_enabled", "hsts_enabled", "content_security_policy"],
    "network": ["bind_address", "dns_server", "proxy", "mtu"],
}


def _random_ini_value(key: str, rng: random.Random) -> str:
    """Generate a plausible value for a config key."""
    k = key.lower()
    if "port" in k:
        return str(rng.choice([80, 443, 3306, 5432, 6379, 8080, 8443, 27017]))
    if "host" in k or "address" in k or "server" in k:
        return rng.choice(["localhost", "127.0.0.1", "0.0.0.0", "db.internal", "cache.local"])
    if "timeout" in k or "ttl" in k or "expiry" in k or "interval" in k:
        return str(rng.choice([30, 60, 120, 300, 600, 3600]))
    if "size" in k or "count" in k or "limit" in k or "workers" in k or "concurrency" in k:
        return str(rng.randint(1, 1000))
    if "level" in k:
        return rng.choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    if "enabled" in k or "debug" in k or "console" in k or "use_" in k:
        return rng.choice(["true", "false", "yes", "no", "1", "0"])
    if "path" in k or "file" in k:
        return rng.choice(["/var/log/app.log", "/tmp/data", "/opt/app/storage", "./logs/output.log"])
    if "url" in k:
        return rng.choice(["https://api.example.com", "http://localhost:8080", "https://cdn.example.org"])
    if "key" in k or "secret" in k or "password" in k:
        return "".join(rng.choices(string.ascii_letters + string.digits, k=32))
    if "backend" in k or "provider" in k or "broker" in k:
        return rng.choice(["redis", "memcached", "rabbitmq", "postgresql", "sqlite", "s3"])
    if "version" in k:
        return f"v{rng.randint(1, 5)}"
    if "region" in k:
        return rng.choice(["us-east-1", "eu-west-1", "ap-southeast-1"])
    if "format" in k:
        return rng.choice(["json", "%(asctime)s %(levelname)s %(message)s", "text"])
    return f"value_{rng.randint(0, 999)}"


def generate_kv_samples(
    n: int = 10000, seed: int = 42
) -> Generator[dict, None, None]:
    """Generate INI/properties/key-value config files and XML snippets."""
    rng = random.Random(seed)

    count = 0
    while count < n:
        style = rng.randint(0, 2)

        if style == 0:
            # INI-style
            num_sections = rng.randint(2, 6)
            sections = rng.sample(_INI_SECTIONS, k=min(num_sections, len(_INI_SECTIONS)))
            lines = []
            for section in sections:
                lines.append(f"[{section}]")
                keys = _INI_KEYS.get(section, ["key1", "key2", "key3"])
                num_keys = rng.randint(2, len(keys))
                for key in rng.sample(keys, k=num_keys):
                    val = _random_ini_value(key, rng)
                    lines.append(f"{key} = {val}")
                lines.append("")
            text = "\n".join(lines)
            sub_type = "ini"

        elif style == 1:
            # Properties / key-value style
            lines = []
            num_entries = rng.randint(8, 40)
            # Optional comment header
            if rng.random() < 0.5:
                lines.append(f"# Configuration generated on 2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}")
                lines.append("")
            section = rng.choice(_INI_SECTIONS)
            keys = _INI_KEYS.get(section, ["key1", "key2"])
            for _ in range(num_entries):
                prefix = rng.choice(["app", "service", section, "config"])
                key = rng.choice(keys)
                sep = rng.choice(["=", ": ", " = "])
                val = _random_ini_value(key, rng)
                lines.append(f"{prefix}.{key}{sep}{val}")
            text = "\n".join(lines)
            sub_type = "key_value"

        else:
            # Simple XML config
            root_tag = rng.choice(["configuration", "settings", "config", "application"])
            lines = [f'<?xml version="1.0" encoding="UTF-8"?>', f"<{root_tag}>"]
            num_sections = rng.randint(2, 5)
            sections = rng.sample(_INI_SECTIONS, k=min(num_sections, len(_INI_SECTIONS)))
            for section in sections:
                lines.append(f"  <{section}>")
                keys = _INI_KEYS.get(section, ["key1", "key2"])
                num_keys = rng.randint(2, min(5, len(keys)))
                for key in rng.sample(keys, k=num_keys):
                    val = _random_ini_value(key, rng)
                    lines.append(f"    <{key}>{val}</{key}>")
                lines.append(f"  </{section}>")
            lines.append(f"</{root_tag}>")
            text = "\n".join(lines)
            sub_type = "xml"

        sample = emit_sample(text, "structured", sub_type, "generated_kv")
        if sample is not None:
            yield sample
            count += 1


# ---------------------------------------------------------------------------
# HuggingFace streaming sources (artifact)
# ---------------------------------------------------------------------------


def _get_hf_token() -> str | None:
    """Get HuggingFace token from environment if available."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _safe_stream(dataset_name: str, **kwargs):
    """Attempt to stream a HuggingFace dataset, returning None on failure."""
    try:
        import datasets

        token = _get_hf_token()
        return datasets.load_dataset(
            dataset_name, streaming=True, token=token, **kwargs
        )
    except Exception as e:
        kwarg_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items() if k != "token")
        print(
            f"WARNING: Failed to stream {dataset_name}({kwarg_str}): "
            f"{type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return None


def finepdfs_stream_candidates() -> list[dict]:
    """Return schema-tolerant candidate configs for FinePDFs."""
    return [
        {
            "dataset_name": "HuggingFaceFW/finepdfs_100BT",
            "kwargs": {
                "split": "train",
                "trust_remote_code": True,
            },
            "text_field": "text",
        },
        {
            "dataset_name": "HuggingFaceFW/finepdfs_100BT",
            "kwargs": {
                "split": "train",
                "data_files": {"train": "data/*.parquet"},
            },
            "text_field": "text",
        },
        {
            "dataset_name": "HuggingFaceFW/finepdfs_100BT",
            "kwargs": {
                "split": "train",
                "data_files": {"train": "data/000_00000.parquet"},
                "trust_remote_code": True,
            },
            "text_field": "text",
        },
    ]


def stack_dataset_candidates(kind: str) -> list[dict]:
    """Return stack dataset candidates for a specific sampling purpose.

    The Stack Dedup organizes data by programming language subdirectories
    under data/. Each language dir contains parquet shards with fields:
    content, size, lang, ext, avg_line_length, max_line_length,
    alphanum_fraction, hexsha, and repo metadata fields.
    """
    if kind == "configs":
        return [
            {
                "dataset_name": "bigcode/the-stack-dedup",
                "kwargs": {"split": "train", "data_dir": data_dir},
                "sub_type": sub_type,
            }
            for data_dir, sub_type in [
                ("data/json", "json"),
                ("data/yaml", "yaml"),
                ("data/toml", "toml"),
                ("data/ini", "ini"),
            ]
        ]
    raise ValueError(f"Unknown stack dataset kind: {kind}")


def load_first_available_stream(candidates: list[dict]):
    """Try candidate dataset configurations until one streams successfully."""
    for candidate in candidates:
        dataset = _safe_stream(candidate["dataset_name"], **candidate["kwargs"])
        if dataset is not None:
            return dataset, candidate
    return None, None


def sample_finepdfs(n: int = 8000) -> Generator[dict, None, None]:
    """Load from HuggingFaceFW/finepdfs_100BT via direct parquet read.

    Uses polars with hf:// protocol to bypass the datasets library schema
    unification issue — different parquet shards have different columns, but
    we only need the 'text' column which is present in all of them.
    """
    count = 0

    # Strategy 1: polars direct parquet via hf:// (bypasses schema mismatch)
    try:
        import polars as pl

        # Read a single shard to avoid downloading 358GB — one shard has ~200K rows
        # which is more than enough for our 10K target
        df = pl.read_parquet(
            "hf://datasets/HuggingFaceFW/finepdfs_100BT/data/000_00000.parquet",
            columns=["text"],
            storage_options={"token": _get_hf_token() or ""},
        )
        for text in df["text"]:
            if count >= n:
                return
            if text is None:
                continue
            sample = emit_sample(str(text), "artifact", "pdf_dump", "finepdfs")
            if sample is not None:
                yield sample
                count += 1
        if count >= n:
            return
    except Exception as e:
        print(f"WARNING: polars finepdfs read failed: {e}", file=sys.stderr)

    # Strategy 2: try the pre-shuffled variant (simpler schema)
    if count < n:
        ds = _safe_stream("HuggingFaceFW/finepdfs_100BT-shuffled", split="train")
        if ds is not None:
            for row in ds:
                if count >= n:
                    return
                text = row.get("text", "")
                sample = emit_sample(text, "artifact", "pdf_dump", "finepdfs_shuffled")
                if sample is not None:
                    yield sample
                    count += 1

    # Strategy 3: fallback to datasets streaming with trust_remote_code
    if count < n:
        ds, _candidate = load_first_available_stream(finepdfs_stream_candidates())
        if ds is not None:
            for row in ds:
                if count >= n:
                    return
                text = row.get("text", "")
                sample = emit_sample(text, "artifact", "pdf_dump", "finepdfs")
                if sample is not None:
                    yield sample
                    count += 1


def sample_scientific_papers(n: int = 5000) -> Generator[dict, None, None]:
    """Stream full-article text via ccdv/arxiv-summarization.

    The 'article' field contains full paper body text extracted from PDFs,
    which carries real extraction noise (broken equations, citation artifacts,
    section numbering remnants) — exactly the signal we need for pdf_dump.

    Two configs available: 'document' (full papers) and 'section' (per-section).
    We use 'document' for longer, more realistic pdf_dump samples.
    """
    # Try 'document' config first (full papers), then fall back to default
    ds = _safe_stream("ccdv/arxiv-summarization", name="document", split="train")
    if ds is None:
        ds = _safe_stream("ccdv/arxiv-summarization", split="train")
    if ds is None:
        return

    count = 0
    for row in ds:
        if count >= n:
            break
        text = row.get("article", "")
        sample = emit_sample(text, "artifact", "pdf_dump", "arxiv_summarization")
        if sample is not None:
            yield sample
            count += 1


def sample_boilerplate(n: int = 10000) -> Generator[dict, None, None]:
    """Stream LICENSE/NOTICE/COPYING files from bigcode/the-stack-dedup.

    The Stack organizes files by programming language, not by file type.
    LICENSE files appear across many language dirs. We stream several
    high-volume language dirs and filter for license-related paths.
    The 'text' language dir contains plain text files which is where
    most LICENSE/NOTICE files are categorized.
    """
    count = 0

    # Languages most likely to contain LICENSE/NOTICE/COPYING files
    # 'text' is the primary home; others have license files too
    boilerplate_lang_dirs = ["text", "markdown", "restructuredtext"]

    for lang in boilerplate_lang_dirs:
        if count >= n:
            break
        ds = _safe_stream(
            "bigcode/the-stack-dedup",
            data_dir=f"data/{lang}",
            split="train",
        )
        if ds is None:
            continue

        for row in ds:
            if count >= n:
                break
            content = row.get("content", "")
            # Check if it looks like a license/boilerplate file
            # Use both path-based and content-based detection
            path = (row.get("max_stars_repo_path", "") or "").upper()
            is_license_path = any(
                kw in path
                for kw in ["LICENSE", "NOTICE", "COPYING", "COPYRIGHT", "PATENTS"]
            )
            # Content-based detection: look for common license preambles
            content_upper = content[:500].upper() if content else ""
            is_license_content = any(
                kw in content_upper
                for kw in [
                    "MIT LICENSE",
                    "APACHE LICENSE",
                    "BSD LICENSE",
                    "GNU GENERAL PUBLIC",
                    "MOZILLA PUBLIC",
                    "PERMISSION IS HEREBY GRANTED",
                    "REDISTRIBUTION AND USE",
                    "THIS SOFTWARE IS PROVIDED",
                    "ALL RIGHTS RESERVED",
                    "CREATIVE COMMONS",
                ]
            )
            if is_license_path or is_license_content:
                sample = emit_sample(
                    content, "artifact", "boilerplate", f"the_stack_{lang}_licenses"
                )
                if sample is not None:
                    yield sample
                    count += 1

    # Supplement with synthetic if we didn't get enough
    if count < n:
        yield from _generate_synthetic_boilerplate(n - count)


_LICENSE_TEMPLATES = [
    # MIT
    "MIT License\n\nCopyright (c) {year} {author}\n\n"
    "Permission is hereby granted, free of charge, to any person obtaining a copy "
    "of this software and associated documentation files (the \"Software\"), to deal "
    "in the Software without restriction, including without limitation the rights "
    "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell "
    "copies of the Software, and to permit persons to whom the Software is "
    "furnished to do so, subject to the following conditions:\n\n"
    "The above copyright notice and this permission notice shall be included in all "
    "copies or substantial portions of the Software.\n\n"
    "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR "
    "IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, "
    "FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.",
    # Apache 2.0 header
    "Copyright {year} {author}\n\n"
    "Licensed under the Apache License, Version 2.0 (the \"License\");\n"
    "you may not use this file except in compliance with the License.\n"
    "You may obtain a copy of the License at\n\n"
    "    http://www.apache.org/licenses/LICENSE-2.0\n\n"
    "Unless required by applicable law or agreed to in writing, software "
    "distributed under the License is distributed on an \"AS IS\" BASIS, "
    "WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n"
    "See the License for the specific language governing permissions and "
    "limitations under the License.",
    # BSD 3-Clause
    "Copyright (c) {year}, {author}\nAll rights reserved.\n\n"
    "Redistribution and use in source and binary forms, with or without "
    "modification, are permitted provided that the following conditions are met:\n\n"
    "1. Redistributions of source code must retain the above copyright notice, "
    "this list of conditions and the following disclaimer.\n\n"
    "2. Redistributions in binary form must reproduce the above copyright notice, "
    "this list of conditions and the following disclaimer in the documentation "
    "and/or other materials provided with the distribution.\n\n"
    "3. Neither the name of the copyright holder nor the names of its "
    "contributors may be used to endorse or promote products derived from "
    "this software without specific prior written permission.",
    # GPL header
    "This program is free software: you can redistribute it and/or modify "
    "it under the terms of the GNU General Public License as published by "
    "the Free Software Foundation, either version 3 of the License, or "
    "(at your option) any later version.\n\n"
    "This program is distributed in the hope that it will be useful, "
    "but WITHOUT ANY WARRANTY; without even the implied warranty of "
    "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the "
    "GNU General Public License for more details.\n\n"
    "You should have received a copy of the GNU General Public License "
    "along with this program. If not, see <https://www.gnu.org/licenses/>.\n\n"
    "Copyright (C) {year} {author}",
]

_AUTHORS = [
    "The Project Contributors", "John Doe", "Example Corp",
    "Open Source Foundation", "The Developers", "Software Inc.",
    "Community Contributors", "Tech Solutions Ltd.",
]


def _generate_synthetic_boilerplate(
    n: int, seed: int = 42
) -> Generator[dict, None, None]:
    """Generate synthetic boilerplate from license templates."""
    rng = random.Random(seed)
    count = 0
    while count < n:
        template = rng.choice(_LICENSE_TEMPLATES)
        text = template.format(
            year=rng.randint(2000, 2025),
            author=rng.choice(_AUTHORS),
        )
        sample = emit_sample(text, "artifact", "boilerplate", "generated_boilerplate")
        if sample is not None:
            yield sample
            count += 1


# ---------------------------------------------------------------------------
# HuggingFace streaming sources (prose)
# ---------------------------------------------------------------------------


def sample_wikipedia_paragraphs(n: int = 15000) -> Generator[dict, None, None]:
    """Stream from agentlans/wikipedia-paragraphs.

    Field name is 'text' (not 'paragraph') — the dataset has two fields:
    'title' and 'text'. Dataset has ~21.8K rows total.
    """
    ds = _safe_stream("agentlans/wikipedia-paragraphs", split="train")
    if ds is None:
        return

    count = 0
    for row in ds:
        if count >= n:
            break
        text = row.get("text", "")
        sample = emit_sample(text, "prose", "plain", "wikipedia_paragraphs")
        if sample is not None:
            yield sample
            count += 1


def sample_wikipedia_full(n: int = 10000) -> Generator[dict, None, None]:
    """Stream from wikimedia/wikipedia, extracting random sections.

    The wikimedia/wikipedia dataset uses 'name' (not 'language') as the
    config parameter. The config name is '20231101.en' for English.
    """
    ds = _safe_stream(
        "wikimedia/wikipedia", name="20231101.en", split="train"
    )
    if ds is None:
        return

    rng = random.Random(42)
    count = 0
    for row in ds:
        if count >= n:
            break
        text = row.get("text", "")
        if not text:
            continue
        # Split into sections and pick a random one
        sections = text.split("\n\n")
        sections = [s.strip() for s in sections if len(s.strip()) >= 50]
        if not sections:
            continue
        section = rng.choice(sections)
        sample = emit_sample(section, "prose", "plain", "wikipedia_full")
        if sample is not None:
            yield sample
            count += 1


def sample_arxiv_abstracts(n: int = 10000) -> Generator[dict, None, None]:
    """Stream from gfissore/arxiv-abstracts-2021."""
    ds = _safe_stream("gfissore/arxiv-abstracts-2021", split="train")
    if ds is None:
        return

    count = 0
    for row in ds:
        if count >= n:
            break
        text = row.get("abstract", "")
        sample = emit_sample(text, "prose", "plain", "arxiv_abstracts")
        if sample is not None:
            yield sample
            count += 1


# ---------------------------------------------------------------------------
# HuggingFace streaming sources (structured)
# ---------------------------------------------------------------------------


def sample_stack_configs(n: int = 15000) -> Generator[dict, None, None]:
    """Stream real config files from bigcode/the-stack-dedup.

    Distributes the quota evenly across json, yaml, toml, and ini dirs
    so we get sub-type diversity, not just 15K JSON files.
    """
    candidates = stack_dataset_candidates("configs")
    per_lang = n // len(candidates)
    remainder = n - per_lang * len(candidates)

    total_count = 0
    for i, candidate in enumerate(candidates):
        lang_target = per_lang + (1 if i < remainder else 0)
        ds = _safe_stream(candidate["dataset_name"], **candidate["kwargs"])
        if ds is None:
            continue
        lang_count = 0
        for row in ds:
            if lang_count >= lang_target:
                break
            text = row.get("content", row.get("text", ""))
            sample = emit_sample(
                text,
                "structured",
                candidate["sub_type"],
                f"the_stack_{candidate['sub_type']}",
            )
            if sample is not None:
                yield sample
                lang_count += 1
                total_count += 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sample real-world text from HuggingFace datasets"
    )
    parser.add_argument(
        "--output",
        default="data/real_samples_v2.jsonl",
        help="Output JSONL file path (default: data/real_samples_v2.jsonl)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print source plan without downloading",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    if args.dry_run:
        print(format_source_plan(SOURCE_PLANS))
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_target = sum(plan.target_count for plan in SOURCE_PLANS)
    total_written = 0

    with open(output_path, "w") as f:
        with tqdm(total=total_target, desc="Sampling", unit="samples") as pbar:
            for plan in SOURCE_PLANS:
                func = globals()[plan.name]
                # Pass seed to generators that accept it
                if plan.name.startswith("generate_"):
                    gen = func(n=plan.target_count, seed=args.seed)
                else:
                    gen = func(n=plan.target_count)

                source_count = 0
                try:
                    for sample in gen:
                        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        source_count += 1
                        total_written += 1
                        pbar.update(1)
                except Exception as e:
                    print(
                        f"\nERROR: {plan.name} failed after {source_count}/{plan.target_count} "
                        f"samples: {type(e).__name__}: {e}",
                        file=sys.stderr,
                    )

                shortfall = plan.target_count - source_count
                if shortfall > 0:
                    # Advance progress bar for missed samples so total stays accurate
                    pbar.update(shortfall)
                    level = "ERROR" if source_count == 0 else "WARNING"
                    print(
                        f"\n{level}: {plan.name} yielded {source_count}/{plan.target_count} "
                        f"samples (shortfall: {shortfall})",
                        file=sys.stderr,
                    )

                pbar.set_postfix(source=plan.name, n=source_count)

    # Summary with per-source accounting
    print(f"\nWrote {total_written:,} samples to {output_path}")
    if total_written < total_target:
        print(
            f"WARNING: {total_target - total_written:,} samples short of "
            f"{total_target:,} target",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
