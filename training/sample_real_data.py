#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["datasets", "tqdm"]
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
from pathlib import Path
from typing import Generator

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Core helper
# ---------------------------------------------------------------------------


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
        print(f"WARNING: Failed to stream {dataset_name}: {e}", file=sys.stderr)
        return None


def sample_finepdfs(n: int = 8000) -> Generator[dict, None, None]:
    """Stream from HuggingFaceFW/finepdfs_100BT for PDF dump text."""
    ds = _safe_stream("HuggingFaceFW/finepdfs_100BT", split="train")
    if ds is None:
        return

    count = 0
    for row in ds:
        if count >= n:
            break
        text = row.get("text", "")
        sample = emit_sample(text, "artifact", "pdf_dump", "finepdfs")
        if sample is not None:
            yield sample
            count += 1


def sample_scientific_papers(n: int = 7000) -> Generator[dict, None, None]:
    """Stream from armanc/scientific_papers (arxiv) for PDF dump text."""
    ds = _safe_stream("armanc/scientific_papers", name="arxiv", split="train")
    if ds is None:
        return

    count = 0
    for row in ds:
        if count >= n:
            break
        text = row.get("article", "")
        sample = emit_sample(text, "artifact", "pdf_dump", "scientific_papers")
        if sample is not None:
            yield sample
            count += 1


def sample_boilerplate(n: int = 10000) -> Generator[dict, None, None]:
    """Stream from bigcode/the-stack for LICENSE/NOTICE files."""
    # Try the-stack first, then the-stack-dedup
    for dataset_name in ["bigcode/the-stack", "bigcode/the-stack-dedup"]:
        ds = _safe_stream(dataset_name, split="train", data_dir="data/license")
        if ds is not None:
            break
    else:
        # If both fail, generate synthetic boilerplate
        yield from _generate_synthetic_boilerplate(n)
        return

    count = 0
    for row in ds:
        if count >= n:
            break
        path = row.get("path", "").upper()
        if any(kw in path for kw in ["LICENSE", "NOTICE", "COPYING", "COPYRIGHT"]):
            text = row.get("content", "")
            sample = emit_sample(text, "artifact", "boilerplate", "the_stack_licenses")
            if sample is not None:
                yield sample
                count += 1

    # If we didn't get enough from HF, supplement with synthetic
    if count < n:
        for s in _generate_synthetic_boilerplate(n - count):
            yield s


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
    """Stream from agentlans/wikipedia-paragraphs."""
    ds = _safe_stream("agentlans/wikipedia-paragraphs", split="train")
    if ds is None:
        return

    count = 0
    for row in ds:
        if count >= n:
            break
        text = row.get("paragraph", "")
        sample = emit_sample(text, "prose", "plain", "wikipedia_paragraphs")
        if sample is not None:
            yield sample
            count += 1


def sample_wikipedia_full(n: int = 10000) -> Generator[dict, None, None]:
    """Stream from wikimedia/wikipedia, extracting random sections."""
    ds = _safe_stream(
        "wikimedia/wikipedia", language="20231101.en", split="train"
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
    """Stream from bigcode/the-stack for config files."""
    ext_to_subtype = {
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
    }

    for dataset_name in ["bigcode/the-stack", "bigcode/the-stack-dedup"]:
        ds = _safe_stream(dataset_name, split="train")
        if ds is not None:
            break
    else:
        return

    count = 0
    for row in ds:
        if count >= n:
            break
        path = row.get("path", "")
        for ext, sub_type in ext_to_subtype.items():
            if path.endswith(ext):
                text = row.get("content", "")
                sample = emit_sample(
                    text, "structured", sub_type, "the_stack_configs"
                )
                if sample is not None:
                    yield sample
                    count += 1
                break


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# All source functions with their target counts
ALL_SOURCES = [
    # Artifact sources (~48K)
    ("sample_finepdfs", 8000),
    ("sample_scientific_papers", 7000),
    ("sample_boilerplate", 10000),
    ("generate_skip_samples", 8000),
    ("generate_ocr_garbage", 15000),
    # Prose sources (~44K)
    ("sample_wikipedia_paragraphs", 15000),
    ("sample_wikipedia_full", 10000),
    ("sample_arxiv_abstracts", 10000),
    ("generate_prose_variants", 9000),
    # Structured sources (~46K)
    ("sample_stack_configs", 15000),
    ("generate_csv_samples", 15000),
    ("generate_log_samples", 6000),
    ("generate_kv_samples", 10000),
]


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
        total = sum(count for _, count in ALL_SOURCES)
        print(f"Data sourcing plan ({total:,} total samples):")
        print()
        for name, count in ALL_SOURCES:
            print(f"  {name}: {count:,}")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_target = sum(count for _, count in ALL_SOURCES)
    total_written = 0

    with open(output_path, "w") as f:
        with tqdm(total=total_target, desc="Sampling", unit="samples") as pbar:
            for source_name, target_count in ALL_SOURCES:
                func = globals()[source_name]
                # Pass seed to generators that accept it
                if source_name.startswith("generate_"):
                    gen = func(n=target_count, seed=args.seed)
                else:
                    gen = func(n=target_count)

                source_count = 0
                for sample in gen:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    source_count += 1
                    total_written += 1
                    pbar.update(1)

                pbar.set_postfix(source=source_name, n=source_count)

    print(f"\nWrote {total_written:,} samples to {output_path}")


if __name__ == "__main__":
    main()
