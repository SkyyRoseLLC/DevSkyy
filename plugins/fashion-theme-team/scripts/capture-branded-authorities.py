#!/usr/bin/env python3
"""Capture TLS-authenticated primary-authority metadata for branded skills."""

from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import subprocess
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

MAX_CAPTURE_BYTES = 10_000_000
PUBLISHER_HOSTS = {
    "WooCommerce": {"developer.woocommerce.com"},
    "Google Search Central": {"developers.google.com"},
    "Federal Communications Commission": {"docs.fcc.gov"},
    "Schema.org": {"schema.org"},
    "Yahoo": {"senders.yahooinc.com"},
    "Google": {"support.google.com"},
    "Meta": {"transparency.meta.com"},
    "Microsoft Bing Webmaster": {"www.bing.com"},
    "U.S. Copyright Office": {"www.copyright.gov"},
    "U.S. Consumer Product Safety Commission": {"www.cpsc.gov"},
    "Federal Trade Commission": {"www.ftc.gov"},
    "LinkedIn": {"www.linkedin.com"},
    "TikTok": {"www.tiktok.com"},
    "United States Patent and Trademark Office": {"www.uspto.gov"},
    "W3C": {"www.w3.org"},
}
PUBLISHER_MARKERS = {
    "WooCommerce": [b"woocommerce"],
    "Google Search Central": [b"google search"],
    "Federal Communications Commission": [b"federal communications commission"],
    "Schema.org": [b"schema.org"],
    "Yahoo": [b"yahoo"],
    "Google": [b"google"],
    "Meta": [b"meta"],
    "Microsoft Bing Webmaster": [b"bing"],
    "U.S. Copyright Office": [b"copyright"],
    "U.S. Consumer Product Safety Commission": [
        b"consumer product safety commission",
        b"cpsc",
    ],
    "Federal Trade Commission": [b"federal trade commission", b"ftc"],
    "LinkedIn": [b"linkedin"],
    "TikTok": [b"tiktok"],
    "United States Patent and Trademark Office": [
        b"patent and trademark office",
        b"uspto",
    ],
    "W3C": [b"w3c", b"web content accessibility guidelines"],
}


def authenticated_capture(
    publisher: str, requested_url: str, final_url: str, body: bytes
) -> dict[str, object]:
    requested_host = (urlparse(requested_url).hostname or "").lower()
    final_host = (urlparse(final_url).hostname or "").lower()
    allowed_hosts = PUBLISHER_HOSTS.get(publisher, set())
    if requested_host not in allowed_hosts or final_host not in allowed_hosts:
        raise ValueError(f"publisher host mismatch: requested={requested_host} final={final_host}")
    if len(body) > MAX_CAPTURE_BYTES:
        raise ValueError(f"authority body exceeds {MAX_CAPTURE_BYTES} bytes")
    lowered = body.lower()
    if publisher != "Federal Communications Commission" and not any(
        marker in lowered for marker in PUBLISHER_MARKERS[publisher]
    ):
        raise ValueError("expected publisher/document identity marker absent")
    if publisher == "Federal Communications Commission" and not body.startswith(b"%PDF"):
        raise ValueError("expected FCC PDF identity absent")
    return {
        "status": "VERIFIED_PINNED_PRIMARY_ARTIFACT",
        "requested_host": requested_host,
        "final_host": final_host,
        "host_allowlist": sorted(allowed_hosts),
        "captured_bytes": len(body),
        "capture_complete": True,
        "content_sha256": hashlib.sha256(body).hexdigest(),
        "identity_markers_verified": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorities", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=20)
    args = parser.parse_args()
    source = json.loads(args.authorities.read_text(encoding="utf-8"))
    requested: dict[str, str] = {}
    for domain in source["domains"]:
        for authority in domain["primary_authorities"]:
            if authority["url"].startswith("https://"):
                requested[authority["url"]] = authority["publisher"]
    context = ssl.create_default_context()
    captures = []
    for url, publisher in sorted(requested.items()):
        record = {"publisher": publisher, "requested_url": url}
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "SkyyRose-Authority-Validator/1.0",
                    "Accept-Encoding": "identity",
                },
            )
            with urllib.request.urlopen(request, timeout=args.timeout, context=context) as response:
                body = response.read(MAX_CAPTURE_BYTES + 1)
                record.update(
                    {
                        "http_status": response.status,
                        "final_url": response.geturl(),
                        "content_type": response.headers.get("Content-Type", "UNKNOWN"),
                        "etag": response.headers.get("ETag"),
                        "last_modified": response.headers.get("Last-Modified"),
                    }
                )
                record.update(authenticated_capture(publisher, url, response.geturl(), body))
        except Exception as exc:  # use a second CA-verified client before blocking
            try:
                completed = subprocess.run(
                    [
                        "curl",
                        "--fail",
                        "--location",
                        "--compressed",
                        "--silent",
                        "--show-error",
                        "--max-time",
                        str(args.timeout),
                        "--max-filesize",
                        str(MAX_CAPTURE_BYTES),
                        "--output",
                        "-",
                        "--write-out",
                        "\n%{http_code}\n%{url_effective}",
                        url,
                    ],
                    check=True,
                    capture_output=True,
                    timeout=args.timeout + 5,
                )
                body, status, final_url = completed.stdout.rsplit(b"\n", 2)
                record.update(
                    {
                        "http_status": int(status),
                        "final_url": final_url.decode("utf-8"),
                        "content_type": "UNKNOWN_CURL_FALLBACK",
                        "etag": None,
                        "last_modified": None,
                        "client": "curl-ca-verified-fallback",
                        "primary_client_error": f"{type(exc).__name__}: {str(exc)[:300]}",
                    }
                )
                record.update(
                    authenticated_capture(publisher, url, final_url.decode("utf-8"), body)
                )
            except Exception as fallback_exc:
                record.update(
                    {
                        "status": "BLOCKED",
                        "error_type": type(fallback_exc).__name__,
                        "error": str(fallback_exc)[:500],
                        "primary_client_error": f"{type(exc).__name__}: {str(exc)[:300]}",
                    }
                )
        captures.append(record)
    now = datetime.now(UTC).replace(microsecond=0).isoformat()
    report = {
        "schema_version": "1.0.0",
        "captured_at": now,
        "method": "CA-verified TLS; publisher and final hosts pinned; complete artifact up to 10 MB hashed; publisher identity marker checked",
        "claim_boundary": "Artifact authentication only; every material claim still requires a current claim-specific anchor and applicability check at use time.",
        "requested_count": len(captures),
        "verified_count": sum(
            item["status"] == "VERIFIED_PINNED_PRIMARY_ARTIFACT" for item in captures
        ),
        "blocked_count": sum(item["status"] == "BLOCKED" for item in captures),
        "captures": captures,
        "verdict": (
            "PASS"
            if all(item["status"] == "VERIFIED_PINNED_PRIMARY_ARTIFACT" for item in captures)
            else "BLOCKED"
        ),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "requested_count",
                    "verified_count",
                    "blocked_count",
                    "verdict",
                )
            }
        )
    )
    return 0 if report["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
