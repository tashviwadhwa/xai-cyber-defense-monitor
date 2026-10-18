"""
ip_reputation.py  (Feature 2 - IP reputation / geolocation lookup)

When an alert fires, an analyst wants to know more about the source than its
address: where it is and whether it is already known for abuse. This module
adds that context to each alert.

For a PUBLIC source IP it queries two free services:
  - ip-api.com        -> country and network owner (no key required)
  - AbuseIPDB         -> an abuse-confidence score, IF an API key is set in
                         the ABUSEIPDB_KEY environment variable (free tier).
If no key is set, the AbuseIPDB part is skipped and only geolocation is shown.

PRIVATE / local addresses (10.x, 172.16-31.x, 192.168.x, 127.x, and the
reserved lab ranges 203.0.113.x / 198.51.100.x) are never sent to any
service: they are not routable on the internet, so a lookup is meaningless
and would leak internal structure. These return a clear "private / lab
address" result instead.

Results are cached in memory and persisted to a small JSON file so the same
IP is not looked up twice. Every network call has a short timeout and fails
safe: if a service is unreachable, the alert still shows, just without the
extra context.

Standalone test:  python ip_reputation.py 8.8.8.8
"""

import ipaddress
import json
import os
import time
from pathlib import Path

import requests

from common import ARTIFACTS

CACHE_FILE = ARTIFACTS / "live" / "ip_reputation_cache.json"
TIMEOUT = 4            # seconds per request; keep the monitor responsive
ABUSE_KEY = os.environ.get("ABUSEIPDB_KEY", "").strip()

# Reserved documentation ranges used for the lab attackers, treated as local.
_LAB_NETS = [ipaddress.ip_network("203.0.113.0/24"),
             ipaddress.ip_network("198.51.100.0/24")]

_cache = {}
if CACHE_FILE.exists():
    try:
        _cache = json.loads(CACHE_FILE.read_text())
    except Exception:
        _cache = {}


def _save_cache():
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps(_cache, indent=2))
    except Exception:
        pass


def is_public(ip):
    """True only for a normal, internet-routable address."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
        return False
    if any(addr in net for net in _LAB_NETS):
        return False
    return True


def _geo(ip):
    r = requests.get(f"https://ip-api.com/json/{ip}",
                     params={"fields": "status,country,isp,org"}, timeout=TIMEOUT)
    d = r.json()
    if d.get("status") != "success":
        return {}
    return {"country": d.get("country", "?"),
            "network": d.get("org") or d.get("isp") or "?"}


def _abuse(ip):
    if not ABUSE_KEY:
        return {}
    r = requests.get("https://api.abuseipdb.com/api/v2/check",
                     headers={"Key": ABUSE_KEY, "Accept": "application/json"},
                     params={"ipAddress": ip, "maxAgeInDays": 90}, timeout=TIMEOUT)
    d = r.json().get("data", {})
    return {"abuse_score": d.get("abuseConfidenceScore"),
            "total_reports": d.get("totalReports"),
            "last_reported": d.get("lastReportedAt")}


def lookup(ip):
    """
    Return a reputation dict for one IP. Keys: scope ('public'|'local'),
    and for public IPs: country, network, and (if a key is set) abuse_score,
    total_reports. Always returns a 'summary' one-liner for the dashboard.
    """
    if ip in _cache:
        return _cache[ip]

    if not is_public(ip):
        result = {"scope": "local",
                  "summary": "Private / lab address - not looked up"}
        _cache[ip] = result
        _save_cache()
        return result

    result = {"scope": "public"}
    try:
        result.update(_geo(ip))
    except Exception:
        pass
    try:
        result.update(_abuse(ip))
    except Exception:
        pass

    # Build a readable one-liner from whatever we got.
    parts = []
    if result.get("country"):
        parts.append(result["country"])
    if result.get("network"):
        parts.append(result["network"])
    score = result.get("abuse_score")
    if score is not None:
        verdict = ("known malicious" if score >= 50
                   else "some reports" if score > 0 else "clean")
        parts.append(f"abuse {score}/100 ({verdict})")
    elif not ABUSE_KEY:
        parts.append("abuse score: set ABUSEIPDB_KEY to enable")
    result["summary"] = " - ".join(parts) if parts else "lookup unavailable"

    _cache[ip] = result
    _save_cache()
    return result


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "8.8.8.8"
    print(f"{target}:")
    print(json.dumps(lookup(target), indent=2))
