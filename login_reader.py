"""
login_reader.py
Reads REAL failed-login events from the computer this runs on.

  Windows : Security event log, event ID 4625 ("An account failed to log on").
            Needs the terminal to be run as Administrator.
  Linux   : /var/log/auth.log (or /var/log/secure), "Failed password" lines
            written by the SSH server.
  macOS   : not supported (returns nothing).

Each event becomes {"timestamp", "src_ip", "user"}. A failed login typed at
the machine itself has no network address, so it is recorded as
"local console".
"""

import platform
import re
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

LOCAL_NAMES = {"-", "", "127.0.0.1", "::1", "localhost"}
EVENT_NS = {"e": "http://schemas.microsoft.com/win/2004/08/events/event"}


def clean_ip(ip):
    ip = (ip or "").strip()
    if ip.startswith("::ffff:"):          # IPv4 written in IPv6 form
        ip = ip[7:]
    return "local console" if ip in LOCAL_NAMES else ip


def parse_windows_time(text):
    """'2026-09-18T10:12:33.1234567Z' (UTC) -> local datetime without timezone."""
    text = re.sub(r"(\.\d{6})\d*", r"\1", text).replace("Z", "+00:00")
    return datetime.fromisoformat(text).astimezone().replace(tzinfo=None)


class LoginReader:
    def __init__(self, linux_log=None):
        self.system = platform.system()
        self.seen_ids = set()             # Windows: event record IDs already read
        self.linux_log = Path(linux_log) if linux_log else None
        self.linux_offset = None          # Linux: how far into the file we have read
        self.status = "starting"

        if self.system == "Linux" and self.linux_log is None:
            for candidate in ("/var/log/auth.log", "/var/log/secure"):
                if Path(candidate).exists():
                    self.linux_log = Path(candidate)
                    break

    # -- Windows --------------------------------------------------------------
    def _poll_windows(self):
        query = "*[System[(EventID=4625) and TimeCreated[timediff(@SystemTime) <= 300000]]]"
        result = subprocess.run(
            ["wevtutil", "qe", "Security", f"/q:{query}", "/f:xml", "/rd:true", "/c:500"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            self.status = ("Cannot read the Windows Security log. Run the terminal as "
                           "Administrator.")
            return []
        self.status = "reading Windows Security log (event 4625)"

        events = []
        root = ET.fromstring(f"<Events>{result.stdout}</Events>")
        for ev in root.findall("e:Event", EVENT_NS):
            record_id = ev.findtext("e:System/e:EventRecordID", namespaces=EVENT_NS)
            if record_id in self.seen_ids:
                continue
            self.seen_ids.add(record_id)
            data = {d.get("Name"): (d.text or "") for d in ev.findall("e:EventData/e:Data", EVENT_NS)}
            when = ev.find("e:System/e:TimeCreated", EVENT_NS).get("SystemTime")
            events.append({
                "timestamp": parse_windows_time(when),
                "src_ip": clean_ip(data.get("IpAddress")),
                "user": data.get("TargetUserName", "?"),
            })
        return events

    # -- Linux ----------------------------------------------------------------
    def _poll_linux(self):
        if self.linux_log is None or not self.linux_log.exists():
            self.status = "no auth log found (install/enable an SSH server to get login events)"
            return []
        try:
            with open(self.linux_log, errors="ignore") as f:
                if self.linux_offset is None:          # first call: skip old history
                    f.seek(0, 2)
                    self.linux_offset = f.tell()
                    self.status = f"reading {self.linux_log}"
                    return []
                f.seek(self.linux_offset)
                lines = f.readlines()
                self.linux_offset = f.tell()
        except PermissionError:
            self.status = f"no permission to read {self.linux_log} (run with sudo)"
            return []

        self.status = f"reading {self.linux_log}"
        pattern = re.compile(r"Failed password for (?:invalid user )?(\S+) from (\S+)")
        events = []
        for line in lines:
            m = pattern.search(line)
            if m:
                events.append({"timestamp": datetime.now(), "src_ip": clean_ip(m.group(2)),
                               "user": m.group(1)})
        return events

    # -- public ---------------------------------------------------------------
    def poll(self):
        """Return failed logins that happened since the last call."""
        try:
            if self.system == "Windows":
                return self._poll_windows()
            if self.system == "Linux":
                return self._poll_linux()
            self.status = f"login monitoring not supported on {self.system}"
            return []
        except Exception as exc:                      # never crash the monitor
            self.status = f"login reader error: {exc}"
            return []
