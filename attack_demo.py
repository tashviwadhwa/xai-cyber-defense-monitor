"""
attack_demo.py - run this on the SECOND laptop to attack the FIRST one
(the one running live_monitor.py). Both must be on the same Wi-Fi/hotspot.

ONLY USE THIS AGAINST YOUR OWN DEVICES. Scanning or brute-forcing a
machine you don't own or have permission to test is illegal.

    python attack_demo.py 192.168.43.10 scan      port scan (ports 1-1024)
    python attack_demo.py 192.168.43.10 brute     failed Windows logins over SMB
    python attack_demo.py 192.168.43.10 flood     10 s burst of UDP packets
    python attack_demo.py 192.168.43.10 all       scan, then brute + scan together

Find the first laptop's IP: it is printed when live_monitor.py starts
(or run `ipconfig` on it and look for "IPv4 Address").
"""

import platform
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor


def scan(target, ports=range(1, 1025)):
    """TCP connect scan: try to open a connection to every port."""
    print(f"Scanning {target} ports {ports.start}-{ports.stop - 1} ...")

    def try_port(port):
        with socket.socket() as s:
            s.settimeout(0.5)
            return port if s.connect_ex((target, port)) == 0 else None

    with ThreadPoolExecutor(max_workers=100) as pool:
        open_ports = [p for p in pool.map(try_port, ports) if p]
    print(f"  done. Open ports found: {open_ports or 'none'}")


def brute(target, attempts=15):
    """Wrong-password logins over SMB. Each one creates a real event 4625
    on the target. Uses a made-up username so no real account gets locked."""
    if platform.system() != "Windows":
        print("The brute-force demo uses Windows' `net use`. Run it from a Windows laptop, "
              "or type a wrong password at the monitored laptop's lock screen instead.")
        return
    print(f"Trying {attempts} wrong passwords against \\\\{target} ...")
    for i in range(attempts):
        subprocess.run(["net", "use", rf"\\{target}\IPC$", f"/user:demo_attacker", f"wrong{i}"],
                       capture_output=True)
        print(f"  attempt {i + 1}/{attempts}", end="\r")
    print("\n  done.")


def flood(target, seconds=10, port=9999):
    """Send a burst of small UDP packets (about 1,000 per second)."""
    print(f"Flooding {target}:{port} with UDP for {seconds}s ...")
    end = time.time() + seconds
    sent = 0
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        while time.time() < end:
            s.sendto(b"x" * 64, (target, port))
            sent += 1
            time.sleep(0.001)
    print(f"  done, {sent:,} packets sent.")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[2] not in ("scan", "brute", "flood", "all"):
        print(__doc__)
        sys.exit(1)
    target, action = sys.argv[1], sys.argv[2]
    if action == "scan":
        scan(target)
    elif action == "brute":
        brute(target)
    elif action == "flood":
        flood(target)
    else:
        scan(target)
        time.sleep(6)
        brute(target)
        scan(target, range(1, 300))    # keep traffic going so both signals line up
