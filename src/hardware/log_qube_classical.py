"""Log telemetry from arduino/qube_classical_runner over serial to CSV."""

from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path

import serial
from serial.tools import list_ports

LINE_PATTERN = re.compile(
    r"mode=(?P<mode>\w+)\s+"
    r"theta=(?P<theta>[-+]?\d*\.?\d+)\s+"
    r"alpha=(?P<alpha>[-+]?\d*\.?\d+)\s+"
    r"thetaDot=(?P<theta_dot>[-+]?\d*\.?\d+)\s+"
    r"alphaDot=(?P<alpha_dot>[-+]?\d*\.?\d+)\s+"
    r"voltage=(?P<voltage>[-+]?\d*\.?\d+)\s+"
    r"ampFault=(?P<amp_fault>\d+)\s+"
    r"stall=(?P<stall>\d+)\s+"
    r"stallError=(?P<stall_error>\d+)"
)


def parse_line(raw_line: str) -> dict[str, object] | None:
    match = LINE_PATTERN.search(raw_line.strip())
    if match is None:
        return None
    groups = match.groupdict()
    return {
        "mode": groups["mode"],
        "theta": float(groups["theta"]),
        "alpha": float(groups["alpha"]),
        "theta_dot": float(groups["theta_dot"]),
        "alpha_dot": float(groups["alpha_dot"]),
        "voltage": float(groups["voltage"]),
        "amp_fault": int(groups["amp_fault"]),
        "stall": int(groups["stall"]),
        "stall_error": int(groups["stall_error"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default="COM3")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--timeout", type=float, default=0.2)
    parser.add_argument("--startup-grace", type=float, default=5.0)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--echo-raw", action="store_true")
    parser.add_argument("--list-ports", action="store_true")
    args = parser.parse_args()

    if args.list_ports:
        for port in list_ports.comports():
            print(f"{port.device} {port.description}")
        return

    csv_path = args.csv
    if csv_path is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = Path(f"results/qube_classical_log_{stamp}.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "time_s",
        "parsed",
        "mode",
        "theta",
        "alpha",
        "theta_dot",
        "alpha_dot",
        "voltage",
        "amp_fault",
        "stall",
        "stall_error",
        "raw_line",
    ]

    start = time.monotonic()
    deadline = start + args.startup_grace + args.duration
    first_data_time: float | None = None
    rows_written = 0
    parsed_rows = 0

    with serial.Serial(args.port, args.baudrate, timeout=args.timeout) as ser, csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        print(f"Logging from {args.port} at {args.baudrate} baud to {csv_path}")
        print(f"Duration: {args.duration:.1f} s, startup grace: {args.startup_grace:.1f} s")

        while time.monotonic() < deadline:
            raw_bytes = ser.readline()
            if not raw_bytes:
                continue

            raw_line = raw_bytes.decode("utf-8", errors="replace").strip()
            if first_data_time is None:
                first_data_time = time.monotonic()
            if args.echo_raw:
                print(raw_line)

            parsed = parse_line(raw_line)
            time_s = (time.monotonic() - first_data_time) if first_data_time is not None else 0.0
            row = {
                "time_s": round(time_s, 6),
                "parsed": int(parsed is not None),
                "mode": "",
                "theta": "",
                "alpha": "",
                "theta_dot": "",
                "alpha_dot": "",
                "voltage": "",
                "amp_fault": "",
                "stall": "",
                "stall_error": "",
                "raw_line": raw_line,
            }
            if parsed is not None:
                row.update(parsed)
                parsed_rows += 1
            writer.writerow(row)
            file.flush()
            rows_written += 1

            if parsed is not None:
                print(
                    f"{time_s:7.3f}s mode={row['mode']} "
                    f"theta={float(row['theta']):+.4f} alpha={float(row['alpha']):+.4f} "
                    f"theta_dot={float(row['theta_dot']):+.4f} alpha_dot={float(row['alpha_dot']):+.4f} "
                    f"voltage={float(row['voltage']):+.4f}"
                )
            else:
                print(f"{time_s:7.3f}s raw={raw_line}")

            if first_data_time is not None and (time.monotonic() - first_data_time) >= args.duration:
                break

    print(f"Wrote {rows_written} rows to {csv_path} ({parsed_rows} parsed)")
    if rows_written == 0:
        print("No serial data received. Check COM port, cable, and that no other program has the port open.")
    elif parsed_rows == 0:
        print("Serial data was received, but no telemetry lines matched the parser. Use --echo-raw and inspect raw_line.")


if __name__ == "__main__":
    main()
