"""Durable outputs: CSV for spreadsheets, SQLite for queries, JSON for tools."""
from __future__ import annotations

import csv
import datetime as dt
import json
import sqlite3
from pathlib import Path


def bucket_label(start_time, start_seconds):
    if start_time is None:
        return None
    return (start_time + dt.timedelta(seconds=start_seconds)).isoformat(timespec="seconds")


def write_flow_outputs(output_dir, crossings, buckets, zones, fps, start_time=None):
    output_dir = Path(output_dir)
    for row in buckets:
        row["start_time"] = bucket_label(start_time, row["start_seconds"])
    (output_dir / "crossings.json").write_text(json.dumps(crossings, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "totals.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bucket", "start_time", "start_seconds", "end_seconds", "in", "out", "occupancy_end"])
        for row in buckets:
            writer.writerow([row["bucket"], row["start_time"] or "", row["start_seconds"], row["end_seconds"], row["in"], row["out"], row["occupancy_end"]])
    db_path = output_dir / "counts.sqlite"
    if db_path.exists():
        db_path.unlink()
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("CREATE TABLE crossings (id INTEGER PRIMARY KEY, frame INTEGER NOT NULL, seconds REAL NOT NULL, "
                           "timestamp TEXT, track_id INTEGER NOT NULL, line TEXT NOT NULL, direction TEXT NOT NULL, x REAL, y REAL)")
        connection.executemany("INSERT INTO crossings VALUES (?,?,?,?,?,?,?,?,?)",
                               [(i + 1, c["frame"], c["frame"] / fps, bucket_label(start_time, c["frame"] / fps), c["track_id"], c["line"],
                                 c["direction"], c["point"][0], c["point"][1]) for i, c in enumerate(crossings)])
        connection.execute("CREATE TABLE buckets (bucket INTEGER PRIMARY KEY, start_time TEXT, start_seconds REAL, end_seconds REAL, "
                           "entered INTEGER, exited INTEGER, occupancy_end INTEGER)")
        connection.executemany("INSERT INTO buckets VALUES (?,?,?,?,?,?,?)",
                               [(r["bucket"], r["start_time"], r["start_seconds"], r["end_seconds"], r["in"], r["out"], r["occupancy_end"]) for r in buckets])
        connection.execute("CREATE TABLE zones (name TEXT PRIMARY KEY, frames INTEGER, peak INTEGER, mean REAL, occupied_fraction REAL, area_m2 REAL)")
        connection.executemany("INSERT INTO zones VALUES (?,?,?,?,?,?)",
                               [(name, z["frames"], z["peak"], z["mean"], z["occupied_fraction"], z["area_m2"]) for name, z in zones.items()])
        connection.commit()
    finally:
        connection.close()  # an open handle keeps the file locked on Windows


def parse_start_time(text):
    if not text:
        return None
    return dt.datetime.fromisoformat(text)
