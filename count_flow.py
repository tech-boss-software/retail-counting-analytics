"""Count people crossing gate lines in a video and measure zone occupancy; write CSV, SQLite and a dashboard."""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

from counting import LineCounter, ZoneOccupancy, bucket_totals, ignored, load_config
from detectors import load_detector
from evaluate_flow import evaluate
from frames import VideoSink, open_source
from geometry import box_anchor
from homography import FloorPlane
from render import draw_flow_frame, floor_plan_svg, flow_dashboard
from store import parse_start_time, write_flow_outputs
from tracker import Tracker


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="video file or folder of frames")
    parser.add_argument("--config", required=True, help="JSON with lines, zones and optional floor calibration")
    parser.add_argument("--detector", required=True, help="onnx:<model.onnx> or torchvision-person")
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--fps", type=float, default=None, help="frame rate for image folders")
    parser.add_argument("--score", type=float, default=0.5, help="tracker high-confidence threshold")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--bucket", type=float, default=3600.0, help="totals bucket length in seconds (hourly by default)")
    parser.add_argument("--start-time", default=None, help="ISO timestamp of frame 0, used to label buckets and crossings")
    parser.add_argument("--snapshots", default="", help="comma-separated frame indices to save as annotated PNGs")
    parser.add_argument("--truth", default=None, help="truth.json from synthetic.py to score the crossings")
    args = parser.parse_args(argv)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(json.loads(Path(args.config).read_text(encoding="utf-8")))
    plane = FloorPlane(config["floor"]) if config["space"] == "floor" else None
    counter = LineCounter(config["lines"], config["margin"])
    zones = ZoneOccupancy(config["zones"], floor_units=plane.units if plane else None)
    tracker = Tracker(high_score=args.score, low_score=min(0.1, args.score))
    detector = load_detector(args.detector, args.threads, args.cache_dir)
    person_class = next((k for k, v in detector.labels.items() if v == "person"), None)
    if person_class is None:
        raise ValueError(f"detector has no 'person' class; its classes are {sorted(detector.labels.values())}")
    frames, fps, size, description = open_source(args.input, args.fps)
    annotated = VideoSink(output / "annotated.mp4", fps, size)
    wanted = {int(v) for v in args.snapshots.split(",") if v.strip()}
    snapshots, trails, floor_trails, last_seen = [], {}, {}, {}
    detector_ms, frame_ms = [], []
    started = time.perf_counter()
    frame_index = -1
    for frame_index, image in frames:
        if args.max_frames and frame_index >= args.max_frames:
            frame_index -= 1
            break
        t0 = time.perf_counter()
        detections = detector.detect(image)
        t1 = time.perf_counter()
        tracks = tracker.update([d for d in detections if d["class_id"] == person_class], frame_index)
        anchors, points = {}, {}
        kept = []
        for track in tracks:
            anchor = box_anchor(track["box_xyxy"], config["anchor"])
            point = plane.floor(anchor) if plane else anchor
            if ignored(point, config["ignore_zones"]):
                continue
            kept.append(track)
            anchors[track["track_id"]] = anchor
            points[track["track_id"]] = point
            trails.setdefault(track["track_id"], []).append(anchor)
            floor_trails.setdefault(track["track_id"], []).append(points[track["track_id"]])
            last_seen[track["track_id"]] = frame_index
        tracks = kept
        counter.update(points, frame_index)
        zone_counts = zones.update(points, frame_index)
        stale = [tid for tid, seen in last_seen.items() if frame_index - seen > 100]
        counter.forget(stale)
        for tid in stale:
            del last_seen[tid]
        canvas = draw_flow_frame(image, config, plane, tracks, anchors, trails, counter, zone_counts, frame_index, fps)
        annotated.write(canvas)
        if frame_index in wanted:
            name = f"frame_{frame_index:04d}.png"
            canvas.save(output / name)
            snapshots.append((frame_index, name))
        detector_ms.append((t1 - t0) * 1000)
        frame_ms.append((time.perf_counter() - t0) * 1000)
    annotated_path = annotated.close()
    start_time = parse_start_time(args.start_time)
    buckets = bucket_totals(counter.crossings, fps, args.bucket, 0, frame_index)
    zone_summary = zones.summary()
    write_flow_outputs(output, counter.crossings, buckets, zone_summary, fps, start_time)
    run = {"source": description, "frames": frame_index + 1, "fps": fps, "size": list(size), "detector": detector.info(),
           "config": config, "floor_residual_px": round(plane.residual_px, 4) if plane else None,
           "tracker": {"high_score": tracker.high_score, "low_score": tracker.low_score, "iou_threshold": tracker.iou_threshold,
                       "min_hits": tracker.min_hits, "max_lost": tracker.max_lost},
           "bucket_seconds": args.bucket, "start_time": args.start_time, "totals": counter.totals,
           "zones": zone_summary, "crossings": len(counter.crossings), "tracks_seen": len(trails),
           "timing": {"detector_median_ms": statistics.median(detector_ms) if detector_ms else None,
                      "frame_median_ms": statistics.median(frame_ms) if frame_ms else None,
                      "wall_seconds": time.perf_counter() - started, "threads": args.threads, "cpu": platform.processor() or platform.machine()},
           "annotated_video": Path(annotated_path).name}
    if args.truth:
        truth = json.loads(Path(args.truth).read_text(encoding="utf-8"))
        report = evaluate(counter.crossings, truth["crossings"], frame_index + 1)
        (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        run["evaluation"] = {k: v for k, v in report.items() if k not in ("matches", "unmatched_predicted", "unmatched_truth")}
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    svg = floor_plan_svg(config, floor_trails, counter.crossings) if plane else floor_plan_svg(config, floor_trails, counter.crossings, scale=1)
    (output / "dashboard.html").write_text(flow_dashboard(run, counter.crossings, buckets, zone_summary, svg, snapshots), encoding="utf-8")
    print(json.dumps({"frames": run["frames"], "totals": counter.totals, "zones": {k: v["peak"] for k, v in zone_summary.items()},
                      "detector_median_ms": run["timing"]["detector_median_ms"],
                      "evaluation": {k: run["evaluation"][k] for k in ("true_positive", "false_positive", "false_negative", "median_latency_frames")} if args.truth else None}))
    return run


if __name__ == "__main__":
    main()
