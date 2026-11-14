"""Count products and find empty slots on shelf photos; write per-photo JSON, annotated images and a dashboard."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from detectors import load_detector
from frames import IMAGE_SUFFIXES
from render import draw_shelf_audit, shelf_dashboard
from shelf import audit, evaluate_shelves


def list_images(path):
    path = Path(path)
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if path.suffix.lower() in IMAGE_SUFFIXES:
        return [path]
    raise ValueError(f"{path} is neither an image nor a folder of images")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="shelf photo or folder of photos")
    parser.add_argument("--detector", required=True, help="onnx:<model.onnx> with a 'product' class")
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--score", type=float, default=0.5)
    parser.add_argument("--min-gap-ratio", type=float, default=0.8, help="gap / median product width that counts as an empty slot")
    parser.add_argument("--row-gap-ratio", type=float, default=0.5, help="bottom-edge difference / median height that starts a new row")
    parser.add_argument("--extent", default=None, help="shelf left,right in pixels; default is the span of detected products")
    parser.add_argument("--truth", default=None, help="truth.json from synthetic.py --shelves")
    args = parser.parse_args(argv)

    output = Path(args.output)
    (output / "annotated").mkdir(parents=True, exist_ok=True)
    detector = load_detector(args.detector, args.threads)
    product_class = next((k for k, v in detector.labels.items() if v == "product"), None)
    if product_class is None:
        raise ValueError(f"detector has no 'product' class; its classes are {sorted(detector.labels.values())}")
    extent = tuple(float(v) for v in args.extent.split(",")) if args.extent else None
    from PIL import Image

    reports, results, timings = [], {}, []
    truth = json.loads(Path(args.truth).read_text(encoding="utf-8"))["shelves"] if args.truth else {}
    for path in list_images(args.input):
        with Image.open(path) as image:
            image = image.convert("RGB")
        t0 = time.perf_counter()
        detections = [d for d in detector.detect(image) if d["class_id"] == product_class and d["score"] >= args.score]
        timings.append((time.perf_counter() - t0) * 1000)
        report = audit([d["box_xyxy"] for d in detections], extent, args.min_gap_ratio, args.row_gap_ratio)
        annotated = f"annotated/{path.stem}.png"
        draw_shelf_audit(image, report, detections).save(output / annotated)
        entry = {"image": path.name, "detections": len(detections), "audit": report, "annotated": annotated,
                 "products": [{"box_xyxy": [round(v, 2) for v in d["box_xyxy"]], "score": round(d["score"], 4)} for d in detections]}
        if path.name in truth:
            entry["truth_items"] = truth[path.name]["items"]
            entry["truth_slots"] = len(truth[path.name]["slots"])
        reports.append(entry)
        results[path.name] = {"items": report["items"], "slots": [s["box_xyxy"] for s in report["slots"]], "rows": [r["items"] for r in report["rows"]]}
    run = {"images": len(reports), "detector": detector.info(), "score": args.score,
           "settings": {"min_gap_ratio": args.min_gap_ratio, "row_gap_ratio": args.row_gap_ratio, "extent": extent},
           "timing": {"detector_median_ms": statistics.median(timings) if timings else None, "threads": args.threads},
           "totals": {"items": sum(r["audit"]["items"] for r in reports), "empty_slots": sum(r["audit"]["empty_slots"] for r in reports)}}
    if truth:
        truths = {name: {"items": t["items"], "slots": t["slots"], "rows": t["rows"]} for name, t in truth.items()}
        run["evaluation"] = evaluate_shelves(results, truths)
        oracle = {name: (lambda a: {"items": a["items"], "slots": [s["box_xyxy"] for s in a["slots"]], "rows": [r["items"] for r in a["rows"]]})(
            audit(t["boxes"], extent, args.min_gap_ratio, args.row_gap_ratio)) for name, t in truth.items()}
        run["evaluation_oracle"] = evaluate_shelves(oracle, truths)
        (output / "evaluation.json").write_text(json.dumps({"detector": run["evaluation"], "oracle": run["evaluation_oracle"]}, indent=2) + "\n", encoding="utf-8")
        for key in ("evaluation", "evaluation_oracle"):
            run[key] = {k: v for k, v in run[key].items() if k != "per_image"}
    (output / "shelf_report.json").write_text(json.dumps({"run": run, "shelves": reports}, indent=2) + "\n", encoding="utf-8")
    (output / "dashboard.html").write_text(shelf_dashboard(run, reports), encoding="utf-8")
    print(json.dumps({"images": run["images"], "totals": run["totals"], "detector_median_ms": run["timing"]["detector_median_ms"],
                      "evaluation": {k: run["evaluation"][k] for k in ("count_mae", "exact_count_fraction", "slot_precision", "slot_recall")} if truth else None}))
    return run


if __name__ == "__main__":
    main()
