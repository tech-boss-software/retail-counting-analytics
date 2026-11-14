"""Shelf audit from product boxes: rows, counts and empty slots. Pure Python.

Products standing on a shelf share a bottom edge, so rows are found by clustering
box bottoms. Inside a row, a horizontal gap between neighbouring products (or
between the shelf edge and the first/last product) that is wider than
`min_gap_ratio` times the row's median product width is an empty slot; a gap is
reported as several slots when it would hold several products.
"""
from __future__ import annotations

from geometry import box_iou, validate_box

DEFAULTS = {"min_gap_ratio": 0.8, "row_gap_ratio": 0.5}


def group_rows(boxes, row_gap_ratio=0.5):
    """Cluster boxes into shelf rows by their bottom edge; returns rows top to bottom, each sorted left to right."""
    boxes = [validate_box(b) for b in boxes]
    if not boxes:
        return []
    heights = sorted(b[3] - b[1] for b in boxes)
    median_height = heights[len(heights) // 2]
    rows, current = [], []
    for box in sorted(boxes, key=lambda b: b[3]):
        if current and box[3] - sum(b[3] for b in current) / len(current) > row_gap_ratio * median_height:
            rows.append(current)
            current = []
        current.append(box)
    rows.append(current)
    return [sorted(row, key=lambda b: b[0]) for row in rows]


def find_empty_slots(row, extent, min_gap_ratio=0.8):
    """Empty slot boxes in one row; `extent` is (shelf_x0, shelf_x1)."""
    if not row:
        return []
    widths = sorted(b[2] - b[0] for b in row)
    median_width = widths[len(widths) // 2]
    top = min(b[1] for b in row)
    bottom = max(b[3] for b in row)
    edges = [extent[0]] + [v for b in row for v in (b[0], b[2])] + [extent[1]]
    # Products are placed on a pitch (width plus a small spacing); the spacing is estimated from
    # the gaps that are too narrow to be slots, so a gap of k products is read as k slots.
    small = sorted(b[0] - a[2] for a, b in zip(row, row[1:]) if 0 <= b[0] - a[2] < min_gap_ratio * median_width)
    spacing = small[len(small) // 2] if small else 0.0
    slots = []
    for left, right in zip(edges[0::2], edges[1::2]):
        gap = right - left
        if gap >= min_gap_ratio * median_width and gap > 0:
            count = max(1, int(round((gap + spacing) / (median_width + spacing))))
            step = gap / count
            for i in range(count):
                slots.append([round(left + i * step, 2), round(top, 2), round(left + (i + 1) * step, 2), round(bottom, 2)])
    return slots


def audit(boxes, extent=None, min_gap_ratio=0.8, row_gap_ratio=0.5):
    """Count products per row and find empty slots. `extent` defaults to the span of all products."""
    rows = group_rows(boxes, row_gap_ratio)
    if extent is None and rows:
        extent = (min(b[0] for row in rows for b in row), max(b[2] for row in rows for b in row))
    report_rows, slots = [], []
    for index, row in enumerate(rows, 1):
        row_slots = find_empty_slots(row, extent, min_gap_ratio)
        slots.extend({"row": index, "box_xyxy": s} for s in row_slots)
        report_rows.append({"row": index, "items": len(row), "empty_slots": len(row_slots),
                            "bottom": round(max(b[3] for b in row), 2), "boxes": [[round(v, 2) for v in b] for b in row]})
    return {"items": sum(r["items"] for r in report_rows), "empty_slots": len(slots), "rows": report_rows, "slots": slots,
            "extent": [round(v, 2) for v in extent] if extent else None,
            "settings": {"min_gap_ratio": min_gap_ratio, "row_gap_ratio": row_gap_ratio}}


def match_slots(predicted, truth, min_iou=0.3):
    """Greedy one-to-one IoU matching of slot boxes; returns (tp, fp, fn)."""
    candidates = sorted(((box_iou(p, t), i, j) for i, p in enumerate(predicted) for j, t in enumerate(truth)), key=lambda c: -c[0])
    used_p, used_t = set(), set()
    for iou, i, j in candidates:
        if iou < min_iou:
            break
        if i in used_p or j in used_t:
            continue
        used_p.add(i)
        used_t.add(j)
    tp = len(used_p)
    return tp, len(predicted) - tp, len(truth) - tp


def evaluate_shelves(results, truths, min_iou=0.3):
    """results/truths: {image_id: {"items": n, "slots": [[x0,y0,x1,y1]...], "rows": [counts...]}}."""
    per_image, abs_errors, rel_errors, row_hits, row_total = [], [], [], 0, 0
    tp = fp = fn = 0
    for image_id, truth in sorted(truths.items()):
        result = results.get(image_id)
        if result is None:
            continue
        error = result["items"] - truth["items"]
        abs_errors.append(abs(error))
        rel_errors.append(abs(error) / truth["items"] if truth["items"] else 0.0)
        t, f, n = match_slots(result["slots"], truth["slots"], min_iou)
        tp, fp, fn = tp + t, fp + f, fn + n
        row_total += 1
        if result["rows"] == truth["rows"]:
            row_hits += 1
        per_image.append({"image": image_id, "truth_items": truth["items"], "predicted_items": result["items"], "count_error": error,
                          "truth_slots": len(truth["slots"]), "predicted_slots": len(result["slots"]), "slot_tp": t, "slot_fp": f, "slot_fn": n,
                          "rows_exact": result["rows"] == truth["rows"]})
    count = len(per_image)
    return {"images": count,
            "count_mae": sum(abs_errors) / count if count else None,
            "count_mape": sum(rel_errors) / count if count else None,
            "exact_count_fraction": sum(1 for e in abs_errors if e == 0) / count if count else None,
            "rows_exact_fraction": row_hits / row_total if row_total else None,
            "slot_precision": tp / (tp + fp) if tp + fp else None,
            "slot_recall": tp / (tp + fn) if tp + fn else None,
            "slot_tp": tp, "slot_fp": fp, "slot_fn": fn, "min_iou": min_iou, "per_image": per_image}
