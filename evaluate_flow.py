"""Score predicted gate crossings and occupancy against scripted truth."""
from __future__ import annotations

from counting import occupancy_curve


def match_crossings(predicted, truth, tolerance_frames=15):
    """Greedy one-to-one matching by line, direction and nearest frame within the tolerance."""
    candidates = []
    for i, p in enumerate(predicted):
        for j, t in enumerate(truth):
            if p["line"] == t["line"] and p["direction"] == t["direction"] and abs(p["frame"] - t["frame"]) <= tolerance_frames:
                candidates.append((abs(p["frame"] - t["frame"]), i, j))
    candidates.sort()
    used_p, used_t, pairs = set(), set(), []
    for distance, i, j in candidates:
        if i in used_p or j in used_t:
            continue
        used_p.add(i)
        used_t.add(j)
        pairs.append((i, j, predicted[i]["frame"] - truth[j]["frame"]))
    return pairs, [i for i in range(len(predicted)) if i not in used_p], [j for j in range(len(truth)) if j not in used_t]


def evaluate(predicted, truth, frames, tolerance_frames=15):
    pairs, false_positive, false_negative = match_crossings(predicted, truth, tolerance_frames)
    per_direction = {}
    for direction in ("in", "out"):
        p = sum(1 for c in predicted if c["direction"] == direction)
        t = sum(1 for c in truth if c["direction"] == direction)
        tp = sum(1 for i, j, _ in pairs if truth[j]["direction"] == direction)
        per_direction[direction] = {"truth": t, "predicted": p, "true_positive": tp, "false_positive": p - tp, "false_negative": t - tp,
                                    "count_error": p - t}
    latencies = sorted(delta for _, _, delta in pairs)
    pred_curve = occupancy_curve(predicted, frames)
    truth_curve = occupancy_curve(truth, frames)
    abs_diff = [abs(a - b) for a, b in zip(pred_curve, truth_curve)]
    tp = len(pairs)
    return {"tolerance_frames": tolerance_frames, "truth": len(truth), "predicted": len(predicted), "true_positive": tp,
            "false_positive": len(false_positive), "false_negative": len(false_negative),
            "precision": tp / len(predicted) if predicted else None, "recall": tp / len(truth) if truth else None,
            "median_latency_frames": latencies[len(latencies) // 2] if latencies else None,
            "max_abs_latency_frames": max(abs(v) for v in latencies) if latencies else None,
            "per_direction": per_direction,
            "net_count_error": (per_direction["in"]["predicted"] - per_direction["out"]["predicted"]) - (per_direction["in"]["truth"] - per_direction["out"]["truth"]),
            "occupancy_mean_abs_error": sum(abs_diff) / len(abs_diff) if abs_diff else None,
            "occupancy_frames_exact": sum(1 for d in abs_diff if d == 0) / len(abs_diff) if abs_diff else None,
            "matches": [{"predicted": predicted[i], "truth": truth[j], "latency_frames": delta} for i, j, delta in pairs],
            "unmatched_predicted": [predicted[i] for i in false_positive],
            "unmatched_truth": [truth[j] for j in false_negative]}
