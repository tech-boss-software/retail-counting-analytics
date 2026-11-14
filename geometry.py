"""Dependency-free box, polygon and line geometry."""
from __future__ import annotations

import math


def validate_box(box):
    if len(box) != 4:
        raise ValueError("box must be [xmin, ymin, xmax, ymax]")
    box = [float(v) for v in box]
    if any(not math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("box must have finite coordinates and positive area")
    return box


def box_iou(a, b) -> float:
    a, b = validate_box(a), validate_box(b)
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union


def box_anchor(box, kind="bottom_center"):
    box = validate_box(box)
    if kind == "bottom_center":
        return ((box[0] + box[2]) / 2, box[3])
    if kind == "center":
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    raise ValueError(f"unknown anchor kind: {kind}")


def validate_point(point):
    if len(point) != 2 or any(not math.isfinite(float(v)) for v in point):
        raise ValueError("point must be a finite [x, y] pair")
    return (float(point[0]), float(point[1]))


def validate_polygon(polygon):
    if len(polygon) < 3:
        raise ValueError("polygon needs at least three points")
    return [validate_point(p) for p in polygon]


def polygon_area(polygon) -> float:
    """Shoelace area; in floor coordinates this is square metres."""
    points = validate_polygon(polygon)
    total = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]):
        total += x1 * y2 - x2 * y1
    return abs(total) / 2


def point_in_polygon(point, polygon) -> bool:
    """Ray casting; points on an edge count as inside."""
    x, y = validate_point(point)
    points = validate_polygon(polygon)
    inside = False
    count = len(points)
    for i in range(count):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % count]
        if _on_segment(x, y, x1, y1, x2, y2):
            return True
        if (y1 > y) != (y2 > y):
            cross = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < cross:
                inside = not inside
    return inside


def _on_segment(x, y, x1, y1, x2, y2, eps=1e-9):
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(cross) > eps:
        return False
    return min(x1, x2) - eps <= x <= max(x1, x2) + eps and min(y1, y2) - eps <= y <= max(y1, y2) + eps


def signed_distance(point, a, b) -> float:
    """Perpendicular distance of point from the infinite line a->b; positive on the left of the direction a->b."""
    (x, y), (ax, ay), (bx, by) = validate_point(point), validate_point(a), validate_point(b)
    length = math.hypot(bx - ax, by - ay)
    if length == 0:
        raise ValueError("line endpoints must differ")
    return ((bx - ax) * (y - ay) - (by - ay) * (x - ax)) / length


def _orientation(p, q, r):
    value = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    if abs(value) < 1e-12:
        return 0
    return 1 if value > 0 else -1


def segments_intersect(p1, p2, q1, q2) -> bool:
    """True when segment p1-p2 touches segment q1-q2 (collinear overlaps included)."""
    p1, p2, q1, q2 = (validate_point(v) for v in (p1, p2, q1, q2))
    o1, o2 = _orientation(p1, p2, q1), _orientation(p1, p2, q2)
    o3, o4 = _orientation(q1, q2, p1), _orientation(q1, q2, p2)
    if o1 != o2 and o3 != o4:
        return True
    for a, b, c, o in ((p1, p2, q1, o1), (p1, p2, q2, o2), (q1, q2, p1, o3), (q1, q2, p2, o4)):
        if o == 0 and _on_segment(c[0], c[1], a[0], a[1], b[0], b[1]):
            return True
    return False
