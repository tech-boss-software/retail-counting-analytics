"""Plane homography between image pixels and floor metres, in pure Python.

Four or more measured floor points (a tape measure on site, or a known tile grid)
paired with their pixel positions fix the 3x3 matrix by the direct linear
transform. With exactly four pairs the 8x8 system is solved directly; with more,
the normal equations give the least-squares fit. Foot points of people on the floor
can then be measured in metres, which is what makes counting lines and occupancy
zones independent of the camera's perspective.
"""
from __future__ import annotations

import math

from geometry import validate_point


def _solve(matrix, vector):
    """Gaussian elimination with partial pivoting on a small dense system."""
    n = len(vector)
    rows = [list(matrix[i]) + [vector[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(rows[r][col]))
        if abs(rows[pivot][col]) < 1e-12:
            raise ValueError("degenerate point configuration (collinear or repeated points)")
        rows[col], rows[pivot] = rows[pivot], rows[col]
        for r in range(n):
            if r == col:
                continue
            factor = rows[r][col] / rows[col][col]
            if factor:
                for c in range(col, n + 1):
                    rows[r][c] -= factor * rows[col][c]
    return [rows[i][n] / rows[i][i] for i in range(n)]


def fit_homography(source_points, target_points):
    """Return H (3x3 nested lists, H[2][2] = 1) with target ~ H * source for at least four pairs."""
    src = [validate_point(p) for p in source_points]
    dst = [validate_point(p) for p in target_points]
    if len(src) != len(dst) or len(src) < 4:
        raise ValueError("need at least four source/target point pairs of equal count")
    equations, values = [], []
    for (x, y), (u, v) in zip(src, dst):
        equations.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        values.append(u)
        equations.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        values.append(v)
    if len(src) == 4:
        h = _solve(equations, values)
    else:
        normal = [[sum(equations[k][i] * equations[k][j] for k in range(len(equations))) for j in range(8)] for i in range(8)]
        rhs = [sum(equations[k][i] * values[k] for k in range(len(equations))) for i in range(8)]
        h = _solve(normal, rhs)
    return [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]]


def apply_homography(h, point):
    x, y = validate_point(point)
    w = h[2][0] * x + h[2][1] * y + h[2][2]
    if abs(w) < 1e-12:
        raise ValueError("point maps to infinity under this homography")
    return ((h[0][0] * x + h[0][1] * y + h[0][2]) / w, (h[1][0] * x + h[1][1] * y + h[1][2]) / w)


def invert_homography(h):
    a, b, c = h[0]
    d, e, f = h[1]
    g, i, j = h[2]
    det = a * (e * j - f * i) - b * (d * j - f * g) + c * (d * i - e * g)
    if abs(det) < 1e-15:
        raise ValueError("homography is singular")
    adj = [[e * j - f * i, c * i - b * j, b * f - c * e],
           [f * g - d * j, a * j - c * g, c * d - a * f],
           [d * i - e * g, b * g - a * i, a * e - b * d]]
    inv = [[v / det for v in row] for row in adj]
    scale = inv[2][2]
    if abs(scale) < 1e-15:
        return inv
    return [[v / scale for v in row] for row in inv]


class FloorPlane:
    """Image <-> floor mapping built from a calibration dict: {image_points: [[u, v]...], floor_points: [[x, y]...]}."""

    def __init__(self, calibration):
        image_points = calibration.get("image_points") or []
        floor_points = calibration.get("floor_points") or []
        self.to_floor = fit_homography(image_points, floor_points)
        self.to_image = invert_homography(self.to_floor)
        self.units = calibration.get("units", "m")
        self.residual_px = self._residual(image_points, floor_points)

    def _residual(self, image_points, floor_points):
        worst = 0.0
        for image_point, floor_point in zip(image_points, floor_points):
            back = apply_homography(self.to_image, floor_point)
            worst = max(worst, math.hypot(back[0] - image_point[0], back[1] - image_point[1]))
        return worst

    def floor(self, image_point):
        return apply_homography(self.to_floor, image_point)

    def image(self, floor_point):
        return apply_homography(self.to_image, floor_point)
