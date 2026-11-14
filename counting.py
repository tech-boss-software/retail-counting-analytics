"""Line-crossing counter and zone occupancy over tracked points.

A counting line is a directed segment a->b. The side where the cross product
(b - a) x (p - a) is positive is the `in` side: with y pointing up (floor metres)
that is the left of a->b, with y pointing down (image pixels) it is the right.
Rule of thumb: walk from a to b and keep the inside of the store on your left in
the floor plan; swap a and b to flip. Two protections against double counting:

- hysteresis: a track's side is only confirmed once it is at least `margin` away
  from the line, so a person stopping on the threshold does not count twice;
- extent: the path between the last confirmed position and the current one must
  actually cross the segment, so walking around the end of a gate does not count.

Points may be pixels or floor metres; the caller decides by passing a FloorPlane.
"""
from __future__ import annotations

from geometry import point_in_polygon, polygon_area, segments_intersect, signed_distance, validate_point, validate_polygon

DEFAULT_CONFIG = {
    "anchor": "bottom_center",
    "space": "image",            # "image" (pixels) or "floor" (metres through the floor calibration)
    "margin": 6.0,               # hysteresis distance in the chosen space's units
    "lines": [],
    "zones": [],
    "ignore_zones": [],          # polygons (same space as lines) where tracks are dropped: mirrors, screens, staff areas
    "floor": None,
}


def load_config(config):
    merged = dict(DEFAULT_CONFIG)
    merged.update(config or {})
    if merged["space"] not in ("image", "floor"):
        raise ValueError("space must be 'image' or 'floor'")
    if merged["space"] == "floor" and not merged.get("floor"):
        raise ValueError("space 'floor' needs a floor calibration with image_points and floor_points")
    if float(merged["margin"]) < 0:
        raise ValueError("margin must be non-negative")
    names = set()
    for line in merged["lines"]:
        if not line.get("name") or line["name"] in names:
            raise ValueError("every line needs a unique name")
        names.add(line["name"])
        a, b = validate_point(line["a"]), validate_point(line["b"])
        if a == b:
            raise ValueError(f"line {line['name']} has identical endpoints")
    zone_names = set()
    for zone in merged["zones"]:
        if not zone.get("name") or zone["name"] in zone_names:
            raise ValueError("every zone needs a unique name")
        zone_names.add(zone["name"])
        validate_polygon(zone["polygon"])
    for zone in merged["ignore_zones"]:
        validate_polygon(zone["polygon"])
    return merged


def ignored(point, ignore_zones):
    return any(point_in_polygon(point, zone["polygon"]) for zone in ignore_zones)


class LineCounter:
    def __init__(self, lines, margin=6.0):
        self.lines = [{"name": line["name"], "a": validate_point(line["a"]), "b": validate_point(line["b"])} for line in lines]
        self.margin = float(margin)
        self.state = {}        # (track_id, line name) -> {"side": +1/-1/None, "point": last confirmed point}
        self.crossings = []
        self.totals = {line["name"]: {"in": 0, "out": 0} for line in self.lines}

    def update(self, points, frame_index):
        """points: {track_id: (x, y)} for the tracks visible this frame. Returns the crossings this frame."""
        new = []
        for track_id, point in points.items():
            point = validate_point(point)
            for line in self.lines:
                distance = signed_distance(point, line["a"], line["b"])
                if abs(distance) < self.margin:
                    continue            # on the threshold: no decision yet
                side = 1 if distance > 0 else -1
                key = (track_id, line["name"])
                state = self.state.get(key)
                if state is None:
                    self.state[key] = {"side": side, "point": point}
                    continue
                if side != state["side"]:
                    crossed = segments_intersect(state["point"], point, line["a"], line["b"])
                    if crossed:
                        direction = "in" if side > 0 else "out"
                        record = {"frame": frame_index, "track_id": track_id, "line": line["name"], "direction": direction,
                                  "point": [round(point[0], 3), round(point[1], 3)]}
                        self.crossings.append(record)
                        self.totals[line["name"]][direction] += 1
                        new.append(record)
                    state["side"] = side
                state["point"] = point
        return new

    def forget(self, track_ids):
        for key in [k for k in self.state if k[0] in track_ids]:
            del self.state[key]


class ZoneOccupancy:
    def __init__(self, zones, floor_units=None):
        self.zones = []
        for zone in zones:
            polygon = validate_polygon(zone["polygon"])
            area = polygon_area(polygon) if floor_units else None
            self.zones.append({"name": zone["name"], "polygon": polygon, "area": area})
        self.series = {zone["name"]: [] for zone in self.zones}

    def update(self, points, frame_index):
        counts = {}
        for zone in self.zones:
            inside = [track_id for track_id, point in points.items() if point_in_polygon(point, zone["polygon"])]
            counts[zone["name"]] = len(inside)
            self.series[zone["name"]].append((frame_index, len(inside)))
        return counts

    def summary(self):
        result = {}
        for zone in self.zones:
            values = [count for _, count in self.series[zone["name"]]]
            entry = {"frames": len(values), "peak": max(values) if values else 0,
                     "mean": sum(values) / len(values) if values else 0.0,
                     "occupied_fraction": sum(1 for v in values if v > 0) / len(values) if values else 0.0,
                     "area_m2": zone["area"]}
            if zone["area"]:
                entry["peak_density_per_m2"] = entry["peak"] / zone["area"]
            result[zone["name"]] = entry
        return result


def bucket_totals(crossings, fps, bucket_seconds=3600.0, start_frame=0, end_frame=None, initial_occupancy=0):
    """Per-bucket in/out totals plus occupancy at the end of each bucket (initial + in - out)."""
    if fps <= 0 or bucket_seconds <= 0:
        raise ValueError("fps and bucket_seconds must be positive")
    last = end_frame if end_frame is not None else max([c["frame"] for c in crossings] + [start_frame])
    per_frame_bucket = lambda frame: int((frame - start_frame) / fps // bucket_seconds)
    bucket_count = per_frame_bucket(last) + 1
    rows = [{"bucket": i, "start_seconds": i * bucket_seconds, "end_seconds": (i + 1) * bucket_seconds, "in": 0, "out": 0}
            for i in range(bucket_count)]
    for crossing in crossings:
        index = per_frame_bucket(crossing["frame"])
        if 0 <= index < bucket_count:
            rows[index][crossing["direction"]] += 1
    occupancy = initial_occupancy
    for row in rows:
        occupancy += row["in"] - row["out"]
        row["occupancy_end"] = occupancy
    return rows


def occupancy_curve(crossings, frames, initial_occupancy=0):
    """Occupancy after each frame as a list of length `frames`."""
    deltas = [0] * frames
    for crossing in crossings:
        if 0 <= crossing["frame"] < frames:
            deltas[crossing["frame"]] += 1 if crossing["direction"] == "in" else -1
    curve, level = [], initial_occupancy
    for delta in deltas:
        level += delta
        curve.append(level)
    return curve
