import csv
import datetime as dt
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from counting import LineCounter, ZoneOccupancy, bucket_totals, load_config, occupancy_curve
from evaluate_flow import evaluate
from homography import FloorPlane
from store import write_flow_outputs
from tracker import Tracker

GATE = {"name": "door", "a": [0, 0], "b": [10, 0]}     # positive side is y>0 (left of a->b with y up): y<0 -> y>0 is "in"


def walk(counter, track_id, points, start=0):
    out = []
    for i, p in enumerate(points):
        out += counter.update({track_id: p}, start + i)
    return out


class LineCounterTests(unittest.TestCase):
    def test_straight_crossing_counts_once_in_the_right_direction(self):
        counter = LineCounter([GATE], margin=0.5)
        crossings = walk(counter, 1, [(5, -3), (5, -2), (5, -1), (5, 0.2), (5, 1), (5, 2), (5, 3)])
        self.assertEqual([(c["direction"], c["frame"]) for c in crossings], [("in", 4)])
        self.assertEqual(counter.totals["door"], {"in": 1, "out": 0})
        back = walk(counter, 1, [(5, 2), (5, 1), (5, -1), (5, -2)], start=10)
        self.assertEqual([c["direction"] for c in back], ["out"])
        self.assertEqual(counter.totals["door"], {"in": 1, "out": 1})

    def test_hesitation_on_the_threshold_does_not_count(self):
        counter = LineCounter([GATE], margin=0.5)
        wobble = [(5, -2), (5, -0.4), (5, 0.3), (5, -0.2), (5, 0.4), (5, -0.3), (5, -2)]
        self.assertEqual(walk(counter, 1, wobble), [])
        self.assertEqual(counter.totals["door"], {"in": 0, "out": 0})
        counter_no_margin = LineCounter([GATE], margin=0.0)
        self.assertGreaterEqual(len(walk(counter_no_margin, 1, wobble)), 3)   # the margin is what prevents the flicker

    def test_walking_around_the_end_of_the_gate_does_not_count(self):
        counter = LineCounter([GATE], margin=0.5)
        self.assertEqual(walk(counter, 1, [(12, -3), (12, -1), (12, 1), (12, 3)]), [])
        self.assertEqual(walk(counter, 2, [(-1, -3), (-1, 1)]), [])
        self.assertEqual(walk(counter, 3, [(9.5, -3), (9.5, 1)]), [{"frame": 1, "track_id": 3, "line": "door", "direction": "in", "point": [9.5, 1.0]}])

    def test_diagonal_crossing_through_a_frame_gap_is_still_detected(self):
        counter = LineCounter([GATE], margin=0.5)
        crossings = walk(counter, 7, [(2, -4), (8, 4)])      # one big step across the segment
        self.assertEqual([c["direction"] for c in crossings], ["in"])

    def test_two_people_and_two_lines_are_independent(self):
        second = {"name": "aisle", "a": [0, 10], "b": [10, 10]}   # positive side is y>10: crossing from y<10 to y>10 is "in"
        counter = LineCounter([GATE, second], margin=0.5)
        for frame, (p1, p2) in enumerate(zip([(3, -2), (3, 2), (3, 12)], [(6, 12), (6, 8), (6, 2)])):
            counter.update({1: p1, 2: p2}, frame)
        self.assertEqual(counter.totals, {"door": {"in": 1, "out": 0}, "aisle": {"in": 1, "out": 1}})
        counter.forget([1])
        self.assertFalse(any(key[0] == 1 for key in counter.state))

    def test_floor_space_counting_matches_metric_truth_despite_perspective(self):
        calibration = {"units": "m", "floor_points": [[0, 0], [4, 0], [4, 6], [0, 6]],
                       "image_points": [[100, 300], [500, 300], [400, 100], [200, 100]]}
        plane = FloorPlane(calibration)
        gate = {"name": "g", "a": [4, 3], "b": [0, 3]}
        counter = LineCounter([gate], margin=0.25)
        pixels = [plane.image((2.0, y)) for y in (5.5, 4.5, 3.6, 3.2, 2.8, 2.0, 1.0)]
        crossings = []
        for frame, pixel in enumerate(pixels):
            crossings += counter.update({1: plane.floor(pixel)}, frame)
        self.assertEqual([(c["direction"], c["frame"]) for c in crossings], [("in", 5)])   # 3.2 and 2.8 are inside the 0.25 m margin
        self.assertAlmostEqual(crossings[0]["point"][0], 2.0, places=2)


class OccupancyTests(unittest.TestCase):
    def test_zone_counts_series_and_density(self):
        zones = ZoneOccupancy([{"name": "promo", "polygon": [[0, 0], [2, 0], [2, 2.5], [0, 2.5]]}], floor_units="m")
        zones.update({1: (1, 1), 2: (5, 5)}, 0)
        zones.update({1: (1, 1), 2: (1.5, 2)}, 1)
        zones.update({}, 2)
        summary = zones.summary()["promo"]
        self.assertEqual((summary["peak"], summary["frames"]), (2, 3))
        self.assertAlmostEqual(summary["mean"], 1.0)
        self.assertAlmostEqual(summary["occupied_fraction"], 2 / 3)
        self.assertAlmostEqual(summary["area_m2"], 5.0)
        self.assertAlmostEqual(summary["peak_density_per_m2"], 0.4)
        pixel_zones = ZoneOccupancy([{"name": "p", "polygon": [[0, 0], [10, 0], [10, 10]]}])
        self.assertIsNone(pixel_zones.summary()["p"]["area_m2"])

    def test_bucket_totals_and_occupancy_curve(self):
        crossings = [{"frame": 5, "direction": "in"}, {"frame": 12, "direction": "in"}, {"frame": 25, "direction": "out"},
                     {"frame": 61, "direction": "in"}]
        rows = bucket_totals(crossings, fps=10, bucket_seconds=3.0, end_frame=70)
        self.assertEqual([(r["in"], r["out"], r["occupancy_end"]) for r in rows], [(2, 1, 1), (0, 0, 1), (1, 0, 2)])
        self.assertEqual(rows[2]["start_seconds"], 6.0)
        curve = occupancy_curve(crossings, 30)
        self.assertEqual((curve[4], curve[5], curve[12], curve[25], curve[29]), (0, 1, 2, 1, 1))
        with self.assertRaises(ValueError):
            bucket_totals(crossings, fps=0, bucket_seconds=1)

    def test_config_validation(self):
        good = load_config({"lines": [GATE], "zones": [{"name": "z", "polygon": [[0, 0], [1, 0], [1, 1]]}]})
        self.assertEqual(good["space"], "image")
        for bad in ({"space": "metres"}, {"space": "floor"}, {"lines": [GATE, GATE]}, {"lines": [{"name": "x", "a": [0, 0], "b": [0, 0]}]},
                    {"margin": -1}, {"zones": [{"name": "z", "polygon": [[0, 0], [1, 1]]}]}):
            with self.assertRaises(ValueError):
                load_config(bad)

    def test_outputs_land_in_csv_and_sqlite_with_timestamps(self):
        crossings = [{"frame": 5, "track_id": 1, "line": "door", "direction": "in", "point": [1.0, 2.0]},
                     {"frame": 50, "track_id": 2, "line": "door", "direction": "out", "point": [1.5, 2.0]}]
        buckets = bucket_totals(crossings, 10, 3.0, end_frame=60)
        zones = {"promo": {"frames": 61, "peak": 1, "mean": 0.2, "occupied_fraction": 0.2, "area_m2": 5.0}}
        with tempfile.TemporaryDirectory() as folder:
            write_flow_outputs(folder, crossings, buckets, zones, 10, dt.datetime(2026, 11, 14, 9, 0))
            with (Path(folder) / "totals.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["start_time"], "2026-11-14T09:00:00")
            self.assertEqual(rows[1]["start_time"], "2026-11-14T09:00:03")
            self.assertEqual([r["occupancy_end"] for r in rows], ["1", "0", "0"])
            connection = sqlite3.connect(Path(folder) / "counts.sqlite")
            try:
                self.assertEqual(connection.execute("SELECT direction, timestamp FROM crossings ORDER BY id").fetchall(),
                                 [("in", "2026-11-14T09:00:00"), ("out", "2026-11-14T09:00:05")])
                self.assertEqual(connection.execute("SELECT SUM(entered), SUM(exited) FROM buckets").fetchone(), (1, 1))
                self.assertEqual(connection.execute("SELECT area_m2 FROM zones").fetchone(), (5.0,))
            finally:
                connection.close()


class EvaluationTests(unittest.TestCase):
    def test_crossings_are_matched_by_direction_within_tolerance(self):
        truth = [{"frame": 100, "track_id": 1, "line": "door", "direction": "in"},
                 {"frame": 200, "track_id": 2, "line": "door", "direction": "out"},
                 {"frame": 300, "track_id": 3, "line": "door", "direction": "in"}]
        predicted = [{"frame": 104, "track_id": 9, "line": "door", "direction": "in"},
                     {"frame": 230, "track_id": 8, "line": "door", "direction": "out"},   # too late
                     {"frame": 301, "track_id": 7, "line": "door", "direction": "out"}]   # wrong direction
        report = evaluate(predicted, truth, 400, tolerance_frames=15)
        self.assertEqual((report["true_positive"], report["false_positive"], report["false_negative"]), (1, 2, 2))
        self.assertEqual(report["median_latency_frames"], 4)
        self.assertEqual(report["per_direction"]["out"]["count_error"], 1)
        self.assertEqual(report["net_count_error"], -2)
        self.assertLess(report["occupancy_frames_exact"], 1.0)


class TrackerCountingIntegrationTests(unittest.TestCase):
    def test_tracked_boxes_through_the_counter_give_one_count_per_person(self):
        tracker = Tracker(min_hits=2, max_lost=3)
        counter = LineCounter([{"name": "door", "a": [200, 150], "b": [0, 150]}], margin=4)
        for frame in range(20):
            y = 100 + frame * 6.0
            boxes = [{"box_xyxy": [50, y - 60, 80, y], "score": 0.9, "class_id": 0},
                     {"box_xyxy": [120, 250 - y + 60 - 60, 150, 250 - y + 60], "score": 0.9, "class_id": 0}]
            tracks = tracker.update(boxes, frame)
            counter.update({t["track_id"]: ((t["box_xyxy"][0] + t["box_xyxy"][2]) / 2, t["box_xyxy"][3]) for t in tracks}, frame)
        self.assertEqual(counter.totals["door"], {"in": 1, "out": 1})


if __name__ == "__main__":
    unittest.main()
