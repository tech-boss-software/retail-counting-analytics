import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from counting import LineCounter
from geometry import point_in_polygon
from homography import FloorPlane
from synthetic import (FPS, GATE, HEIGHT, PROMO_ZONE, WIDTH, floor_calibration, interpolate, person_box, project, scenario,
                       truth_crossings, walker_state, yolo_line)


class SyntheticEntranceTests(unittest.TestCase):
    def test_projection_perspective_and_calibration_consistency(self):
        near, far = person_box(3.0, 1.0)[2], person_box(3.0, 8.0)[2]
        self.assertGreater(near, far * 2)                       # closer people are taller in pixels
        plane = FloorPlane(floor_calibration())
        self.assertLess(plane.residual_px, 1e-6)
        for x, y in ((2.0, 2.0), (4.5, 6.0), (1.0, 3.0)):
            u, v = project((x, y, 0.0))
            fx, fy = plane.floor((u, v))
            self.assertAlmostEqual(fx, x, places=4)             # the homography and the camera agree on the floor
            self.assertAlmostEqual(fy, y, places=4)
        with self.assertRaises(ValueError):
            project((0.0, -5.0, 0.0))

    def test_interpolation_clamps(self):
        keys = [(10, 0, 0), (20, 1, 2)]
        self.assertEqual(interpolate(keys, 0), (0, 0))
        self.assertEqual(interpolate(keys, 15), (0.5, 1.0))
        self.assertEqual(interpolate(keys, 30), (1, 2))

    def test_scripted_truth_has_seven_crossings_and_the_decoys_have_none(self):
        spec = scenario()
        crossings = truth_crossings(spec)
        self.assertEqual(sorted((c["track_id"], c["direction"]) for c in crossings),
                         [(1, "in"), (2, "out"), (4, "in"), (5, "in"), (8, "out"), (9, "in"), (10, "out")])
        self.assertEqual(sorted(c["frame"] for c in crossings), [c["frame"] for c in crossings])
        for c in crossings:
            self.assertEqual(c["point"][1], GATE["a"][1])
            self.assertTrue(min(GATE["a"][0], GATE["b"][0]) <= c["point"][0] <= max(GATE["a"][0], GATE["b"][0]))
        self.assertNotIn(3, {c["track_id"] for c in crossings})   # hesitates at the threshold
        self.assertNotIn(6, {c["track_id"] for c in crossings})   # walks around the gate's end
        self.assertNotIn(7, {c["track_id"] for c in crossings})   # loiterer

    def test_counter_on_exact_floor_paths_reproduces_the_truth(self):
        spec = scenario()
        counter = LineCounter([GATE], margin=0.3)
        promo_frames = 0
        for frame in range(spec["frames"]):
            points = {}
            for walker in spec["walkers"]:
                state = walker_state(walker, frame)
                if state is not None:
                    points[state["id"]] = state["floor"]
            counter.update(points, frame)
            promo_frames += any(point_in_polygon(p, PROMO_ZONE["polygon"]) for p in points.values())
        truth = truth_crossings(spec)
        self.assertEqual([(c["track_id"], c["direction"]) for c in counter.crossings], [(c["track_id"], c["direction"]) for c in truth])
        for predicted, expected in zip(counter.crossings, truth):
            self.assertLessEqual(abs(predicted["frame"] - expected["frame"]), 10)   # 0.3 m hysteresis at walking pace: under 1 s
        self.assertGreater(promo_frames, 30 * FPS)

    def test_walker_boxes_stay_inside_the_frame_and_yolo_lines_are_normalised(self):
        spec = scenario()
        for walker in spec["walkers"]:
            for frame in range(0, spec["frames"], 7):
                state = walker_state(walker, frame)
                if state is None:
                    continue
                x0, y0, x1, y1 = state["box"]
                self.assertTrue(0 <= x0 < x1 <= WIDTH and 0 <= y0 < y1 <= HEIGHT)
        parts = yolo_line(0, [10, 20, 30, 60]).split()
        self.assertEqual(parts[0], "0")
        self.assertTrue(all(0 <= float(v) <= 1 for v in parts[1:]))


if __name__ == "__main__":
    unittest.main()
