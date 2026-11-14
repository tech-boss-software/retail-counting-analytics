import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geometry import box_anchor, box_iou, point_in_polygon, polygon_area, segments_intersect, signed_distance, validate_polygon
from homography import FloorPlane, apply_homography, fit_homography, invert_homography

SQUARE = [[0, 0], [10, 0], [10, 10], [0, 10]]


class GeometryTests(unittest.TestCase):
    def test_polygon_membership_area_and_validation(self):
        self.assertTrue(point_in_polygon((5, 5), SQUARE))
        self.assertFalse(point_in_polygon((15, 5), SQUARE))
        self.assertTrue(point_in_polygon((10, 5), SQUARE))
        self.assertEqual(polygon_area(SQUARE), 100.0)
        self.assertEqual(polygon_area([[0, 0], [4, 0], [0, 3]]), 6.0)
        with self.assertRaises(ValueError):
            validate_polygon([[0, 0], [1, 1]])

    def test_iou_and_anchor(self):
        self.assertAlmostEqual(box_iou([0, 0, 2, 2], [1, 1, 3, 3]), 1 / 7)
        self.assertEqual(box_anchor([0, 0, 4, 10]), (2.0, 10.0))
        self.assertEqual(box_anchor([0, 0, 4, 10], "center"), (2.0, 5.0))
        with self.assertRaises(ValueError):
            box_anchor([0, 0, 4, 10], "left_ear")

    def test_signed_distance_is_positive_on_the_left_of_the_direction(self):
        a, b = (0, 0), (10, 0)
        self.assertAlmostEqual(signed_distance((5, 3), a, b), 3.0)     # above the x axis = left of a->b
        self.assertAlmostEqual(signed_distance((5, -3), a, b), -3.0)
        self.assertAlmostEqual(signed_distance((5, 3), b, a), -3.0)    # reversing the line flips the sign
        self.assertAlmostEqual(signed_distance((0, 5), (0, 0), (3, 4)), 3.0)
        with self.assertRaises(ValueError):
            signed_distance((1, 1), a, a)

    def test_segment_intersection_including_touching_and_parallel(self):
        self.assertTrue(segments_intersect((0, -1), (0, 1), (-1, 0), (1, 0)))
        self.assertFalse(segments_intersect((2, -1), (2, 1), (-1, 0), (1, 0)))    # beyond the end of the gate
        self.assertTrue(segments_intersect((1, -1), (1, 1), (-1, 0), (1, 0)))     # touches the endpoint
        self.assertFalse(segments_intersect((0, 1), (5, 1), (0, 0), (5, 0)))      # parallel
        self.assertTrue(segments_intersect((0, 0), (5, 0), (3, 0), (8, 0)))       # collinear overlap


class HomographyTests(unittest.TestCase):
    def test_four_point_fit_reproduces_a_known_perspective_map(self):
        floor = [(0, 0), (4, 0), (4, 6), (0, 6), (2, 3), (1, 5)]
        truth = [[2.0, 0.3, 5.0], [0.1, 1.5, -2.0], [0.02, 0.05, 1.0]]
        image = [apply_homography(truth, p) for p in floor]
        h = fit_homography(image[:4], floor[:4])
        for p, q in zip(image, floor):
            back = apply_homography(h, p)
            self.assertAlmostEqual(back[0], q[0], places=6)
            self.assertAlmostEqual(back[1], q[1], places=6)
        inverse = invert_homography(h)
        for p, q in zip(image, floor):
            forward = apply_homography(inverse, q)
            self.assertAlmostEqual(forward[0], p[0], places=6)
            self.assertAlmostEqual(forward[1], p[1], places=6)

    def test_overdetermined_fit_averages_noise(self):
        truth = [[1.2, 0.1, 3.0], [-0.2, 0.9, 1.0], [0.001, 0.002, 1.0]]
        floor = [(x, y) for x in range(0, 7, 2) for y in range(0, 7, 2)]
        image = [apply_homography(truth, p) for p in floor]
        h = fit_homography(image, floor)
        worst = max(math.dist(apply_homography(h, p), q) for p, q in zip(image, floor))
        self.assertLess(worst, 1e-6)

    def test_degenerate_points_are_rejected(self):
        with self.assertRaises(ValueError):
            fit_homography([(0, 0), (1, 1), (2, 2), (3, 3)], [(0, 0), (1, 0), (1, 1), (0, 1)])
        with self.assertRaises(ValueError):
            fit_homography([(0, 0), (1, 0), (1, 1)], [(0, 0), (1, 0), (1, 1)])

    def test_floor_plane_measures_metres_from_pixels(self):
        calibration = {"units": "m", "floor_points": [[0, 0], [4, 0], [4, 6], [0, 6]],
                       "image_points": [[100, 300], [500, 300], [400, 100], [200, 100]]}
        plane = FloorPlane(calibration)
        self.assertLess(plane.residual_px, 1e-6)
        x, y = plane.floor((300, 300))
        self.assertAlmostEqual(x, 2.0, places=6)
        self.assertAlmostEqual(y, 0.0, places=6)
        u, v = plane.image((2.0, 6.0))
        self.assertAlmostEqual(u, 300.0, places=6)
        self.assertAlmostEqual(v, 100.0, places=6)
        mid = plane.floor((300, 200))
        self.assertLess(mid[1], 3.0)   # perspective: the pixel midpoint is nearer than the metric midpoint


if __name__ == "__main__":
    unittest.main()
