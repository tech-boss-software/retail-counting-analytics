import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shelf import audit, evaluate_shelves, find_empty_slots, group_rows, match_slots
from synthetic import sample_shelf, shelf_truth


def row(bottom, xs, width=20, height=40):
    return [[x, bottom - height, x + width, bottom] for x in xs]


class ShelfLogicTests(unittest.TestCase):
    def test_rows_are_grouped_by_bottom_edge_and_sorted(self):
        boxes = row(100, [60, 10, 35]) + row(200, [10, 40]) + [[12, 150, 30, 203]]   # last box: taller, same shelf as row 2
        rows = group_rows(boxes)
        self.assertEqual([len(r) for r in rows], [3, 3])
        self.assertEqual([b[0] for b in rows[0]], [10, 35, 60])
        self.assertEqual(group_rows([]), [])

    def test_gaps_become_slots_only_when_wide_enough_and_split_by_product_width(self):
        boxes = row(100, [0, 24, 48, 120, 144])       # 48..72 product, gap 72..120 = 2 products, small 4px gaps elsewhere
        slots = find_empty_slots(sorted(boxes), (0, 164), min_gap_ratio=0.8)
        self.assertEqual(len(slots), 2)
        self.assertEqual(slots[0][0], 68.0)
        self.assertEqual(slots[1][2], 120.0)
        self.assertEqual(slots[0][1], 60.0)
        self.assertEqual(find_empty_slots(sorted(boxes), (0, 164), min_gap_ratio=3.0), [])
        self.assertEqual(find_empty_slots(sorted(boxes), (-30, 190), min_gap_ratio=0.8)[0][0], -30.0)   # end gaps count too

    def test_audit_counts_rows_items_and_slots(self):
        report = audit(row(100, [10, 34, 58, 130]) + row(200, [10, 82, 106]))
        self.assertEqual(report["items"], 7)
        self.assertEqual([r["items"] for r in report["rows"]], [4, 3])
        self.assertEqual(report["empty_slots"], 5)      # row 1: 78..130 = 2 slots; row 2: 30..82 = 2 slots, 126..150 = 1
        self.assertEqual(report["extent"], [10.0, 150.0])
        self.assertEqual(audit([])["items"], 0)

    def test_slot_matching_and_evaluation(self):
        tp, fp, fn = match_slots([[0, 0, 10, 10], [50, 0, 60, 10]], [[1, 0, 11, 10], [100, 0, 110, 10]], min_iou=0.3)
        self.assertEqual((tp, fp, fn), (1, 1, 1))
        results = {"a.png": {"items": 10, "slots": [[0, 0, 10, 10]], "rows": [5, 5]}, "b.png": {"items": 7, "slots": [], "rows": [7]}}
        truths = {"a.png": {"items": 10, "slots": [[0, 0, 10, 10]], "rows": [5, 5]}, "b.png": {"items": 8, "slots": [[20, 0, 30, 10]], "rows": [8]}}
        report = evaluate_shelves(results, truths)
        self.assertEqual(report["images"], 2)
        self.assertAlmostEqual(report["count_mae"], 0.5)
        self.assertEqual(report["exact_count_fraction"], 0.5)
        self.assertEqual(report["rows_exact_fraction"], 0.5)
        self.assertEqual((report["slot_precision"], report["slot_recall"]), (1.0, 0.5))


class ShelfSyntheticTests(unittest.TestCase):
    def test_audit_on_truth_boxes_recovers_the_scripted_slots(self):
        rng = random.Random(3)
        exact_counts, slot_hits, slot_total = 0, 0, 0
        for _ in range(30):
            scene = sample_shelf(rng)
            truth = shelf_truth(scene)
            report = audit(truth["boxes"], tuple(scene["extent"]))
            self.assertEqual(report["items"], truth["items"])
            self.assertEqual([r["items"] for r in report["rows"]], truth["rows"])
            tp, fp, fn = match_slots([s["box_xyxy"] for s in report["slots"]], truth["slots"])
            exact_counts += report["empty_slots"] == len(truth["slots"])
            slot_hits += tp
            slot_total += len(truth["slots"])
        self.assertGreaterEqual(exact_counts, 26)
        self.assertGreaterEqual(slot_hits / max(1, slot_total), 0.95)

    def test_truth_boxes_do_not_overlap_and_slots_sit_between_products(self):
        scene = sample_shelf(random.Random(8))
        boxes = [p["box"] for p in scene["products"]]
        for i, a in enumerate(boxes):
            for b in boxes[i + 1:]:
                overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
                self.assertEqual(overlap, 0)
        for slot in scene["slots"]:
            s = slot["box_xyxy"]
            for box in boxes:
                self.assertEqual(max(0, min(s[2], box[2]) - max(s[0], box[0])) * max(0, min(s[3], box[3]) - max(s[1], box[1])), 0)


if __name__ == "__main__":
    unittest.main()
