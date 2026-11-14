"""Synthetic retail fixtures with exact ground truth: shelf photos and an entrance clip.

Shelf photos: a shelving unit with several rows of drawn products (boxes, bottles,
cans) and deliberate empty slots; truth holds every product box, every empty slot
and the per-row counts. Entrance clip: a pinhole camera looks along a store
entrance; walkers move on a measured floor plane, so the floor homography, the
counting line in metres and every crossing are known exactly. Random scenes of both
kinds, in YOLO layout, train the demo detector (classes person, product).

None of this is a photograph. The point is a fully known answer to score against.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from homography import fit_homography

WIDTH, HEIGHT, FPS = 480, 360, 10
CLASSES = ["person", "product"]
PERSON_HEIGHT_M = 1.7

# Camera: 2.8 m above the floor, 1 m behind the floor origin, pitched 25 degrees down.
CAMERA = {"position": (3.0, -1.0, 2.8), "pitch_deg": 25.0, "focal": 300.0, "cx": WIDTH / 2, "cy": HEIGHT / 2}
FLOOR_CALIBRATION_POINTS = [(1.0, 1.0), (5.0, 1.0), (5.0, 8.0), (1.0, 8.0)]
GATE = {"name": "entrance", "a": [5.0, 3.0], "b": [1.0, 3.0]}      # right-to-left when seen from the camera; in = towards the camera
PROMO_ZONE = {"name": "promo_stand", "polygon": [[3.5, 4.0], [5.5, 4.0], [5.5, 6.5], [3.5, 6.5]]}


# ---------------------------------------------------------------- camera model

def project(world_point, camera=CAMERA):
    """Pinhole projection of a world point (x right, y away from the camera, z up) to pixels."""
    px, py, pz = camera["position"]
    pitch = math.radians(camera["pitch_deg"])
    rx, ry, rz = world_point[0] - px, world_point[1] - py, world_point[2] - pz
    forward = ry * math.cos(pitch) - rz * math.sin(pitch)
    down = -ry * math.sin(pitch) - rz * math.cos(pitch)
    if forward <= 1e-6:
        raise ValueError("point is behind the camera")
    return (camera["focal"] * rx / forward + camera["cx"], camera["focal"] * down / forward + camera["cy"])


def floor_calibration(camera=CAMERA):
    return {"units": "m", "floor_points": [list(p) for p in FLOOR_CALIBRATION_POINTS],
            "image_points": [[round(v, 3) for v in project((x, y, 0.0), camera)] for x, y in FLOOR_CALIBRATION_POINTS]}


def person_box(floor_x, floor_y, camera=CAMERA, height_m=PERSON_HEIGHT_M, width_ratio=0.36):
    """Image box of a standing person at a floor position; feet at the bottom edge."""
    foot = project((floor_x, floor_y, 0.0), camera)
    head = project((floor_x, floor_y, height_m), camera)
    height = foot[1] - head[1]
    width = height * width_ratio
    return [foot[0] - width / 2, head[1], foot[0] + width / 2, foot[1]], foot, height


def clip_to_frame(box, width=WIDTH, height=HEIGHT):
    return [max(0.0, min(width, box[0])), max(0.0, min(height, box[1])), max(0.0, min(width, box[2])), max(0.0, min(height, box[3]))]


def visible(box, width=WIDTH, height=HEIGHT, min_fraction=0.25):
    clipped = clip_to_frame(box, width, height)
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return False
    area = (box[2] - box[0]) * (box[3] - box[1])
    return (clipped[2] - clipped[0]) * (clipped[3] - clipped[1]) >= min_fraction * area


# ---------------------------------------------------------------- entrance scenario

def interpolate(keyframes, frame):
    """keyframes: [(frame, x, y), ...] in floor metres; linear, clamped at both ends."""
    if frame <= keyframes[0][0]:
        return keyframes[0][1], keyframes[0][2]
    for (f0, x0, y0), (f1, x1, y1) in zip(keyframes, keyframes[1:]):
        if f0 <= frame <= f1:
            t = (frame - f0) / (f1 - f0) if f1 > f0 else 0.0
            return x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
    return keyframes[-1][1], keyframes[-1][2]


def scenario():
    """Ten walkers over 60 s: six real crossings, one hesitation at the gate, one walk around its end, one loiterer."""
    s = FPS
    return {
        "frames": 60 * s,
        "walkers": [
            {"id": 1, "color": (40, 90, 160), "keyframes": [(0, 3.0, 8.0), (12 * s, 2.5, 1.2)], "note": "in"},
            {"id": 2, "color": (150, 60, 60), "keyframes": [(3 * s, 4.0, 1.5), (16 * s, 3.5, 8.0)], "note": "out"},
            {"id": 3, "color": (60, 130, 80), "keyframes": [(10 * s, 2.0, 8.0), (18 * s, 2.0, 3.15), (21 * s, 2.0, 3.15), (29 * s, 2.0, 8.0)],
             "note": "hesitates on the threshold and returns; no crossing"},
            {"id": 4, "color": (120, 90, 150), "keyframes": [(20 * s, 1.6, 8.0), (32 * s, 1.4, 1.2)], "note": "in, side by side with 5"},
            {"id": 5, "color": (180, 120, 40), "keyframes": [(20 * s, 2.4, 8.0), (32 * s, 2.6, 1.2)], "note": "in, side by side with 4"},
            {"id": 6, "color": (90, 90, 90), "keyframes": [(30 * s, 5.8, 7.5), (40 * s, 5.6, 1.5)], "note": "passes beyond the gate's end; no crossing"},
            {"id": 7, "color": (200, 80, 140), "keyframes": [(15 * s, 4.6, 5.4), (50 * s, 4.4, 5.0)], "note": "loiters at the promo stand"},
            {"id": 8, "color": (70, 150, 170), "keyframes": [(42 * s, 3.2, 1.3), (55 * s, 3.4, 8.0)], "note": "out"},
            {"id": 9, "color": (160, 160, 60), "keyframes": [(45 * s, 2.8, 8.0), (58 * s, 3.0, 1.3)], "note": "in, meets 8 at the gate"},
            {"id": 10, "color": (100, 60, 30), "keyframes": [(50 * s, 1.2, 1.3), (59 * s, 1.0, 8.0)], "note": "out"},
        ],
    }


def walker_state(walker, frame, camera=CAMERA):
    first, last = walker["keyframes"][0][0], walker["keyframes"][-1][0]
    if frame < first or frame > last:
        return None
    x, y = interpolate(walker["keyframes"], frame)
    box, foot, height = person_box(x, y, camera)
    if not visible(box):
        return None
    return {"id": walker["id"], "floor": (x, y), "box": clip_to_frame(box), "foot": foot, "height": height, "color": walker["color"]}


def truth_crossings(spec, gate=GATE):
    """Exact gate crossings from the continuous floor paths (no tracker, no hysteresis)."""
    ax, ay = gate["a"]
    bx, by = gate["b"]
    crossings = []
    for walker in spec["walkers"]:
        keys = walker["keyframes"]
        for (f0, x0, y0), (f1, x1, y1) in zip(keys, keys[1:]):
            d0 = (bx - ax) * (y0 - ay) - (by - ay) * (x0 - ax)
            d1 = (bx - ax) * (y1 - ay) - (by - ay) * (x1 - ax)
            if d0 == 0 or d1 == 0 or (d0 > 0) == (d1 > 0):
                continue
            t = d0 / (d0 - d1)
            cx, cy = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
            along = ((cx - ax) * (bx - ax) + (cy - ay) * (by - ay)) / ((bx - ax) ** 2 + (by - ay) ** 2)
            if not 0.0 <= along <= 1.0:
                continue
            crossings.append({"frame": int(round(f0 + (f1 - f0) * t)), "track_id": walker["id"], "line": gate["name"],
                              "direction": "in" if d1 > 0 else "out", "point": [round(cx, 3), round(cy, 3)]})
    crossings.sort(key=lambda c: (c["frame"], c["track_id"]))
    return crossings


# ---------------------------------------------------------------- drawing

def _floor_grid(draw, camera, color=(150, 152, 146)):
    for x in range(0, 7):
        a, b = project((x, 1.0, 0), camera), project((x, 9.0, 0), camera)
        draw.line([a, b], fill=color, width=1)
    for y in range(1, 10):
        a, b = project((0.0, y, 0), camera), project((6.0, y, 0), camera)
        draw.line([a, b], fill=color, width=1)


def draw_entrance(states, rng, camera=CAMERA, size=(WIDTH, HEIGHT), gate=GATE, promo=PROMO_ZONE):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter

    width, height = size
    image = Image.new("RGB", size, (178, 176, 170))
    draw = ImageDraw.Draw(image, "RGBA")
    horizon = project((3.0, 9.0, 0.0), camera)[1]
    draw.rectangle([0, 0, width, horizon], fill=(96, 100, 108))                         # far wall
    door = [project((2.0, 9.0, 0.0), camera), project((4.0, 9.0, 2.2), camera)]
    draw.rectangle([door[0][0], door[1][1], door[1][0], door[0][1]], fill=(214, 226, 236))  # bright doorway
    _floor_grid(draw, camera)
    polygon = [project((x, y, 0.0), camera) for x, y in promo["polygon"]]
    draw.polygon(polygon, fill=(60, 120, 200, 35), outline=(60, 120, 200, 120))         # floor mat at the promo stand
    stand = [project((5.3, 5.0, 0.0), camera), project((5.6, 5.4, 1.1), camera)]
    draw.rectangle([stand[0][0] - 10, stand[1][1], stand[1][0] + 10, stand[0][1]], fill=(200, 160, 60))  # promo stand (distractor)
    ga, gb = project((gate["a"][0], gate["a"][1], 0.0), camera), project((gate["b"][0], gate["b"][1], 0.0), camera)
    draw.line([ga, gb], fill=(90, 90, 90), width=2)                                     # a floor strip where the gate is
    for state in sorted(states, key=lambda s: s["foot"][1]):
        _draw_person(draw, state)
    if rng.random() < 0.4:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.9)))
    noise = np.random.default_rng(rng.randint(0, 2**31 - 1)).integers(-8, 9, size=(height, width, 3))
    pixels = np.asarray(image, dtype=np.int16) + noise
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), "RGB")


def _draw_person(draw, state):
    fx, fy = state["foot"]
    h = state["height"]
    x0, y0, x1, y1 = state["box"]
    color = state["color"]
    torso = [fx - h * 0.14, fy - h * 0.78, fx + h * 0.14, fy - h * 0.4]
    draw.rectangle(torso, fill=color)
    leg = max(2, int(h * 0.07))
    draw.line([fx - h * 0.08, torso[3], fx - h * 0.12, fy], fill=(50, 50, 70), width=leg)
    draw.line([fx + h * 0.08, torso[3], fx + h * 0.12, fy], fill=(50, 50, 70), width=leg)
    arm = max(2, int(h * 0.05))
    draw.line([torso[0], torso[1] + 4, fx - h * 0.18, torso[3] - 6], fill=color, width=arm)
    draw.line([torso[2], torso[1] + 4, fx + h * 0.18, torso[3] - 6], fill=color, width=arm)
    r = h * 0.11
    draw.ellipse([fx - r, fy - h + 0.5, fx + r, fy - h + 2 * r], fill=(224, 184, 150))


PRODUCT_KINDS = ("box", "bottle", "can")


def draw_shelf(scene, rng, size=(WIDTH, HEIGHT)):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter

    width, height = size
    image = Image.new("RGB", size, (236, 232, 224))
    draw = ImageDraw.Draw(image, "RGBA")
    x0, x1 = scene["extent"]
    draw.rectangle([x0 - 12, 0, x1 + 12, height], fill=(120, 104, 86))                  # shelving unit
    draw.rectangle([x0, 0, x1, height], fill=(228, 222, 210))                             # back panel
    for row in scene["rows"]:
        draw.rectangle([x0 - 12, row["bottom"], x1 + 12, row["bottom"] + 8], fill=(150, 130, 105))   # shelf board
        for tag in row["tags"]:                                                          # price tags: distractors
            draw.rectangle([tag[0], row["bottom"] + 1, tag[1], row["bottom"] + 7], fill=(250, 250, 240), outline=(200, 190, 160))
            draw.line([tag[0] + 2, row["bottom"] + 4, tag[1] - 2, row["bottom"] + 4], fill=(90, 90, 90), width=1)
    for product in scene["products"]:
        _draw_product(draw, product)
    shade = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(shade).rectangle([0, 0, width, height], fill=(0, 0, 0, rng.randint(0, 30)))   # exposure variation
    image = Image.alpha_composite(image.convert("RGBA"), shade).convert("RGB")
    if rng.random() < 0.4:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.8)))
    noise = np.random.default_rng(rng.randint(0, 2**31 - 1)).integers(-7, 8, size=(height, width, 3))
    pixels = np.asarray(image, dtype=np.int16) + noise
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), "RGB")


def _draw_product(draw, product):
    x0, y0, x1, y1 = product["box"]
    color = product["color"]
    dark = tuple(max(0, c - 60) for c in color)
    kind = product["kind"]
    if kind == "box":
        draw.rectangle([x0, y0, x1, y1], fill=color, outline=dark)
        draw.rectangle([x0 + 3, y0 + (y1 - y0) * 0.35, x1 - 3, y0 + (y1 - y0) * 0.55], fill=(245, 245, 245))
    elif kind == "bottle":
        neck = (x1 - x0) * 0.3
        draw.rectangle([x0 + (x1 - x0) / 2 - neck / 2, y0, x0 + (x1 - x0) / 2 + neck / 2, y0 + (y1 - y0) * 0.25], fill=dark)
        draw.rounded_rectangle([x0, y0 + (y1 - y0) * 0.22, x1, y1], radius=max(2, int((x1 - x0) * 0.2)), fill=color, outline=dark)
        draw.rectangle([x0 + 2, y0 + (y1 - y0) * 0.5, x1 - 2, y0 + (y1 - y0) * 0.75], fill=(250, 250, 250))
    else:
        draw.rounded_rectangle([x0, y0, x1, y1], radius=max(2, int((x1 - x0) * 0.15)), fill=color, outline=dark)
        draw.ellipse([x0 + 1, y0 - 1, x1 - 1, y0 + (x1 - x0) * 0.35], fill=(200, 200, 205), outline=dark)
        draw.rectangle([x0 + 2, y0 + (y1 - y0) * 0.4, x1 - 2, y0 + (y1 - y0) * 0.6], fill=(250, 250, 250))


# ---------------------------------------------------------------- shelf scenes

def sample_shelf(rng, size=(WIDTH, HEIGHT), empty_probability=0.12):
    """Random shelving unit: 3-4 rows, products of one kind per row, some slots left empty."""
    width, height = size
    x0 = rng.randint(30, 70)
    x1 = width - rng.randint(30, 70)
    row_count = rng.randint(3, 4)
    pitch = (height - 30) / row_count
    rows, products, slots = [], [], []
    for r in range(row_count):
        bottom = int(30 + pitch * (r + 1)) - 10
        kind = rng.choice(PRODUCT_KINDS)
        base_width = rng.randint(22, 46)
        base_height = min(int(pitch) - 16, rng.randint(36, 70))
        spacing = rng.randint(2, 6)
        palette = [tuple(rng.randint(40, 220) for _ in range(3)) for _ in range(rng.randint(1, 3))]
        x = x0 + rng.randint(2, 8)
        cells, tags = [], []
        while True:                      # products sit on a pitch of width + spacing; some cells are left empty
            w = base_width + rng.randint(-3, 3)
            if x + w > x1 - 2:
                break
            cells.append({"x": x, "w": w, "filled": rng.random() >= empty_probability})
            if rng.random() < 0.5:
                tags.append([x + 2, x + w - 2])
            x += w + spacing
        row_boxes = []
        for cell in cells:
            if cell["filled"]:
                h = base_height + rng.randint(-4, 4)
                box = [float(cell["x"]), float(bottom - h), float(cell["x"] + cell["w"]), float(bottom)]
                products.append({"kind": kind, "box": box, "color": rng.choice(palette), "row": r + 1})
                row_boxes.append(box)
        if not row_boxes:          # a row with nothing on it cannot be audited without a reference width; refill it
            continue
        top = min(b[1] for b in row_boxes)
        runs, run = [], []
        for cell in cells:         # a run of k empty cells is exactly k slots: the truth knows k
            if cell["filled"]:
                if run:
                    runs.append(run)
                run = []
            else:
                run.append(cell)
        if run:                    # trailing empty cells merge with the leftover shelf end below
            runs.append(run)
            runs.pop()
        for run in runs:
            g0, g1 = run[0]["x"], run[-1]["x"] + run[-1]["w"]
            step = (g1 - g0) / len(run)
            for i in range(len(run)):
                slots.append({"row": r + 1, "box_xyxy": [round(g0 + i * step, 2), float(top), round(g0 + (i + 1) * step, 2), float(bottom)]})
        # Free space at the shelf end (empty trailing cells plus the leftover under one pitch) holds
        # as many products as fit on the row's pitch; that is the physical definition of an empty slot.
        last_right = max(b[2] for b in row_boxes)
        trailing = x1 - last_right
        if trailing >= 0.8 * base_width:
            count = max(1, int(round((trailing + spacing) / (base_width + spacing))))
            step = trailing / count
            for i in range(count):
                slots.append({"row": r + 1, "box_xyxy": [round(last_right + i * step, 2), float(top), round(last_right + (i + 1) * step, 2), float(bottom)]})
        rows.append({"row": r + 1, "bottom": bottom, "items": len(row_boxes), "empty_slots": sum(1 for s in slots if s["row"] == r + 1),
                     "kind": kind, "tags": tags})
    return {"extent": [x0, x1], "rows": rows, "products": products, "slots": slots, "items": len(products)}


def shelf_truth(scene):
    return {"items": scene["items"], "rows": [row["items"] for row in scene["rows"]],
            "slots": [s["box_xyxy"] for s in scene["slots"]], "extent": scene["extent"],
            "boxes": [[round(v, 2) for v in p["box"]] for p in scene["products"]]}


# ---------------------------------------------------------------- writers

def yolo_line(class_id, box, width=WIDTH, height=HEIGHT):
    box = clip_to_frame(box, width, height)
    return f"{class_id} {(box[0] + box[2]) / 2 / width:.6f} {(box[1] + box[3]) / 2 / height:.6f} {(box[2] - box[0]) / width:.6f} {(box[3] - box[1]) / height:.6f}"


def write_dataset(root: Path, count=400, seed=5):
    """Half shelf scenes, half entrance scenes, YOLO layout, classes person and product."""
    root = Path(root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    (root / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    rng = random.Random(seed)
    totals = {"person": 0, "product": 0, "shelf_images": 0, "entrance_images": 0}
    for index in range(count):
        stem = f"scene_{index:04d}"
        if index % 2 == 0:
            scene = sample_shelf(rng)
            image = draw_shelf(scene, rng)
            lines = [yolo_line(1, p["box"]) for p in scene["products"]]
            totals["product"] += len(lines)
            totals["shelf_images"] += 1
        else:
            states = []
            for walker_id in range(rng.randint(0, 5)):
                x, y = rng.uniform(0.3, 5.7), rng.uniform(1.2, 8.5)
                box, foot, height = person_box(x, y)
                if visible(box):
                    states.append({"id": walker_id, "floor": (x, y), "box": clip_to_frame(box), "foot": foot, "height": height,
                                   "color": tuple(rng.randint(30, 200) for _ in range(3))})
            image = draw_entrance(states, rng)
            lines = [yolo_line(0, s["box"]) for s in states]
            totals["person"] += len(lines)
            totals["entrance_images"] += 1
        image.save(root / "images" / f"{stem}.png")
        (root / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    summary = {"count": count, "seed": seed, "classes": CLASSES, "totals": totals}
    (root / "synthetic.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def write_shelves(root: Path, count=40, seed=99):
    """Held-out shelf photos with truth.json for the audit evaluation (different seed from the training set)."""
    root = Path(root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    truth = {}
    for index in range(count):
        scene = sample_shelf(rng)
        name = f"shelf_{index:03d}.png"
        draw_shelf(scene, rng).save(root / "images" / name)
        truth[name] = shelf_truth(scene)
    (root / "truth.json").write_text(json.dumps({"seed": seed, "count": count, "shelves": truth}, indent=1) + "\n", encoding="utf-8")
    return truth


def write_clip(root: Path, seed=11):
    root = Path(root)
    (root / "frames").mkdir(parents=True, exist_ok=True)
    spec = scenario()
    rng = random.Random(seed)
    per_frame = []
    for frame in range(spec["frames"]):
        states = [s for s in (walker_state(w, frame) for w in spec["walkers"]) if s is not None]
        draw_entrance(states, rng).save(root / "frames" / f"frame_{frame:04d}.png")
        per_frame.append({"frame": frame, "walkers": [{"id": s["id"], "box": [round(v, 2) for v in s["box"]],
                                                        "floor": [round(v, 3) for v in s["floor"]]} for s in states]})
    video = _write_video(root / "frames", root / "clip.mp4")
    config = {"space": "floor", "margin": 0.3, "anchor": "bottom_center", "floor": floor_calibration(),
              "lines": [GATE], "zones": [PROMO_ZONE]}
    crossings = truth_crossings(spec)
    truth = {"fps": FPS, "width": WIDTH, "height": HEIGHT, "frames": spec["frames"], "seed": seed, "camera": CAMERA,
             "crossings": crossings, "per_frame": per_frame, "video": video,
             "notes": {w["id"]: w["note"] for w in spec["walkers"]}}
    (root / "truth.json").write_text(json.dumps(truth, indent=1) + "\n", encoding="utf-8")
    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return truth


def _write_video(frames_dir: Path, path: Path):
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    from PIL import Image

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        return None
    for png in sorted(frames_dir.glob("*.png")):
        with Image.open(png) as image:
            writer.write(cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR))
    writer.release()
    return path.name


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", help="write the scripted entrance clip, truth.json and config.json here")
    parser.add_argument("--shelves", help="write held-out shelf photos with truth.json here")
    parser.add_argument("--dataset", help="write a mixed YOLO training dataset here")
    parser.add_argument("--count", type=int, default=400, help="dataset image count")
    parser.add_argument("--shelf-count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args(argv)
    if not (args.clip or args.shelves or args.dataset):
        parser.error("give --clip, --shelves and/or --dataset")
    if args.dataset:
        print(json.dumps(write_dataset(Path(args.dataset), args.count, args.seed)))
    if args.shelves:
        truth = write_shelves(Path(args.shelves), args.shelf_count, args.seed + 94)
        print(json.dumps({"shelves": len(truth), "items": sum(t["items"] for t in truth.values()), "slots": sum(len(t["slots"]) for t in truth.values())}))
    if args.clip:
        truth = write_clip(Path(args.clip), args.seed + 6)
        print(json.dumps({"frames": truth["frames"], "crossings": len(truth["crossings"]), "video": truth["video"]}))


if __name__ == "__main__":
    main()
