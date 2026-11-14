"""Overlays (Pillow), a top-down floor plan (SVG) and the static HTML dashboards."""
from __future__ import annotations

import html
import json

GREEN = (40, 180, 90)
RED = (220, 60, 40)
BLUE = (60, 120, 200)


def _line_in_image(line, plane):
    a, b = line["a"], line["b"]
    if plane is not None:
        a, b = plane.image(a), plane.image(b)
    return tuple(a), tuple(b)


def _polygon_in_image(polygon, plane):
    return [tuple(plane.image(p)) if plane is not None else tuple(p) for p in polygon]


def _label(draw, position, text, color):
    x, y = position
    draw.rectangle([x - 2, y - 1, x + draw.textlength(text) + 3, y + 12], fill=color + (210,))
    draw.text((x, y), text, fill=(255, 255, 255, 255))


def draw_flow_frame(image, config, plane, tracks, anchors, trails, counter, zone_counts, frame_index, fps):
    from PIL import ImageDraw

    canvas = image.copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    for zone in config["zones"]:
        points = _polygon_in_image(zone["polygon"], plane)
        draw.polygon(points, fill=BLUE + (35,), outline=BLUE + (255,))
        _label(draw, (points[0][0] + 3, points[0][1] + 2), f"{zone['name']}: {zone_counts.get(zone['name'], 0)}", BLUE)
    for line in config["lines"]:
        a, b = _line_in_image(line, plane)
        draw.line([a, b], fill=RED + (255,), width=3)
        totals = counter.totals[line["name"]]
        _label(draw, (min(a[0], b[0]) + 4, max(a[1], b[1]) + 4), f"{line['name']}  in {totals['in']}  out {totals['out']}", RED)
    for track in tracks:
        x0, y0, x1, y1 = track["box_xyxy"]
        draw.rectangle([x0, y0, x1, y1], outline=GREEN + (255,), width=2)
        draw.rectangle([x0, max(0, y0 - 13), x0 + 26, max(0, y0 - 13) + 13], fill=GREEN + (230,))
        draw.text((x0 + 3, max(0, y0 - 13)), f"#{track['track_id']}", fill=(255, 255, 255, 255))
        trail = trails.get(track["track_id"], [])
        if len(trail) > 1:
            draw.line([tuple(p) for p in trail[-30:]], fill=GREEN + (160,), width=2)
    net = sum(t["in"] - t["out"] for t in counter.totals.values())
    banner = f"frame {frame_index}  t={frame_index / fps:5.1f}s  tracks {len(tracks)}  net {net:+d}"
    draw.rectangle([0, 0, draw.textlength(banner) + 8, 14], fill=(0, 0, 0, 160))
    draw.text((4, 1), banner, fill=(255, 255, 255, 255))
    return canvas


def draw_shelf_audit(image, report, products):
    from PIL import ImageDraw

    canvas = image.copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    for product in products:
        x0, y0, x1, y1 = product["box_xyxy"]
        draw.rectangle([x0, y0, x1, y1], outline=GREEN + (255,), width=2)
    for slot in report["slots"]:
        x0, y0, x1, y1 = slot["box_xyxy"]
        draw.rectangle([x0, y0, x1, y1], fill=RED + (50,), outline=RED + (255,), width=2)
        step = 8
        x = x0 - (y1 - y0)
        while x < x1:
            draw.line([max(x0, x), y1 if x >= x0 else y1 - (x0 - x), min(x1, x + (y1 - y0)), y0 if x + (y1 - y0) <= x1 else y0 + (x + (y1 - y0) - x1)],
                      fill=RED + (120,), width=1)
            x += step
    for row in report["rows"]:
        text = f"row {row['row']}: {row['items']} items, {row['empty_slots']} empty"
        y = row["bottom"] + 9
        draw.rectangle([2, y, draw.textlength(text) + 8, y + 13], fill=(0, 0, 0, 170))
        draw.text((5, y), text, fill=(255, 255, 255, 255))
    headline = f"{report['items']} items, {report['empty_slots']} empty slots"
    draw.rectangle([0, 0, draw.textlength(headline) + 10, 15], fill=(0, 0, 0, 190))
    draw.text((5, 1), headline, fill=(255, 255, 255, 255))
    return canvas


def floor_plan_svg(config, floor_trails, crossings, bounds=None, scale=60):
    """Top-down plan in floor units: lines, zones, every track's trail and crossing points."""
    points = [p for trail in floor_trails.values() for p in trail]
    for line in config["lines"]:
        points += [line["a"], line["b"]]
    for zone in config["zones"]:
        points += list(zone["polygon"])
    if not points:
        return "<svg xmlns='http://www.w3.org/2000/svg' width='10' height='10'></svg>"
    if bounds is None:
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        bounds = (min(xs) - 0.5, min(ys) - 0.5, max(xs) + 0.5, max(ys) + 0.5)
    x0, y0, x1, y1 = bounds
    width, height = (x1 - x0) * scale, (y1 - y0) * scale

    def sx(x):
        return (x - x0) * scale

    def sy(y):
        return height - (y - y0) * scale     # larger floor y (further from the camera) is drawn higher

    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width:.0f}' height='{height:.0f}' viewBox='0 0 {width:.0f} {height:.0f}' style='background:#f8fafc;border:1px solid #e2e8f0'>"]
    for gx in range(int(x0), int(x1) + 1):
        parts.append(f"<line x1='{sx(gx):.1f}' y1='0' x2='{sx(gx):.1f}' y2='{height:.0f}' stroke='#e2e8f0'/>")
    for gy in range(int(y0), int(y1) + 1):
        parts.append(f"<line x1='0' y1='{sy(gy):.1f}' x2='{width:.0f}' y2='{sy(gy):.1f}' stroke='#e2e8f0'/>")
    for zone in config["zones"]:
        path = " ".join(f"{sx(p[0]):.1f},{sy(p[1]):.1f}" for p in zone["polygon"])
        parts.append(f"<polygon points='{path}' fill='rgba(60,120,200,0.15)' stroke='#3c78c8'/>")
        parts.append(f"<text x='{sx(zone['polygon'][0][0]) + 4:.1f}' y='{sy(zone['polygon'][0][1]) - 4:.1f}' font-size='11' fill='#3c78c8'>{html.escape(zone['name'])}</text>")
    for line in config["lines"]:
        a, b = line["a"], line["b"]
        parts.append(f"<line x1='{sx(a[0]):.1f}' y1='{sy(a[1]):.1f}' x2='{sx(b[0]):.1f}' y2='{sy(b[1]):.1f}' stroke='#dc3c28' stroke-width='3'/>")
        parts.append(f"<text x='{sx(min(a[0], b[0])):.1f}' y='{sy(a[1]) - 6:.1f}' font-size='11' fill='#dc3c28'>{html.escape(line['name'])} (in = towards the bottom)</text>")
    palette = ["#2563eb", "#16a34a", "#d97706", "#7c3aed", "#db2777", "#0891b2", "#65a30d", "#ea580c", "#4f46e5", "#be123c"]
    for index, (track_id, trail) in enumerate(sorted(floor_trails.items())):
        if len(trail) < 2:
            continue
        path = " ".join(f"{sx(p[0]):.1f},{sy(p[1]):.1f}" for p in trail)
        color = palette[index % len(palette)]
        parts.append(f"<polyline points='{path}' fill='none' stroke='{color}' stroke-width='1.5' opacity='0.8'/>")
        parts.append(f"<text x='{sx(trail[-1][0]) + 3:.1f}' y='{sy(trail[-1][1]):.1f}' font-size='10' fill='{color}'>#{track_id}</text>")
    for crossing in crossings:
        px, py = crossing["point"]
        fill = "#16a34a" if crossing["direction"] == "in" else "#dc2626"
        parts.append(f"<circle cx='{sx(px):.1f}' cy='{sy(py):.1f}' r='4' fill='{fill}'><title>#{crossing['track_id']} {crossing['direction']} frame {crossing['frame']}</title></circle>")
    parts.append("</svg>")
    return "".join(parts)


STYLE = """body{font-family:system-ui,sans-serif;margin:24px;color:#1f2937;background:#f8fafc} h1{margin:0 0 4px} h2{margin-top:28px} .muted{color:#64748b}
table{border-collapse:collapse;margin:12px 0 20px;background:#fff} td,th{border:1px solid #e2e8f0;padding:6px 10px;text-align:left;vertical-align:top}
img{max-width:480px;display:block} .in{color:#16a34a;font-weight:600} .out{color:#dc2626;font-weight:600} code{background:#eef2ff;padding:1px 4px}
.big{font-size:28px;font-weight:700} .cards{display:flex;gap:18px;flex-wrap:wrap} .card{background:#fff;border:1px solid #e2e8f0;padding:12px 18px;min-width:120px}
.gallery{display:flex;flex-wrap:wrap;gap:18px} .tile{background:#fff;border:1px solid #e2e8f0;padding:10px}"""


def flow_dashboard(run, crossings, buckets, zones, svg, snapshots):
    totals = run["totals"]
    net = sum(t["in"] - t["out"] for t in totals.values())
    cards = "".join(f"<div class='card'><div class='muted'>{html.escape(name)} in</div><div class='big in'>{t['in']}</div>"
                    f"<div class='muted'>{html.escape(name)} out</div><div class='big out'>{t['out']}</div></div>" for name, t in totals.items())
    cards += f"<div class='card'><div class='muted'>net change</div><div class='big'>{net:+d}</div></div>"
    bucket_rows = "".join(f"<tr><td>{r['bucket']}</td><td>{html.escape(r.get('start_time') or '')}</td><td>{r['start_seconds']:.0f}-{r['end_seconds']:.0f}s</td>"
                          f"<td class='in'>{r['in']}</td><td class='out'>{r['out']}</td><td>{r['occupancy_end']}</td></tr>" for r in buckets)
    zone_rows = "".join(f"<tr><td>{html.escape(name)}</td><td>{z['peak']}</td><td>{z['mean']:.2f}</td><td>{z['occupied_fraction']:.0%}</td>"
                        f"<td>{'' if z['area_m2'] is None else f'{z['area_m2']:.1f}'}</td><td>{'' if z.get('peak_density_per_m2') is None else f'{z['peak_density_per_m2']:.2f}'}</td></tr>"
                        for name, z in zones.items())
    crossing_rows = "".join(f"<tr><td>{c['frame']}</td><td>{c['frame'] / run['fps']:.1f}s</td><td>#{c['track_id']}</td><td>{html.escape(c['line'])}</td>"
                            f"<td class='{c['direction']}'>{c['direction']}</td></tr>" for c in crossings)
    snaps = "".join(f"<div class='tile'><img src='{html.escape(path)}' alt='frame {frame}'><div class='muted'>frame {frame}</div></div>" for frame, path in snapshots)
    evaluation = ""
    if run.get("evaluation"):
        e = run["evaluation"]
        evaluation = (f"<h2>Against scripted truth</h2><p>{e['true_positive']} of {e['truth']} crossings matched within {e['tolerance_frames']} frames, "
                      f"{e['false_positive']} false, {e['false_negative']} missed; median latency {e['median_latency_frames']} frames; "
                      f"occupancy curve exact on {e['occupancy_frames_exact']:.0%} of frames.</p>")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>People counting</title><style>{STYLE}</style></head><body>
<h1>People counting and occupancy</h1>
<p class="muted">Source: {html.escape(run['source'])} · {run['frames']} frames at {run['fps']:.1f} fps · detector {html.escape(run['detector']['backend'])}
· space <code>{run['config']['space']}</code> · median detector time {run['timing']['detector_median_ms']:.1f} ms/frame · annotated video <code>{html.escape(run['annotated_video'])}</code></p>
<div class="cards">{cards}</div>
<h2>Totals per {run['bucket_seconds']:.0f} s bucket</h2>
<table><tr><th>#</th><th>Start</th><th>Window</th><th>In</th><th>Out</th><th>Occupancy at end</th></tr>{bucket_rows}</table>
<p class="muted">Also in <code>totals.csv</code> and the <code>buckets</code> table of <code>counts.sqlite</code>; occupancy is the running in-minus-out from the start of the recording.</p>
<h2>Zones</h2>
<table><tr><th>Zone</th><th>Peak</th><th>Mean</th><th>Occupied</th><th>Area m²</th><th>Peak per m²</th></tr>{zone_rows}</table>
<h2>Floor plan</h2>
<p class="muted">Foot points mapped through the floor homography (calibration residual {run['floor_residual_px'] if run.get('floor_residual_px') is not None else 'n/a'} px); trails per track, dots are counted crossings.</p>
{svg}
<h2>Crossings</h2>
<table><tr><th>Frame</th><th>Time</th><th>Track</th><th>Line</th><th>Direction</th></tr>{crossing_rows}</table>
{evaluation}
<h2>Snapshots</h2><div class="gallery">{snaps}</div>
</body></html>
"""


def shelf_dashboard(run, reports):
    tiles = []
    for report in reports:
        rows = "".join(f"<tr><td>{r['row']}</td><td>{r['items']}</td><td>{r['empty_slots']}</td></tr>" for r in report["audit"]["rows"])
        truth = ""
        if report.get("truth_items") is not None:
            truth = f"<div class='muted'>truth: {report['truth_items']} items, {report['truth_slots']} empty slots</div>"
        tiles.append(f"<div class='tile'><img src='{html.escape(report['annotated'])}' alt='{html.escape(report['image'])}'>"
                     f"<div class='big'>{report['audit']['items']} items, {report['audit']['empty_slots']} empty slots</div>{truth}"
                     f"<div class='muted'>{html.escape(report['image'])} · {report['detections']} detections ≥ {run['score']}</div>"
                     f"<table><tr><th>Row</th><th>Items</th><th>Empty</th></tr>{rows}</table></div>")
    evaluation = ""
    if run.get("evaluation"):
        e = run["evaluation"]
        o = run.get("evaluation_oracle") or {}
        evaluation = (f"<h2>Against truth ({e['images']} shelves)</h2><table><tr><th></th><th>Count MAE</th><th>Exact counts</th><th>Rows exact</th><th>Slot precision</th><th>Slot recall</th></tr>"
                      f"<tr><td>detector + audit</td><td>{e['count_mae']:.2f}</td><td>{e['exact_count_fraction']:.0%}</td><td>{e['rows_exact_fraction']:.0%}</td><td>{e['slot_precision']:.2f}</td><td>{e['slot_recall']:.2f}</td></tr>"
                      + (f"<tr><td>truth boxes + audit</td><td>{o['count_mae']:.2f}</td><td>{o['exact_count_fraction']:.0%}</td><td>{o['rows_exact_fraction']:.0%}</td><td>{o['slot_precision']:.2f}</td><td>{o['slot_recall']:.2f}</td></tr>" if o else "")
                      + "</table><p class='muted'>The second row runs the same audit on the ground-truth boxes, so it isolates the row/gap logic from detector errors.</p>")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Shelf audit</title><style>{STYLE}</style></head><body>
<h1>Shelf audit</h1>
<p class="muted">{len(reports)} photos · detector {html.escape(run['detector']['backend'])} · score ≥ {run['score']} · median detector time {run["timing"]["detector_median_ms"]:.1f} ms per photo
· empty slot = gap ≥ {run['settings']['min_gap_ratio']} × median product width in the row</p>
{evaluation}
<div class="gallery">{''.join(tiles)}</div>
</body></html>
"""
