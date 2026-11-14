# System card: retail counting analytics

**Project owner:** Tech Boss. **Components:** one detector (fine-tuned by Tech Boss with detector-training-pipeline; alternatively the unchanged official TorchVision COCO person detector for flow mode), a pure-Python tracker, line counter, zone occupancy, floor homography and shelf audit written by Tech Boss, and CSV/SQLite/HTML reporting. **Inputs used for the numbers below:** drawn shelf photos and a drawn entrance clip, not photographs or CCTV.

## Intended use

Answer two retail questions on an on-premise CPU machine: how many products are on a shelf and where the empty slots are (a photo at a time), and how many people entered and left through a gate per time bucket, with occupancy of marked zones (a camera feed). Counts are operational estimates for stock checks and footfall reports, not an inventory of record and not a security system.

## Demo detector

TorchVision `fasterrcnn_mobilenet_v3_large_320_fpn` fine-tuned from `FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1` on 400 synthetic scenes (200 shelves with 6,162 product boxes, 200 entrances with 544 person boxes), classes `person` and `product`, 6 epochs, exported to ONNX (opset 11) with ONNX Runtime parity within 3e-4 px. Full training log, held-out metrics and export details: `examples/model/`.

| Detector input | Test mAP@0.5 | person AP@0.5 | product AP@0.5 | Test mAP@[.5:.95] | ONNX Runtime median (480x360, 4 threads) |
|---|---|---|---|---|---|
| 320 px (default) | 0.945 | 0.943 | 0.948 | 0.672 | 43.8 ms |
| 512 px (used) | 0.985 | 0.980 | 0.990 | 0.852 | 57.0 ms |

Products in these scenes are 19 to 49 px wide. At the default 320-px internal resize the narrow ones (under about 35 px) were missed on 114 of 1,204 held-out products, which the shelf audit then reported as empty slots; the 512-px model misses 3. Object size relative to the detector's input is the first thing to settle on a client's photos.

## Shelf audit on 40 held-out shelves

1,204 products and 212 empty slots in truth. Slots are matched by IoU at least 0.3. The audit was run with the default shelf extent (the span of detected products), as a client photo would be.

| Pipeline | Count MAE | Exact counts | Rows exact | Slot precision | Slot recall |
|---|---|---|---|---|---|
| 320-px detector + audit | 2.03 | 20% | 15% | 0.62 | 0.83 |
| 512-px detector + audit (used) | 0.08 | 95% | 95% | 0.99 | 0.83 |
| truth boxes + audit (logic only) | 0.00 | 100% | 100% | 1.00 | 0.83 |

All 37 slots the logic-only run misses are at the right end of a row, where nothing marks the shelf edge once the last product is gone; with the shelf extent given (`--extent`, known for a fixed camera) the same logic reaches 211 of 212 slots with 3 false ones. The two false slots of the 512-px run come from its three missed products. Median detector time 86 ms per 480x360 photo.

## People counting on the scripted clip

600 frames at 10 fps, ten walkers, seven true gate crossings (4 in, 3 out), one walker stopping 15 cm short of the line and turning back, one walking past the end of the gate, one standing 35 s at the promo stand. Counting in floor metres through the homography fitted from four calibration points (residual 0.0 px); hysteresis margin 0.3 m; crossings matched to truth by line, direction and a 15-frame tolerance.

| | Truth | Counted | False | Missed | Median latency | Max latency |
|---|---|---|---|---|---|---|
| in | 4 | 4 | 0 | 0 | 6 frames | 7 frames |
| out | 3 | 3 | 0 | 0 | 6 frames | 7 frames |

Net count error 0; the occupancy curve (in minus out) matches the truth on 94% of frames, the rest being the hysteresis delay of under a second after each crossing. The two decoys produced no count. The promo zone (5 m²) was occupied 66% of the time with a peak of one person, as scripted. The tracker opened 12 identities for 10 walkers: two re-identifications after a walker left and re-entered the view, none of them at the gate, so no double count. Per-bucket totals are in `examples/flow/totals.csv`. The 320-px detector gave the same 7/7 result on this clip (people are 57 to 150 px tall here); the committed run uses the 512-px model so both modes share one file. Median detector time 70 ms and 74 ms per frame end to end (detect, track, count, overlay, encode) on 4 threads; 46 s for the 60-second clip.

## Real footage: a mall walkway (Pexels 1338594)

An 8-second hand-held clip of a busy mall walkway (1920x1080, 30 fps, free Pexels licence; source in `examples/FOOTAGE_SOURCES.md`), counted across a vertical line drawn across the walkway in pixel space with the official COCO Faster R-CNN person backend at a 640 px resize. No floor calibration is possible (the camera moves), so this is a pixel-space count only.

| Setting | Result |
|---|---|
| First attempt: 12 px hysteresis, no ignore zone | 13 crossings (3 in, 10 out) from 66 tracks: people pausing on the line were counted back and forth, and a mirrored pillar produced a phantom track |
| Shipped settings: 40 px hysteresis, the pillar as an ignore zone | 5 crossings (1 in, 4 out) from 65 tracks; by eye, four to five people walk past the line to the right during the clip, and the single `in` is one person stepping back at the line |

Cost: 0.42 s per 1080p frame on 4 CPU threads. The clip shows what the synthetic entrance could not: a hand-held camera, a crowd with children partly hidden behind adults, and reflections. It is a check that the pipeline runs on real video and that the two guards (hysteresis, ignore zones) matter; a counting accuracy claim needs a fixed camera over a doorway with a hand count, which is the first thing to record at a client site.

## What was not measured

- Any photograph or real footage. Real shelves have overlapping, leaning and rotated products, glare, shadows, price-tag clutter and depth stacking; real people occlude each other, carry bags and trolleys, and walk in groups that a tracker can merge.
- Product identity: the audit counts instances and gaps, not SKUs or planogram compliance.
- The `torchvision-person` backend on these drawings or on real video; it is wired and checked for its output contract only.
- Camera motion, re-calibration drift, lens distortion (the homography assumes an undistorted image of a planar floor), night lighting, RTSP reconnects, continuous operation and storage rotation.
- Crowds: hysteresis and tracking were exercised with at most four people in view.

## Privacy and licence

No face or identity recognition; tracks are numbered per run and the outputs hold boxes, counts and the snapshots the operator chooses to keep. TorchVision code and checkpoints are BSD-3-Clause, ONNX Runtime and OpenCV are MIT/Apache; original code is MIT; no AGPL component. Public shelf datasets with research-only licences (SKU-110K) were not used.

## Before client deployment

Collect shelf photos from the actual cameras or phones and label products (and, if wanted, SKUs); record entrance footage with the store's consent, measure four floor points per camera with a tape, label people and count a reference hour by hand; retrain with `--image-size` chosen from the measured product and person sizes; set `--extent` per shelf camera; re-tune `margin`, `min_gap_ratio` and tracker settings on that data; agree how occupancy is reset each day; and measure latency on the target machine.
