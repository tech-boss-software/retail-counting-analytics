# Data and output provenance

## Fixtures

No photographs or camera footage are used. `synthetic.py` draws everything with Pillow and records the answer alongside:

- **Shelf photos** (480x360): a shelving unit of 3 to 4 rows; each row holds products of one kind (box, bottle or can) placed on a pitch of product width plus a small spacing, with cells left empty at random (12% each) and price tags on the shelf edge as distractors. Truth per photo: every product box, the per-row item counts and every empty slot (a run of k empty cells is k slots; free space at the shelf end wide enough for a product is a slot too). The 40 held-out photos (`--shelves`, seed 99) are generated apart from the training scenes.
- **Entrance clip** (600 frames, 60 s at 10 fps): a pinhole camera 2.8 m above the floor pitched 25 degrees down, a floor grid, a doorway, a promo stand and ten walkers moving along scripted floor paths in metres. The floor calibration in `config.json` (four floor points and their projections) is derived from the same camera, so the homography the counter fits is exact; a real site replaces it with four measured points. Truth: per-frame boxes and floor positions, and seven gate crossings computed from the continuous paths (4 in, 3 out). Two decoys do not cross: one walker stops 15 cm short of the line and turns back, another passes beyond the end of the gate; a third stands at the promo stand for 35 s.
- **Training set** (`--dataset`, seed 5): 200 random shelf scenes and 200 random entrance scenes in YOLO layout, classes `person` and `product`. Seeded blur, exposure and pixel noise make images differ.

## Real footage

`mall_rush_pexels_1338594.mp4`: "Rush hour in the mall" by Richarles Moral on Pexels (Pexels License: free for commercial use and modification, attribution optional), 8 s, 1920x1080, 29.97 fps, downloaded on 14 November 2026; author and URL in `examples/FOOTAGE_SOURCES.md`. Hand-held, so no floor calibration; counted in pixel space.

## Demo detector

Trained with the sibling `detector-training-pipeline` project (TorchVision Faster R-CNN MobileNetV3-Large 320 FPN fine-tuned from the official COCO checkpoint) on the 400 scenes with `--image-size 512`, exported to ONNX. Training log, held-out metrics and export parity are in `examples/model/`; the weights are not committed. A first model trained at the default 320-px input is kept as `examples/model/evaluation-320px.json`, and its shelf audit as `examples/shelf-320px/evaluation.json`. The `torchvision-person` backend uses the official `SSDLite320_MobileNet_V3_Large_Weights.COCO_V1` checkpoint unchanged and was not exercised on these drawings.

## Executed run

The committed `examples/` were produced by one actual local run on 14 November 2026 on a laptop CPU (Python 3.12.14, torch 2.15.0+cpu, torchvision 0.29.2+cpu, onnxruntime 1.32.0, OpenCV 5.0.0 headless):

```powershell
python synthetic.py --dataset artifacts/dataset --shelves artifacts/shelf-test --clip artifacts/clip --count 400 --seed 5
python ..\detector-training-pipeline\pipeline.py --data artifacts\dataset --output artifacts\model-512 --epochs 6 --threads 4 --image-size 512 --cache-dir 'D:\Tech Boss\cv-cache'
python audit_shelf.py --input artifacts\shelf-test\images --detector onnx:artifacts\model-512\model.onnx --output artifacts\shelf-run --truth artifacts\shelf-test\truth.json --threads 4
python count_flow.py --input artifacts\clip\clip.mp4 --config artifacts\clip\config.json --detector onnx:artifacts\model-512\model.onnx --output artifacts\flow-run --truth artifacts\clip\truth.json --bucket 10 --start-time 2026-11-14T09:00:00 --snapshots 90,290,550 --threads 4
```

- `examples/shelf/`: `dashboard.html`, `shelf_report.json`, `evaluation.json` (detector + audit, and the audit on truth boxes) and the annotated held-out photos.
- `examples/flow/`: `dashboard.html`, `run.json` (config, calibration residual, timings, detector identity), `crossings.json`, `totals.csv`, `counts.sqlite`, `evaluation.json`, three snapshot frames and `annotated.mp4` (mp4v).
- `examples/model/`: `training_log.json`, `evaluation.json`, `export.json`, `dataset_report.json`, `MODEL_CARD.md`, `pr_curves.png` from the training pipeline.

- `examples/flow-real/`: the mall clip run (`run.json`, `crossings.json`, `totals.csv`, `counts.sqlite`, `dashboard.html`, snapshots as JPEG, a half-size annotated clip).

Timings are from this one machine and run and are not a hardware benchmark.
