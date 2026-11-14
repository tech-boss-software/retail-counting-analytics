# Model card: fine-tuned detector

**Project owner:** Tech Boss. **Base checkpoint:** official TorchVision `fasterrcnn_mobilenet_v3_large_320_fpn` (FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1). **Fine-tuning performed by Tech Boss:** yes, on the dataset described below.

## Intended use

Demonstrate a reproducible fine-tuning, evaluation and ONNX export pipeline for custom object classes. The numbers below describe the synthetic demonstration dataset only; they say nothing about any client domain until the pipeline is re-run on representative, licensed client data.

## Data

- Source: `artifacts\dataset` (operator-supplied dataset at artifacts\dataset)
- Images: 400 (0 without objects), objects: 6706
- Objects per class: person 544, product 6162
- Split (seed 0): train 280, val 60, test 60 images; the test split was not used for model selection
- Dataset fingerprint: `047b6a925092bb60`

## Training

- Epochs 6, batch 4, SGD lr 0.01, momentum 0.9, weight decay 0.0005, seed 0, device cpu, threads 4
- Augmentation: random horizontal flip (p=0.5)
- Internal resize: shorter side 512 px (the base checkpoint was trained at 320)
- Best epoch by validation mAP@0.5: 6 (val mAP@0.5 0.985)
- Wall time: 1108 s; software: torch 2.15.0+cpu, torchvision 0.29.2+cpu

| Epoch | Train loss | Val mAP@0.5 |
|---|---|---|
| 1 | 0.7090 | 0.984 |
| 2 | 0.5976 | 0.983 |
| 3 | 0.4360 | 0.985 |
| 4 | 0.5316 | 0.975 |
| 5 | 0.3337 | 0.984 |
| 6 | 0.3183 | 0.985 |

## Evaluation on the held-out test split

- mAP@0.5: **0.985**, mAP@[0.5:0.95]: **0.852** over 2 classes with ground truth
- Method: class-aware greedy one-to-one matching, 101-point interpolated AP averaged over IoU 0.50:0.95; no crowd/ignore handling; classes without ground truth are excluded, not scored zero

| Class | Ground truth | Predictions | AP@0.5 | AP@[0.5:0.95] |
|---|---|---|---|---|
| person | 65 | 65 | 0.980 | 0.822 |
| product | 1020 | 1021 | 0.990 | 0.883 |

Confusion matrix at score >= 0.5 and IoU >= 0.5 (rows = ground truth, columns = predictions):

| | person | product | background |
|---|---|---|---|
| **person** | 62 | 0 | 3 |
| **product** | 0 | 1018 | 2 |
| **background** | 0 | 0 | 0 |

## ONNX export

- Exporter: torch.onnx.export (TorchScript-based, dynamo=False), traced on a labelled training image, opset 11, dynamic image size; file size 76.0 MB, SHA-256 `c498411a292e1a3b` (weights are not committed)
- Parity on 8 test images: max box difference 0.00021 px, max score difference 0.000042, label mismatches 0
- Latency (single image, 480x360, 4 CPU threads, 20 runs after warm-up): PyTorch eager median 110.9 ms (p95 133.0), ONNX Runtime median 57.0 ms (p95 69.9) on Intel64 Family 6 Model 154 Stepping 3, GenuineIntel
- Latency is a local measurement on one machine; it is not a benchmark of hardware.

## Licence note

TorchVision model code and the COCO-pretrained checkpoint are BSD-3-Clause licensed; original pipeline code is MIT. No AGPL component is used, so the exported model can be deployed in a client product without copyleft obligations. Confirm the client dataset's own terms separately.

## Limitations

- The demonstration dataset is synthetic: flat shapes on noisy backgrounds. High scores here do not transfer to photographs, occlusion, small objects, motion blur or class imbalance.
- Scores are not calibrated probabilities; the confusion matrix threshold is a choice, not a fact.
- Greedy matching without crowd/ignore semantics differs slightly from pycocotools on crowded scenes.
- CPU training uses a small input size (320 px); fine detail is lost.
- No tracking, segmentation, drift monitoring or serving layer is included.

## Before client deployment

Re-run the pipeline on representative licensed data with every object labelled, inspect per-class and per-slice results, compare the ONNX latency on the target hardware, choose thresholds with a human-review boundary, and keep the dataset fingerprint and metrics JSON with the delivered model.
