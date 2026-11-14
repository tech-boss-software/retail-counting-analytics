"""Detector backends behind one interface: detect(PIL image) -> list of {box_xyxy, score, class_id}.

- onnx:<path>: a model exported by detector-training-pipeline (class names in its metadata),
  run with ONNX Runtime only. For this project: classes person and product.
- torchvision-person: the official COCO-pretrained SSDLite320 MobileNetV3 from
  TorchVision, keeping only the person class. It needs no training and serves the
  people-counting mode on real footage.
- torchvision-person-frcnn[:min_size]: the official COCO Faster R-CNN MobileNetV3
  with an adjustable resize, better for small or distant people.
- torchvision-coco:<categories>:<class>[:min_size]: the same Faster R-CNN keeping any
  COCO categories under one name, e.g. torchvision-coco:bottle:product:1024 lets the
  shelf audit count bottles on real photos without training. Products that are not
  a COCO category still need an ONNX model trained on the client's photos.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


class OnnxDetector:
    def __init__(self, path, threads=4, score_threshold=0.05):
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = int(threads)
        self.session = onnxruntime.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        meta = self.session.get_modelmeta().custom_metadata_map
        if "classes" not in meta:
            raise ValueError("ONNX model lacks 'classes' metadata; export it with detector-training-pipeline")
        names = json.loads(meta["classes"])
        self.labels = dict(enumerate(names))
        self.score_threshold = score_threshold
        self.path = Path(path)

    def detect(self, image):
        import numpy as np

        array = np.ascontiguousarray((np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0).transpose(2, 0, 1))
        boxes, labels, scores = self.session.run(None, {"image": array})
        width, height = image.size
        result = []
        for box, label, score in zip(boxes.tolist(), labels.tolist(), scores.tolist()):
            if label == 0 or score < self.score_threshold:
                continue
            clipped = [max(0.0, min(width, box[0])), max(0.0, min(height, box[1])), max(0.0, min(width, box[2])), max(0.0, min(height, box[3]))]
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                continue
            result.append({"box_xyxy": clipped, "score": float(score), "class_id": int(label) - 1})
        return result

    def info(self):
        import onnxruntime

        return {"backend": "onnx", "model": str(self.path), "sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
                "classes": self.labels, "onnxruntime": onnxruntime.__version__}


class TorchvisionPersonDetector:
    def __init__(self, threads=4, cache_dir=None, score_threshold=0.05):
        if cache_dir:
            os.environ["TORCH_HOME"] = str(Path(cache_dir).resolve())
        import torch
        import torchvision
        from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights

        torch.set_num_threads(int(threads))
        self.weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
        self.model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=self.weights).eval()
        self.person_index = self.weights.meta["categories"].index("person")
        self.labels = {0: "person"}
        self.score_threshold = score_threshold
        self.torch = torch

    def detect(self, image):
        import numpy as np

        tensor = self.torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()
        with self.torch.no_grad():
            output = self.model([tensor])[0]
        width, height = image.size
        result = []
        for box, label, score in zip(output["boxes"].tolist(), output["labels"].tolist(), output["scores"].tolist()):
            if label != self.person_index or score < self.score_threshold:
                continue
            clipped = [max(0.0, min(width, box[0])), max(0.0, min(height, box[1])), max(0.0, min(width, box[2])), max(0.0, min(height, box[3]))]
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                continue
            result.append({"box_xyxy": clipped, "score": float(score), "class_id": 0})
        return result

    def info(self):
        import torchvision

        return {"backend": "torchvision-person", "weights": str(self.weights), "url": self.weights.url,
                "classes": self.labels, "torchvision": torchvision.__version__}


class TorchvisionCocoDetector(TorchvisionPersonDetector):
    """The official COCO Faster R-CNN MobileNetV3-Large 320 FPN with an adjustable internal resize,
    keeping one or more COCO categories and exposing them under one class name.

    `torchvision-person-frcnn[:size]` keeps `person`; `torchvision-coco:bottle+cup:product[:size]`
    keeps bottles and cups and calls them `product`, which lets the shelf audit run on real
    photos of drink shelves without any training. `min_size` is the shorter side after the
    internal resize; 320 is the checkpoint default, larger keeps small objects.
    """

    def __init__(self, threads=4, cache_dir=None, score_threshold=0.05, min_size=640, categories=("person",), label="person", backend="torchvision-person-frcnn"):
        if cache_dir:
            os.environ["TORCH_HOME"] = str(Path(cache_dir).resolve())
        import torch
        import torchvision
        from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights

        torch.set_num_threads(int(threads))
        self.weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
        self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=self.weights).eval()
        self.model.transform.min_size = (int(min_size),)
        self.model.transform.max_size = int(min_size) * 2
        names = self.weights.meta["categories"]
        unknown = [c for c in categories if c not in names]
        if unknown:
            raise ValueError(f"not COCO categories: {unknown}")
        self.kept = {names.index(c) for c in categories}
        self.categories = list(categories)
        self.backend = backend
        self.labels = {0: label}
        self.score_threshold = score_threshold
        self.torch = torch

    def detect(self, image):
        import numpy as np

        tensor = self.torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()
        with self.torch.no_grad():
            output = self.model([tensor])[0]
        width, height = image.size
        result = []
        for box, label, score in zip(output["boxes"].tolist(), output["labels"].tolist(), output["scores"].tolist()):
            if label not in self.kept or score < self.score_threshold:
                continue
            clipped = [max(0.0, min(width, box[0])), max(0.0, min(height, box[1])), max(0.0, min(width, box[2])), max(0.0, min(height, box[3]))]
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                continue
            result.append({"box_xyxy": clipped, "score": float(score), "class_id": 0})
        return result

    def info(self):
        import torchvision

        return {"backend": self.backend, "weights": str(self.weights), "url": self.weights.url, "min_size": self.model.transform.min_size[0],
                "coco_categories": self.categories, "classes": self.labels, "torchvision": torchvision.__version__}


class TorchvisionPersonFRCNN(TorchvisionCocoDetector):
    def __init__(self, threads=4, cache_dir=None, score_threshold=0.05, min_size=640):
        super().__init__(threads, cache_dir, score_threshold, min_size, ("person",), "person", "torchvision-person-frcnn")


def load_detector(spec, threads=4, cache_dir=None):
    if spec.startswith("onnx:"):
        return OnnxDetector(spec[5:], threads)
    if spec == "torchvision-person":
        return TorchvisionPersonDetector(threads, cache_dir)
    if spec.startswith("torchvision-person-frcnn"):
        size = int(spec.split(":", 1)[1]) if ":" in spec else 640
        return TorchvisionPersonFRCNN(threads, cache_dir, min_size=size)
    if spec.startswith("torchvision-coco:"):
        parts = spec.split(":")
        if len(parts) < 3:
            raise ValueError("torchvision-coco needs '<coco categories joined by +>:<class name>[:min_size]'")
        size = int(parts[3]) if len(parts) > 3 else 640
        return TorchvisionCocoDetector(threads, cache_dir, min_size=size, categories=tuple(parts[1].split("+")), label=parts[2], backend="torchvision-coco")
    raise ValueError("detector must be 'onnx:<model.onnx>', 'torchvision-person', 'torchvision-person-frcnn[:min_size]' or 'torchvision-coco:<categories>:<class>[:min_size]'")
