"""Frame input and video output; OpenCV is optional, image folders always work."""
from __future__ import annotations

from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


def open_source(path, fps_hint=None):
    """Return (iterator of (index, PIL RGB image), fps, (width, height), description)."""
    from PIL import Image

    path = Path(path)
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            raise ValueError(f"no images in {path}")
        with Image.open(files[0]) as first:
            size = first.size

        def frames():
            for index, file in enumerate(files):
                with Image.open(file) as image:
                    yield index, image.convert("RGB")
        return frames(), float(fps_hint or 10.0), size, f"{len(files)} images from {path}"
    if path.suffix.lower() in VIDEO_SUFFIXES:
        import cv2

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"cannot open video {path}")
        fps = capture.get(cv2.CAP_PROP_FPS) or fps_hint or 10.0
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        def frames():
            index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                yield index, Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                index += 1
            capture.release()
        return frames(), float(fps), size, f"video {path.name}"
    raise ValueError(f"unsupported input: {path} (give a video file or a folder of images)")


class VideoSink:
    """Writes MP4 through OpenCV when available, otherwise a folder of PNG frames."""

    def __init__(self, path, fps, size):
        self.path = Path(path)
        self.fps = fps
        self.size = size
        self.count = 0
        self.writer = None
        try:
            import cv2

            self.path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            if writer.isOpened():
                self.writer = writer
                self.cv2 = cv2
        except ImportError:
            pass
        if self.writer is None:
            self.path = self.path.with_suffix("")
            self.path.mkdir(parents=True, exist_ok=True)

    def write(self, image):
        if self.writer is not None:
            import numpy as np

            self.writer.write(self.cv2.cvtColor(np.asarray(image.convert("RGB")), self.cv2.COLOR_RGB2BGR))
        else:
            image.save(self.path / f"frame_{self.count:05d}.png")
        self.count += 1

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        return str(self.path)
