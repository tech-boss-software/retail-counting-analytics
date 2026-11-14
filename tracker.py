"""ByteTrack-style multi-object tracker in pure Python (shared with safety-video-analytics).

Two association stages: confident detections first, then low-confidence ones for
tracks that would otherwise be lost. Greedy IoU matching keeps the behaviour easy to
inspect and test; Kalman filtering is replaced by a constant-velocity box prediction.
"""
from __future__ import annotations

from geometry import box_iou, validate_box


class Track:
    def __init__(self, track_id, box, score, class_id, frame_index):
        self.track_id = track_id
        self.box = validate_box(box)
        self.velocity = [0.0, 0.0, 0.0, 0.0]
        self.score = float(score)
        self.class_id = int(class_id)
        self.hits = 1
        self.lost = 0
        self.first_frame = frame_index
        self.last_frame = frame_index

    def predicted_box(self):
        # Extrapolate over every frame missed so a track coasts through an occlusion.
        steps = self.lost + 1
        box = [self.box[i] + self.velocity[i] * steps for i in range(4)]
        if box[2] <= box[0] or box[3] <= box[1]:
            return list(self.box)
        return box

    def update(self, box, score, frame_index, smoothing=0.5):
        box = validate_box(box)
        gap = max(1, frame_index - self.last_frame)
        self.velocity = [smoothing * self.velocity[i] + (1 - smoothing) * (box[i] - self.box[i]) / gap for i in range(4)]
        self.box = box
        self.score = float(score)
        self.hits += 1
        self.lost = 0
        self.last_frame = frame_index

    def as_dict(self, confirmed):
        return {"track_id": self.track_id, "box_xyxy": list(self.box), "score": self.score, "class_id": self.class_id,
                "hits": self.hits, "lost": self.lost, "first_frame": self.first_frame, "last_frame": self.last_frame,
                "confirmed": confirmed}


class Tracker:
    def __init__(self, iou_threshold=0.3, high_score=0.5, low_score=0.1, min_hits=3, max_lost=10, duplicate_iou=0.4):
        if not 0 <= low_score <= high_score <= 1:
            raise ValueError("need 0 <= low_score <= high_score <= 1")
        self.iou_threshold = float(iou_threshold)
        self.duplicate_iou = float(duplicate_iou)
        self.high_score = float(high_score)
        self.low_score = float(low_score)
        self.min_hits = int(min_hits)
        self.max_lost = int(max_lost)
        self.tracks: list[Track] = []
        self.next_id = 1

    def update(self, detections, frame_index):
        """Advance one frame. Detections are dicts with box_xyxy, score, class_id.

        Returns the confirmed tracks visible this frame (matched or coasting).
        """
        high = [d for d in detections if float(d["score"]) >= self.high_score]
        low = [d for d in detections if self.low_score <= float(d["score"]) < self.high_score]
        unmatched_tracks = list(self.tracks)
        matched = set()

        pairs, unmatched_tracks, unmatched_high = self._match(unmatched_tracks, high)
        for track, det in pairs:
            track.update(det["box_xyxy"], det["score"], frame_index)
            matched.add(track.track_id)
        # Second stage: low-confidence detections may still explain an existing track
        # (partial occlusion, motion blur) and stop it from being dropped prematurely.
        # Low-confidence boxes may only sustain tracks that already earned confirmation;
        # letting them feed tentative tracks turns detector noise into phantom people.
        confirmed = [t for t in unmatched_tracks if t.hits >= self.min_hits]
        tentative = [t for t in unmatched_tracks if t.hits < self.min_hits]
        pairs, confirmed, _ = self._match(confirmed, low)
        for track, det in pairs:
            track.update(det["box_xyxy"], det["score"], frame_index)
            matched.add(track.track_id)
        # Third stage: a sudden shape change (a person falling, crouching or turning)
        # breaks IoU but not position; a confirmed track adopts a confident detection
        # whose centre lies inside the track's last box. The jump is not motion, so the
        # velocity is reset after the update rather than learned from it.
        pairs, confirmed, unmatched_high = self._match_by_center(confirmed, unmatched_high)
        for track, det in pairs:
            track.update(det["box_xyxy"], det["score"], frame_index)
            track.velocity = [0.0, 0.0, 0.0, 0.0]
            matched.add(track.track_id)
        unmatched_tracks = confirmed + tentative
        for track in unmatched_tracks:
            track.lost += 1
        updated = [t for t in self.tracks if t.track_id in matched]
        for det in unmatched_high:
            # A second box on an already-tracked person is a duplicate, not a new person.
            if any(t.class_id == int(det["class_id"]) and box_iou(t.box, det["box_xyxy"]) > self.duplicate_iou for t in updated):
                continue
            self.tracks.append(Track(self.next_id, det["box_xyxy"], det["score"], det["class_id"], frame_index))
            self.next_id += 1
        self.tracks = [t for t in self.tracks if t.lost <= self.max_lost and not (t.hits < self.min_hits and t.lost > 0)]
        return [t.as_dict(True) for t in self.tracks if t.hits >= self.min_hits and t.lost <= 1]

    def _match_by_center(self, tracks, detections):
        used_t, used_d, pairs = set(), set(), []
        for ti, track in enumerate(tracks):
            if track.hits < self.min_hits:
                continue
            box = track.predicted_box()
            best = None
            for di, det in enumerate(detections):
                if di in used_d or int(det["class_id"]) != track.class_id:
                    continue
                d = det["box_xyxy"]
                cx, cy = (d[0] + d[2]) / 2, (d[1] + d[3]) / 2
                if box[0] <= cx <= box[2] and box[1] <= cy <= box[3]:
                    overlap = box_iou(box, d)
                    if best is None or overlap > best[0]:
                        best = (overlap, di)
            if best is not None:
                used_t.add(ti)
                used_d.add(best[1])
                pairs.append((track, detections[best[1]]))
        rest_tracks = [t for i, t in enumerate(tracks) if i not in used_t]
        rest_dets = [d for i, d in enumerate(detections) if i not in used_d]
        return pairs, rest_tracks, rest_dets

    def _match(self, tracks, detections):
        candidates = []
        for ti, track in enumerate(tracks):
            predicted = track.predicted_box()
            for di, det in enumerate(detections):
                if int(det["class_id"]) != track.class_id:
                    continue
                overlap = box_iou(predicted, det["box_xyxy"])
                if overlap >= self.iou_threshold:
                    candidates.append((overlap, ti, di))
        candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
        used_t, used_d, pairs = set(), set(), []
        for overlap, ti, di in candidates:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            pairs.append((tracks[ti], detections[di]))
        rest_tracks = [t for i, t in enumerate(tracks) if i not in used_t]
        rest_dets = [d for i, d in enumerate(detections) if i not in used_d]
        return pairs, rest_tracks, rest_dets
