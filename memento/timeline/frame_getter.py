import bisect

import cv2
import numpy as np

from memento.caching import MetadataCache, ReadersCache
from memento.manifest import parse_time


class FrameGetter:
    """Navigates captures by logical time and manifest.

    ``self.captures`` is the session-ordered list of live capture entries
    (sorted by time). Its position is a *session-only* logical index; the
    permanent identity used everywhere is ``capture_id``.
    """

    def __init__(self, window_size, manifest=None):
        self.window_size = window_size

        from memento.caching import load_manifest

        self.manifest = manifest or load_manifest()
        self.readers_cache = ReadersCache(self.manifest)
        self.metadata_cache = MetadataCache(self.manifest)
        self.annotations = {}
        self.current_ret_annotated = 0
        self._build_capture_index()
        self.nb_results = 0
        self.debug_mode = False

        self.current_displayed_capture_id = None
        self.current_displayed_frame = None

    def _build_capture_index(self):
        self.captures = self.manifest.sorted_captures()
        self.nb_frames = len(self.captures)
        self._position_by_capture = {}
        times = []
        for pos, frame in enumerate(self.captures):
            self._position_by_capture[frame["capture_id"]] = pos
            times.append(parse_time(frame["time"]))
        self._times = times

    # ------------------------------------------------------------- accessors

    def capture_at_position(self, pos):
        if 0 <= pos < len(self.captures):
            return self.captures[pos]
        return None

    def position_of(self, capture_id):
        return self._position_by_capture.get(int(capture_id))

    def position_at_or_before_time(self, dt):
        pos = bisect.bisect_right(self._times, dt) - 1
        return max(0, pos)

    def captures_in_time_range(self, start_dt, end_dt):
        lo = bisect.bisect_left(self._times, start_dt)
        hi = bisect.bisect_left(self._times, end_dt)
        return self.captures[lo:hi]

    def latest_time(self):
        return self._times[-1] if self._times else None

    # ---------------------------------------------------------------- frames

    def toggle_debug_mode(self):
        self.debug_mode = not self.debug_mode
        self.clear_annotations()

    def get_frame(self, capture_id, resize=None):
        im = self.current_displayed_frame

        # Resize frame if needed, still use cache
        if im is not None and resize != im.shape:
            self.current_displayed_frame = None

        # Avoid resizing and converting the same frame each time
        if (
            capture_id != self.current_displayed_capture_id
            or self.current_displayed_frame is None
        ):
            im = self.readers_cache.get_frame(capture_id)
            self.process_debug(capture_id)
            im = self.annotate_frame(capture_id, im)
            if resize:
                im = cv2.resize(im, resize)
            else:
                im = cv2.resize(im, self.window_size)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).swapaxes(0, 1)
            self.current_displayed_frame = im
            self.current_displayed_capture_id = capture_id
        return im

    def process_debug(self, capture_id):
        if self.debug_mode:
            self.clear_annotations()
            frame_metadata = self.metadata_cache.get_frame_metadata(capture_id)
            if "bbs" in frame_metadata:
                res = []
                for i in range(len(frame_metadata["bbs"])):
                    entry = {}
                    bb = frame_metadata["bbs"][i]
                    text = frame_metadata["text"][i]
                    entry["bb"] = {
                        "x": bb["x"],
                        "y": bb["y"],
                        "w": bb["w"],
                        "h": bb["h"],
                    }
                    entry["text"] = text
                    res.append(entry)
                self.add_annotation(capture_id, res)

    def annotate_frame(self, capture_id, frame):
        if str(capture_id) in self.annotations.keys():
            entries = self.annotations[str(capture_id)]
            for entry in entries:
                bb = entry["bb"]
                x = int(bb["x"])
                y = int(bb["y"])
                w = int(bb["w"])
                h = int(bb["h"])
                text = entry["text"]

                red_rect = np.ones((h, w, 3), dtype=np.uint8)
                red_rect[:, :, 0] = 0
                red_rect *= 200
                sub_img = frame[y : y + h, x : x + w]
                res = cv2.addWeighted(sub_img, 0.5, red_rect, 0.5, 1.0)
                if res is None:
                    continue
                frame[y : y + h, x : x + w] = res
                frame = cv2.putText(
                    frame,
                    text,
                    (x, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2,
                )
            frame = cv2.putText(
                frame,
                f"{self.nb_results} results",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        elif self.nb_results == -1:
            frame = cv2.putText(
                frame,
                "No result",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        return frame

    def get_next_annotated_frame_i(self):
        """Return the next annotated capture id (cycling), or None."""
        if len(self.annotations.keys()) > 0:
            frame_id = list(self.annotations.keys())[self.current_ret_annotated]
            self.current_ret_annotated += 1
            if self.current_ret_annotated >= len(self.annotations.keys()):
                self.current_ret_annotated = 0
            return int(frame_id)
        else:
            return None

    def set_annotations(self, annotations):
        self.annotations = annotations
        self.current_displayed_frame = None

    def get_annotations(self):
        return self.annotations

    def get_annotated_frames(self):
        frames = []
        for frame_id in list(self.annotations.keys())[:10]:
            frames.append(self.get_frame(int(frame_id)))

        return frames

    def get_annotations_text(self):
        text = ""
        for entries in self.annotations.values():
            for entry in entries:
                text += entry["text"] + "\n"
        return text

    def add_annotation(self, capture_id, annotations):
        if str(capture_id) not in self.annotations.keys():
            self.annotations[str(capture_id)] = []

        for annotation in annotations:
            self.annotations[str(capture_id)].append(annotation)
            self.nb_results += 1
        self.current_displayed_frame = None

    def is_annotated(self, capture_id):
        return str(capture_id) in self.annotations.keys()

    def clear_annotations(self):
        self.annotations = {}
        self.current_ret_annotated = 0
        self.nb_results = 0
        self.current_displayed_frame = None
