"""Stable capture identity and physical layout manifest.

The manifest is the single source of truth mapping a stable ``capture_id``
(allocated once, never reused) to its *current* physical location:

    capture_id -> segment uid + frame index inside the segment mp4

Segment files are addressed by uid and carry a generation number. When a
segment is partially purged, the retained frames are rewritten into a new
generation file and the manifest atomically starts pointing at it; the old
generation becomes an unreferenced file that the reclaimer removes.

On-disk layout (cache root)::

    manifest.json
    memento.db
    media/<seg_uid>.gen<n>.mp4
    media/<seg_uid>.gen<n>.json
    purge_reports/<plan_id>.json
    <chroma files ...>
"""

import datetime
import json
import os
import tempfile

MANIFEST_NAME = "manifest.json"
MEDIA_DIR = "media"
REPORTS_DIR = "purge_reports"

MANIFEST_VERSION = 2
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def normalize_time(value):
    """Legacy metadata stored JSON-encoded timestamps (with quotes)."""
    if value is None:
        return None
    return str(value).strip().strip('"')


def parse_time(value):
    return datetime.datetime.strptime(normalize_time(value), TIME_FORMAT)


def fsync_dir(path):
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_bytes(path, data):
    """Write ``data`` to ``path`` atomically (tmp file + rename + fsync)."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        fsync_dir(directory)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_write_json(path, obj):
    atomic_write_bytes(path, json.dumps(obj, indent=2).encode("utf-8"))


class Manifest:
    def __init__(self, cache_path, data):
        self.cache_path = cache_path
        self.data = data
        self._rebuild_indexes()

    # ------------------------------------------------------------------ load

    @classmethod
    def load(cls, cache_path):
        os.makedirs(cache_path, exist_ok=True)
        manifest_path = os.path.join(cache_path, MANIFEST_NAME)
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                data = json.load(f)
            manifest = cls(cache_path, data)
            manifest.save()
            return manifest

        # Legacy dense layout: <video_id>.mp4 + <video_id>.json
        if os.path.exists(os.path.join(cache_path, "0.mp4")):
            return cls._migrate_legacy(cache_path)

        manifest = cls(cache_path, cls._fresh_data())
        manifest.save()
        return manifest

    @staticmethod
    def _fresh_data():
        return {
            "version": MANIFEST_VERSION,
            "next_capture_id": 0,
            "next_segment_id": 1,
            "segments": [],
        }

    @classmethod
    def _migrate_legacy(cls, cache_path):
        """Convert numbered dense files to media/seg_<n>.gen1 files.

        Legacy integer frame ids are kept as capture_ids, so existing SQLite
        rows, JSON content and Chroma metadata stay valid.
        """
        media_dir = os.path.join(cache_path, MEDIA_DIR)
        os.makedirs(media_dir, exist_ok=True)

        legacy_ids = sorted(
            int(f[:-4]) for f in os.listdir(cache_path) if f.endswith(".mp4")
        )

        segments = []
        max_capture_id = -1
        for legacy_i in legacy_ids:
            json_path = os.path.join(cache_path, str(legacy_i) + ".json")
            metadata = {}
            if os.path.exists(json_path):
                with open(json_path) as f:
                    metadata = json.load(f)

            ordered_ids = sorted((int(k) for k in metadata.keys()))
            frames = []
            for index, capture_id in enumerate(ordered_ids):
                entry = metadata[str(capture_id)]
                frames.append(
                    {
                        "capture_id": capture_id,
                        "index": index,
                        "time": normalize_time(entry.get("time")),
                        "window_title": entry.get("window_title"),
                    }
                )
                max_capture_id = max(max_capture_id, capture_id)

            seg_uid = "seg_%d" % (legacy_i + 1)
            seg = {
                "uid": seg_uid,
                "gen": 1,
                "media_path": "%s/%s.gen1.mp4" % (MEDIA_DIR, seg_uid),
                "meta_path": "%s/%s.gen1.json" % (MEDIA_DIR, seg_uid),
                "sealed": True,
                "frames": frames,
            }
            segments.append(seg)

            os.rename(
                os.path.join(cache_path, str(legacy_i) + ".mp4"),
                os.path.join(cache_path, seg["media_path"]),
            )
            if os.path.exists(json_path):
                os.rename(json_path, os.path.join(cache_path, seg["meta_path"]))

        data = {
            "version": MANIFEST_VERSION,
            "next_capture_id": max_capture_id + 1,
            "next_segment_id": len(legacy_ids) + 1,
            "segments": segments,
        }
        manifest = cls(cache_path, data)
        manifest.save()
        fsync_dir(media_dir)
        print("Migrated %d legacy segments to manifest layout" % len(segments))
        return manifest

    # -------------------------------------------------------------- indexing

    def _rebuild_indexes(self):
        self._segments_by_uid = {s["uid"]: s for s in self.data["segments"]}
        self._frame_by_capture = {}
        for seg in self.data["segments"]:
            for frame in seg["frames"]:
                self._frame_by_capture[frame["capture_id"]] = (seg, frame)

    def save(self):
        atomic_write_json(os.path.join(self.cache_path, MANIFEST_NAME), self.data)

    # ------------------------------------------------------------ allocation

    def allocate_capture_id(self):
        capture_id = self.data["next_capture_id"]
        self.data["next_capture_id"] += 1
        return capture_id

    def allocate_segment(self):
        seg_n = self.data["next_segment_id"]
        self.data["next_segment_id"] += 1
        seg_uid = "seg_%d" % seg_n
        seg = {
            "uid": seg_uid,
            "gen": 1,
            "media_path": "%s/%s.gen1.mp4" % (MEDIA_DIR, seg_uid),
            "meta_path": "%s/%s.gen1.json" % (MEDIA_DIR, seg_uid),
            "sealed": False,
            "frames": [],
        }
        self.data["segments"].append(seg)
        self._segments_by_uid[seg_uid] = seg
        os.makedirs(os.path.join(self.cache_path, MEDIA_DIR), exist_ok=True)
        return seg

    def seal(self, seg_uid):
        self._segments_by_uid[seg_uid]["sealed"] = True

    def register_frame(self, seg, capture_id, window_title, time_str):
        frame = {
            "capture_id": capture_id,
            "index": len(seg["frames"]),
            "time": normalize_time(time_str),
            "window_title": window_title,
        }
        seg["frames"].append(frame)
        self._frame_by_capture[capture_id] = (seg, frame)
        return frame

    # -------------------------------------------------------------- lookups

    @property
    def segments(self):
        return list(self.data["segments"])

    def get_segment(self, seg_uid):
        return self._segments_by_uid[seg_uid]

    def open_segment(self):
        """The segment currently being recorded (sealed == False), if any."""
        for seg in self.data["segments"]:
            if not seg["sealed"]:
                return seg
        return None

    def locate(self, capture_id):
        """Return (segment, frame_entry) for a live capture_id."""
        return self._frame_by_capture[capture_id]

    def has_capture(self, capture_id):
        return capture_id in self._frame_by_capture

    def sorted_captures(self):
        """All frames sorted by logical time (time, then capture_id)."""
        frames = []
        for seg in self.data["segments"]:
            frames.extend(seg["frames"])
        frames.sort(key=lambda f: (f["time"] or "", f["capture_id"]))
        return frames

    def next_generation_paths(self, seg):
        new_gen = seg["gen"] + 1
        uid = seg["uid"]
        return (
            new_gen,
            "%s/%s.gen%d.mp4" % (MEDIA_DIR, uid, new_gen),
            "%s/%s.gen%d.json" % (MEDIA_DIR, uid, new_gen),
        )

    def replace_generation(self, seg, new_gen, media_path, meta_path, frames):
        """Point a segment at a freshly written generation."""
        seg["gen"] = new_gen
        seg["media_path"] = media_path
        seg["meta_path"] = meta_path
        seg["frames"] = frames
        self._rebuild_indexes()

    def remove_segment(self, seg_uid):
        seg = self._segments_by_uid.pop(seg_uid)
        self.data["segments"].remove(seg)
        self._rebuild_indexes()
        return seg

    # ------------------------------------------------------------- paths io

    def abs_path(self, rel_path):
        return os.path.join(self.cache_path, rel_path)

    def reports_dir(self):
        path = os.path.join(self.cache_path, REPORTS_DIR)
        os.makedirs(path, exist_ok=True)
        return path
