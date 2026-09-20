import json
import os
import time

import av

from memento.manifest import Manifest
from memento.utils import FRAME_CACHE_SIZE


def load_manifest(cache_path=None):
    """Process-wide Manifest singleton keyed by cache path."""
    if cache_path is None:
        from memento.utils import CACHE_PATH

        cache_path = CACHE_PATH
    return Manifest.load(cache_path)


class Reader:
    def __init__(self, filename):
        self.container = av.open(filename)
        self.stream = self.container.streams.video[0]
        self.frames = []
        for frame in self.container.decode(self.stream):
            self.frames.append(frame)

    def get_index(self, index):
        if index < len(self.frames):
            return self.frames[index].to_ndarray(format="bgr24")
        return None

    def close(self):
        try:
            self.container.close()
        except Exception:
            pass


class ReadersCache:
    """Decode cache keyed by *segment uid*, never by frame arithmetic."""

    def __init__(self, manifest=None):
        self.manifest = manifest or load_manifest()
        self.readers = {}
        self.readers_ids = []  # LRU order of segment uids
        self.cache_size = FRAME_CACHE_SIZE

    def get_reader(self, seg_uid):
        if seg_uid not in self.readers:
            start = time.time()
            seg = self.manifest.get_segment(seg_uid)
            self.readers[seg_uid] = Reader(self.manifest.abs_path(seg["media_path"]))
            self.readers_ids.append(seg_uid)
            if len(self.readers) > self.cache_size:
                dumped_id = self.readers_ids[0]
                self.readers_ids = self.readers_ids[1:]
                self.readers[dumped_id].close()
                del self.readers[dumped_id]
            # print("Caching reader", seg_uid, "took", time.time() - start)
        return self.readers[seg_uid]

    def get_frame(self, capture_id):
        seg, frame = self.manifest.locate(capture_id)
        return self.get_reader(seg["uid"]).get_index(frame["index"])

    def drop(self, seg_uid=None):
        if seg_uid is None:
            for reader in self.readers.values():
                reader.close()
            self.readers = {}
            self.readers_ids = []
        elif seg_uid in self.readers:
            self.readers[seg_uid].close()
            del self.readers[seg_uid]
            self.readers_ids.remove(seg_uid)


class Metadata:
    """Per-segment JSON metadata, keyed by stable capture ids."""

    def __init__(self, file_path):
        self.file_path = file_path
        if not os.path.exists(self.file_path):
            self.metadata = {}
        else:
            with open(self.file_path) as f:
                self.metadata = json.load(f)

    def get_frame(self, capture_id):
        return self.metadata[str(capture_id)]

    def write(self, capture_id, data):
        self.metadata[str(capture_id)] = data
        with open(self.file_path, "w") as f:
            json.dump(self.metadata, f)


class MetadataCache:
    def __init__(self, manifest=None):
        self.manifest = manifest or load_manifest()
        self.cache = {}
        self.cache_size = FRAME_CACHE_SIZE
        self.cache_ids = []

    def _load_metadata(self, seg):
        metadata_id = seg["uid"]
        if metadata_id not in self.cache:
            self.cache[metadata_id] = Metadata(
                self.manifest.abs_path(seg["meta_path"])
            )
            self.cache_ids.append(metadata_id)
            if len(self.cache) > self.cache_size:
                dumped_id = self.cache_ids[0]
                self.cache_ids = self.cache_ids[1:]
                del self.cache[dumped_id]
        return self.cache[metadata_id]

    def get_metadata(self, capture_id):
        seg, _frame = self.manifest.locate(capture_id)
        return self._load_metadata(seg)

    def get_frame_metadata(self, capture_id):
        metadata = self.get_metadata(capture_id)
        return metadata.get_frame(capture_id)

    def write(self, capture_id, data):
        metadata = self.get_metadata(capture_id)
        metadata.write(capture_id, data)

    def drop(self, seg_uid=None):
        if seg_uid is None:
            self.cache = {}
            self.cache_ids = []
        elif seg_uid in self.cache:
            del self.cache[seg_uid]
            self.cache_ids.remove(seg_uid)
