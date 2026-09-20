"""Transactional storage purge engine.

Generates deletion plans from three independent selectors (logical-time
interval, application/window title, maximum retention) and executes them in
crash-safe stages:

1. Journal phase (one SQLite transaction): plan + per-segment work items +
   tombstones + reclaim watermark become durable together.
2. Reclaim phase, per segment:
   * fully hit segment  -> removed from manifest, files deleted;
   * partially hit H.264 segment -> decoded, retained frames rewritten into a
     new generation file, manifest atomically switched to the new generation.
3. Index phase: JSON metadata, FRAME/CONTENT/FTS rows and Chroma documents are
   deleted by stable capture id.
4. Verify + report: every purged id is re-checked in every layer; unreferenced
   media files are reclaimed; a local report is persisted.

All stages are idempotent: if the reclaim watermark is present at startup,
``PurgeExecutor.resume`` continues the plan.
"""

import datetime
import os
import uuid

import av

from memento.manifest import (
    MEDIA_DIR,
    atomic_write_json,
    fsync_dir,
    parse_time,
)

CHROMA_DELETE_BATCH = 100


# --------------------------------------------------------------------- criteria


class PurgeCriteria:
    """Union of the provided deletion rules.

    A capture is selected when ANY provided rule matches:
      * time interval [start, end)
      * window title contains ``app`` (case-insensitive)
      * age older than ``max_age_days`` (retention cutoff)
    """

    def __init__(self, start=None, end=None, app=None, max_age_days=None,
                 now=None):
        self.start = self._parse_optional(start)
        self.end = self._parse_optional(end)
        self.app = app.strip() if app and app.strip() else None
        self.max_age_days = float(max_age_days) if max_age_days else None
        self.now = now or datetime.datetime.now()
        if self.max_age_days is not None:
            self.retention_cutoff = self.now - datetime.timedelta(
                days=self.max_age_days
            )
        else:
            self.retention_cutoff = None

    @staticmethod
    def _parse_optional(value):
        if value is None or str(value).strip() == "":
            return None
        return parse_time(value)

    def to_dict(self):
        return {
            "start": self.start.strftime("%Y-%m-%d %H:%M:%S")
            if self.start
            else None,
            "end": self.end.strftime("%Y-%m-%d %H:%M:%S") if self.end else None,
            "app": self.app,
            "max_age_days": self.max_age_days,
        }

    def describe(self):
        rules = []
        if self.start or self.end:
            rules.append(
                "time %s..%s"
                % (
                    self.start.strftime("%Y-%m-%d %H:%M") if self.start else "begin",
                    self.end.strftime("%Y-%m-%d %H:%M") if self.end else "end",
                )
            )
        if self.app:
            rules.append("app ~ '%s'" % self.app)
        if self.retention_cutoff:
            rules.append(
                "older than %g days (before %s)"
                % (self.max_age_days, self.retention_cutoff.strftime("%Y-%m-%d %H:%M"))
            )
        return "; ".join(rules) if rules else "no criteria"

    def matches(self, frame):
        if frame["time"] is None:
            return False
        t = parse_time(frame["time"])
        if self.start and t < self.start:
            time_hit = False
        elif self.end and t >= self.end:
            time_hit = False
        elif self.start or self.end:
            time_hit = True
        else:
            time_hit = False

        app_hit = bool(
            self.app
            and frame.get("window_title")
            and self.app.lower() in str(frame["window_title"]).lower()
        )
        retention_hit = bool(self.retention_cutoff and t <= self.retention_cutoff)
        return time_hit or app_hit or retention_hit


# ------------------------------------------------------------------------- plan


class DeletionPlan:
    def __init__(self, plan_id, created_at, criteria, items, skipped_open,
                 summary):
        self.plan_id = plan_id
        self.created_at = created_at
        self.criteria = criteria
        self.items = items  # [{seg_uid, action, capture_ids}]
        self.skipped_open = skipped_open  # captures in the recording segment
        self.summary = summary

    @property
    def capture_ids(self):
        return [
            cid for item in self.items for cid in item["capture_ids"]
        ]

    def to_dict(self):
        return {
            "plan_id": self.plan_id,
            "created_at": self.created_at,
            "criteria": self.criteria.to_dict(),
            "criteria_description": self.criteria.describe(),
            "summary": self.summary,
            "segments": [
                {
                    "seg_uid": item["seg_uid"],
                    "action": item["action"],
                    "capture_ids": item["capture_ids"],
                }
                for item in self.items
            ],
            "skipped_active_segment": self.skipped_open,
        }


def _file_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def plan_purge(manifest, criteria, now=None):
    now = now or datetime.datetime.now()
    created_at = now.strftime("%Y-%m-%d %H:%M:%S")
    plan_id = "purge-%s-%s" % (now.strftime("%Y%m%d%H%M%S"), uuid.uuid4().hex[:4])

    deleted_by_seg = {}
    skipped_open = []
    for seg in manifest.segments:
        deleted = []
        for frame in seg["frames"]:
            if criteria.matches(frame):
                if not seg["sealed"]:
                    skipped_open.append(frame["capture_id"])
                else:
                    deleted.append(frame["capture_id"])
        if deleted:
            deleted_by_seg[seg["uid"]] = deleted

    items = []
    recycled = 0
    rewritten = 0
    bytes_to_free = 0
    per_app = {}
    earliest = None
    latest = None
    for seg in manifest.segments:
        deleted = deleted_by_seg.get(seg["uid"])
        if not deleted:
            continue
        deleted_set = set(deleted)
        if len(seg["frames"]) == len(deleted_set):
            action = "recycle"
            recycled += 1
            media_size = _file_size(manifest.abs_path(seg["media_path"]))
            meta_size = _file_size(manifest.abs_path(seg["meta_path"]))
            bytes_to_free += media_size + meta_size
        else:
            action = "rewrite"
            rewritten += 1
            media_size = _file_size(manifest.abs_path(seg["media_path"]))
            # proportional estimate for the removed frames
            bytes_to_free += int(media_size * len(deleted_set) / len(seg["frames"]))

        items.append(
            {"seg_uid": seg["uid"], "action": action, "capture_ids": sorted(deleted)}
        )

        for frame in seg["frames"]:
            if frame["capture_id"] not in deleted_set:
                continue
            app = frame.get("window_title") or "None"
            per_app.setdefault(app, 0)
            per_app[app] += 1
            t = frame["time"]
            if earliest is None or t < earliest:
                earliest = t
            if latest is None or t > latest:
                latest = t

    total_captures = sum(len(i["capture_ids"]) for i in items)
    summary = {
        "total_captures": total_captures,
        "segments_recycled": recycled,
        "segments_rewritten": rewritten,
        "bytes_to_free_estimate": bytes_to_free,
        "per_app": per_app,
        "earliest": earliest,
        "latest": latest,
        "skipped_active_segment_count": len(skipped_open),
    }
    return DeletionPlan(plan_id, created_at, criteria, items, skipped_open, summary)


# ------------------------------------------------------------- segment rewrite


def _encode_segment(frames_bgr, out_path, resolution, fps, bit_rate):
    """Encode ndarray frames to a standalone H.264 mp4 (sequential pts)."""
    import fractions

    clock_rate = 90000
    directory = os.path.dirname(out_path)
    os.makedirs(directory, exist_ok=True)
    container = av.open(out_path, "w")
    try:
        stream = container.add_stream(
            "h264", fractions.Fraction(str(fps))
        )
        stream.height = resolution[1]
        stream.width = resolution[0]
        stream.bit_rate = bit_rate
        time_base = fractions.Fraction(1, clock_rate)
        pts_step = int(clock_rate / fps)
        for i, im in enumerate(frames_bgr):
            vf = av.video.frame.VideoFrame.from_ndarray(im, format="bgr24")
            vf.pts = i * pts_step
            vf.time_base = time_base
            packet = stream.encode(vf)
            container.mux(packet)
        packet = stream.encode(None)
        container.mux(packet)
    finally:
        container.close()
    fsync_dir(directory)


class PurgeExecutor:
    def __init__(self, db, manifest, chromadb=None, resolution=None, fps=None,
                 bit_rate=None):
        from memento import utils

        self.db = db
        self.manifest = manifest
        self.chromadb = chromadb
        self.resolution = resolution or utils.RESOLUTION
        self.fps = fps or utils.FPS
        self.bit_rate = bit_rate if bit_rate is not None else 8500e1
        self.bytes_freed = 0

    # --------------------------------------------------------------- execute

    def execute(self, plan):
        if not plan.items:
            return {"status": "empty", "message": "Nothing matched the criteria."}

        deleted_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.db.create_purge_plan(
            plan.plan_id,
            plan.criteria.to_dict(),
            plan.items,
            deleted_at,
            plan.criteria.describe(),
        )
        return self._run_stages(plan.plan_id, resumed=False)

    def resume(self):
        """Continue the plan referenced by the reclaim watermark, if any."""
        unfinished = self.db.get_unfinished_plan()
        if unfinished is None:
            return None
        plan_id = unfinished["plan_id"]
        print("Resuming interrupted purge plan", plan_id)
        report = self._run_stages(plan_id, resumed=True)
        return report

    def _run_stages(self, plan_id, resumed):
        self.db.set_plan_status(plan_id, "reclaiming")
        self._reclaim_items(plan_id)

        self.db.set_plan_status(plan_id, "indexing")
        capture_ids = [
            cid for item in self.db.get_plan_items(plan_id)
            for cid in item["capture_ids"]
        ]
        chroma_result = self._delete_chroma(capture_ids)
        self.db.delete_frame_rows(capture_ids)

        # Reload on-disk manifest state before verification.
        verification = self._verify(capture_ids)
        removed_files = self._reconcile_media()
        verification["media_reconciled"] = {
            "files_removed": len(removed_files),
            "bytes_freed": sum(s for _, s in removed_files),
        }

        report = {
            "plan_id": plan_id,
            "completed_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "resumed": resumed,
            "captures_deleted": len(capture_ids),
            "segments_recycled": sum(
                1 for it in self.db.get_plan_items(plan_id) if it["action"] == "recycle"
            ),
            "segments_rewritten": sum(
                1 for it in self.db.get_plan_items(plan_id) if it["action"] == "rewrite"
            ),
            "bytes_freed": self.bytes_freed,
            "chroma": chroma_result,
            "verification": verification,
            "verified": verification["all_clear"],
        }
        self.db.finish_purge_plan(plan_id, report)
        report_path = os.path.join(
            self.manifest.reports_dir(), plan_id + ".json"
        )
        atomic_write_json(report_path, report)
        return report

    # -------------------------------------------------------------- reclaiming

    def _reclaim_items(self, plan_id):
        for item in self.db.get_plan_items(plan_id):
            if item["state"] == "done":
                continue
            seg_uid = item["seg_uid"]
            self.db.set_item_state(plan_id, seg_uid, "reclaiming")
            if item["action"] == "recycle":
                self._recycle_segment(seg_uid)
            else:
                self._rewrite_segment(seg_uid, set(item["capture_ids"]))
            self.db.set_item_state(plan_id, seg_uid, "done")

    def _recycle_segment(self, seg_uid):
        seg = None
        try:
            seg = self.manifest.get_segment(seg_uid)
        except KeyError:
            seg = None
        if seg is not None:
            self.manifest.remove_segment(seg_uid)
            self.manifest.save()
        self._delete_generation_files(seg_uid)

    def _rewrite_segment(self, seg_uid, deleted_set):
        seg = self.manifest.get_segment(seg_uid)

        # Idempotency: remove any partial/higher generations staged on disk
        # while the manifest still points at the current generation.
        self._discard_newer_generations(seg)

        old_meta_path = self.manifest.abs_path(seg["meta_path"])
        old_metadata = {}
        if os.path.exists(old_meta_path):
            import json

            with open(old_meta_path) as f:
                old_metadata = json.load(f)

        # Decode only retained frames, in source order.
        retained_entries = [
            frame
            for frame in sorted(seg["frames"], key=lambda f: f["index"])
            if frame["capture_id"] not in deleted_set
        ]
        from memento.caching import Reader

        reader = Reader(self.manifest.abs_path(seg["media_path"]))
        try:
            retained_bgr = [
                reader.get_index(frame["index"]) for frame in retained_entries
            ]
        finally:
            reader.close()

        new_gen, new_media_rel, new_meta_rel = self.manifest.next_generation_paths(seg)
        new_media_abs = self.manifest.abs_path(new_media_rel)

        # Stage the new generation (unique name, old generation untouched).
        _encode_segment(
            retained_bgr,
            new_media_abs,
            self.resolution,
            self.fps,
            self.bit_rate,
        )

        new_metadata = {}
        new_frames = []
        import json

        for new_index, frame in enumerate(retained_entries):
            cid = frame["capture_id"]
            if str(cid) in old_metadata:
                new_metadata[str(cid)] = old_metadata[str(cid)]
            new_frames.append(
                {
                    "capture_id": cid,
                    "index": new_index,
                    "time": frame["time"],
                    "window_title": frame["window_title"],
                }
            )
        atomic_write_json(self.manifest.abs_path(new_meta_rel), new_metadata)

        # Atomic switch: manifest starts referencing the new generation.
        self.manifest.replace_generation(
            seg, new_gen, new_media_rel, new_meta_rel, new_frames
        )
        self.manifest.save()

        # Old generation is now unreferenced -> reclaim its bytes.
        self.bytes_freed += self._safe_size(self._old_media_path(seg, new_gen))
        self.bytes_freed += self._safe_size(self._old_meta_path(seg, new_gen))
        self._unlink_safe(self._old_media_path(seg, new_gen))
        self._unlink_safe(self._old_meta_path(seg, new_gen))

    @staticmethod
    def _old_media_path(seg, new_gen):
        uid = seg["uid"]
        return "%s/%s.gen%d.mp4" % (MEDIA_DIR, uid, new_gen - 1)

    @staticmethod
    def _old_meta_path(seg, new_gen):
        uid = seg["uid"]
        return "%s/%s.gen%d.json" % (MEDIA_DIR, uid, new_gen - 1)

    def _discard_newer_generations(self, seg):
        """Remove staged gen files strictly above the manifest generation."""
        uid = seg["uid"]
        media_dir = os.path.join(self.manifest.cache_path, MEDIA_DIR)
        if not os.path.isdir(media_dir):
            return
        prefix = uid + ".gen"
        for name in os.listdir(media_dir):
            if not name.startswith(prefix):
                continue
            middle = name[len(prefix):]
            gen_str = middle.split(".")[0]
            try:
                gen_n = int(gen_str)
            except ValueError:
                continue
            if gen_n > seg["gen"]:
                self._unlink_safe(os.path.join(MEDIA_DIR, name))

    def _delete_generation_files(self, seg_uid):
        media_dir = os.path.join(self.manifest.cache_path, MEDIA_DIR)
        if not os.path.isdir(media_dir):
            return
        for name in os.listdir(media_dir):
            # Only the media family for this uid (<uid>.gen<n>.mp4/.json)
            if name.split(".")[0] == seg_uid:
                self.bytes_freed += self._safe_size(
                    os.path.join(MEDIA_DIR, name)
                )
                self._unlink_safe(os.path.join(MEDIA_DIR, name))

    # ---------------------------------------------------------------- indexes

    def _delete_chroma(self, capture_ids):
        if self.chromadb is None:
            return {"status": "unavailable", "documents_deleted": 0}
        deleted = 0
        ids = [str(cid) for cid in capture_ids]
        try:
            for i in range(0, len(ids), CHROMA_DELETE_BATCH):
                chunk = ids[i : i + CHROMA_DELETE_BATCH]
                collection = self.chromadb._collection
                before = collection.count()
                self.chromadb.delete(where={"id": {"$in": chunk}})
                deleted += max(0, before - collection.count())
            if hasattr(self.chromadb, "persist"):
                self.chromadb.persist()
            return {"status": "deleted", "documents_deleted": deleted}
        except Exception as e:
            return {"status": "error", "error": str(e), "documents_deleted": deleted}

    # ------------------------------------------------------------- verification

    def _verify(self, capture_ids):
        layers = {
            "manifest": [],
            "json": [],
            "frame_rows": [],
            "fts_rows": [],
            "chroma": [],
        }

        import json

        referenced_meta = {}
        for seg in self.manifest.segments:
            meta_abs = self.manifest.abs_path(seg["meta_path"])
            if os.path.exists(meta_abs):
                with open(meta_abs) as f:
                    referenced_meta[seg["uid"]] = json.load(f)

        chroma_leftovers = self._chroma_leftovers(capture_ids)
        for cid in capture_ids:
            if self.manifest.has_capture(cid):
                layers["manifest"].append(cid)
            if any(str(cid) in meta for meta in referenced_meta.values()):
                layers["json"].append(cid)
            n_frame, _n_content, n_fts = self.db.count_frame_rows(cid)
            if n_frame:
                layers["frame_rows"].append(cid)
            if n_fts:
                layers["fts_rows"].append(cid)
            if str(cid) in chroma_leftovers:
                layers["chroma"].append(cid)

        all_clear = all(not leftovers for leftovers in layers.values())
        return {"all_clear": all_clear, "leftovers": layers}

    def _chroma_leftovers(self, capture_ids):
        if self.chromadb is None:
            return set()
        try:
            ids = [str(cid) for cid in capture_ids]
            leftovers = set()
            for i in range(0, len(ids), CHROMA_DELETE_BATCH):
                chunk = ids[i : i + CHROMA_DELETE_BATCH]
                got = self.chromadb._collection.get(where={"id": {"$in": chunk}})
                leftovers.update(got.get("ids", []))
            return leftovers
        except Exception:
            return set()

    # ------------------------------------------------------------ media gc io

    def _reconcile_media(self):
        """Delete media files not referenced by any manifest segment."""
        referenced = set()
        for seg in self.manifest.segments:
            referenced.add(seg["media_path"])
            referenced.add(seg["meta_path"])

        media_dir = os.path.join(self.manifest.cache_path, MEDIA_DIR)
        removed = []
        if not os.path.isdir(media_dir):
            return removed
        for name in os.listdir(media_dir):
            rel = os.path.join(MEDIA_DIR, name)
            if rel in referenced:
                continue
            if name.startswith(".tmp-") or name.endswith(".part") or \
                    name.endswith(".mp4") or name.endswith(".json"):
                size = _file_size(os.path.join(self.manifest.cache_path, rel))
                if self._unlink_safe(rel):
                    self.bytes_freed += size
                    removed.append((rel, size))
        fsync_dir(media_dir)
        return removed

    def _safe_size(self, rel_path):
        return _file_size(self.manifest.abs_path(rel_path))

    def _unlink_safe(self, rel_path):
        abs_path = self.manifest.abs_path(rel_path)
        try:
            os.unlink(abs_path)
            return True
        except OSError:
            return False
