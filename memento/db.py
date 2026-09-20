import json
import os
import sqlite3

from memento.utils import CACHE_PATH


class Db:
    def __init__(self):
        db_path = os.path.join(CACHE_PATH, "memento.db")
        create_tables = not os.path.isfile(db_path)

        self.conn = sqlite3.connect(db_path)
        # WAL makes the journal updates (tombstones/watermark) durable and
        # visible to other processes without blocking readers.
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")

        self._create_schema()
        if not create_tables:
            self._migrate_schema()

    def _create_schema(self):
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS FRAME
                (id INT PRIMARY KEY NOT NULL,
                window_title TEXT NOT NULL,
                time DATETIME NOT NULL);
        """
        )
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS CONTENT
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_id INT NOT NULL,
                text TEXT NOT NULL,
                x INT NOT NULL,
                y INT NOT NULL,
                w INT NOT NULL,
                h INT NOT NULL
                );
        """
        )
        self.conn.execute(
            """CREATE VIRTUAL TABLE IF NOT EXISTS CONTENT_FTS
                USING fts5(frame_id, text, x, y, w, h)"""
        )
        self.conn.execute(
            """CREATE INDEX IF NOT EXISTS CONTENT_FRAME_ID ON CONTENT(frame_id)"""
        )

        # Stable tombstones: a capture id present here has been purged and can
        # never be re-resolved through the manifest.
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS TOMBSTONE
                (capture_id INT PRIMARY KEY NOT NULL,
                deleted_at DATETIME NOT NULL,
                plan_id TEXT NOT NULL,
                reason TEXT NOT NULL);
        """
        )
        # Purge plans and their per-segment work items.
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS PURGE_PLAN
                (plan_id TEXT PRIMARY KEY NOT NULL,
                created_at DATETIME NOT NULL,
                status TEXT NOT NULL,
                criteria_json TEXT NOT NULL,
                report_json TEXT);
        """
        )
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS PURGE_ITEM
                (plan_id TEXT NOT NULL,
                seg_uid TEXT NOT NULL,
                action TEXT NOT NULL,
                capture_ids_json TEXT NOT NULL,
                state TEXT NOT NULL,
                PRIMARY KEY (plan_id, seg_uid));
        """
        )
        # Reclaim watermark / generic key-value metadata.
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS META
                (key TEXT PRIMARY KEY NOT NULL,
                value TEXT);
        """
        )

        # The insert trigger only has to be created once (IF NOT EXISTS is not
        # supported for triggers, so guard by querying sqlite_master).
        row = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='trigger' AND name='insert_content'"
        ).fetchone()
        if row is None:
            self.conn.execute(
                """CREATE TRIGGER insert_content
                        AFTER INSERT ON CONTENT
                        BEGIN
                        INSERT INTO CONTENT_FTS (frame_id, text, x, y, w, h)
                        VALUES (new.frame_id, new.text, new.x, new.y, new.w, new.h);
                        end;
                        """
            )
        self.conn.commit()

    def _migrate_schema(self):
        # Schema evolution for databases created before the manifest layout.
        self._create_schema()

    def add_texts(self, texts, bbs, frame_i, window_title, time):
        self.conn.execute(
            "INSERT INTO FRAME (id, window_title, time) VALUES (?, ?, ?)",
            (str(frame_i), window_title, time),
        )

        for i in range(len(texts)):
            self.conn.execute(
                "INSERT INTO CONTENT (frame_id, text, x, y, w, h) VALUES (?, ?, ?, ?, ?, ?)",
                (frame_i, texts[i], bbs[i]["x"], bbs[i]["y"], bbs[i]["w"], bbs[i]["h"]),
            )

        self.conn.commit()

    def search(self, query):
        cursor = self.conn.execute(
            "SELECT rank, frame_id, text, x, y, w, h FROM CONTENT_FTS WHERE text MATCH ? ORDER BY rank DESC",
            (query,),
        )

        results = {}
        for row in cursor:
            rank = row[0]
            frame_id = str(row[1])
            if frame_id not in results:
                results[frame_id] = []

            results[frame_id].append(
                {
                    "text": row[2],
                    "bb": {"x": row[3], "y": row[4], "w": row[5], "h": row[6]},
                    "score": 1.0,
                }
            )

        return results

    # ------------------------------------------------------- purge journaling

    def create_purge_plan(self, plan_id, criteria, items, deleted_at, reason):
        """Persist plan + per-segment items + tombstones in one transaction.

        ``items`` is a list of dicts: seg_uid, action, capture_ids.
        """
        with self.conn:
            self.conn.execute(
                "INSERT INTO PURGE_PLAN (plan_id, created_at, status, criteria_json) "
                "VALUES (?, ?, 'planned', ?)",
                (plan_id, deleted_at, json.dumps(criteria)),
            )
            for item in items:
                self.conn.execute(
                    "INSERT INTO PURGE_ITEM (plan_id, seg_uid, action, capture_ids_json, state) "
                    "VALUES (?, ?, ?, ?, 'pending')",
                    (
                        plan_id,
                        item["seg_uid"],
                        item["action"],
                        json.dumps(item["capture_ids"]),
                    ),
                )
            self.conn.executemany(
                "INSERT OR REPLACE INTO TOMBSTONE (capture_id, deleted_at, plan_id, reason) "
                "VALUES (?, ?, ?, ?)",
                [(cid, deleted_at, plan_id, reason) for cid in sorted({
                    cid for item in items for cid in item["capture_ids"]
                })],
            )
            self.conn.execute(
                "INSERT OR REPLACE INTO META (key, value) VALUES ('reclaim_watermark', ?)",
                (plan_id,),
            )

    def get_unfinished_plan(self):
        """Return the plan referenced by the reclaim watermark, if unfinished."""
        row = self.conn.execute(
            "SELECT value FROM META WHERE key='reclaim_watermark'"
        ).fetchone()
        if row is None:
            return None
        plan_id = row[0]
        row = self.conn.execute(
            "SELECT status, criteria_json FROM PURGE_PLAN WHERE plan_id=?",
            (plan_id,),
        ).fetchone()
        if row is None or row[0] == "completed":
            return None
        return {"plan_id": plan_id, "status": row[0], "criteria": json.loads(row[1])}

    def get_plan_items(self, plan_id):
        rows = self.conn.execute(
            "SELECT seg_uid, action, capture_ids_json, state FROM PURGE_ITEM WHERE plan_id=?",
            (plan_id,),
        ).fetchall()
        return [
            {
                "seg_uid": r[0],
                "action": r[1],
                "capture_ids": json.loads(r[2]),
                "state": r[3],
            }
            for r in rows
        ]

    def set_item_state(self, plan_id, seg_uid, state):
        with self.conn:
            self.conn.execute(
                "UPDATE PURGE_ITEM SET state=? WHERE plan_id=? AND seg_uid=?",
                (state, plan_id, seg_uid),
            )

    def set_plan_status(self, plan_id, status):
        with self.conn:
            self.conn.execute(
                "UPDATE PURGE_PLAN SET status=? WHERE plan_id=?", (status, plan_id)
            )

    def finish_purge_plan(self, plan_id, report):
        with self.conn:
            self.conn.execute(
                "UPDATE PURGE_PLAN SET status='completed', report_json=? WHERE plan_id=?",
                (json.dumps(report), plan_id),
            )
            self.conn.execute("DELETE FROM META WHERE key='reclaim_watermark'")

    def delete_frame_rows(self, capture_ids):
        """Remove FRAME / CONTENT / FTS rows for the given capture ids."""
        if not capture_ids:
            return
        with self.conn:
            for cid in capture_ids:
                self.conn.execute("DELETE FROM CONTENT WHERE frame_id=?", (cid,))
                self.conn.execute("DELETE FROM CONTENT_FTS WHERE frame_id=?", (cid,))
                self.conn.execute("DELETE FROM FRAME WHERE id=?", (cid,))

    def count_frame_rows(self, capture_id):
        return (
            self.conn.execute(
                "SELECT "
                "(SELECT COUNT(*) FROM FRAME WHERE id=?), "
                "(SELECT COUNT(*) FROM CONTENT WHERE frame_id=?), "
                "(SELECT COUNT(*) FROM CONTENT_FTS WHERE frame_id=?)",
                (capture_id, capture_id, capture_id),
            ).fetchone()
        )
