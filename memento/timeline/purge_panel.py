import datetime

import pygame
import pygame_textinput

import memento.timeline.text_utils as text_utils
from memento.purge import (
    PurgeCriteria,
    PurgeExecutor,
    plan_purge,
)

STATE_FORM = "form"
STATE_PREVIEW = "preview"
STATE_WORKING = "working"
STATE_REPORT = "report"

FIELDS = [
    ("start", "Start time  YYYY-MM-DD HH:MM:SS (empty = none)"),
    ("end", "End time    YYYY-MM-DD HH:MM:SS (empty = none)"),
    ("app", "App/window title substring (empty = any)"),
    ("max_age_days", "Max retention days (empty = unlimited)"),
]


def human_bytes(n):
    n = float(n)
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024 or unit == "GB":
            return "%.1f %s" % (n, unit)
        n /= 1024


class PurgePanel:
    """Modal UI: build criteria, preview impact, execute, read the report."""

    def __init__(self, timeline):
        self.timeline = timeline
        self.open = False
        self.needs_reload = False
        self.state = STATE_FORM
        self.values = {key: "" for key, _ in FIELDS}
        self.field_index = 0
        self.textinput = pygame_textinput.TextInputManager()
        self.error = ""
        self.plan = None
        self.report = None

    # ------------------------------------------------------------- lifecycle

    def open_form(self):
        self.open = True
        self.needs_reload = False
        self.state = STATE_FORM
        self.values = {key: "" for key, _ in FIELDS}
        self.field_index = 0
        self.error = ""
        self.plan = None
        self.report = None
        self._load_field_into_input()

    def close(self):
        self.open = False
        self.state = STATE_FORM

    def _load_field_into_input(self):
        self.textinput.value = self.values[FIELDS[self.field_index][0]]

    def _store_input_field(self):
        self.values[FIELDS[self.field_index][0]] = self.textinput.value

    # -------------------------------------------------------------- handling

    def handle(self, events):
        if not self.open:
            return
        if self.state == STATE_FORM:
            self.textinput.update(events)
        for event in events:
            if event.type != pygame.KEYDOWN:
                continue
            if event.key == pygame.K_ESCAPE:
                if self.state == STATE_PREVIEW:
                    self.state = STATE_FORM
                    self._load_field_into_input()
                else:
                    self.close()
                return
            if self.state == STATE_FORM:
                if event.key == pygame.K_TAB:
                    self._store_input_field()
                    self.field_index = (self.field_index + 1) % len(FIELDS)
                    self._load_field_into_input()
                elif event.key == pygame.K_RETURN:
                    self._store_input_field()
                    self._preview()
            elif self.state == STATE_PREVIEW:
                if event.key in (pygame.K_RETURN, pygame.K_y):
                    self._execute()
            elif self.state == STATE_REPORT:
                if event.key in (pygame.K_e, pygame.K_RETURN):
                    self.close()

    def _build_criteria(self):
        return PurgeCriteria(
            start=self.values["start"],
            end=self.values["end"],
            app=self.values["app"],
            max_age_days=self.values["max_age_days"],
        )

    def _preview(self):
        try:
            criteria = self._build_criteria()
        except (ValueError, TypeError) as e:
            self.error = "Invalid input: %s" % e
            return
        if criteria.start is None and criteria.end is None and criteria.app is None \
                and criteria.max_age_days is None:
            self.error = "Please provide at least one criterion."
            return
        self.error = ""
        fg = self.timeline.frame_getter
        self.plan = plan_purge(fg.manifest, criteria)
        self.state = STATE_PREVIEW

    def _execute(self):
        self.state = STATE_WORKING
        fg = self.timeline.frame_getter
        chromadb = getattr(self.timeline.chat, "chromadb", None)
        executor = PurgeExecutor(fg.db_conn(), fg.manifest, chromadb)
        self.report = executor.execute(self.plan)
        self.state = STATE_REPORT
        if isinstance(self.report, dict) and self.report.get("status") == "empty":
            self.state = STATE_PREVIEW
            self.error = "Nothing matched the criteria."
        else:
            # Physical layout changed; timeline reloads frames/timebar/search.
            self.needs_reload = True

    # ---------------------------------------------------------------- drawing

    def draw(self, screen):
        if not self.open:
            return
        ws = self.timeline.window_size
        overlay = pygame.Surface(ws, flags=pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        screen.blit(overlay, (0, 0))

        pw, ph = int(ws[0] * 0.6), int(ws[1] * 0.7)
        px, py = (ws[0] - pw) // 2, (ws[1] - ph) // 2
        panel = pygame.Surface((pw, ph), flags=pygame.SRCALPHA)
        pygame.draw.rect(panel, (245, 245, 245), (0, 0, pw, ph),
                         border_radius=12)
        screen.blit(panel, (px, py))
        pygame.draw.rect(screen, (0, 0, 0), (px, py, pw, ph), width=3)

        title_font = pygame.font.SysFont("Arial", 26, bold=True)
        font = pygame.font.SysFont("Arial", 20)
        pad = 24

        if self.state == STATE_FORM:
            self._draw_form(screen, px, py, pw, title_font, font, pad)
        elif self.state == STATE_PREVIEW:
            self._draw_preview(screen, px, py, pw, title_font, font, pad)
        elif self.state == STATE_WORKING:
            t = title_font.render("Purging ... rewriting H.264 segments", True, (0, 0, 0))
            screen.blit(t, (px + pad, py + pad))
        elif self.state == STATE_REPORT:
            self._draw_report(screen, px, py, pw, title_font, font, pad)

    def _draw_form(self, screen, px, py, pw, title_font, font, pad):
        t = title_font.render("Local purge plan", True, (0, 0, 0))
        screen.blit(t, (px + pad, py + pad))
        hint = font.render(
            "Rules are combined with OR. Tab to switch fields, Enter to preview.",
            True, (90, 90, 90),
        )
        screen.blit(hint, (px + pad, py + pad + 40))

        y = py + pad + 90
        for i, (key, label) in enumerate(FIELDS):
            color = (0, 90, 200) if i == self.field_index else (60, 60, 60)
            screen.blit(font.render(label, True, color), (px + pad, y))
            value = self.textinput.value if i == self.field_index else self.values[key]
            screen.blit(font.render(value or "(empty)", True,
                                    (0, 0, 0) if value else (160, 160, 160)),
                        (px + pad, y + 26))
            y += 78

        if self.error:
            text_utils.render_text(
                screen, self.error, font, px + pad, y, pw - pad * 2, (200, 0, 0)
            )

    def _draw_preview(self, screen, px, py, pw, title_font, font, pad):
        t = title_font.render("Purge impact preview", True, (0, 0, 0))
        screen.blit(t, (px + pad, py + pad))
        lines = self._preview_lines()
        y = py + pad + 46
        for color, line in lines:
            text_utils.render_text(
                screen, line, font, px + pad, y, pw - pad * 2, color
            )
            y += text_utils.get_text_height(line, font, pw - pad * 2) + 6

    def _preview_lines(self):
        p = self.plan
        s = p.summary
        lines = [(0, "Rules: " + p.criteria.describe())]
        lines.append((0, "%d captures will be deleted" % s["total_captures"]))
        lines.append((0, "Time span: %s -> %s" % (s["earliest"], s["latest"])))
        lines.append(
            (0,
             "Segments: %d fully recycled, %d partially rewritten (H.264 decode/rewrite)"
             % (s["segments_recycled"], s["segments_rewritten"]))
        )
        lines.append(
            (0, "Estimated space to free: %s" % human_bytes(s["bytes_to_free_estimate"]))
        )
        if s["per_app"]:
            lines.append((0, "Per application:"))
            for app, n in sorted(s["per_app"].items(), key=lambda kv: -kv[1]):
                lines.append((60, "  - %s: %d captures" % (app, n)))
        if s["skipped_active_segment_count"]:
            lines.append(
                (200, "Note: %d capture(s) in the active recording segment "
                      "cannot be purged until the segment closes (max 10s)."
                 % s["skipped_active_segment_count"])
            )
        if s["total_captures"] == 0:
            lines.append((200, "No capture matches these criteria."))
        else:
            lines.append((0, "Layers purged: media files, segment JSON, "
                             "FRAME/CONTENT/FTS, Chroma; tombstones written."))
            lines.append(((0, 110, 0), "Enter/Y: confirm purge    Esc: back"))
        return lines

    def _draw_report(self, screen, px, py, pw, title_font, font, pad):
        t = title_font.render("Local purge report", True, (0, 0, 0))
        screen.blit(t, (px + pad, py + pad))
        r = self.report
        lines = [
            (0, "Plan: %s   completed %s%s" % (
                r["plan_id"], r["completed_at"],
                " (resumed after crash)" if r.get("resumed") else "")),
            (0, "Captures deleted: %d (%d segments recycled, %d rewritten)"
             % (r["captures_deleted"], r["segments_recycled"],
                r["segments_rewritten"])),
            (0, "Space freed: %s" % human_bytes(r["bytes_freed"])),
        ]
        chroma = r.get("chroma", {})
        if chroma.get("status") == "deleted":
            lines.append((0, "Chroma documents deleted: %d"
                          % chroma.get("documents_deleted", 0)))
        elif chroma.get("status") == "error":
            lines.append((200, "Chroma error: %s" % chroma.get("error")))
        else:
            lines.append((90, "Chroma: unavailable (no API key), nothing to clear"))

        v = r["verification"]
        if r["verified"]:
            lines.append(((0, 110, 0),
                          "Verification: all purged ids are gone "
                          "from manifest, JSON, FRAME/CONTENT/FTS and Chroma"))
        else:
            lines.append((200, "Verification: leftovers detected:"))
            for layer, leftovers in v["leftovers"].items():
                if leftovers:
                    lines.append((200, "  - %s: %s" % (layer, leftovers[:10])))
        lines.append((90, "Report saved at purge_reports/%s.json" % r["plan_id"]))
        lines.append((0, "Press Enter/E to close"))

        y = py + pad + 46
        for color, line in lines:
            text_utils.render_text(
                screen, line, font, px + pad, y, pw - pad * 2, color
            )
            y += text_utils.get_text_height(line, font, pw - pad * 2) + 6
