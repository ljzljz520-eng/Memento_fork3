import datetime

import cv2
import numpy as np
import pygame

import memento.utils as utils
from memento.manifest import parse_time
from memento.timeline.apps import Apps

FRAME_PERIOD = 1.0 / utils.FPS


class TimeBar:
    """Time-based navigation over the sparse capture index.

    The window is defined in *seconds*; captures are located by logical time
    through the FrameGetter/manifest. Nothing here assumes dense integer
    frame ids.
    """

    def __init__(self, frame_getter):
        self.frame_getter = frame_getter
        self.manifest = frame_getter.manifest
        self.show_bar = True

        # Actual graphical window size
        ws = self.frame_getter.window_size
        self.h = ws[1] // 40
        self.x = 20
        self.y = ws[1] - self.h - ws[1] // 10
        self.w = ws[0] - self.x * 2

        # Time window size in seconds
        self.min_tws = 40.0
        self.max_tws = utils.MAX_TWS / utils.FPS
        self.tws = min(
            max(self.min_tws, 10 * utils.SECONDS_PER_REC), self.max_tws
        )
        self.offset = 0.0  # seconds between latest capture and window end

        self.compute_time_window()

        last_pos = self.frame_getter.nb_frames - 1
        self.current_pos = max(0, last_pos)
        self.preview_capture_id = None
        self.preview_surf = None
        self.apps = Apps(self.frame_getter)
        self.today = datetime.datetime.now().strftime("%Y-%m-%d")

    # --------------------------------------------------------- window helpers

    def zoom(self, direction):
        center = self.tw_start + datetime.timedelta(seconds=self.tws / 2)
        self.tws = min(
            self.max_tws, max(self.min_tws, self.tws * (1 + direction * 0.1))
        )
        self.tw_start = center - datetime.timedelta(seconds=self.tws / 2)
        self._recompute_end_and_clamp()

    def compute_time_window(self):
        latest = self.frame_getter.latest_time()
        if latest is not None:
            self.tw_end = latest - datetime.timedelta(seconds=self.offset)
        else:
            self.tw_end = datetime.datetime.now()
        self.tw_start = self.tw_end - datetime.timedelta(seconds=self.tws)

    def _recompute_end_and_clamp(self):
        latest = self.frame_getter.latest_time()
        if latest is None:
            self.compute_time_window()
            return
        self.tw_end = self.tw_start + datetime.timedelta(seconds=self.tws)
        self.offset = max(0.0, (latest - self.tw_end).total_seconds())
        self.tw_end = latest - datetime.timedelta(seconds=self.offset)
        self.tw_start = self.tw_end - datetime.timedelta(seconds=self.tws)

    def get_friendly_date(self, date):
        day = date.split(" ")[0]
        hr = date.split(" ")[1]
        if day == self.today:
            return "Today" + " " + hr
        elif day == (datetime.datetime.now() - datetime.timedelta(days=1)).strftime(
            "%Y-%m-%d"
        ):
            return "Yesterday" + " " + hr
        elif day == (datetime.datetime.now() - datetime.timedelta(days=2)).strftime(
            "%Y-%m-%d"
        ):
            return "2 days ago" + " " + hr
        else:
            return date

    @property
    def current_capture_id(self):
        cap = self.frame_getter.capture_at_position(self.current_pos)
        return cap["capture_id"] if cap else None

    def jump_to_capture(self, capture_id):
        pos = self.frame_getter.position_of(capture_id)
        if pos is None:
            return
        self.current_pos = pos
        self._ensure_visible()

    def move_cursor(self, delta):
        if self.frame_getter.nb_frames == 0:
            return
        self.current_pos = max(
            0,
            min(self.frame_getter.nb_frames - 1, self.current_pos + delta),
        )
        self._ensure_visible()

    def _ensure_visible(self):
        cap = self.frame_getter.capture_at_position(self.current_pos)
        if cap is None:
            return
        t = parse_time(cap["time"])
        edge = 0.15
        if t < self.tw_start + datetime.timedelta(seconds=self.tws * edge):
            self.tw_start = t - datetime.timedelta(seconds=self.tws * edge)
            self._recompute_end_and_clamp()
        elif t > self.tw_end - datetime.timedelta(seconds=self.tws * edge):
            self.tw_end = t + datetime.timedelta(seconds=self.tws * edge)
            latest = self.frame_getter.latest_time()
            self.tw_end = min(self.tw_end, latest)
            self.offset = max(0.0, (latest - self.tw_end).total_seconds())
            self.tw_end = latest - datetime.timedelta(seconds=self.offset)
            self.tw_start = self.tw_end - datetime.timedelta(seconds=self.tws)

    def click_select(self, mouse_pos):
        t = self.get_time(mouse_pos)
        self.current_pos = self.frame_getter.position_at_or_before_time(t)
        self._ensure_visible()

    def get_time(self, mouse_pos):
        frac = (mouse_pos[0] - self.x) / self.w
        frac = max(0.0, min(1.0, frac))
        return self.tw_start + datetime.timedelta(seconds=frac * self.tws)

    def hover(self, mouse_pos):
        return utils.in_rect((self.x, self.y, self.w, self.h), mouse_pos)

    def draw_cursor(self, screen):
        cap = self.frame_getter.capture_at_position(self.current_pos)
        if cap is None:
            return
        t = parse_time(cap["time"])
        frac = (t - self.tw_start).total_seconds() / self.tws
        cursor_x = self.x + frac * self.w
        pygame.draw.line(
            screen,
            (255, 255, 255),
            (cursor_x, self.y - self.h),
            (cursor_x, self.y + self.h * 2),
            5,
        )

        self.draw_time(screen, (cursor_x, self.y))

    def draw_bar(self, screen, mouse_pos):
        visible = self.frame_getter.captures_in_time_range(
            self.tw_start, self.tw_end + datetime.timedelta(seconds=FRAME_PERIOD)
        )
        if not visible:
            pygame.draw.rect(
                screen, (40, 40, 40), (self.x, self.y, self.w, self.h),
                border_radius=self.h // 4,
            )
            return

        # Midpoint boundaries between consecutive visible captures.
        times = [parse_time(f["time"]) for f in visible]
        boundaries = []
        for i, t in enumerate(times):
            if i == 0:
                left = self.tw_start
            else:
                left = times[i - 1] + (t - times[i - 1]) / 2
            if i == len(times) - 1:
                right = self.tw_end
            else:
                tn = times[i + 1]
                right = t + (tn - t) / 2
            boundaries.append((left, right))

        groups = []  # consecutive same-app runs
        for frame, (left, right) in zip(visible, boundaries):
            app = frame["window_title"]
            if groups and groups[-1]["app"] == app:
                groups[-1]["right"] = right
            else:
                groups.append({"app": app, "left": left, "right": right})

        for group in groups:
            seg_x = self.x + (
                (group["left"] - self.tw_start).total_seconds() / self.tws
            ) * self.w
            seg_w = (
                (group["right"] - group["left"]).total_seconds() / self.tws
            ) * self.w
            pygame.draw.rect(
                screen,
                self.apps.get_color(group["app"]),
                (seg_x, self.y, max(1, seg_w), self.h),
                border_radius=self.h // 4,
            )

        current_cid = self.current_capture_id
        for frame, (left, right) in zip(visible, boundaries):
            app = frame["window_title"]
            middle = left + (right - left) / 2
            seg_x = self.x + (
                (left - self.tw_start).total_seconds() / self.tws
            ) * self.w
            seg_w = (
                (right - left).total_seconds() / self.tws
            ) * self.w

            if self.apps.get_icon(app) is not None:
                if frame["capture_id"] == current_cid or utils.in_rect(
                    (seg_x, self.y, seg_w, self.h), mouse_pos
                ):
                    screen.blit(
                        self.apps.get_icon(app, small=False),
                        (
                            self.x
                            + ((middle - self.tw_start).total_seconds() / self.tws)
                            * self.w
                            - self.apps.ig.size,
                            self.y - self.apps.ig.size // 2,
                        ),
                    )
                else:
                    screen.blit(
                        self.apps.get_icon(app, small=True),
                        (
                            self.x
                            + ((middle - self.tw_start).total_seconds() / self.tws)
                            * self.w
                            - self.apps.ig.size // 2,
                            self.y,
                        ),
                    )

    # TODO cache previews ?
    def draw_preview(self, screen, mouse_pos):
        if not self.hover(mouse_pos):
            return
        t = self.get_time(mouse_pos)
        pos = self.frame_getter.position_at_or_before_time(t)
        cap = self.frame_getter.capture_at_position(pos)
        if cap is None:
            return
        capture_id = cap["capture_id"]
        if capture_id != self.preview_capture_id or self.preview_surf is None:
            self.preview_capture_id = capture_id
            frame = self.frame_getter.get_frame(capture_id)
            frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)
            self.preview_surf = pygame.surfarray.make_surface(frame)
        screen.blit(
            self.preview_surf,
            [mouse_pos[0], self.y]
            - np.array(
                [self.preview_surf.get_size()[0] // 2, self.preview_surf.get_size()[1]]
            ),
        )

    def draw_time(self, screen, pos):
        if not self.hover(pos):
            return
        t = self.get_time(pos)
        capture_pos = self.frame_getter.position_at_or_before_time(t)
        cap = self.frame_getter.capture_at_position(capture_pos)
        if cap is None:
            return
        time = cap["time"]
        time = self.get_friendly_date(time)
        font = pygame.font.SysFont("Arial", 20)
        text = font.render(time, True, (0, 0, 0))
        text_size = text.get_size()
        border = 10
        pygame.draw.rect(
            screen,
            (255, 255, 255),
            (
                pos[0] - text_size[0] // 2 - border,
                self.y + text_size[1] + self.h - border,
                text_size[0] + border * 2,
                text_size[1] + border * 2,
            ),
            border_radius=self.h // 2,
        )

        screen.blit(text, [pos[0] - text_size[0] // 2, self.y + text_size[1] + self.h])
        pygame.draw.line(
            screen,
            (255, 255, 255),
            (pos[0], self.y),
            (pos[0], self.y + self.h),
            5,
        )

    def show(self):
        self.show_bar = True

    def hide(self):
        self.show_bar = False

    def draw(self, screen, mouse_pos):
        if self.show_bar:
            self.draw_preview(screen, mouse_pos)
            self.draw_bar(screen, mouse_pos)
            self.draw_cursor(screen)
            self.draw_time(screen, mouse_pos)
