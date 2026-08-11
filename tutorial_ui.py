"""Startup tutorial UI for Focus Tracker.

This module is intentionally independent from ui_ver3.py.
Add only the following to main_ver3.py:

    import tutorial_ui

and at the start of main():

    if not tutorial_ui.run_startup_tutorial(
        window_name=ui.WINDOW_NAME,
        width=ui.CANVAS_W,
        height=ui.CANVAS_H,
    ):
        return

No camera is opened by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
import sys
import traceback

import cv2
import numpy as np

import kr_text

Rect = Tuple[int, int, int, int]
Color = Tuple[int, int, int]


def _report_error(context: str, exc: BaseException, log_path: str = "tutorial_ui_error.log") -> None:
    """Print a full traceback and best-effort append it to a local log file."""
    message = (
        f"[TutorialUI ERROR] {context}: {type(exc).__name__}: {exc}\n"
        + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    )
    print(message, file=sys.stderr, flush=True)

    # Logging must never cause a second crash.
    try:
        Path(log_path).open("a", encoding="utf-8").write(message + "\n")
    except Exception as log_exc:
        print(
            f"[TutorialUI WARNING] Could not write {log_path}: "
            f"{type(log_exc).__name__}: {log_exc}",
            file=sys.stderr,
            flush=True,
        )



@dataclass(frozen=True)
class TutorialPage:
    title: str
    subtitle: str
    lines: Sequence[str]
    visual: str


DEFAULT_PAGES: Tuple[TutorialPage, ...] = (
    TutorialPage(
        title="1. 카메라 설정",
        subtitle="이 트래커는 두 개의 카메라를 사용합니다.",
        lines=(
            "얼굴 캠 : 노트북 내장 카메라",
            "책상 캠 : 보조 웹캠 또는 폰 카메라",
            "책상 캠은 휴대폰 / 책 / 노트북이 보이도록 배치하세요.",
            "얼굴 캠은 평소 공부하는 자리 정면에 두세요.",
        ),
        visual="camera",
    ),
    TutorialPage(
        title="2. 캘리브레이션",
        subtitle="튜토리얼이 끝나면 캘리브레이션이 시작됩니다.",
        lines=(
            "마이크: 기준값을 측정하는 동안 조용히 해주세요.",
            "시선: 캘리브레이션 중에는 얼굴 캠을 정면으로 응시하세요.",
            "평소 앉는 자세로 시선 기준을 잡아주세요.",
            "완료되면 자동으로 트래커가 시작됩니다.",
        ),
        visual="calibration",
    ),
    TutorialPage(
        title="3. 공부 시작",
        subtitle="과목을 선택하고 학습 세션을 시작하세요.",
        lines=(
            "과목을 선택하고 재생 버튼을 누르세요.",
            "오른쪽 패널에 타이머, 점수, 10초 그래프가 표시됩니다.",
            "일시정지하면 과목 목록으로 돌아갑니다.",
            "종료하려면 트래커에서 q 또는 X 버튼을 누르세요.",
        ),
        visual="study",
    ),
)


def point_in_rect(x: int, y: int, rect: Optional[Rect]) -> bool:
    if rect is None:
        return False
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]


def _draw_text_centered(
    canvas: np.ndarray,
    text: str,
    cx: int,
    cy: int,
    scale: float,
    color: Color,
    thickness: int,
) -> None:
    kr_text.put_text(
        canvas,
        text,
        (cx, cy),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
        center=True,
    )


def _draw_button(
    canvas: np.ndarray,
    rect: Rect,
    text: str,
    fill: Color = (55, 55, 55),
    border: Color = (180, 180, 180),
) -> None:
    cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), fill, -1)
    cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), border, 1)
    _draw_text_centered(
        canvas,
        text,
        (rect[0] + rect[2]) // 2,
        (rect[1] + rect[3]) // 2,
        0.55,
        (245, 245, 245),
        1,
    )


class TutorialUI:
    """Small stateful OpenCV tutorial that can also be rendered headlessly for tests."""

    def __init__(
        self,
        window_name: str = "Focus Tracker Tutorial",
        width: int = 1280,
        height: int = 600,
        pages: Sequence[TutorialPage] = DEFAULT_PAGES,
    ) -> None:
        if width < 900 or height < 500:
            raise ValueError("Tutorial window must be at least 900x500.")
        if not pages:
            raise ValueError("At least one tutorial page is required.")

        self.window_name = window_name
        self.width = int(width)
        self.height = int(height)
        self.pages = tuple(pages)
        self.page_index = 0
        self.done = False
        self.cancelled = False
        self.buttons: Dict[str, Optional[Rect]] = {
            "back": None,
            "next": None,
            "skip": None,
        }

    def _panel_layout(self) -> Tuple[Rect, int]:
        left_margin = 40
        top = 70
        bottom = self.height - 100
        left_panel_w = max(430, int(self.width * 0.405))
        left_rect = (left_margin, top, left_margin + left_panel_w, bottom)
        text_x = left_rect[2] + 60
        return left_rect, text_x

    def _draw_visual(self, canvas: np.ndarray, page: TutorialPage, rect: Rect) -> None:
        x0, y0, x1, y1 = rect
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (28, 28, 28), -1)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (70, 70, 70), 1)

        center_x = (x0 + x1) // 2
        panel_w = x1 - x0

        if page.visual == "camera":
            pad = max(35, panel_w // 10)
            top_box = (x0 + pad, y0 + 55, x1 - pad, y0 + 195)
            bot_box = (x0 + pad, y0 + 245, x1 - pad, y0 + 385)
            for box in (top_box, bot_box):
                cv2.rectangle(canvas, box[:2], box[2:], (40, 40, 40), -1)
                cv2.rectangle(canvas, box[:2], box[2:], (100, 100, 100), 1)
            _draw_text_centered(canvas, "책상 캠", center_x, top_box[1] + 28, 0.7, (0, 230, 255), 2)
            _draw_text_centered(canvas, "휴대폰 / 책 / 노트북", center_x, top_box[1] + 82, 0.56, (210, 210, 210), 1)
            _draw_text_centered(canvas, "얼굴 캠", center_x, bot_box[1] + 28, 0.7, (0, 230, 255), 2)
            _draw_text_centered(canvas, "얼굴 / 시선 / 눈", center_x, bot_box[1] + 82, 0.56, (210, 210, 210), 1)

        elif page.visual == "calibration":
            label_x = x0 + int(panel_w * 0.27)
            visual_x = x0 + int(panel_w * 0.62)
            _draw_text_centered(canvas, "마이크", label_x, y0 + 105, 1.0, (0, 230, 255), 2)
            cv2.line(canvas, (visual_x - 90, y0 + 105), (visual_x + 90, y0 + 105), (120, 120, 120), 3)
            _draw_text_centered(canvas, "조용히", visual_x, y0 + 150, 0.65, (210, 210, 210), 1)

            _draw_text_centered(canvas, "시선", label_x, y0 + 270, 0.9, (0, 230, 255), 2)
            cv2.circle(canvas, (visual_x, y0 + 270), 48, (110, 110, 110), 2)
            cv2.circle(canvas, (visual_x, y0 + 270), 12, (0, 230, 255), -1)
            _draw_text_centered(canvas, "정면 응시", visual_x, y0 + 355, 0.6, (210, 210, 210), 1)

        else:
            pad = max(35, panel_w // 10)
            labels = ("1. 과목 선택", "2. 재생 버튼 클릭", "3. 점수 / 그래프 확인")
            box_h = 64
            gap = 40
            y = y0 + 60
            for label in labels:
                box = (x0 + pad, y, x1 - pad, y + box_h)
                cv2.rectangle(canvas, box[:2], box[2:], (45, 45, 45), -1)
                cv2.rectangle(canvas, box[:2], box[2:], (100, 100, 100), 1)
                _draw_text_centered(canvas, label, center_x, y + box_h // 2, 0.60, (230, 230, 230), 1)
                y += box_h + gap

    def render_page(self, page_index: Optional[int] = None) -> np.ndarray:
        """Return one tutorial frame without opening a window (useful for testing)."""
        if page_index is None:
            page_index = self.page_index
        if not 0 <= page_index < len(self.pages):
            raise IndexError("Tutorial page index out of range.")

        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:] = (18, 18, 18)
        page = self.pages[page_index]
        left_rect, tx = self._panel_layout()
        self._draw_visual(canvas, page, left_rect)

        kr_text.put_text(canvas, "집중력 트래커 튜토리얼", (tx, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (160, 160, 160), 1, cv2.LINE_AA)
        kr_text.put_text(canvas, page.title, (tx, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        kr_text.put_text(canvas, page.subtitle, (tx, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 220, 255), 1, cv2.LINE_AA)

        y = 245
        for line in page.lines:
            kr_text.put_text(canvas, line, (tx, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
            y += 34

        for i in range(len(self.pages)):
            color = (0, 220, 255) if i == page_index else (75, 75, 75)
            cv2.circle(canvas, (tx + i * 28, self.height - 175), 6, color, -1)

        skip_rect = (self.width - 125, 18, self.width - 25, 52)
        back_rect = (tx, self.height - 115, tx + 120, self.height - 65)
        next_rect = (self.width - 190, self.height - 115, self.width - 35, self.height - 65)

        self.buttons["skip"] = skip_rect
        self.buttons["back"] = back_rect if page_index > 0 else None
        self.buttons["next"] = next_rect

        _draw_button(canvas, skip_rect, "건너뛰기", fill=(42, 42, 42))
        if page_index > 0:
            _draw_button(canvas, back_rect, "이전")
        _draw_button(
            canvas,
            next_rect,
            "시작" if page_index == len(self.pages) - 1 else "다음",
            fill=(45, 90, 55),
        )

        kr_text.put_text(
            canvas,
            "키보드: ← / → / Enter    ESC: 건너뛰기    Q: 취소",
            (tx, self.height - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (130, 130, 130),
            1,
            cv2.LINE_AA,
        )
        return canvas

    def next_page(self) -> None:
        if self.page_index >= len(self.pages) - 1:
            self.done = True
        else:
            self.page_index += 1

    def previous_page(self) -> None:
        self.page_index = max(0, self.page_index - 1)

    def skip(self) -> None:
        self.done = True

    def cancel(self) -> None:
        self.cancelled = True
        self.done = True

    def handle_key(self, key: int) -> None:
        # OpenCV arrow codes differ by platform, so support common codes + A/D.
        left_codes = {81, 2424832, ord("a"), ord("A")}
        right_codes = {83, 2555904, ord("d"), ord("D")}
        if key in left_codes:
            self.previous_page()
        elif key in right_codes or key in {13, 10}:
            self.next_page()
        elif key == 27:  # ESC: skip tutorial and continue tracker
            self.skip()
        elif key in {ord("q"), ord("Q")}:
            self.cancel()

    def handle_click(self, x: int, y: int) -> None:
        if point_in_rect(x, y, self.buttons.get("skip")):
            self.skip()
        elif point_in_rect(x, y, self.buttons.get("back")):
            self.previous_page()
        elif point_in_rect(x, y, self.buttons.get("next")):
            self.next_page()

    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            self.handle_click(x, y)

    def run(self) -> bool:
        """Show the tutorial. Return True to continue tracker, False to cancel.

        Unexpected exceptions are intentionally allowed to propagate to
        run_startup_tutorial(), which records a full traceback and decides
        whether the tracker should continue.
        """
        window_created = False
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            window_created = True
            cv2.resizeWindow(self.window_name, self.width, self.height)
            cv2.setMouseCallback(self.window_name, self._mouse_callback)

            while not self.done:
                canvas = self.render_page()
                cv2.imshow(self.window_name, canvas)
                key = cv2.waitKeyEx(20)
                if key != -1:
                    self.handle_key(key)

                try:
                    visible = cv2.getWindowProperty(
                        self.window_name, cv2.WND_PROP_VISIBLE
                    )
                except cv2.error as exc:
                    # Some OpenCV backends can transiently fail this query.
                    # Keep the tutorial alive but make the issue visible.
                    print(
                        f"[TutorialUI WARNING] window visibility check failed: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    visible = 1.0

                if visible < 1:
                    self.cancel()

            return not self.cancelled
        finally:
            if window_created:
                try:
                    cv2.destroyWindow(self.window_name)
                except cv2.error as exc:
                    print(
                        f"[TutorialUI WARNING] destroyWindow failed: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )


def run_startup_tutorial(
    window_name: str = "Focus Tracker Tutorial",
    width: int = 1280,
    height: int = 600,
    *,
    fail_open: bool = True,
    error_log: str = "tutorial_ui_error.log",
) -> bool:
    """Run the startup tutorial safely.

    Returns:
        True: continue to the tracker. This includes Skip/ESC and, when
              fail_open=True, an unexpected tutorial error.
        False: user explicitly cancelled with Q / window close, or an
               unexpected error occurred while fail_open=False.

    Any unexpected error is printed with a full traceback and also appended
    to ``error_log`` when possible. The default ``fail_open=True`` prevents
    an optional tutorial bug from taking down the main tracker.
    """
    try:
        return TutorialUI(
            window_name=window_name,
            width=width,
            height=height,
        ).run()
    except Exception as exc:
        _report_error("startup tutorial failed", exc, error_log)
        if fail_open:
            print(
                "[TutorialUI] Tutorial skipped because of the error; "
                "continuing to the main tracker.",
                file=sys.stderr,
                flush=True,
            )
            return True
        return False


__all__ = [
    "TutorialPage",
    "TutorialUI",
    "DEFAULT_PAGES",
    "run_startup_tutorial",
]
