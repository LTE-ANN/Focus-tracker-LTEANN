import cv2
import time
import datetime
import numpy as np
import sys
import traceback

from cam1_ver5 import Camera1
from cam2_ver5 import Camera2
import ui_ver5 as ui

# Tutorial is an optional module. If a teammate accidentally omits/breaks it,
# the tracker still starts, but the full traceback is shown immediately.
try:
    import tutorial_ui
except Exception as exc:
    tutorial_ui = None
    print(
        f"[Main WARNING] tutorial_ui import failed: {type(exc).__name__}: {exc}",
        file=sys.stderr,
        flush=True,
    )
    traceback.print_exc()

ALPHA = 0.5              # 총점 EMA 스무딩 계수
GRAPH_SAMPLE_SEC = 5.0  # 그래프에 점 하나 찍는 주기(초) - 5초 평균


def _run_tutorial_safely():
    """Run the optional startup tutorial without taking down the tracker.

    Returns:
        True  -> continue to calibration/tracker
        False -> user explicitly cancelled tutorial (Q/window close)
    """
    if tutorial_ui is None:
        print(
            "[Main WARNING] Tutorial unavailable; continuing without tutorial.",
            file=sys.stderr,
            flush=True,
        )
        return True

    try:
        return tutorial_ui.run_startup_tutorial(
            window_name=ui.WINDOW_NAME,
            width=ui.CANVAS_W,
            height=ui.CANVAS_H,
            fail_open=True,
            error_log="tutorial_ui_error.log",
        )
    except Exception as exc:
        # Defensive fallback. tutorial_ui already catches its own unexpected
        # errors, but this makes main_ver5 robust even if that module changes.
        print(
            f"[Main ERROR] Unexpected tutorial failure: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        print(
            "[Main] Continuing to the tracker without tutorial.",
            file=sys.stderr,
            flush=True,
        )
        return True


def main():
    # Tutorial runs before opening cameras/audio.
    # Skip/ESC -> continue, Q/window close -> cancel tracker startup.
    if not _run_tutorial_safely():
        print("Tracker startup cancelled from tutorial.")
        return

    cam1 = Camera1()  # 책상(핸드폰) 카메라 - YOLO 물체 감지
    cam2 = Camera2()  # 얼굴(노트북) 카메라 - 시선/눈 감김/소음 분석

    if cam2.start_audio():
        cam2.calibrate_audio_baseline()
    else:
        print("Warning: audio disabled; noise detection will be off.")
    cam2.calibrate_gaze()

    active_subject = None  # 현재 활성 과목 이름 (메모리에만 유지, 활성 과목 없으면 None)

    def flush_leftover_sample():
        # sample_buffer는 GRAPH_SAMPLE_SEC(10초)마다 한 번씩 흘려보내지는데,
        # 일시정지/종료가 그 주기 중간에 일어나면 아직 흘려보내지 못한 표본이 남아있다.
        # 여기서 세션이 끝나기 직전에 그 나머지를 마저 기록해서 유실을 막는다.
        if not ui.sample_buffer:
            return
        leftover_avg = sum(ui.sample_buffer) / len(ui.sample_buffer)
        ui.graph_points.append(leftover_avg)
        if active_subject is not None:
            ui.record_subject_score(active_subject, leftover_avg)
        ui.sample_buffer.clear()
        ui.last_flush_t = time.time()

    def enter_summary():
        """공부 종료/윈도우 종료 요청을 가로채서 바로 끄지 않고 종합 평가 화면으로 전환."""
        if state_active_subject_running():
            flush_leftover_sample()
            ui.stop_active()
        ui.prepare_summary(ui.graph_points, ui.subject_breakdown())
        ui.state["screen"] = "summary"
        ui.state["should_exit"] = False

    def state_active_subject_running():
        return ui.state["screen"] == "active" and ui.state["active_idx"] is not None

    cv2.namedWindow(ui.WINDOW_NAME)
    cv2.setMouseCallback(ui.WINDOW_NAME, ui.mouse_callback)

    smoothed_score = None
    prev_t = time.time()
    fps = 0.0

    print("통합 트래커 실행 중! (종료하려면 화면 내 X 버튼 또는 'q' 입력)")

    while True:
        today = datetime.date.today()
        if today != ui.state["today"]:
            ui.state["today"] = today
            ui.reset_daily_state()

        # --- 종합 평가(summary) 화면: 카메라를 더 이상 읽지 않고 결과만 보여준다 ---
        if ui.state["screen"] == "summary":
            canvas = np.zeros((ui.CANVAS_H, ui.CANVAS_W, 3), dtype=np.uint8)
            ui.draw_summary_screen(canvas)
            cv2.imshow(ui.WINDOW_NAME, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                ui.handle_key(key)
            if ui.state["should_exit"] or cv2.getWindowProperty(ui.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
            continue

        # --- 활성 과목 추적: 재생 중이면 이름을 기억, 아니면 남은 표본을 흘려보낸다 ---
        if state_active_subject_running():
            active_subject = ui.subjects[ui.state["active_idx"]]["name"]
        elif active_subject is not None:
            flush_leftover_sample()
            active_subject = None

        # --- [Step A] 책상 캠: YOLO 감지 ---
        frame1, factor, boxes = cam1.process_frame()
        if frame1 is None:
            print("Camera1 frame grab failed; exiting.")
            enter_summary()
            continue
        # 박스 좌표는 원본 해상도 기준이므로, 리사이즈 전에 원본 프레임 위에 먼저 그린다.
        # (리사이즈 후에 그리면 좌표계가 어긋나 박스 위치가 실제 물체와 맞지 않는다.)
        ui.draw_detection_boxes(frame1, boxes, cam1.model.names, cam1.DISTRACTION, cam1.STUDY_ITEMS, cam1.CONF_TH)
        frame1 = cv2.resize(frame1, (ui.CAM_W, ui.CAM1_H))
        ui.draw_pane_label(frame1, "Desk Cam")

        # --- [Step B] 얼굴 캠: MediaPipe 시선/눈 감김/소음 분석 ---
        face_data = cam2.process_frame()
        if face_data is None:
            print("Camera2 frame grab failed; exiting.")
            enter_summary()
            continue
        if face_data["should_exit"]:
            print(f"No face detected for {cam2.NO_FACE_SECONDS} seconds. Exiting...")
            enter_summary()
            continue
        frame2 = cv2.resize(face_data["frame"], (ui.CAM_W, ui.CAM2_H))
        ui.draw_face_mesh(frame2, face_data["face_landmarks"])
        ui.draw_face_overlay(frame2, face_data)
        ui.draw_pane_label(frame2, "Face Cam")

        # --- [Step C] 점수 병합 ---
        raw_score = factor * face_data["score"]
        smoothed_score = raw_score if smoothed_score is None else ALPHA * raw_score + (1 - ALPHA) * smoothed_score

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_t, 1e-6))
        prev_t = now

        if state_active_subject_running():
            ui.sample_buffer.append(smoothed_score)
            if now - ui.last_flush_t >= GRAPH_SAMPLE_SEC and ui.sample_buffer:
                sample_avg = sum(ui.sample_buffer) / len(ui.sample_buffer)
                ui.graph_points.append(sample_avg)
                ui.sample_buffer.clear()
                ui.last_flush_t = now
                # 공부 중 집중도 데이터를 메모리에 기록 (10초 평균 단위, 재시작하면 사라짐)
                ui.record_subject_score(active_subject, sample_avg)

        # --- [Step D] 캔버스 및 UI 렌더링 ---
        cv2.putText(frame1, f"FPS:{fps:.0f}", (ui.CAM_W - 110, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2, cv2.LINE_AA)

        canvas = np.zeros((ui.CANVAS_H, ui.CANVAS_W, 3), dtype=np.uint8)
        canvas[0:ui.CAM1_H, 0:ui.CAM_W] = frame1
        canvas[ui.CAM1_H:ui.CAM_H, 0:ui.CAM_W] = frame2
        cv2.line(canvas, (0, ui.CAM1_H), (ui.CAM_W, ui.CAM1_H), (60, 60, 60), 2)

        if ui.state["screen"] == "active":
            ui.draw_active_screen(canvas, smoothed_score, factor)
        else:
            ui.draw_list_screen(canvas)

        cv2.imshow(ui.WINDOW_NAME, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            ui.handle_key(key)
        if ui.state["should_exit"] or cv2.getWindowProperty(ui.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            # 바로 끄지 않고 종합 평가 화면을 먼저 보여준다.
            enter_summary()
            continue

    cam1.cleanup()
    cam2.cleanup()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
