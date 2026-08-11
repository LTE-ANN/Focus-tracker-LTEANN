"""FastAPI web entry point for Focus Tracker (local-only).

Reuses cam1_ver5 / cam2_ver5 / ui_ver5 / tutorial_ui unchanged.
Everything that used to be a local cv2 window (tutorial, gaze calibration,
the tracker itself) now runs in a background thread and is pushed to the
browser as an MJPEG stream; clicks/keys from the browser are forwarded to
the same handlers (ui.mouse_callback/handle_key, TutorialUI.handle_click/key)
that already did the button-hit-testing, so none of that logic changes.

One session = tutorial -> calibration -> tracker -> summary, driven by a
single "시작" click. Assumes the server and the webcams are on the same
local machine (you open http://127.0.0.1:8000 on the same PC that's
running this).

Run with:  python main_web.py
"""

import sys
import threading
import time
import datetime
import traceback

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

from cam1_ver5 import Camera1
from cam2_ver5 import Camera2
import ui_ver5 as ui

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
GRAPH_SAMPLE_SEC = 10.0  # 그래프에 점 하나 찍는 주기(초) - 10초 평균
WAIT_MESSAGE = "대기 중.. 현재 자세와 시선을 유지하세요"

app = FastAPI()

# ---------------- 프레임 버퍼 (백그라운드 세션 -> /stream) ----------------
_frame_lock = threading.Lock()
_latest_jpeg = None

# ---------------- 세션(튜토리얼->캘리브레이션->트래커) 생명주기 ----------------
_run_lock = threading.Lock()
_session_thread = None
_running = False
_stop_requested = False
_phase = "idle"          # idle | tutorial | calibrating | running
_tutorial_obj = None      # phase == "tutorial"일 때 클릭/키를 넘겨줄 대상


def _set_frame(canvas):
    global _latest_jpeg
    ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if ok:
        with _frame_lock:
            _latest_jpeg = buf.tobytes()


def _get_frame():
    with _frame_lock:
        return _latest_jpeg


def _blank_canvas():
    return np.zeros((ui.CANVAS_H, ui.CANVAS_W, 3), dtype=np.uint8)


# ============================================================
# 1) 튜토리얼 (웹) — tutorial_ui.TutorialUI는 이미 cv2 창 없이
#    render_page()/handle_click()/handle_key()만으로 동작 가능하게 만들어져
#    있어서 그대로 재사용한다.
# ============================================================
def _run_tutorial_phase():
    """Returns True to continue to calibration, False if the user cancelled."""
    global _phase, _tutorial_obj

    if tutorial_ui is None:
        print("[Web] tutorial_ui unavailable; skipping tutorial.", file=sys.stderr, flush=True)
        return True

    _phase = "tutorial"
    tut = tutorial_ui.TutorialUI(width=ui.CANVAS_W, height=ui.CANVAS_H)
    _tutorial_obj = tut
    try:
        while not tut.done and not _stop_requested:
            _set_frame(tut.render_page())
            time.sleep(0.03)
    finally:
        _tutorial_obj = None

    if _stop_requested:
        return False
    return not tut.cancelled


# ============================================================
# 2) 캘리브레이션 (웹) — cam2.calibrate_gaze()는 자체 cv2 창을 띄우므로
#    웹에서는 쓰지 않고, 같은 계산을 여기서 직접 반복하면서 매 프레임을
#    스트림 버퍼로 내보낸다. "대기 중.. 현재 자세와 시선을 유지하세요" 오버레이 표시.
# ============================================================
def _run_calibration_phase(cam2):
    global _phase
    _phase = "calibrating"

    canvas = _blank_canvas()
    canvas[:] = (18, 18, 18)
    ui.draw_text_centered(canvas, WAIT_MESSAGE, ui.CANVAS_W // 2, ui.CANVAS_H // 2 - 20, 0.9, (0, 230, 255), 2)
    ui.draw_text_centered(canvas, "마이크 소음 기준을 측정 중입니다...", ui.CANVAS_W // 2, ui.CANVAS_H // 2 + 20, 0.6, (200, 200, 200), 1)
    _set_frame(canvas)

    if cam2.start_audio():
        cam2.calibrate_audio_baseline()
    else:
        print("Warning: audio disabled; noise detection will be off.")

    calib_x, calib_y = [], []
    start = time.time()
    while time.time() - start < cam2.CALIB_SECONDS and not _stop_requested:
        ret, frame = cam2.cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mesh = cam2.face_mesh.process(rgb)
        det = cam2.face_detection.process(rgb)
        if mesh.multi_face_landmarks and det.detections:
            avgx, avgy = cam2.get_iris_avg(mesh.multi_face_landmarks[0].landmark)
            if avgx is not None:
                calib_x.append(avgx)
                calib_y.append(avgy)

        canvas = _blank_canvas()
        canvas[:] = (18, 18, 18)
        disp = cv2.resize(frame, (ui.CAM_W, ui.CAM_H))
        cx0 = (ui.CANVAS_W - ui.CAM_W) // 2
        canvas[0:ui.CAM_H, cx0:cx0 + ui.CAM_W] = disp
        ui.draw_text_centered(canvas, WAIT_MESSAGE, ui.CANVAS_W // 2, ui.CANVAS_H - 40, 0.85, (0, 230, 255), 2)
        _set_frame(canvas)

    if _stop_requested:
        return False

    if not calib_x:
        print("Calibration failed: no face/iris detected. Retry with better lighting/position.")
        canvas = _blank_canvas()
        canvas[:] = (18, 18, 18)
        ui.draw_text_centered(canvas, "캘리브레이션 실패: 얼굴을 인식하지 못했습니다.", ui.CANVAS_W // 2, ui.CANVAS_H // 2, 0.7, (0, 0, 255), 2)
        _set_frame(canvas)
        time.sleep(2.0)
        return False

    cam2.baseline_x = float(np.mean(calib_x))
    cam2.baseline_y = float(np.mean(calib_y))
    print(f"Calibration complete. Baseline iris center = ({cam2.baseline_x:.3f}, {cam2.baseline_y:.3f})")
    return True


# ============================================================
# 3) 트래커 루프 — main_ver5.py와 동일한 로직. cv2 창 대신 프레임을 버퍼에 저장.
# ============================================================
def _run_tracker_phase(cam1, cam2):
    global _phase
    _phase = "running"

    active_subject = None  # 현재 활성 과목 이름 (메모리에만 유지, 활성 과목 없으면 None)
    ui.reset_daily_state()

    smoothed_score = None
    prev_t = time.time()
    fps = 0.0

    def state_active_subject_running():
        return ui.state["screen"] == "active" and ui.state["active_idx"] is not None

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
        if state_active_subject_running():
            flush_leftover_sample()
            ui.stop_active()
        ui.prepare_summary(ui.graph_points, ui.subject_breakdown())
        ui.state["screen"] = "summary"
        ui.state["should_exit"] = False

    print("통합 트래커(웹) 실행 중!")

    # 이 단계부터는 _stop_requested(즉시 중단용, 튜토리얼/캘리브레이션 취소에 씀)가 아니라
    # ui.state["should_exit"] + screen 전이로만 종료를 제어한다. "종료" 버튼을 눌러도
    # _stop_requested가 True가 되긴 하지만 여기서는 보지 않으므로, 요약 화면을 보여준 뒤
    # 다시 한번 확인(X/Enter)해야 진짜로 끝나는 기존 데스크톱 버전과 동일한 흐름이 된다.
    while True:
        today = datetime.date.today()
        if today != ui.state["today"]:
            ui.state["today"] = today
            ui.reset_daily_state()

        # --- 종합 평가 화면: 카메라를 더 이상 읽지 않고 결과만 스트리밍 ---
        if ui.state["screen"] == "summary":
            canvas = _blank_canvas()
            ui.draw_summary_screen(canvas)
            _set_frame(canvas)
            if ui.state["should_exit"]:
                break
            time.sleep(0.05)
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
        ui.draw_pane_label(frame1, "책상 캠")

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
        ui.draw_pane_label(frame2, "얼굴 캠")

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

        # --- [Step D] 캔버스 렌더링 -> JPEG 스트림 버퍼 ---
        cv2.putText(frame1, f"FPS:{fps:.0f}", (ui.CAM_W - 110, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2, cv2.LINE_AA)

        canvas = _blank_canvas()
        canvas[0:ui.CAM1_H, 0:ui.CAM_W] = frame1
        canvas[ui.CAM1_H:ui.CAM_H, 0:ui.CAM_W] = frame2
        cv2.line(canvas, (0, ui.CAM1_H), (ui.CAM_W, ui.CAM1_H), (60, 60, 60), 2)

        if ui.state["screen"] == "active":
            ui.draw_active_screen(canvas, smoothed_score, factor)
        else:
            ui.draw_list_screen(canvas)

        _set_frame(canvas)

        if ui.state["should_exit"]:
            enter_summary()
            continue


def _session_worker():
    global _running, _stop_requested, _phase

    cam1 = None
    cam2 = None
    try:
        if not _run_tutorial_phase():
            print("[Web] Tutorial cancelled; session aborted before opening cameras.")
            return

        cam1 = Camera1()  # 책상(핸드폰) 카메라 - YOLO 물체 감지
        cam2 = Camera2()  # 얼굴(노트북) 카메라 - 시선/눈 감김/소음 분석

        if not _run_calibration_phase(cam2):
            print("[Web] Calibration cancelled/failed; session aborted.")
            return

        _run_tracker_phase(cam1, cam2)
    except Exception:
        traceback.print_exc()
    finally:
        if cam1 is not None:
            cam1.cleanup()
        if cam2 is not None:
            cam2.cleanup()
        _running = False
        _stop_requested = False
        _phase = "idle"
        print("[Web] Session ended; server still running.")


# ============================================================
# HTTP / 스트리밍 엔드포인트
# ============================================================
@app.post("/session/start")
def start_session():
    global _session_thread, _running, _stop_requested
    with _run_lock:
        if _running:
            return JSONResponse({"status": "already_running"})
        _stop_requested = False
        _running = True
        _session_thread = threading.Thread(target=_session_worker, daemon=True)
        _session_thread.start()
    return JSONResponse({"status": "started"})


@app.post("/session/stop")
def stop_session():
    global _stop_requested
    _stop_requested = True
    ui.state["should_exit"] = True
    return JSONResponse({"status": "stopping"})


@app.get("/status")
def status():
    return JSONResponse({
        "running": _running,
        "phase": _phase,
        "screen": ui.state["screen"] if _phase == "running" else None,
    })


@app.post("/input/click")
def input_click(x: int, y: int):
    if _phase == "tutorial" and _tutorial_obj is not None:
        _tutorial_obj.handle_click(x, y)
    elif _phase == "running":
        ui.mouse_callback(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
    return JSONResponse({"ok": True})


@app.post("/input/key")
def input_key(code: int):
    if _phase == "tutorial" and _tutorial_obj is not None:
        _tutorial_obj.handle_key(code)
    elif _phase == "running":
        ui.handle_key(code)
    return JSONResponse({"ok": True})


def _mjpeg_generator():
    boundary = b"--frame"
    while True:
        frame = _get_frame()
        if frame is not None:
            yield (boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.03)


@app.get("/stream")
def stream():
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


INDEX_HTML = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>집중력 트래커 (웹)</title>
<style>
  html, body { height: 100%; margin: 0; }
  body {
    background: #ffffff;
    color: #111;
    font-family: sans-serif;
    display: flex;
    flex-direction: column;
  }
  #topbar {
    flex: 0 0 auto;
    padding: 10px;
    text-align: center;
    border-bottom: 1px solid #ddd;
    background: #ffffff;
  }
  #streamWrap {
    flex: 1 1 auto;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    background: #ffffff;
    min-height: 0;
  }
  #stream {
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
    cursor: pointer;
  }
  button {
    padding: 8px 16px;
    font-size: 14px;
    margin: 0 6px;
    cursor: pointer;
  }
  #statusLabel { margin-left: 10px; color: #555; }
</style>
</head>
<body>
  <div id="topbar">
    <button id="startBtn">시작</button>
    <button id="stopBtn">종료</button>
    <span id="statusLabel"></span>
  </div>
  <div id="streamWrap">
    <img id="stream" src="/stream">
  </div>
<script>
const img = document.getElementById('stream');
const statusLabel = document.getElementById('statusLabel');

const PHASE_LABEL = {
  idle: '정지됨',
  tutorial: '튜토리얼',
  calibrating: '캘리브레이션 중 (자세를 꼭 유지하세요!)',
  running: 'Focus Tracker 실행 중!',
};

img.addEventListener('click', (e) => {
  const iw = img.naturalWidth, ih = img.naturalHeight;
  if (!iw || !ih) return;
  const rect = img.getBoundingClientRect();

  // #stream is scaled up to fill its box with object-fit:contain, so unless
  // the box's aspect ratio exactly matches the stream's, the actual pixels
  // are letterboxed (empty bars on the sides or top/bottom). Map the click
  // through the letterboxed rect, not the raw element box.
  const boxRatio = rect.width / rect.height;
  const imgRatio = iw / ih;
  let renderW, renderH, offsetX, offsetY;
  if (boxRatio > imgRatio) {
    renderH = rect.height;
    renderW = renderH * imgRatio;
    offsetX = (rect.width - renderW) / 2;
    offsetY = 0;
  } else {
    renderW = rect.width;
    renderH = renderW / imgRatio;
    offsetX = 0;
    offsetY = (rect.height - renderH) / 2;
  }

  const px = e.clientX - rect.left - offsetX;
  const py = e.clientY - rect.top - offsetY;
  if (px < 0 || py < 0 || px > renderW || py > renderH) return; // 여백(레터박스) 클릭은 무시

  const x = Math.round(px * (iw / renderW));
  const y = Math.round(py * (ih / renderH));
  fetch(`/input/click?x=${x}&y=${y}`, { method: 'POST' });
});

document.addEventListener('keydown', (e) => {
  let code = null;
  if (e.key === 'Enter') code = 13;
  else if (e.key === 'Escape') code = 27;
  else if (e.key === 'Backspace') code = 8;
  else if (e.key === 'ArrowLeft') code = 81;
  else if (e.key === 'ArrowRight') code = 83;
  else if (e.key.length === 1) code = e.key.charCodeAt(0);
  if (code === null) return;
  fetch(`/input/key?code=${code}`, { method: 'POST' });
  e.preventDefault();
});

document.getElementById('startBtn').addEventListener('click', () => {
  fetch('/session/start', { method: 'POST' });
});
document.getElementById('stopBtn').addEventListener('click', () => {
  fetch('/session/stop', { method: 'POST' });
});

setInterval(() => {
  fetch('/status').then(r => r.json()).then(d => {
    let label = PHASE_LABEL[d.phase] || d.phase;
    if (d.phase === 'running' && d.screen) label += ` - ${d.screen}`;
    statusLabel.textContent = label;
  });
}, 1000);
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(INDEX_HTML)


if __name__ == "__main__":
    import uvicorn

    print("브라우저에서 http://127.0.0.1:8000 을 열고 '시작' 버튼을 누르세요.")
    uvicorn.run(app, host="127.0.0.1", port=8000)
