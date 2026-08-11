import time
import datetime
import cv2
import numpy as np
import mediapipe as mp

import kr_text

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
CAM_W, CAM_H = 800, 600
CAM1_H = CAM_H // 2   # 위쪽: cam1 (책상/물체 감지)
CAM2_H = CAM_H - CAM1_H  # 아래쪽: cam2 (얼굴/시선)
RIGHT_W = 480
CANVAS_H = CAM_H
CANVAS_W = CAM_W + RIGHT_W
WINDOW_NAME = "Focus Tracker"

# ---- list 화면 레이아웃 ----
ROW_START_Y = 230
ROW_H = 70
ROW_INNER_PAD = 8

# ---- active(공부중) 화면 레이아웃 ----
GRAPH_MARGIN = 20

SUBJECTS_INIT = [
    {"name": "Math",    "color": (100, 220, 100)},
    {"name": "Calc",    "color": (70, 90, 130)}
]

def make_subjects():
    return [{"name": s["name"], "color": s["color"], "elapsed": 0.0} for s in SUBJECTS_INIT]

subjects = make_subjects()
graph_points = []
sample_buffer = []
subject_totals = {}  # subject name -> [sum, count], 오늘 하루치 (재시작하면 초기화)
last_flush_t = time.time()

state = {
    "screen": "list",
    "active_idx": None,
    "resume_t": None,
    "should_exit": False,
    "today": datetime.date.today(),
    "edit_mode": False,
    "input_mode": None,
    "input_buffer": "",
    "rename_idx": None,
}

btn_rects = {
    "play": [], "delete": [], "name": [], "add": None,
    "edit_toggle": None, "exit": None, "pause": None,
}

# ============================================================
# 종합 평가(summary) 화면 데이터 — 집중 효율 5단계 + 오늘 전체 그래프
# main_ver5.py가 종료 직전에 graph_points/subject_totals(메모리)를 prepare_summary()로 넘겨준다.
# 재시작하면 초기화되는 값이라, 디스크에 남는 기록은 없다.
# ============================================================
FOCUS_LEVELS = [
    (20,  "매우 낮음", (60, 60, 220)),
    (40,  "낮음",      (0, 130, 230)),
    (60,  "보통",      (0, 210, 230)),
    (80,  "높음",      (0, 210, 120)),
    (101, "매우 높음", (0, 200, 0)),
]

def classify_focus_level(avg_score):
    """평균 점수(0~100)를 5단계 집중 효율 등급으로 변환. (label, color) 반환."""
    for threshold, label, color in FOCUS_LEVELS:
        if avg_score < threshold:
            return label, color
    return FOCUS_LEVELS[-1][1], FOCUS_LEVELS[-1][2]

summary_data = {
    "scores": [],             # 오늘 하루 전체 10초 평균 샘플 (시간순)
    "avg_score": 0.0,
    "level_label": "",
    "level_color": (200, 200, 200),
    "subject_breakdown": [],  # [(name, avg_score, sample_count), ...]
}

def prepare_summary(scores, subject_breakdown=None):
    """DB에서 읽어온 오늘의 점수들로 summary_data를 채운다."""
    global summary_data
    avg = sum(scores) / len(scores) if scores else 0.0
    label, color = classify_focus_level(avg)
    summary_data = {
        "scores": list(scores),
        "avg_score": avg,
        "level_label": label,
        "level_color": color,
        "subject_breakdown": list(subject_breakdown) if subject_breakdown else [],
    }

# ============================================================
# UI 유틸리티 함수들 (cam1_ver2.py)
# ============================================================
def point_in_rect(x, y, rect):
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]

def reset_daily_state():
    global subjects, graph_points, sample_buffer, subject_totals, last_flush_t
    subjects = make_subjects()
    graph_points, sample_buffer = [], []
    subject_totals = {}
    last_flush_t = time.time()
    state.update({"screen": "list", "active_idx": None, "resume_t": None, "should_exit": False, "edit_mode": False, "input_mode": None, "input_buffer": "", "rename_idx": None})

def record_subject_score(subject_name, score):
    """과목별 오늘 점수 합계/샘플 수를 메모리에 누적한다(디스크 저장 없음)."""
    totals = subject_totals.setdefault(subject_name, [0.0, 0])
    totals[0] += score
    totals[1] += 1

def subject_breakdown():
    """[(name, avg_score, sample_count), ...] 과목명 오름차순."""
    return sorted(
        ((name, total / count, count) for name, (total, count) in subject_totals.items()),
        key=lambda row: row[0],
    )

def start_subject(idx):
    state.update({"active_idx": idx, "resume_t": time.time(), "screen": "active"})
    sample_buffer.clear()

def stop_active():
    if state["active_idx"] is not None and state["resume_t"] is not None:
        subjects[state["active_idx"]]["elapsed"] += time.time() - state["resume_t"]
    state.update({"active_idx": None, "resume_t": None, "screen": "list"})
    # sample_buffer는 여기서 비우지 않는다. main(_web).py가 다음 루프 반복에서
    # "활성 -> 비활성" 전환을 감지해 10초 주기가 안 됐어도 남은 표본을 DB에 저장한 뒤
    # 비운다(flush_leftover_sample) — 여기서 먼저 지워버리면 그 구간 점수가 유실된다.

def subject_live_elapsed(i):
    base = subjects[i]["elapsed"]
    if state["active_idx"] == i and state["resume_t"]:
        base += time.time() - state["resume_t"]
    return base

def total_today_elapsed():
    total = sum(s["elapsed"] for s in subjects)
    if state["active_idx"] is not None and state["resume_t"] is not None:
        total += time.time() - state["resume_t"]
    return total

def mouse_callback(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN or state["input_mode"]: return
    if btn_rects["exit"] and point_in_rect(x, y, btn_rects["exit"]):
        state["should_exit"] = True; return
    if state["screen"] == "summary":
        return
    if state["screen"] == "active":
        if btn_rects["pause"] and point_in_rect(x, y, btn_rects["pause"]): stop_active()
        return

    if not state["edit_mode"]:
        for rect, idx in btn_rects["play"]:
            if point_in_rect(x, y, rect): start_subject(idx); return
    else:
        for rect, idx in btn_rects["delete"]:
            if point_in_rect(x, y, rect) and len(subjects) > 1 and state["active_idx"] != idx:
                subjects.pop(idx); return
        for rect, idx in btn_rects["name"]:
            if point_in_rect(x, y, rect):
                state.update({"input_mode": "rename", "rename_idx": idx, "input_buffer": subjects[idx]["name"]})
                return
        if btn_rects["add"] and point_in_rect(x, y, btn_rects["add"]):
            state.update({"input_mode": "add", "input_buffer": ""})
            return

    if btn_rects["edit_toggle"] and point_in_rect(x, y, btn_rects["edit_toggle"]):
        state["edit_mode"] = not state["edit_mode"]

def handle_key(key):
    if state["input_mode"] is not None:
        if key in (13, 10):
            text = state["input_buffer"].strip()
            if text:
                if state["input_mode"] == "add":
                    palette = [(100, 220, 100), (0, 215, 235), (60, 60, 230), (200, 160, 60), (180, 90, 200)]
                    subjects.append({"name": text, "color": palette[len(subjects) % len(palette)], "elapsed": 0.0})
                elif state["input_mode"] == "rename":
                    subjects[state["rename_idx"]]["name"] = text
            state.update({"input_mode": None, "input_buffer": "", "rename_idx": None})
        elif key == 27:
            state.update({"input_mode": None, "input_buffer": "", "rename_idx": None})
        elif key in (8, 127):
            state["input_buffer"] = state["input_buffer"][:-1]
        elif 32 <= key <= 126 and len(state["input_buffer"]) < 20:
            state["input_buffer"] += chr(key)
        return
    if state["screen"] == "summary" and key in (13, 10):
        state["should_exit"] = True
        return
    if key == ord('q'): state["should_exit"] = True

def format_time(sec):
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def draw_text_centered(img, text, cx, cy, scale, color, thickness):
    kr_text.put_text(img, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA, center=True)

def draw_exit_button(canvas):
    rect = (CANVAS_W - 60, 10, CANVAS_W - 10, 45)
    cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 160), -1)
    cv2.rectangle(canvas, (rect[0], rect[1]), (rect[2], rect[3]), (255, 255, 255), 1)
    draw_text_centered(canvas, "X", (rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2, 0.6, (255, 255, 255), 2)
    btn_rects["exit"] = rect

def draw_input_bar(canvas):
    x0, x1, y0, y1 = CAM_W + 20, CANVAS_W - 20, CANVAS_H - 60, CANVAS_H - 20
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (60, 60, 60), -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 255, 255), 1)
    text = ("추가: " if state["input_mode"] == "add" else "이름변경: ") + state["input_buffer"] + "|"
    kr_text.put_text(canvas, text, (x0 + 10, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    kr_text.put_text(canvas, "Enter: 확인   ESC: 취소", (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)

# ============================================================
# cam1 / cam2 데이터 오버레이 (main_ver5.py 가 비디오 프레임 위에 그릴 때 사용)
# ============================================================
CLASS_NAME_KO = {
    "cell phone": "휴대폰",
    "book": "책",
    "laptop": "노트북",
}

def draw_detection_boxes(frame, boxes, names, distraction_set, study_set, conf_th):
    for b in boxes:
        conf = float(b.conf[0])
        if conf <= conf_th:
            continue
        xyxy = b.xyxy[0].cpu().numpy().astype(int)
        cls_name = names[int(b.cls[0])]
        if cls_name in distraction_set:
            color = (0, 0, 255)
        elif cls_name in study_set:
            color = (0, 255, 0)
        else:
            color = (200, 200, 200)
        label_ko = CLASS_NAME_KO.get(cls_name, cls_name)
        cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
        kr_text.put_text(frame, f"{label_ko} {conf:.2f}", (xyxy[0], max(15, xyxy[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return frame

def draw_pane_label(frame, text):
    kr_text.put_text(frame, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return frame

def draw_face_mesh(frame, face_landmarks):
    if face_landmarks is None:
        return frame
    mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                               mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=1, circle_radius=1),
                               mp_drawing.DrawingSpec(color=(0, 150, 255), thickness=1))
    return frame

# cam2_ver5.py가 내부적으로 쓰는 영문 상태값(status/gaze_dir)은 로직에서 그대로
# 비교에 쓰이므로 바꾸지 않고, 화면 표시용 한글 라벨만 별도로 매핑한다.
STATUS_KO = {
    "NO FACE": "얼굴 미인식",
    "OCCLUDED": "가려짐",
    "NOISY": "소음 감지",
    "BLINK": "깜빡임",
    "DISTRACTED": "집중 안 함",
    "CONCENTRATED": "집중 중",
}
GAZE_KO = {
    "CENTER": "정면",
    "LEFT": "왼쪽",
    "RIGHT": "오른쪽",
    "UP": "위",
    "DOWN": "아래",
    "UNKNOWN": "알 수 없음",
    "NO_FACE": "얼굴 없음",
}

def draw_face_overlay(frame, face_data):
    if face_data is None:
        return frame
    status_ko = STATUS_KO.get(face_data["status"], face_data["status"])
    gaze_ko = GAZE_KO.get(face_data["gaze_dir"], face_data["gaze_dir"])
    kr_text.put_text(frame, f"상태: {status_ko}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)
    kr_text.put_text(frame, f"시선: {gaze_ko}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 2, cv2.LINE_AA)
    if face_data["blink_event"]:
        kr_text.put_text(frame, "깜빡임", (200, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    if face_data["noise_flag"]:
        kr_text.put_text(frame, "소음", (200, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    if face_data["eyes_closed_long"]:
        kr_text.put_text(frame, "눈 감음 3초 이상", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    if face_data["status"] == "NO FACE" and face_data["no_face_elapsed"] > 0:
        kr_text.put_text(frame, f"얼굴 미인식: {int(face_data['no_face_elapsed'])}초", (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    return frame

def draw_list_screen(canvas):
    x0 = CAM_W
    btn_rects.update({"play": [], "delete": [], "name": [], "add": None})
    cv2.rectangle(canvas, (x0, 0), (CANVAS_W, CANVAS_H), (18, 18, 18), -1)
    kr_text.put_text(canvas, state["today"].strftime("%Y-%m-%d"), (x0 + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 170), 1, cv2.LINE_AA)
    draw_text_centered(canvas, format_time(total_today_elapsed()), x0 + RIGHT_W // 2, 90, 1.3, (255, 255, 255), 3)
    kr_text.put_text(canvas, "오늘 총 시간", (x0 + RIGHT_W // 2 - 55, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.line(canvas, (x0 + 20, 165), (CANVAS_W - 20, 165), (60, 60, 60), 1)

    for i, s in enumerate(subjects):
        y0 = ROW_START_Y + i * ROW_H
        y1 = y0 + ROW_H - ROW_INNER_PAD
        cy = (y0 + y1) // 2
        play_rect = (x0 + 30, y0, x0 + 82, y1)
        cx = (play_rect[0] + play_rect[2]) // 2
        cv2.circle(canvas, (cx, cy), 26, s["color"], -1)
        if not state["edit_mode"]:
            pts = np.array([[cx - 7, cy - 12], [cx - 7, cy + 12], [cx + 12, cy]], np.int32)
            cv2.fillPoly(canvas, [pts], (20, 20, 20))
            btn_rects["play"].append((play_rect, i))

        name_rect = (x0 + 100, y0, x0 + 300, y1)
        kr_text.put_text(canvas, s["name"], (name_rect[0], cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (235, 235, 235), 2, cv2.LINE_AA)
        if state["edit_mode"]:
            btn_rects["name"].append((name_rect, i))
            cv2.rectangle(canvas, (name_rect[0] - 5, y0 + 5), (name_rect[2], y1 - 5), (80, 80, 80), 1)

        time_str = format_time(subject_live_elapsed(i))
        (tw, _), _ = cv2.getTextSize(time_str, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        kr_text.put_text(canvas, time_str, (CANVAS_W - 60 - tw, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0) if state["active_idx"] == i else (200, 200, 200), 2, cv2.LINE_AA)

        if state["edit_mode"]:
            del_rect = (CANVAS_W - 40, cy - 15, CANVAS_W - 10, cy + 15)
            cv2.rectangle(canvas, (del_rect[0], del_rect[1]), (del_rect[2], del_rect[3]), (0, 0, 150), -1)
            draw_text_centered(canvas, "x", (del_rect[0] + del_rect[2]) // 2, cy, 0.6, (255, 255, 255), 2)
            btn_rects["delete"].append((del_rect, i))
        cv2.line(canvas, (x0 + 20, y1 + 5), (CANVAS_W - 20, y1 + 5), (35, 35, 35), 1)

    next_y = ROW_START_Y + len(subjects) * ROW_H + 15
    if state["edit_mode"]:
        add_rect = (x0 + 30, next_y, CANVAS_W - 30, next_y + 40)
        cv2.rectangle(canvas, (add_rect[0], add_rect[1]), (add_rect[2], add_rect[3]), (40, 70, 40), -1)
        cv2.rectangle(canvas, (add_rect[0], add_rect[1]), (add_rect[2], add_rect[3]), (255, 255, 255), 1)
        draw_text_centered(canvas, "+ 과목 추가", (add_rect[0] + add_rect[2]) // 2, (add_rect[1] + add_rect[3]) // 2, 0.6, (255, 255, 255), 2)
        btn_rects["add"] = add_rect
        next_y += 55

    edit_rect = (x0 + 30, next_y, CANVAS_W - 30, next_y + 40)
    cv2.rectangle(canvas, (edit_rect[0], edit_rect[1]), (edit_rect[2], edit_rect[3]), (60, 60, 90) if state["edit_mode"] else (55, 55, 55), -1)
    cv2.rectangle(canvas, (edit_rect[0], edit_rect[1]), (edit_rect[2], edit_rect[3]), (255, 255, 255), 1)
    draw_text_centered(canvas, "완료" if state["edit_mode"] else "과목 편집", (edit_rect[0] + edit_rect[2]) // 2, (edit_rect[1] + edit_rect[3]) // 2, 0.6, (255, 255, 255), 2)
    btn_rects["edit_toggle"] = edit_rect
    if state["input_mode"] is not None: draw_input_bar(canvas)
    draw_exit_button(canvas)

def draw_active_screen(canvas, score, factor):
    x0 = CAM_W
    btn_rects["pause"] = None
    cv2.rectangle(canvas, (x0, 0), (CANVAS_W, CANVAS_H), (18, 18, 18), -1)
    kr_text.put_text(canvas, "오늘 총 시간", (x0 + 25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    timer_str = format_time(total_today_elapsed())
    (tw, th), _ = cv2.getTextSize(timer_str, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)
    kr_text.put_text(canvas, timer_str, (x0 + 25, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)

    pause_cx, pause_cy = x0 + 25 + tw + 45, 100 - th // 2
    cv2.circle(canvas, (pause_cx, pause_cy), 30, (230, 230, 230), -1)
    cv2.rectangle(canvas, (pause_cx - 11, pause_cy - 11), (pause_cx - 5, pause_cy + 11), (20, 20, 20), -1)
    cv2.rectangle(canvas, (pause_cx + 5, pause_cy - 11), (pause_cx + 11, pause_cy + 11), (20, 20, 20), -1)
    btn_rects["pause"] = (pause_cx - 30, pause_cy - 30, pause_cx + 30, pause_cy + 30)

    if state["active_idx"] is not None:
        s = subjects[state["active_idx"]]
        cv2.circle(canvas, (x0 + 35, 150), 10, s["color"], -1)
        kr_text.put_text(canvas, f'{s["name"]}  {format_time(subject_live_elapsed(state["active_idx"]))}', (x0 + 55, 156), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

    # 한글 라벨은 Hershey 폰트보다 줄 높이가 커서(kr_text.py 참고) 세로 간격을 더 넉넉히 둔다.
    status, scolor = ("집중 안 함", (0, 0, 255)) if factor <= 0.5 else ("공부 중", (0, 255, 0)) if factor >= 1.0 else ("보통", (0, 200, 255))
    cv2.circle(canvas, (x0 + 35, 192), 7, scolor, -1)
    kr_text.put_text(canvas, status, (x0 + 50, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.55, scolor, 1, cv2.LINE_AA)

    score = max(0.0, min(100.0, score))
    score_color = (0, 200, 0) if score >= 60 else (0, 140, 255) if score >= 40 else (0, 80, 200)
    kr_text.put_text(canvas, f"현재 점수: {score:.0f}%", (x0 + 25, 228), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, score_color, 2, cv2.LINE_AA)

    gx0, gy0, gx1, gy1 = x0 + GRAPH_MARGIN, 266, CANVAS_W - GRAPH_MARGIN, CANVAS_H - GRAPH_MARGIN
    cv2.rectangle(canvas, (gx0, gy0), (gx1, gy1), (25, 25, 25), -1)
    cv2.rectangle(canvas, (gx0, gy0), (gx1, gy1), (90, 90, 90), 1)
    kr_text.put_text(canvas, "집중도 점수 (10초 평균)", (gx0, gy0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    for val in (0, 50, 100):
        y = int(gy1 - (val / 100.0) * (gy1 - gy0))
        cv2.line(canvas, (gx0, y), (gx1, y), (55, 55, 55), 1)
        kr_text.put_text(canvas, str(val), (gx1 + 4, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

    if len(graph_points) >= 2:
        pts = np.array([[(gx0 + int(i / (len(graph_points) - 1) * (gx1 - gx0))), int(gy1 - (max(0.0, min(100.0, val)) / 100.0) * (gy1 - gy0))] for i, val in enumerate(graph_points)], dtype=np.int32)
        cv2.polylines(canvas, [pts], False, (0, 255, 0), 2, cv2.LINE_AA)
    elif len(graph_points) == 1:
        cv2.circle(canvas, (gx0, int(gy1 - (max(0.0, min(100.0, graph_points[0])) / 100.0) * (gy1 - gy0))), 3, (0, 255, 0), -1)
    draw_exit_button(canvas)

# ============================================================
# 종합 평가(summary) 화면 — 카메라 종료 후 전체 캔버스를 사용해 그린다.
# ============================================================
def draw_summary_screen(canvas):
    canvas[:] = (18, 18, 18)
    kr_text.put_text(canvas, "오늘의 집중도 종합 평가", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    kr_text.put_text(canvas, state["today"].strftime("%Y-%m-%d"), (40, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 170), 1, cv2.LINE_AA)

    avg = summary_data["avg_score"]
    label = summary_data["level_label"]
    color = summary_data["level_color"]
    draw_text_centered(canvas, f"{avg:.0f}%", CANVAS_W // 2, 145, 1.4, color, 3)
    draw_text_centered(canvas, f"집중 등급: {label}", CANVAS_W // 2, 190, 0.75, color, 2)

    gx0, gy0, gx1, gy1 = 60, 230, CANVAS_W - 60, CANVAS_H - 150
    cv2.rectangle(canvas, (gx0, gy0), (gx1, gy1), (25, 25, 25), -1)
    cv2.rectangle(canvas, (gx0, gy0), (gx1, gy1), (90, 90, 90), 1)
    kr_text.put_text(canvas, "시간별 점수 (10초 평균)", (gx0, gy0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    for val in (0, 50, 100):
        y = int(gy1 - (val / 100.0) * (gy1 - gy0))
        cv2.line(canvas, (gx0, y), (gx1, y), (55, 55, 55), 1)
        kr_text.put_text(canvas, str(val), (gx1 + 4, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

    scores = summary_data["scores"]
    if len(scores) >= 2:
        pts = np.array([[gx0 + int(i / (len(scores) - 1) * (gx1 - gx0)),
                          int(gy1 - (max(0.0, min(100.0, v)) / 100.0) * (gy1 - gy0))]
                         for i, v in enumerate(scores)], dtype=np.int32)
        cv2.polylines(canvas, [pts], False, color, 2, cv2.LINE_AA)
    elif len(scores) == 1:
        cv2.circle(canvas, (gx0, int(gy1 - (max(0.0, min(100.0, scores[0])) / 100.0) * (gy1 - gy0))), 4, color, -1)
    else:
        draw_text_centered(canvas, "오늘 기록된 데이터가 없습니다", (gx0 + gx1) // 2, (gy0 + gy1) // 2, 0.6, (150, 150, 150), 1)

    y = gy1 + 32
    kr_text.put_text(canvas, "과목별:", (gx0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    for name, subj_avg, cnt in summary_data["subject_breakdown"]:
        y += 24
        kr_text.put_text(canvas, f"- {name}: {subj_avg:.0f}%  (샘플 {cnt}개)", (gx0 + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    draw_text_centered(canvas, "닫으려면 Enter 또는 X 버튼을 누르세요", CANVAS_W // 2, CANVAS_H - 25, 0.5, (150, 150, 150), 1)
    draw_exit_button(canvas)
