import cv2
from collections import deque
from ultralytics import YOLO


class Camera1():
    def __init__(self,
                 cam_index=3,
                 model_path="yolo11n.pt",
                 DISTRACTION=None,
                 STUDY_ITEMS=None,
                 CONF_TH=0.5,
                 INFER_EVERY=3,
                 HISTORY_LEN=15,
                 PRESENT_TH=5):
        self.model = YOLO(model_path)

        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("웹캠을 열 수 없음. 카메라 인덱스나 권한 확인해봐.")

        self.DISTRACTION = DISTRACTION if DISTRACTION is not None else {"cell phone"}
        self.STUDY_ITEMS = STUDY_ITEMS if STUDY_ITEMS is not None else {"book", "laptop"}
        self.CONF_TH = CONF_TH
        self.INFER_EVERY = INFER_EVERY
        self.HISTORY_LEN = HISTORY_LEN
        self.PRESENT_TH = PRESENT_TH

        self.distraction_hist = deque(maxlen=HISTORY_LEN)
        self.study_hist = deque(maxlen=HISTORY_LEN)
        self.frame_count = 0
        self.factor = 0.8
        self.latest_boxes = []

    def process_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            return None, None, None

        # 몇 프레임에 한 번만 추론하고, 라벨로 factor 결정
        if self.frame_count % self.INFER_EVERY == 0:
            results = self.model(frame, verbose=False)
            labels = {self.model.names[int(b.cls[0])]
                      for b in results[0].boxes
                      if float(b.conf[0]) > self.CONF_TH}

            # 최근 프레임 누적으로 깜빡임 없이 "있다/없다" 판정
            self.distraction_hist.append(bool(self.DISTRACTION & labels))
            self.study_hist.append(bool(self.STUDY_ITEMS & labels))

            if sum(self.distraction_hist) >= self.PRESENT_TH:
                self.factor = 0.4
            elif sum(self.study_hist) >= self.PRESENT_TH:
                self.factor = 1.0
            else:
                self.factor = 0.8

            self.latest_boxes = results[0].boxes

        self.frame_count += 1
        return frame, self.factor, self.latest_boxes

    def cleanup(self):
        self.cap.release()
