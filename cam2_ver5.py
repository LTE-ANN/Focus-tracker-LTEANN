# cam2_ver5.py
import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import time
import threading
from collections import deque

import kr_text


class Camera2():
    def __init__(self,
                 CALIB_SECONDS=3.0,
                 EAR_BLINK_THRESHOLD=0.1,
                 BLINK_CONSEC_FRAMES=2,
                 FACE_DET_CONF=0.45,
                 EYE_VARIANCE_THRESHOLD=200.0,   # occlusion check: variance too low => likely covered
                 EYE_MEAN_DARK=45.0,             # occlusion check: mean too dark => covered
                 GAZE_X_DELTA=0.09,              # threshold around calibrated center for LEFT/RIGHT
                 GAZE_Y_DELTA=0.09,              # threshold around calibrated center for UP/DOWN
                 SCORE_SMOOTH=6,
                 NOISE_SENSITIVITY=2.0,          # how much above baseline RMS counts as noise
                 AUDIO_CALIB_SECONDS=1.0,
                 AUDIO_SR=22050,
                 AUDIO_BLOCKSIZE=1024,
                 EYES_CLOSED_SECONDS=3.0,        # <-- display eyes closed if closed for this long
                 NO_FACE_SECONDS=10.0,           # <-- exit if no face detected this many seconds
                 cam_index=0):
        # ---------------- CONFIG ----------------
        self.CALIB_SECONDS = CALIB_SECONDS
        self.EAR_BLINK_THRESHOLD = EAR_BLINK_THRESHOLD
        self.BLINK_CONSEC_FRAMES = BLINK_CONSEC_FRAMES
        self.FACE_DET_CONF = FACE_DET_CONF
        self.EYE_VARIANCE_THRESHOLD = EYE_VARIANCE_THRESHOLD
        self.EYE_MEAN_DARK = EYE_MEAN_DARK
        self.GAZE_X_DELTA = GAZE_X_DELTA
        self.GAZE_Y_DELTA = GAZE_Y_DELTA
        self.SCORE_SMOOTH = SCORE_SMOOTH
        self.NOISE_SENSITIVITY = NOISE_SENSITIVITY
        self.AUDIO_CALIB_SECONDS = AUDIO_CALIB_SECONDS
        self.AUDIO_SR = AUDIO_SR
        self.AUDIO_BLOCKSIZE = AUDIO_BLOCKSIZE
        self.EYES_CLOSED_SECONDS = EYES_CLOSED_SECONDS
        self.NO_FACE_SECONDS = NO_FACE_SECONDS
        # ----------------------------------------

        # MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detection = mp.solutions.face_detection

        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.4)

        # landmark indices
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        self.LEFT_IRIS = 468
        self.RIGHT_IRIS = 473

        # audio globals (thread-safe)
        self._audio_rms = 0.0
        self._audio_lock = threading.Lock()
        self._audio_stream = None
        self._audio_baseline = 1e-6

        # ---------- Camera ----------
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("ERROR: Cannot open camera. Close other apps or check device.")
            raise SystemExit

        # gaze calibration baseline (filled in by calibrate_gaze)
        self.baseline_x = None
        self.baseline_y = None

        # ---------- Main variables ----------
        self.score_buf = deque(maxlen=self.SCORE_SMOOTH)
        self.frame_no = 0
        self.blink_frames = 0
        self.last_blink_time = 0.0
        self.BLINK_MIN_SEP = 0.35  # seconds between blink events

        # --- Timers we add (initialized here so they're in scope) ---
        self.eyes_closed_start = None
        self.no_face_start = None

    def audio_callback(self, indata, frames, time_info, status):
        mono = np.mean(indata, axis=1) if indata.ndim > 1 else indata[:, 0]
        rms = float(np.sqrt(np.mean(np.square(mono))))
        with self._audio_lock:
            self._audio_rms = rms

    def start_audio(self):
        try:
            self._audio_stream = sd.InputStream(callback=self.audio_callback,
                                                 blocksize=self.AUDIO_BLOCKSIZE,
                                                 samplerate=self.AUDIO_SR,
                                                 channels=1)
            self._audio_stream.start()
            return True
        except Exception as e:
            print("Audio stream start failed:", e)
            return False

    def calibrate_audio_baseline(self):
        try:
            print(f"Calibrating microphone for {self.AUDIO_CALIB_SECONDS:.1f}s — please be quiet...")
            rec = sd.rec(int(self.AUDIO_CALIB_SECONDS * self.AUDIO_SR), samplerate=self.AUDIO_SR, channels=1, dtype='float64')
            sd.wait()
            mono = rec[:, 0]
            self._audio_baseline = max(1e-6, float(np.sqrt(np.mean(np.square(mono)))))
            print(f"Audio baseline RMS = {self._audio_baseline:.6f}")
        except Exception as e:
            print("Audio calibration failed:", e)
            self._audio_baseline = 1e-6

    # ---------- Helpers ----------
    def eye_aspect_ratio(self, landmarks, eye_indices, w, h):
        try:
            pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
            A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
            B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
            C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
            if C <= 1e-6:
                return 0.0
            return (A + B) / (2.0 * C)
        except Exception:
            return 0.0

    def eye_region_stats(self, gray, landmarks, eye_indices, w, h, pad=6):
        xs = [int(landmarks[i].x * w) for i in eye_indices]
        ys = [int(landmarks[i].y * h) for i in eye_indices]
        x1 = max(min(xs) - pad, 0); x2 = min(max(xs) + pad, w-1)
        y1 = max(min(ys) - pad, 0); y2 = min(max(ys) + pad, h-1)
        if x2 <= x1 or y2 <= y1:
            return None
        region = gray[y1:y2, x1:x2]
        if region.size == 0:
            return None
        return float(np.mean(region)), float(np.var(region))

    def get_iris_avg(self, landmarks):
        try:
            return (landmarks[self.LEFT_IRIS].x + landmarks[self.RIGHT_IRIS].x) / 2.0, \
                   (landmarks[self.LEFT_IRIS].y + landmarks[self.RIGHT_IRIS].y) / 2.0
        except Exception:
            return None, None

    def compute_concentration(self, gaze_ok, head_ok, blink_recent, occluded, noise_flag):
        # weights (tunable): gaze 40%, head 30%, blink penalty 20%, noise 10%
        gaze_score = 1.0 if gaze_ok else 0.0
        head_score = 1.0 if head_ok else 0.0
        blink_pen = 0.0 if not blink_recent else 0.5
        noise_pen = 1.0 if noise_flag else 0.0
        base = 0.4*gaze_score + 0.3*head_score + 0.2*(1.0 - blink_pen) + 0.1*(1.0 - noise_pen)
        if occluded:
            base *= 0.2
        return int(np.clip(base * 100.0, 0, 100))

    # ---------- Camera and calibration ----------
    def calibrate_gaze(self):
        cv2.namedWindow("Concentration Tracker", cv2.WINDOW_NORMAL)
        print("Camera opened. Starting gaze calibration — look straight at the camera now.")

        calib_x = []
        calib_y = []
        calib_start = time.time()
        while time.time() - calib_start < self.CALIB_SECONDS:
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            det = self.face_detection.process(rgb)
            mesh = self.face_mesh.process(rgb)
            if mesh.multi_face_landmarks and det.detections:
                landmarks = mesh.multi_face_landmarks[0].landmark
                avgx, avgy = self.get_iris_avg(landmarks)
                if avgx is not None:
                    calib_x.append(avgx); calib_y.append(avgy)
            cv2.putText(frame, "Calibrating gaze (keep eyes on camera)...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Concentration Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.cap.release()
                cv2.destroyAllWindows()
                raise SystemExit

        if not calib_x:
            print("Calibration failed: no face/iris detected. Retry with better lighting/position.")
            self.cap.release()
            cv2.destroyAllWindows()
            raise SystemExit

        self.baseline_x = float(np.mean(calib_x))
        self.baseline_y = float(np.mean(calib_y))
        print(f"Calibration complete. Baseline iris center = ({self.baseline_x:.3f}, {self.baseline_y:.3f})")
        cv2.destroyAllWindows()

    # ---------- Per-frame processing (no UI drawing here; see ui_ver5.py) ----------
    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        self.frame_no += 1
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # face detection confidence and mesh
        det = self.face_detection.process(rgb)
        mesh = self.face_mesh.process(rgb)
        face_conf = 0.0
        if det.detections:
            face_conf = max([d.score[0] for d in det.detections])

        occluded = False
        blink_event = False
        gaze_dir = "UNKNOWN"
        concentration = 0
        noise_flag = False
        eyes_closed_long = False
        no_face_elapsed = 0.0
        should_exit = False

        if mesh.multi_face_landmarks and face_conf >= self.FACE_DET_CONF:
            # Reset no-face timer since we have a face now
            self.no_face_start = None

            lm = mesh.multi_face_landmarks[0].landmark

            # EAR blink detection (temporal)
            left_ear = self.eye_aspect_ratio(lm, self.LEFT_EYE, w, h)
            right_ear = self.eye_aspect_ratio(lm, self.RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear > 0 and avg_ear < self.EAR_BLINK_THRESHOLD:
                self.blink_frames += 1
            else:
                if self.blink_frames >= self.BLINK_CONSEC_FRAMES:
                    now = time.time()
                    if now - self.last_blink_time > self.BLINK_MIN_SEP:
                        blink_event = True
                        self.last_blink_time = now
                self.blink_frames = 0

            # Eyes closed continuous timer
            if avg_ear > 0 and avg_ear < self.EAR_BLINK_THRESHOLD:
                if self.eyes_closed_start is None:
                    self.eyes_closed_start = time.time()
                if time.time() - self.eyes_closed_start >= self.EYES_CLOSED_SECONDS:
                    eyes_closed_long = True
            else:
                self.eyes_closed_start = None

            # Occlusion: check pixel stats inside eye regions
            left_stats = self.eye_region_stats(gray, lm, self.LEFT_EYE, w, h)
            right_stats = self.eye_region_stats(gray, lm, self.RIGHT_EYE, w, h)
            if left_stats is None or right_stats is None:
                occluded = True
            else:
                lmean, lvar = left_stats; rmean, rvar = right_stats
                if (lvar < self.EYE_VARIANCE_THRESHOLD or lmean < self.EYE_MEAN_DARK) and \
                   (rvar < self.EYE_VARIANCE_THRESHOLD or rmean < self.EYE_MEAN_DARK):
                    occluded = True

            # gaze using iris avg and calibrated baseline
            avgx, avgy = self.get_iris_avg(lm)
            if avgx is None:
                gaze_dir = "UNKNOWN"
            else:
                dx = avgx - self.baseline_x
                dy = avgy - self.baseline_y
                # assign direction
                if abs(dx) <= self.GAZE_X_DELTA and abs(dy) <= self.GAZE_Y_DELTA:
                    gaze_dir = "CENTER"
                elif abs(dx) > abs(dy):
                    gaze_dir = "LEFT" if dx < 0 else "RIGHT"
                else:
                    gaze_dir = "UP" if dy < 0 else "DOWN"

            # head center heuristic (nose)
            try:
                nose = lm[1]
                nose_x = nose.x; nose_y = nose.y
                head_ok = (abs(nose_x - 0.5) < 0.22 and abs(nose_y - 0.5) < 0.18)
            except Exception:
                head_ok = False

            # audio noise check (normalized)
            with self._audio_lock:
                current_rms = self._audio_rms
            if self._audio_baseline > 0 and current_rms > self._audio_baseline * self.NOISE_SENSITIVITY:
                noise_flag = True

            # compute concentration
            gaze_ok = (gaze_dir == "CENTER")
            concentration = self.compute_concentration(gaze_ok, head_ok, blink_event, occluded, noise_flag)
        else:
            # no reliable face
            concentration = 0
            occluded = True
            gaze_dir = "NO_FACE"

            # Start/advance no-face timer; caller decides whether to exit
            if self.no_face_start is None:
                self.no_face_start = time.time()
            else:
                no_face_elapsed = time.time() - self.no_face_start
                if no_face_elapsed >= self.NO_FACE_SECONDS:
                    should_exit = True

        self.score_buf.append(concentration)
        smooth_score = int(np.mean(self.score_buf)) if len(self.score_buf) > 0 else concentration
        #점수 스무딩: 0~100 점수를 비선형적으로 변환하여 더 높게 값 전달.(이거 없으면 점수 너무 짬ㅠㅠ)
        smooth_score = int((1-(((100-smooth_score)/100)**2))*100)

        # status label priority
        if not mesh.multi_face_landmarks or face_conf < self.FACE_DET_CONF:
            status = "NO FACE"
        elif occluded:
            status = "OCCLUDED"
        else:
            # noisy has precedence if noise present
            with self._audio_lock:
                curr = self._audio_rms
            noisy = (self._audio_baseline > 0 and curr > self._audio_baseline * self.NOISE_SENSITIVITY)
            if noisy:
                status = "NOISY"
            elif blink_event:
                status = "BLINK"
            elif smooth_score < 55:
                status = "DISTRACTED"
            else:
                status = "CONCENTRATED"

        return {
            "frame": frame,
            "score": smooth_score,
            "status": status,
            "gaze_dir": gaze_dir,
            "blink_event": blink_event,
            "noise_flag": noise_flag,
            "occluded": occluded,
            "eyes_closed_long": eyes_closed_long,
            "no_face_elapsed": no_face_elapsed,
            "should_exit": should_exit,
            "face_landmarks": mesh.multi_face_landmarks[0] if mesh.multi_face_landmarks else None,
        }

    def cleanup(self):
        try:
            if self._audio_stream is not None:
                self._audio_stream.stop()
                self._audio_stream.close()
        except Exception:
            pass

        self.cap.release()
        cv2.destroyAllWindows()
        print("Exited cleanly.")
