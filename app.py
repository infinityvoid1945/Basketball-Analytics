import csv
import cv2
import numpy as np
from collections import deque, defaultdict
from ultralytics import YOLO

MODEL_PATH   = ""
VIDEO_PATH   = ""
OUTPUT_VIDEO = "annotated_output3.mp4"
OUTPUT_CSV   = "shot_zone_stats.csv"

CONF = 0.2
PLAYER, BALL, RIM, BACKBOARD = 0, 1, 2, 3

CALIBRATE_PICK = True
IMAGE_POINTS = []

COURT_POINTS = [(-22.0, -5.25), (22.0, -5.25), (-8.0, 13.75), (8.0, 13.75)]

POSSESSION_PAD   = 30
RIM_SMOOTH       = 30
BALL_PREDICT_MAX = 8

BALL_GATE_BASE   = 90
BALL_GATE_GROW   = 30
BALL_GRAVITY     = 1.5
SHOT_COOLDOWN    = 25
SHOT_TIMEOUT     = 60


def dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


class BallTracker:
    def __init__(self, max_coast=BALL_PREDICT_MAX, gate_base=BALL_GATE_BASE,
                 gate_grow=BALL_GATE_GROW, gravity=BALL_GRAVITY,
                 seed_gate=None, alpha=0.6):
        self.max_coast = max_coast
        self.gate_base = gate_base
        self.gate_grow = gate_grow
        self.gravity = gravity
        self.seed_gate = seed_gate if seed_gate is not None else gate_base * 0.7
        self.alpha = alpha

        self.pos = None
        self.vel = (0.0, 0.0)
        self.missing = 0
        self.pending = None
        self.trail = deque(maxlen=30)

    def _predict(self):
        return (self.pos[0] + self.vel[0], self.pos[1] + self.vel[1])

    def _seed(self, det):
        self.pos = (float(det[0]), float(det[1]))
        self.vel = (0.0, 0.0)
        self.missing = 0
        self.pending = None
        self.trail.append(self.pos)

    def _coast(self):
        if self.pos is None or self.missing >= self.max_coast:
            self.pos = None
            return None
        pred = self._predict()
        self.vel = (self.vel[0], self.vel[1] + self.gravity)
        self.pos = pred
        self.missing += 1
        self.trail.append(pred)
        return pred

    def _age_pending(self):
        if self.pending is not None:
            p, age = self.pending
            self.pending = (p, age + 1)
            if age + 1 > 3:
                self.pending = None

    def update(self, det):
        self._age_pending()

        if self.pos is None:
            if det is None:
                return None
            if self.pending is not None and dist(det, self.pending[0]) <= self.seed_gate:
                self._seed(det)
                return self.pos
            self.pending = (det, 0)
            return None

        if det is None:
            return self._coast()

        pred = self._predict()
        gate = self.gate_base + self.gate_grow * self.missing
        if dist(det, pred) <= gate:
            nx = self.alpha * det[0] + (1 - self.alpha) * pred[0]
            ny = self.alpha * det[1] + (1 - self.alpha) * pred[1]
            self.vel = (nx - self.pos[0], ny - self.pos[1])
            self.pos = (nx, ny)
            self.missing = 0
            self.pending = None
            self.trail.append(self.pos)
            return self.pos

        if self.pending is not None and dist(det, self.pending[0]) <= self.seed_gate:
            self._seed(det)
            return self.pos
        self.pending = (det, 0)
        return self._coast()


NBA_ZONES = ["Restricted Area", "In The Paint (Non-RA)", "Mid-Range",
             "Left Corner 3", "Right Corner 3", "Above the Break 3", "Unknown"]

def classify_zone(x, y):
    d = (x * x + y * y) ** 0.5
    if d <= 4.0:
        return "Restricted Area"
    if y <= 14.0 and abs(x) >= 22.0:
        return "Left Corner 3" if x < 0 else "Right Corner 3"
    if d >= 23.75:
        return "Above the Break 3"
    if abs(x) <= 8.0 and -5.25 <= y <= 13.75:
        return "In The Paint (Non-RA)"
    return "Mid-Range"


def build_homography():
    if len(IMAGE_POINTS) < 4 or len(COURT_POINTS) < 4:
        return None
    src = np.array(IMAGE_POINTS, dtype=np.float32)
    dst = np.array(COURT_POINTS, dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    return H


def image_to_court(H, pt):
    if H is None:
        return None
    p = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
    q = cv2.perspectiveTransform(p, H)[0][0]
    return float(q[0]), float(q[1])


def calibrate(video_path, max_w=1000, max_h=650):
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("Could not read first frame for calibration.")
        return

    h0, w0 = frame.shape[:2]
    scale = min(max_w / w0, max_h / h0, 1.0)
    base = cv2.resize(frame, (int(w0 * scale), int(h0 * scale)))

    labels = ["LEFT corner-3 @ baseline", "RIGHT corner-3 @ baseline",
              "LEFT free-throw-line end", "RIGHT free-throw-line end"]
    picks = []
    picks_disp = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(picks) < 4:
            ox, oy = int(round(x / scale)), int(round(y / scale))
            picks.append((ox, oy))
            picks_disp.append((x, y))
            print(f"  point {len(picks)} ({labels[len(picks)-1]}): ({ox}, {oy})")

    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", on_click)
    print(f"(frame {w0}x{h0} shown at {int(scale*100)}%) Click these 4 points in order:")
    for l in labels:
        print("   -", l)
    while True:
        disp = base.copy()
        for i, p in enumerate(picks_disp):
            cv2.circle(disp, p, 5, (0, 255, 0), -1)
            cv2.putText(disp, str(i + 1), (p[0] + 8, p[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("calibrate", disp)
        if (cv2.waitKey(20) & 0xFF) in (27, ord("q")) or len(picks) == 4:
            break
    cv2.waitKey(300)
    cv2.destroyAllWindows()
    print("\nPaste this into IMAGE_POINTS:\n  IMAGE_POINTS =", picks)


class ShotAnalyzer:
    def __init__(self, H):
        self.H = H
        self.rim_hist = deque(maxlen=RIM_SMOOTH)
        self.ball = BallTracker()
        self.last_possessed_foot = None

        self.active = False
        self.went_above = False
        self.made = False
        self.shot_zone = "Unknown"
        self.cooldown = 0
        self.frames_in_flight = 0

        self.attempts = defaultdict(int)
        self.makes = defaultdict(int)
        self.flash = 0
        self.flash_text = ""

    def parse(self, result):
        players, ball, rim = [], None, None
        best_ball, best_rim = 0.0, 0.0
        if result.boxes is not None:
            xyxy = result.boxes.xyxy.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy().astype(int)
            conf = result.boxes.conf.cpu().numpy()
            for box, c, cf in zip(xyxy, cls, conf):
                box = box.tolist()
                if c == PLAYER:
                    players.append(box)
                elif c == BALL and cf > best_ball:
                    best_ball, ball = cf, box
                elif c == RIM and cf > best_rim:
                    best_rim, rim = cf, box
        return players, ball, rim

    def get_rim(self, rim):
        if rim is not None:
            self.rim_hist.append(rim)
        if not self.rim_hist:
            return None
        return np.median(np.array(self.rim_hist), axis=0).tolist()

    def get_ball(self, ball):
        det = None
        if ball is not None:
            det = ((ball[0] + ball[2]) / 2.0, (ball[1] + ball[3]) / 2.0)
        return self.ball.update(det)

    def update_possession(self, players, ball_c):
        if ball_c is None:
            return None
        owner = None
        best = 1e9
        for p in players:
            inside = (p[0] - POSSESSION_PAD <= ball_c[0] <= p[2] + POSSESSION_PAD and
                      p[1] - POSSESSION_PAD <= ball_c[1] <= p[3] + POSSESSION_PAD)
            if inside:
                cx = (p[0] + p[2]) / 2.0
                d = abs(cx - ball_c[0])
                if d < best:
                    best, owner = d, p
        if owner is not None:
            foot = ((owner[0] + owner[2]) / 2.0, owner[3])
            self.last_possessed_foot = foot
        return owner

    def update_shot(self, rim, ball_c):
        if self.cooldown > 0:
            self.cooldown -= 1
        if rim is None or ball_c is None:
            return

        rx1, ry1, rx2, ry2 = rim
        rim_cy = (ry1 + ry2) / 2.0
        rim_w = max(rx2 - rx1, 1.0)
        rim_h = max(ry2 - ry1, 1.0)
        bx, by = ball_c

        near = (rx1 - rim_w <= bx <= rx2 + rim_w) and (ry1 - 5 * rim_h <= by <= ry2 + 2 * rim_h)

        if not self.active and self.cooldown == 0 and near:
            self.active = True
            self.went_above = False
            self.made = False
            self.frames_in_flight = 0
            if self.H is not None and self.last_possessed_foot is not None:
                cx, cy = image_to_court(self.H, self.last_possessed_foot)
                self.shot_zone = classify_zone(cx, cy)
            else:
                self.shot_zone = "Unknown"

        if self.active:
            self.frames_in_flight += 1
            if by < ry1:
                self.went_above = True
            if self.went_above and by > rim_cy and rx1 <= bx <= rx2:
                self.made = True

            below = by > ry2 + 3 * rim_h
            if below or self.frames_in_flight > SHOT_TIMEOUT:
                self._finalize()

    def _finalize(self):
        self.attempts[self.shot_zone] += 1
        if self.made:
            self.makes[self.shot_zone] += 1
        self.flash = 20
        self.flash_text = "MADE!" if self.made else "MISS"
        self.active = False
        self.cooldown = SHOT_COOLDOWN

    def draw(self, frame, players, ball_c, rim):
        for p in players:
            cv2.rectangle(frame, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), (0, 200, 0), 2)
        if rim is not None:
            cv2.rectangle(frame, (int(rim[0]), int(rim[1])), (int(rim[2]), int(rim[3])), (0, 165, 255), 2)
        trail = self.ball.trail
        for i in range(1, len(trail)):
            a, b = trail[i - 1], trail[i]
            if dist(a, b) < 120:
                cv2.line(frame, tuple(map(int, a)), tuple(map(int, b)), (255, 0, 255), 2)
        if ball_c is not None:
            cv2.circle(frame, tuple(map(int, ball_c)), 6, (255, 0, 255), -1)

        total_a = sum(self.attempts.values())
        total_m = sum(self.makes.values())
        pct = (100.0 * total_m / total_a) if total_a else 0.0
        cv2.putText(frame, f"FG: {total_m}/{total_a}  ({pct:.0f}%)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        if self.flash > 0:
            color = (0, 255, 0) if self.flash_text == "MADE!" else (0, 0, 255)
            cv2.putText(frame, self.flash_text, (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            self.flash -= 1
        return frame


def main():
    if CALIBRATE_PICK:
        calibrate(VIDEO_PATH)
        return

    H = build_homography()
    if H is None:
        print("WARNING: not calibrated -> zones will be 'Unknown'.")
        print("Set CALIBRATE_PICK=True, run once, fill IMAGE_POINTS, then rerun.\n")

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Could not open video:", VIDEO_PATH)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    analyzer = ShotAnalyzer(H)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    print(f"Processing {total or '?'} frames... (press Ctrl+C to stop early)\n")

    import time
    t0 = time.time()
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        result = model.track(frame, persist=True, conf=CONF,
                             classes=[PLAYER, BALL, RIM, BACKBOARD], verbose=False)[0]

        players, ball_box, rim_box = analyzer.parse(result)
        rim = analyzer.get_rim(rim_box)
        ball_c = analyzer.get_ball(ball_box)
        analyzer.update_possession(players, ball_c)
        analyzer.update_shot(rim, ball_c)

        frame = analyzer.draw(frame, players, ball_c, rim)
        writer.write(frame)

        if frame_idx % 15 == 0 or frame_idx == total:
            elapsed = time.time() - t0
            fps_proc = frame_idx / elapsed if elapsed else 0
            if total:
                pct = 100.0 * frame_idx / total
                eta = (total - frame_idx) / fps_proc if fps_proc else 0
                print(f"\r  {frame_idx}/{total} frames ({pct:5.1f}%)  "
                      f"{fps_proc:4.1f} fps  ETA {eta:5.0f}s   ", end="", flush=True)
            else:
                print(f"\r  {frame_idx} frames  {fps_proc:4.1f} fps   ",
                      end="", flush=True)

    print()
    cap.release()
    writer.release()

    print("\n================  FIELD-GOAL % BY ZONE  ================")
    print(f"{'Zone':<24}{'Made':>6}{'Att':>6}{'FG%':>8}")
    print("-" * 44)
    rows = []
    for z in NBA_ZONES:
        a = analyzer.attempts.get(z, 0)
        m = analyzer.makes.get(z, 0)
        if a == 0 and z == "Unknown":
            continue
        pct = (100.0 * m / a) if a else 0.0
        print(f"{z:<24}{m:>6}{a:>6}{pct:>7.1f}%")
        rows.append([z, m, a, round(pct, 1)])
    ta = sum(analyzer.attempts.values())
    tm = sum(analyzer.makes.values())
    tp = (100.0 * tm / ta) if ta else 0.0
    print("-" * 44)
    print(f"{'TOTAL':<24}{tm:>6}{ta:>6}{tp:>7.1f}%")

    with open(OUTPUT_CSV, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["zone", "made", "attempts", "fg_pct"])
        wtr.writerows(rows)
        wtr.writerow(["TOTAL", tm, ta, round(tp, 1)])
    print(f"\nSaved annotated video -> {OUTPUT_VIDEO}")
    print(f"Saved per-zone CSV     -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
