from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import math
import time
import threading

app = Flask(__name__)

# --------------------------------------------------
# MediaPipe
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions


face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1
)

face_landmarker = FaceLandmarker.create_from_options(face_options)
pose_landmarker = PoseLandmarker.create_from_options(pose_options)


# --------------------------------------------------
# Settings
# --------------------------------------------------

EYE_CLOSED_THRESHOLD = 0.20
HEAD_DOWN_THRESHOLD = 0.57
SLEEP_TIME = 5.0

HEAD_ON_DESK_THRESHOLD = 0.10
DEEP_SLEEP_TIME = 5.0

YAWN_THRESHOLD = 1.0
YAWN_TIME = 1.3


# --------------------------------------------------
# Landmark IDs
# --------------------------------------------------

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

NOSE = 1
FOREHEAD = 10
CHIN = 152

UPPER_LIP = 13
LOWER_LIP = 14
LEFT_MOUTH = 61
RIGHT_MOUTH = 291


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def distance(a, b):
    return math.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2
    )


def calculate_ear(landmarks, eye_ids):
    eye = [landmarks[i] for i in eye_ids]

    vertical1 = distance(eye[1], eye[5])
    vertical2 = distance(eye[2], eye[4])
    horizontal = distance(eye[0], eye[3])

    return (vertical1 + vertical2) / (2 * horizontal)


def draw_face_landmarks(frame, landmarks):
    h, w, _ = frame.shape

    # All face landmarks
    for point in landmarks:
        x = int(point.x * w)
        y = int(point.y * h)

        cv2.circle(
            frame,
            (x, y),
            1,
            (0, 255, 0),
            -1
        )

    # Eyes - blue
    for eye_ids in [LEFT_EYE, RIGHT_EYE]:
        for i in eye_ids:
            point = landmarks[i]

            x = int(point.x * w)
            y = int(point.y * h)

            cv2.circle(
                frame,
                (x, y),
                3,
                (255, 0, 0),
                -1
            )

    # Nose - red
    point = landmarks[NOSE]
    cv2.circle(
        frame,
        (int(point.x * w), int(point.y * h)),
        5,
        (0, 0, 255),
        -1
    )

    # Mouth - yellow
    for i in [UPPER_LIP, LOWER_LIP, LEFT_MOUTH, RIGHT_MOUTH]:
        point = landmarks[i]

        cv2.circle(
            frame,
            (int(point.x * w), int(point.y * h)),
            4,
            (0, 255, 255),
            -1
        )


def draw_pose_landmarks(frame, landmarks):
    h, w, _ = frame.shape

    for point in landmarks:
        x = int(point.x * w)
        y = int(point.y * h)

        # Pose landmarks - pink
        cv2.circle(
            frame,
            (x, y),
            5,
            (255, 0, 255),
            -1
        )


# --------------------------------------------------
# Detection status
# --------------------------------------------------

detection_status = {
    "face": "AWAKE",
    "eyes": "OPEN",
    "head": "NORMAL",
    "body": "NORMAL",
    "yawn": "NO YAWN",
    "overall": "AWAKE"
}


# Timers
eyes_closed_start = None
head_on_desk_start = None
yawn_start = None


# --------------------------------------------------
# Camera
# --------------------------------------------------

camera = cv2.VideoCapture(0)

latest_raw_frame = None
latest_detected_frame = None

frame_lock = threading.Lock()


# --------------------------------------------------
# Detection loop
# --------------------------------------------------

def detection_loop():

    global latest_raw_frame
    global latest_detected_frame

    global eyes_closed_start
    global head_on_desk_start
    global yawn_start

    timestamp_ms = 0

    while True:

        success, frame = camera.read()

        if not success:
            time.sleep(0.05)
            continue

        timestamp_ms += 33

        # Clean copy for left camera
        raw_frame = frame.copy()

        # Detection copy for right camera
        detected_frame = frame.copy()

        # --------------------------------------------------
        # MediaPipe image
        # --------------------------------------------------

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        face_result = face_landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        pose_result = pose_landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        now = time.time()

        # --------------------------------------------------
        # Default states
        # --------------------------------------------------

        face_state = "AWAKE"
        eyes_state = "OPEN"
        head_state = "NORMAL"
        body_state = "NORMAL"
        yawn_state = "NO YAWN"

        likely_sleeping = False
        likely_deep_sleeping = False
        yawn_detected = False

        # --------------------------------------------------
        # FACE DETECTION
        # --------------------------------------------------

        if face_result.face_landmarks:

            landmarks = face_result.face_landmarks[0]

            # Draw face points
            draw_face_landmarks(
                detected_frame,
                landmarks
            )

            left_ear = calculate_ear(
                landmarks,
                LEFT_EYE
            )

            right_ear = calculate_ear(
                landmarks,
                RIGHT_EYE
            )

            ear = (left_ear + right_ear) / 2

            eyes_closed = ear < EYE_CLOSED_THRESHOLD

            # Head angle
            nose = landmarks[NOSE]
            forehead = landmarks[FOREHEAD]
            chin = landmarks[CHIN]

            face_height = abs(chin.y - forehead.y)

            if face_height > 0:
                head_ratio = (
                    nose.y - forehead.y
                ) / face_height
            else:
                head_ratio = 0

            head_down = head_ratio > HEAD_DOWN_THRESHOLD

            # --------------------------------------------------
            # Eyes closed timer
            # --------------------------------------------------

            if eyes_closed:

                eyes_state = "EYES CLOSED"

                if eyes_closed_start is None:
                    eyes_closed_start = now

                if now - eyes_closed_start >= SLEEP_TIME:
                    likely_sleeping = True

            else:
                eyes_closed_start = None

            # --------------------------------------------------
            # Head
            # --------------------------------------------------

            if head_down:
                head_state = "HEAD DOWN"

            # Sleeping only when eyes closed
            if head_down and eyes_closed:

                if eyes_closed_start is not None:
                    if now - eyes_closed_start >= SLEEP_TIME:
                        likely_sleeping = True

            # --------------------------------------------------
            # Yawn
            # --------------------------------------------------

            mouth_height = distance(
                landmarks[UPPER_LIP],
                landmarks[LOWER_LIP]
            )

            mouth_width = distance(
                landmarks[LEFT_MOUTH],
                landmarks[RIGHT_MOUTH]
            )

            if mouth_width > 0:

                mouth_ratio = (
                    mouth_height / mouth_width
                )

                if mouth_ratio >= YAWN_THRESHOLD:

                    if yawn_start is None:
                        yawn_start = now

                    if now - yawn_start >= YAWN_TIME:
                        yawn_detected = True

                else:
                    yawn_start = None

            # Face status
            if likely_sleeping:
                face_state = "LIKELY SLEEPING"

        else:

            # No face = reset face timers
            eyes_closed_start = None
            yawn_start = None

        # --------------------------------------------------
        # BODY / POSE DETECTION
        # --------------------------------------------------

        if pose_result.pose_landmarks:

            pose_landmarks = pose_result.pose_landmarks[0]

            # Draw pose points
            draw_pose_landmarks(
                detected_frame,
                pose_landmarks
            )

            nose = pose_landmarks[0]

            left_shoulder = pose_landmarks[11]
            right_shoulder = pose_landmarks[12]

            shoulder_x = (
                left_shoulder.x +
                right_shoulder.x
            ) / 2

            shoulder_y = (
                left_shoulder.y +
                right_shoulder.y
            ) / 2

            shoulder_distance = math.sqrt(
                (nose.x - shoulder_x) ** 2 +
                (nose.y - shoulder_y) ** 2
            )

            if shoulder_distance <= HEAD_ON_DESK_THRESHOLD:

                body_state = "HEAD ON DESK"

                if head_on_desk_start is None:
                    head_on_desk_start = now

                if now - head_on_desk_start >= DEEP_SLEEP_TIME:
                    likely_deep_sleeping = True

            else:
                head_on_desk_start = None

        else:
            head_on_desk_start = None

        if likely_deep_sleeping:
            body_state = "LIKELY DEEP SLEEPING"

        if yawn_detected:
            yawn_state = "YAWNING - SLEEPY"

        # --------------------------------------------------
        # Overall status
        # --------------------------------------------------

        if likely_sleeping or likely_deep_sleeping:
            overall = "SLEEPING"

        elif yawn_detected:
            overall = "SLEEPY"

        else:
            overall = "AWAKE"

        # --------------------------------------------------
        # Update status
        # --------------------------------------------------

        detection_status["face"] = face_state
        detection_status["eyes"] = eyes_state
        detection_status["head"] = head_state
        detection_status["body"] = body_state
        detection_status["yawn"] = yawn_state
        detection_status["overall"] = overall

        # --------------------------------------------------
        # Text on detection camera
        # --------------------------------------------------

        cv2.putText(
            detected_frame,
            f"FACE: {face_state}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            detected_frame,
            f"BODY: {body_state}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            detected_frame,
            f"YAWN: {yawn_state}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        # --------------------------------------------------
        # Encode frames
        # --------------------------------------------------

        _, raw_buffer = cv2.imencode(
            ".jpg",
            raw_frame
        )

        _, detected_buffer = cv2.imencode(
            ".jpg",
            detected_frame
        )

        with frame_lock:

            latest_raw_frame = raw_buffer.tobytes()

            latest_detected_frame = (
                detected_buffer.tobytes()
            )

        time.sleep(0.01)


# Start detection thread
thread = threading.Thread(
    target=detection_loop,
    daemon=True
)

thread.start()


# --------------------------------------------------
# Video generators
# --------------------------------------------------

def generate_raw_frames():

    while True:

        with frame_lock:
            frame = latest_raw_frame

        if frame is not None:

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                frame +
                b"\r\n"
            )

        time.sleep(0.03)


def generate_detected_frames():

    while True:

        with frame_lock:
            frame = latest_detected_frame

        if frame is not None:

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                frame +
                b"\r\n"
            )

        time.sleep(0.03)


# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video/raw")
def video_raw():

    return Response(
        generate_raw_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/video/detected")
def video_detected():

    return Response(
        generate_detected_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/status")
def status():

    return jsonify(detection_status)


# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
    )