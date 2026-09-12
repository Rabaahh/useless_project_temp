import {
    FilesetResolver,
    FaceLandmarker,
    PoseLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs";


// --------------------------------------------------
// Settings
// --------------------------------------------------

const EYE_CLOSED_THRESHOLD = 0.20;
const HEAD_DOWN_THRESHOLD = 0.57;
const SLEEP_TIME = 5.0;

const HEAD_ON_DESK_THRESHOLD = 0.10;
const DEEP_SLEEP_TIME = 5.0;

const YAWN_THRESHOLD = 1.0;
const YAWN_TIME = 1.3;


// --------------------------------------------------
// Landmark IDs
// --------------------------------------------------

const LEFT_EYE = [362, 385, 387, 263, 373, 380];
const RIGHT_EYE = [33, 160, 158, 133, 153, 144];

const NOSE = 1;
const FOREHEAD = 10;
const CHIN = 152;

const UPPER_LIP = 13;
const LOWER_LIP = 14;
const LEFT_MOUTH = 61;
const RIGHT_MOUTH = 291;


// --------------------------------------------------
// Elements
// --------------------------------------------------

const video = document.getElementById("camera");

const normalCanvas =
    document.getElementById("normal-canvas");

const detectionCanvas =
    document.getElementById("detection-canvas");

const normalCtx =
    normalCanvas.getContext("2d");

const detectionCtx =
    detectionCanvas.getContext("2d");

const startButton =
    document.getElementById("start-button");

const startMessage =
    document.getElementById("start-message");


// --------------------------------------------------
// Audio
// --------------------------------------------------

const sleepAudio =
    new Audio("audio/sleep.mp3");

const deepSleepAudio =
    new Audio("audio/deep_sleep.mp3");

const yawnAudio =
    new Audio("audio/yawn.mp3");

let soundEnabled = false;

let previousSleep = false;
let previousDeepSleep = false;
let previousYawn = false;


// --------------------------------------------------
// Timers
// --------------------------------------------------

let sleepStart = null;
let headDownStart = null;
let deepSleepStart = null;
let yawnStart = null;


// --------------------------------------------------
// MediaPipe
// --------------------------------------------------

let faceLandmarker = null;
let poseLandmarker = null;

let running = false;
let lastVideoTime = -1;


// --------------------------------------------------
// Utility
// --------------------------------------------------

function distance(a, b) {

    const dx = a.x - b.x;
    const dy = a.y - b.y;

    return Math.sqrt(
        dx * dx + dy * dy
    );
}


function calculateEAR(eye) {

    const vertical1 =
        distance(eye[1], eye[5]);

    const vertical2 =
        distance(eye[2], eye[4]);

    const horizontal =
        distance(eye[0], eye[3]);

    if (horizontal === 0) {
        return 0;
    }

    return (
        (vertical1 + vertical2) /
        (2 * horizontal)
    );
}


function formatTime(seconds) {

    return seconds.toFixed(1) + "s";
}


function playSound(audio) {

    if (!soundEnabled) {
        return;
    }

    audio.currentTime = 0;

    audio.play().catch(error => {
        console.log(
            "Audio playback blocked:",
            error
        );
    });
}


// --------------------------------------------------
// Status helpers
// --------------------------------------------------

function setStatus(id, value) {

    const element =
        document.getElementById(id);

    if (element) {
        element.textContent = value;
    }
}


function updateStatuses(
    face,
    eyes,
    head,
    body,
    yawn,
    overall
) {

    setStatus("face-status", face);
    setStatus("eyes-status", eyes);
    setStatus("head-status", head);
    setStatus("body-status", body);
    setStatus("yawn-status", yawn);
    setStatus("overall-status", overall);


    // Top overlay

    setStatus("overlay-face", face);
    setStatus("overlay-eyes", eyes);
    setStatus("overlay-head", head);
    setStatus("overlay-body", body);
    setStatus("overlay-yawn", yawn);
}


function updateTimers(
    eyesTime,
    headTime,
    bodyTime,
    yawnTime
) {

    setStatus(
        "overlay-eyes-time",
        formatTime(eyesTime)
    );

    setStatus(
        "overlay-head-time",
        formatTime(headTime)
    );

    setStatus(
        "overlay-body-time",
        formatTime(bodyTime)
    );

    setStatus(
        "overlay-yawn-time",
        formatTime(yawnTime)
    );
}


// --------------------------------------------------
// Draw video
// --------------------------------------------------

function drawVideo() {

    if (!video.videoWidth || !video.videoHeight) {
        return;
    }

    const width = video.videoWidth;
    const height = video.videoHeight;


    normalCanvas.width = width;
    normalCanvas.height = height;

    detectionCanvas.width = width;
    detectionCanvas.height = height;


    normalCtx.drawImage(
        video,
        0,
        0,
        width,
        height
    );


    detectionCtx.drawImage(
        video,
        0,
        0,
        width,
        height
    );
}


// --------------------------------------------------
// Draw points
// --------------------------------------------------

function drawPoint(
    ctx,
    landmark,
    width,
    height,
    color,
    radius = 2
) {

    ctx.fillStyle = color;

    ctx.beginPath();

    ctx.arc(
        landmark.x * width,
        landmark.y * height,
        radius,
        0,
        Math.PI * 2
    );

    ctx.fill();
}


// --------------------------------------------------
// Face landmarks
// --------------------------------------------------

function drawFaceLandmarks(
    landmarks,
    width,
    height
) {

    // Face - green

    for (const point of landmarks) {

        drawPoint(
            detectionCtx,
            point,
            width,
            height,
            "green",
            1.5
        );
    }


    // Eyes - blue

    for (const index of LEFT_EYE) {

        drawPoint(
            detectionCtx,
            landmarks[index],
            width,
            height,
            "blue",
            3
        );
    }

    for (const index of RIGHT_EYE) {

        drawPoint(
            detectionCtx,
            landmarks[index],
            width,
            height,
            "blue",
            3
        );
    }


    // Nose - red

    drawPoint(
        detectionCtx,
        landmarks[NOSE],
        width,
        height,
        "red",
        4
    );


    // Mouth - yellow

    const mouthPoints = [
        UPPER_LIP,
        LOWER_LIP,
        LEFT_MOUTH,
        RIGHT_MOUTH
    ];

    for (const index of mouthPoints) {

        drawPoint(
            detectionCtx,
            landmarks[index],
            width,
            height,
            "yellow",
            4
        );
    }
}


// --------------------------------------------------
// Body landmarks
// --------------------------------------------------

function drawPoseLandmarks(
    landmarks,
    width,
    height
) {

    for (const point of landmarks) {

        drawPoint(
            detectionCtx,
            point,
            width,
            height,
            "magenta",
            3
        );
    }
}


// --------------------------------------------------
// Face detection
// --------------------------------------------------

function processFace(
    landmarks,
    now
) {

    const leftEye =
        LEFT_EYE.map(
            index => landmarks[index]
        );

    const rightEye =
        RIGHT_EYE.map(
            index => landmarks[index]
        );


    const leftEAR =
        calculateEAR(leftEye);

    const rightEAR =
        calculateEAR(rightEye);

    const ear =
        (leftEAR + rightEAR) / 2;


    const eyesClosed =
        ear < EYE_CLOSED_THRESHOLD;


    // --------------------------------------------------
    // Head position
    // --------------------------------------------------

    const headRatio =
        (
            landmarks[NOSE].y -
            landmarks[FOREHEAD].y
        ) /
        Math.abs(
            landmarks[CHIN].y -
            landmarks[FOREHEAD].y
        );


    const headDown =
        headRatio > HEAD_DOWN_THRESHOLD;


    // --------------------------------------------------
    // Eyes closed timer
    // --------------------------------------------------

    if (eyesClosed) {

        if (sleepStart === null) {
            sleepStart = now;
        }

    } else {

        sleepStart = null;
    }


    const eyesClosedTime =
        sleepStart !== null
            ? now - sleepStart
            : 0;


    const sleeping =
        eyesClosedTime >= SLEEP_TIME;


    // --------------------------------------------------
    // Head down timer
    // --------------------------------------------------

    if (headDown) {

        if (headDownStart === null) {
            headDownStart = now;
        }

    } else {

        headDownStart = null;
    }


    const headDownTime =
        headDownStart !== null
            ? now - headDownStart
            : 0;


    // --------------------------------------------------
    // Yawn
    // --------------------------------------------------

    const mouthHeight =
        distance(
            landmarks[UPPER_LIP],
            landmarks[LOWER_LIP]
        );

    const mouthWidth =
        distance(
            landmarks[LEFT_MOUTH],
            landmarks[RIGHT_MOUTH]
        );


    let mouthRatio = 0;

    if (mouthWidth !== 0) {

        mouthRatio =
            mouthHeight / mouthWidth;
    }


    const yawning =
        mouthRatio >= YAWN_THRESHOLD;


    if (yawning) {

        if (yawnStart === null) {
            yawnStart = now;
        }

    } else {

        yawnStart = null;
    }


    const yawnTime =
        yawnStart !== null
            ? now - yawnStart
            : 0;


    const yawnConfirmed =
        yawnTime >= YAWN_TIME;


    return {

        eyesClosed,
        eyesClosedTime,

        headDown,
        headDownTime,

        sleeping,

        yawning: yawnConfirmed,
        yawnTime
    };
}


// --------------------------------------------------
// Body detection
// --------------------------------------------------

function processBody(
    landmarks,
    now
) {

    if (
        !landmarks ||
        landmarks.length === 0
    ) {

        deepSleepStart = null;

        return {
            headOnDesk: false,
            bodyTime: 0,
            deepSleeping: false
        };
    }


    const poseNose =
        landmarks[0];

    const leftShoulder =
        landmarks[11];

    const rightShoulder =
        landmarks[12];


    if (
        !poseNose ||
        !leftShoulder ||
        !rightShoulder
    ) {

        deepSleepStart = null;

        return {
            headOnDesk: false,
            bodyTime: 0,
            deepSleeping: false
        };
    }


    const shoulderCenter = {

        x:
            (
                leftShoulder.x +
                rightShoulder.x
            ) / 2,

        y:
            (
                leftShoulder.y +
                rightShoulder.y
            ) / 2
    };


    const headDeskDistance =
        distance(
            poseNose,
            shoulderCenter
        );


    const headOnDesk =
        headDeskDistance <=
        HEAD_ON_DESK_THRESHOLD;


    // --------------------------------------------------
    // Head-on-desk timer
    // --------------------------------------------------

    if (headOnDesk) {

        if (deepSleepStart === null) {
            deepSleepStart = now;
        }

    } else {

        deepSleepStart = null;
    }


    const bodyTime =
        deepSleepStart !== null
            ? now - deepSleepStart
            : 0;


    const deepSleeping =
        bodyTime >= DEEP_SLEEP_TIME;


    return {

        headOnDesk,
        bodyTime,
        deepSleeping
    };
}


// --------------------------------------------------
// Main detection
// --------------------------------------------------

async function detectFrame() {

    if (!running) {
        return;
    }


    if (
        video.readyState < 2 ||
        video.videoWidth === 0
    ) {

        requestAnimationFrame(
            detectFrame
        );

        return;
    }


    const now =
        performance.now() / 1000;


    drawVideo();


    if (
        video.currentTime !==
        lastVideoTime
    ) {

        lastVideoTime =
            video.currentTime;


        const timestamp =
            performance.now();


        // --------------------------------------------------
        // Face
        // --------------------------------------------------

        const faceResult =
            faceLandmarker.detectForVideo(
                video,
                timestamp
            );


        // --------------------------------------------------
        // Pose
        // --------------------------------------------------

        const poseResult =
            poseLandmarker.detectForVideo(
                video,
                timestamp
            );


        let faceState = "NO FACE";
        let eyesState = "UNKNOWN";
        let headState = "UNKNOWN";

        let bodyState = "NORMAL";
        let yawnState = "NO YAWN";

        let sleeping = false;
        let deepSleeping = false;
        let yawning = false;

        let eyesClosedTime = 0;
        let headDownTime = 0;
        let bodyTime = 0;
        let yawnTime = 0;


        // --------------------------------------------------
        // Face
        // --------------------------------------------------

        if (
            faceResult.faceLandmarks &&
            faceResult.faceLandmarks.length > 0
        ) {

            const landmarks =
                faceResult.faceLandmarks[0];


            drawFaceLandmarks(
                landmarks,
                detectionCanvas.width,
                detectionCanvas.height
            );


            const result =
                processFace(
                    landmarks,
                    now
                );


            eyesState =
                result.eyesClosed
                    ? "CLOSED"
                    : "OPEN";


            headState =
                result.headDown
                    ? "DOWN"
                    : "NORMAL";


            sleeping =
                result.sleeping;


            yawning =
                result.yawning;


            eyesClosedTime =
                result.eyesClosedTime;


            headDownTime =
                result.headDownTime;


            yawnTime =
                result.yawnTime;


            faceState =
                sleeping
                    ? "LIKELY SLEEPING"
                    : "AWAKE";


            if (yawning) {

                yawnState =
                    "YAWNING - SLEEPY";
            }

        } else {

            // No face means reset face-related timers

            sleepStart = null;
            headDownStart = null;
            yawnStart = null;
        }


        // --------------------------------------------------
        // Body
        // --------------------------------------------------

        if (
            poseResult.landmarks &&
            poseResult.landmarks.length > 0
        ) {

            const poseLandmarks =
                poseResult.landmarks[0];


            drawPoseLandmarks(
                poseLandmarks,
                detectionCanvas.width,
                detectionCanvas.height
            );


            const bodyResult =
                processBody(
                    poseLandmarks,
                    now
                );


            deepSleeping =
                bodyResult.deepSleeping;


            bodyTime =
                bodyResult.bodyTime;


            if (bodyResult.headOnDesk) {

                bodyState =
                    deepSleeping
                        ? "LIKELY DEEP SLEEPING"
                        : "HEAD ON DESK";
            }

        } else {

            deepSleepStart = null;
        }


        // --------------------------------------------------
        // Overall state
        // --------------------------------------------------

        let overall = "AWAKE";


        if (
            sleeping ||
            deepSleeping
        ) {

            overall = "SLEEPING";

        } else if (yawning) {

            overall = "SLEEPY";
        }


        // --------------------------------------------------
        // Update cards
        // --------------------------------------------------

        updateStatuses(
            faceState,
            eyesState,
            headState,
            bodyState,
            yawnState,
            overall
        );


        // --------------------------------------------------
        // Update overlay timers
        // --------------------------------------------------

        updateTimers(
            eyesClosedTime,
            headDownTime,
            bodyTime,
            yawnTime
        );


        // --------------------------------------------------
        // Sounds
        // --------------------------------------------------

        if (
            sleeping &&
            !previousSleep
        ) {

            playSound(sleepAudio);
        }


        if (
            deepSleeping &&
            !previousDeepSleep
        ) {

            playSound(deepSleepAudio);
        }


        if (
            yawning &&
            !previousYawn
        ) {

            playSound(yawnAudio);
        }


        previousSleep =
            sleeping;

        previousDeepSleep =
            deepSleeping;

        previousYawn =
            yawning;
    }


    requestAnimationFrame(
        detectFrame
    );
}


// --------------------------------------------------
// Initialize MediaPipe
// --------------------------------------------------

async function initializeMediaPipe() {

    startMessage.textContent =
        "Loading detection models...";


    const vision =
        await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm"
        );


    faceLandmarker =
        await FaceLandmarker.createFromOptions(
            vision,
            {
                baseOptions: {
                    modelAssetPath:
                        "face_landmarker.task"
                },

                runningMode: "VIDEO",

                numFaces: 1
            }
        );


    poseLandmarker =
        await PoseLandmarker.createFromOptions(
            vision,
            {
                baseOptions: {
                    modelAssetPath:
                        "pose_landmarker.task"
                },

                runningMode: "VIDEO",

                numPoses: 1
            }
        );
}


// --------------------------------------------------
// Start detection
// --------------------------------------------------

startButton.addEventListener(
    "click",
    async function () {

        if (running) {
            return;
        }


        try {

            startButton.disabled = true;

            startMessage.textContent =
                "Loading detection models...";


            await initializeMediaPipe();


            startMessage.textContent =
                "Requesting camera permission...";


            const stream =
                await navigator.mediaDevices.getUserMedia(
                    {
                        video: {
                            facingMode: "user"
                        },

                        audio: false
                    }
                );


            video.srcObject = stream;


            await video.play();


            soundEnabled = true;

            running = true;


            startButton.textContent =
                "✓ DETECTION RUNNING";

            startButton.classList.add(
                "started"
            );


            startMessage.textContent =
                "Meme sounds enabled 🔊";


            detectFrame();


        } catch (error) {

            console.error(
                "Could not start detection:",
                error
            );


            startButton.disabled = false;


            startMessage.textContent =
                "Could not start camera or detection. Check camera permission and try again.";

        }

    }
);