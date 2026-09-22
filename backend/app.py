import os
import shutil
import uuid
import time
import tempfile
import asyncio
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

from werkzeug.utils import secure_filename
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import cv2
import torch
import numpy as np
from scipy.spatial.transform import Rotation

from frankmocap.integration.copy_and_paste import integration_copy_paste
from frankmocap.demo.demo_frankmocap import __filter_bbox_list
from frankmocap.handmocap.hand_mocap_api import HandMocap
from frankmocap.bodymocap.body_mocap_api import BodyMocap
from frankmocap.handmocap.hand_bbox_detector import HandBboxDetector

from pygltflib import (
    ANIM_LINEAR, GLTF2, BufferView, Accessor, FLOAT, SCALAR, VEC4,
    AnimationSampler, AnimationChannel, AnimationChannelTarget, ROTATION,
    Animation,
)

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("mocap")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

ALLOWED_EXTENSIONS = {"mp4"}
MAX_UPLOAD_MB = 200
JOB_TTL_SECONDS = 3600

START_INDICES = [
    30, 21, 12, 3, 33, 24, 15, 6, 45, 36,
    72, 69, 66, 81, 78, 75, 90, 87, 84,
    99, 96, 93, 108, 105, 102, 60, 54, 48, 39,
    117, 114, 111, 126, 123, 120, 135, 132, 129,
    144, 141, 138, 153, 150, 147, 63, 57, 51, 42,
    27, 18, 9, 0,
]

RELAXED_POSE = [
    3.1415, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.11167845129966736,
    0.04289207234978676, -0.41644084453582764, 0.10881128907203674,
    -0.06598565727472305, -0.756219744682312, -0.0963931530714035,
    -0.09091583639383316, -0.18845966458320618, -0.11809506267309189,
    0.050943851470947266, -0.5295845866203308, -0.14369848370552063,
    0.055241718888282776, -0.704857349395752, -0.019182899966835976,
    -0.0923367589712143, -0.3379131853580475, -0.45703303813934326,
    -0.1962839663028717, -0.6254575848579407, -0.2146523892879486,
    -0.06599827855825424, -0.5068942308425903, -0.36972442269325256,
    -0.0603446289896965, -0.07949023693799973, -0.14186954498291016,
    -0.08585254102945328, -0.6355276107788086, -0.3033415675163269,
    -0.05788097903132439, -0.6313892006874084, -0.17612087726593018,
    -0.13209305703639984, -0.3733545243740082, 0.850964367389679,
    0.2769227623939514, -0.09154807031154633, -0.4998386800289154,
    0.026556432247161865, 0.052880801260471344, 0.5355585217475891,
    0.045960985124111176, -0.27735769748687744, 0.11167845129966736,
    -0.04289207234978676, 0.41644084453582764, 0.10881128907203674,
    0.06598565727472305, 0.756219744682312, -0.0963931530714035,
    0.09091583639383316, 0.18845966458320618, -0.11809506267309189,
    -0.050943851470947266, 0.5295845866203308, -0.14369848370552063,
    -0.055241718888282776, 0.704857349395752, -0.019182899966835976,
    0.0923367589712143, 0.3379131853580475, -0.45703303813934326,
    0.1962839663028717, 0.6254575848579407, -0.2146523892879486,
    0.06599827855825424, 0.5068942308425903, -0.36972442269325256,
    0.0603446289896965, 0.07949023693799973, -0.14186954498291016,
    0.08585254102945328, 0.6355276107788086, -0.3033415675163269,
    0.05788097903132439, 0.6313892006874084, -0.17612087726593018,
    0.13209305703639984, 0.3733545243740082, 0.850964367389679,
    -0.2769227623939514, 0.09154807031154633, -0.4998386800289154,
    -0.026556432247161865, -0.052880801260471344, 0.5355585217475891,
    -0.045960985124111176, 0.27735769748687744,
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_GLB_PATH = os.path.join(BASE_DIR, "model.glb")
EXTRA_DATA_DIR = os.path.join(BASE_DIR, "frankmocap", "extra_data")
SMPL_DIR = os.path.join(EXTRA_DATA_DIR, "smpl")
BODY_CHECKPOINT = os.path.join(
    EXTRA_DATA_DIR, "body_module", "pretrained_weights",
    "smplx-03-28-46060-w_spin_mlc3d_46582-2089_2020_03_28-21_56_16.pt",
)
HAND_CHECKPOINT = os.path.join(
    EXTRA_DATA_DIR, "hand_module", "pretrained_weights",
    "pose_shape_best.pth",
)

# --------------------------------------------------------------------------- #
# Job model
# --------------------------------------------------------------------------- #

JOBS: dict[str, "Job"] = {}

STAGE_LABELS = {
    "queued":      "Queued",
    "decoding":    "Reading video",
    "loading":     "Loading pose models",
    "tracking":    "Tracking poses",
    "integrating": "Combining body & hands",
    "building":    "Building 3D scene",
    "done":        "Ready",
    "error":       "Failed",
    "cancelled":   "Cancelled",
}

STAGE_ORDER = ["queued", "decoding", "loading", "tracking", "integrating", "building"]


@dataclass
class Job:
    id: str
    filename: str
    status: str = "queued"
    stage: str = "queued"
    message: str = "Queued"
    progress: int = 0
    frames_total: int = 0
    frames_processed: int = 0
    frames_with_pose: int = 0
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    glb_path: Optional[str] = None
    tmp_dir: Optional[str] = None
    cancel_requested: bool = False

    def public(self) -> dict[str, Any]:
        elapsed = (self.finished_at or time.time()) - self.started_at
        return {
            "id": self.id,
            "filename": self.filename,
            "status": self.status,
            "stage": self.stage,
            "message": self.message,
            "progress": self.progress,
            "frames_total": self.frames_total,
            "frames_processed": self.frames_processed,
            "frames_with_pose": self.frames_with_pose,
            "elapsed_seconds": round(elapsed, 1),
            "stages": [
                {
                    "key": k,
                    "label": STAGE_LABELS[k],
                    "state": _stage_state(k, self.stage, self.status),
                }
                for k in STAGE_ORDER
            ],
            "error": (
                {"code": self.error_code, "message": self.error_message}
                if self.error_message else None
            ),
        }


def _stage_state(stage: str, current: str, status: str) -> str:
    if status == "error" and stage == current:
        return "failed"
    if stage == current:
        return "active" if status == "running" else "done"
    try:
        return "done" if STAGE_ORDER.index(stage) < STAGE_ORDER.index(current) else "pending"
    except ValueError:
        return "pending"


def update_job(job: Job, **fields: Any) -> None:
    for k, v in fields.items():
        setattr(job, k, v)
    job.updated_at = time.time()


def reap_old_jobs() -> None:
    now = time.time()
    stale = [
        jid for jid, j in JOBS.items()
        if j.finished_at and now - j.finished_at > JOB_TTL_SECONDS
    ]
    for jid in stale:
        job = JOBS.pop(jid, None)
        if job and job.tmp_dir:
            shutil.rmtree(job.tmp_dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# App + process pool
# --------------------------------------------------------------------------- #

app = FastAPI(title="FrankMocap 3D Viewer API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_headers=["*"],
    expose_headers=["X-Request-Id", "X-Frames-Total", "X-Frames-With-Pose"],
    allow_methods=["*"],
)

_pool: Optional[ProcessPoolExecutor] = None
_manager: Optional[multiprocessing.Manager] = None


def get_pool() -> ProcessPoolExecutor:
    global _pool
    if _pool is None:
        _pool = ProcessPoolExecutor(max_workers=1)
    return _pool


def get_manager() -> multiprocessing.Manager:
    global _manager
    if _manager is None:
        _manager = multiprocessing.Manager()
    return _manager


@app.on_event("shutdown")
def shutdown_pool():
    global _pool, _manager
    if _pool is not None:
        _pool.shutdown(wait=False)
        _pool = None
    if _manager is not None:
        _manager.shutdown()
        _manager = None


def api_error(status_code: int, code: str, message: str, **extra: Any) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, **extra}},
    )


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #

@app.get("/health")
def health():
    issues = []
    if not os.path.exists(MODEL_GLB_PATH):
        issues.append(f"model.glb not found at {MODEL_GLB_PATH}")
    if not os.path.exists(BODY_CHECKPOINT):
        issues.append(f"body checkpoint missing: {BODY_CHECKPOINT}")
    if not os.path.exists(HAND_CHECKPOINT):
        issues.append(f"hand checkpoint missing: {HAND_CHECKPOINT}")
    if not torch.cuda.is_available():
        issues.append("CUDA not available (this service requires a GPU)")

    if issues:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "issues": issues},
        )
    return {"status": "ok"}


# --------------------------------------------------------------------------- #
# Job endpoints
# --------------------------------------------------------------------------- #

@app.post("/jobs")
async def create_job(uploaded_file: UploadFile = File(...)):
    reap_old_jobs()
    request_id = uuid.uuid4().hex[:12]
    log.info("[%s] POST /jobs filename=%s", request_id, uploaded_file.filename)

    if not uploaded_file or not uploaded_file.filename:
        return api_error(400, "missing_file", "No file was uploaded.")
    if not allowed_file(uploaded_file.filename):
        return api_error(400, "bad_extension",
                         f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
    if not torch.cuda.is_available():
        return api_error(503, "gpu_unavailable",
                         "This service requires a CUDA-capable GPU and none is available.")
    if not os.path.exists(MODEL_GLB_PATH):
        return api_error(500, "server_misconfigured",
                         "Base model file is missing on the server.")

    job = Job(id=request_id, filename=secure_filename(uploaded_file.filename))
    JOBS[job.id] = job

    tmp_dir = tempfile.mkdtemp(prefix=f"mocap_{job.id}_")
    job.tmp_dir = tmp_dir
    file_path = os.path.join(tmp_dir, job.filename)

    try:
        with open(file_path, "wb+") as f:
            f.write(await uploaded_file.read())
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        JOBS.pop(job.id, None)
        log.exception("[%s] failed writing upload", job.id)
        return api_error(500, "upload_write_failed", str(e))

    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        JOBS.pop(job.id, None)
        return api_error(413, "file_too_large",
                         f"File exceeds the {MAX_UPLOAD_MB} MB limit.",
                         size_mb=round(size_mb, 1))

    update_job(job, status="running", stage="decoding",
               message="Checking video…", progress=2)

    progress_dict = get_manager().dict()
    asyncio.create_task(_process_job(job, file_path, progress_dict))

    return {"job_id": job.id, "status": "queued"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return api_error(404, "job_not_found", "Unknown or expired job.")
    return job.public()


@app.get("/jobs/{job_id}/result")
def get_result(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return api_error(404, "job_not_found", "Unknown or expired job.")
    if job.status == "error":
        return api_error(500, job.error_code or "mocap_failed",
                         job.error_message or "Conversion failed.")
    if job.status == "cancelled":
        return api_error(409, "cancelled", "Job was cancelled.")
    if job.status != "done" or not job.glb_path:
        return api_error(409, "not_ready", "Job has not finished yet.",
                         status=job.status, stage=job.stage)
    if not os.path.exists(job.glb_path):
        return api_error(410, "result_expired", "Result is no longer available.")

    out_name = f"{os.path.splitext(job.filename)[0]}.glb"
    return FileResponse(
        path=job.glb_path,
        media_type="model/gltf-binary",
        filename=out_name,
        headers={
            "X-Request-Id": job.id,
            "X-Frames-Total": str(job.frames_total),
            "X-Frames-With-Pose": str(job.frames_with_pose),
        },
    )


@app.delete("/jobs/{job_id}")
def cancel_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return api_error(404, "job_not_found", "Unknown or expired job.")
    if job.status in ("done", "error", "cancelled"):
        return {"status": job.status}
    job.cancel_requested = True
    update_job(job, status="cancelled", stage="cancelled",
               message="Cancelled", progress=0, finished_at=time.time())
    if job.tmp_dir:
        shutil.rmtree(job.tmp_dir, ignore_errors=True)
        job.tmp_dir = None
    return {"status": "cancelled"}


# --------------------------------------------------------------------------- #
# Job orchestration
# --------------------------------------------------------------------------- #

async def _process_job(job: Job, file_path: str, progress_dict) -> None:
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            cap.release()
            _fail(job, "undecodable_video",
                  "The file could not be opened as a video. Is it a valid MP4?")
            return
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if frame_count and frame_count < 2:
            _fail(job, "too_short", "The video must contain at least 2 frames.")
            return

        update_job(job, stage="loading", message="Loading pose models…",
                   progress=5, frames_total=frame_count)

        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            get_pool(), _run_frankmocap_worker,
            file_path, frame_count, progress_dict,
        )

        while not future.done():
            await asyncio.sleep(0.5)
            snap = dict(progress_dict)
            if snap:
                update_job(
                    job,
                    stage=snap.get("stage", job.stage),
                    message=snap.get("message", job.message),
                    progress=int(snap.get("progress", job.progress)),
                    frames_total=int(snap.get("frames_total", job.frames_total)),
                    frames_processed=int(snap.get("frames_processed", job.frames_processed)),
                    frames_with_pose=int(snap.get("frames_with_pose", job.frames_with_pose)),
                )
            if job.cancel_requested:
                return

        frames, stats = await future

        if job.cancel_requested:
            return

        update_job(job,
                   stage="building", message="Building 3D scene…",
                   progress=92,
                   frames_total=stats["frames_total"],
                   frames_processed=stats["frames_total"],
                   frames_with_pose=stats["frames_with_pose"])

        if stats["frames_with_pose"] < 1:
            _fail(job, "no_pose_detected",
                  "No human pose could be detected in the video. Make sure the "
                  "subject is clearly visible and well-lit.")
            return

        try:
            glb_path = generate_glb(frames, output_dir=job.tmp_dir)
        except Exception as e:
            log.exception("[%s] glb generation failed", job.id)
            _fail(job, "glb_generation_failed", str(e))
            return

        update_job(job, status="done", stage="done",
                   message="Ready", progress=100,
                   glb_path=glb_path, finished_at=time.time())
        log.info("[%s] done — %s", job.id, glb_path)

    except Exception as e:
        log.exception("[%s] unexpected failure", job.id)
        _fail(job, "mocap_failed", str(e))


def _fail(job: Job, code: str, message: str) -> None:
    update_job(job, status="error", stage="error",
               message=message, error_code=code, error_message=message,
               finished_at=time.time())
    if job.tmp_dir:
        shutil.rmtree(job.tmp_dir, ignore_errors=True)
        job.tmp_dir = None


# --------------------------------------------------------------------------- #
# Worker (runs in a separate process — no shared state with FastAPI)
# --------------------------------------------------------------------------- #

def _run_frankmocap_worker(
    file_path: str, frame_count: int, progress_dict,
) -> tuple[np.ndarray, dict]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available in the worker")

    device = torch.device("cuda")

    def report(**fields):
        try:
            progress_dict.update(fields)
        except Exception:
            pass

    report(stage="loading", message="Loading pose models…",
           progress=8, frames_total=frame_count,
           frames_processed=0, frames_with_pose=0)

    hand_bbox_detector = HandBboxDetector("third_view", device)
    body_mocap = BodyMocap(BODY_CHECKPOINT, SMPL_DIR, device=device, use_smplx=True)
    hand_mocap = HandMocap(HAND_CHECKPOINT, SMPL_DIR, device=device)

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {file_path}")

    fm_output: list[np.ndarray] = []
    frames_total = 0
    frames_with_pose = 0
    last_report = 0.0

    while True:
        ok, img_bgr = cap.read()
        if not ok or img_bgr is None:
            break

        frames_total += 1

        body_bbox_list, _, pred_output_list = run_regress(
            img_bgr, hand_bbox_detector, body_mocap, hand_mocap,
        )

        if len(body_bbox_list) >= 1:
            pred_output = extract_output(pred_output_list)
            if pred_output is not None:
                try:
                    output_list = np.concatenate([
                        pred_output["pred_body_pose"][0][:66],
                        pred_output["pred_left_hand_pose"][0],
                        pred_output["pred_right_hand_pose"][0],
                    ])
                    fm_output.append(output_list)
                    frames_with_pose += 1
                except (KeyError, IndexError):
                    pass

        now = time.time()
        if now - last_report > 0.5:
            last_report = now
            if frame_count > 0:
                pct = 10 + int(78 * frames_total / frame_count)
            else:
                pct = 10 + min(78, frames_total)
            report(
                stage="tracking",
                message=f"Tracking poses… {frames_with_pose} detected",
                progress=min(pct, 88),
                frames_total=frame_count or frames_total,
                frames_processed=frames_total,
                frames_with_pose=frames_with_pose,
            )

    cap.release()

    report(stage="integrating", message="Combining body & hands…",
           progress=90,
           frames_total=frames_total,
           frames_processed=frames_total,
           frames_with_pose=frames_with_pose)

    if not fm_output:
        return np.zeros((0, 162), dtype=np.float32), {
            "frames_total": frames_total,
            "frames_with_pose": 0,
        }

    frames = np.stack(fm_output).astype(np.float32)
    return frames, {
        "frames_total": frames_total,
        "frames_with_pose": frames_with_pose,
    }


# --------------------------------------------------------------------------- #
# GLB generation (unchanged logic)
# --------------------------------------------------------------------------- #

def generate_glb(frames: np.ndarray, output_dir: str) -> str:
    if frames.ndim != 2 or frames.shape[0] == 0:
        raise ValueError(f"Expected 2D frames array, got shape {frames.shape}")

    glb = GLTF2().load(MODEL_GLB_PATH)
    buffer = glb.binary_blob()
    offset = len(buffer)

    times = np.array([i / 12 for i in range(len(frames))], dtype=np.float32)
    times_bytes = times.tobytes()
    animation_data = times_bytes

    time_bv_index = len(glb.bufferViews)
    glb.bufferViews.append(BufferView(buffer=0, byteOffset=offset,
                                      byteLength=len(times_bytes)))

    time_accessor_idx = len(glb.accessors)
    glb.accessors.append(Accessor(
        bufferView=time_bv_index, byteOffset=0, componentType=FLOAT,
        type=SCALAR, min=[float(times.min())], max=[float(times.max())],
        count=len(times),
    ))
    offset += len(times_bytes)

    samplers: list[AnimationSampler] = []
    channels: list[AnimationChannel] = []

    for i in range(52):
        rotations = []
        for frame in frames:
            idx = START_INDICES[i]
            anim_aa = frame[idx: idx + 3]
            relaxed_aa = RELAXED_POSE[idx: idx + 3]
            rot = Rotation.from_rotvec(anim_aa) * Rotation.from_rotvec(relaxed_aa)
            rotations.append(rot.as_quat())

        rotations = np.asarray(rotations, dtype=np.float32)
        rotations_bytes = rotations.tobytes()
        animation_data += rotations_bytes

        rotation_bv_idx = len(glb.bufferViews)
        glb.bufferViews.append(BufferView(
            buffer=0, byteOffset=offset, byteLength=len(rotations_bytes)))

        rotation_accessor_idx = len(glb.accessors)
        glb.accessors.append(Accessor(
            bufferView=rotation_bv_idx, byteOffset=0, componentType=FLOAT,
            count=len(rotations), type=VEC4,
        ))

        samplers.append(AnimationSampler(
            input=time_accessor_idx, output=rotation_accessor_idx,
            interpolation=ANIM_LINEAR,
        ))
        channels.append(AnimationChannel(
            sampler=i, target=AnimationChannelTarget(node=i, path=ROTATION),
        ))
        offset += len(rotations_bytes)

    glb.animations.append(Animation(
        name="fm_animation", samplers=samplers, channels=channels,
    ))

    buffer = buffer + animation_data
    glb.set_binary_blob(buffer)
    glb.buffers[0].byteLength = len(buffer)

    path = os.path.join(output_dir, f"{uuid.uuid4().hex}.glb")
    glb.save(path)
    return path


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def run_regress(img_original_bgr, hand_bbox_detector, body_mocap, hand_mocap):
    _, body_bbox_list = hand_bbox_detector.detect_body_bbox(img_original_bgr.copy())
    if len(body_bbox_list) < 1:
        return [], [], []

    hand_bbox_list = [None] * len(body_bbox_list)
    body_bbox_list, _ = __filter_bbox_list(body_bbox_list, hand_bbox_list, True)

    pred_body_list = body_mocap.regress(img_original_bgr, body_bbox_list)
    assert len(body_bbox_list) == len(pred_body_list)

    hand_bbox_list = body_mocap.get_hand_bboxes(
        pred_body_list, img_original_bgr.shape[:2])
    assert len(pred_body_list) == len(hand_bbox_list)

    pred_hand_list = hand_mocap.regress(
        img_original_bgr, hand_bbox_list, add_margin=True)
    assert len(hand_bbox_list) == len(pred_hand_list)

    integral_output_list = integration_copy_paste(
        pred_body_list, pred_hand_list, body_mocap.smpl, img_original_bgr.shape)
    return body_bbox_list, hand_bbox_list, integral_output_list


def extract_output(pred_output_list):
    pred_output = pred_output_list[0] if pred_output_list else None
    if pred_output is None:
        return None

    saved = {}
    for key in pred_output:
        if key.find("vertices") < 0 or key == "faces":
            saved[key] = pred_output[key]
        elif key != "faces":
            saved[key] = pred_output[key].astype(np.float16)
        else:
            saved[key] = pred_output[key]
    return saved