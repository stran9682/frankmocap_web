import os
import shutil
import uuid
import tempfile
import asyncio
from concurrent.futures import ProcessPoolExecutor
from werkzeug.utils import secure_filename
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTasks

import cv2
import torch
import numpy as np

from frankmocap.integration.copy_and_paste import integration_copy_paste
from frankmocap.demo.demo_frankmocap import __filter_bbox_list
from frankmocap.handmocap.hand_mocap_api import HandMocap
from frankmocap.bodymocap.body_mocap_api import BodyMocap
from frankmocap.handmocap.hand_bbox_detector import HandBboxDetector

import math
from pygltflib import (
    ANIM_LINEAR,
    GLTF2, 
    BufferView, 
    Accessor, 
    FLOAT, 
    SCALAR, 
    VEC4, 
    AnimationSampler, 
    AnimationChannel,
    AnimationChannelTarget,
    ROTATION,
    Animation,
)

UPLOAD_FOLDER = 'temp'
ALLOWED_EXTENSIONS = {'mp4', 'MP4'}

START_INDICES = [
	30,21,12,3,
	33,24,15,6,
	45,36,
	72,69,66,
	81,78,75,
	90,87,84,
	99,96,93,
	108,105,102,
	60,54,48,39,
	117,114,111,
	126,123,120,
	135,132,129,
	144,141,138,
	153,150,147,
	63,57,51,42,
	27,18,9,0,
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"]
)

@app.post("/")
async def mocap(bg_tasks: BackgroundTasks, uploaded_file: UploadFile = File(...),):
    if not uploaded_file or not allowed_file(uploaded_file.filename):
        raise HTTPException(
           status_code=status.HTTP_400_BAD_REQUEST,
           detail="Incorrect File Type, only .mp4 supported"
        )

    filename = secure_filename(uploaded_file.filename)

    tmp_dir = tempfile.mkdtemp()

    try:
        file_path = os.path.join(tmp_dir, filename)

        with open(file_path, "wb+") as file_object:
            file_object.write(await uploaded_file.read())

        loop = asyncio.get_event_loop()

        with ProcessPoolExecutor() as pool:
            frames = await loop.run_in_executor(pool, run_frankmocap, file_path)

        glb_path = generate_glb(frames, output_dir= tmp_dir)

        bg_tasks.add_task(shutil.rmtree, tmp_dir)

        # Stream file back
        return FileResponse(
            path=glb_path,
            background=bg_tasks
        )

    except Exception as e:
        shutil.rmtree(tmp_dir)
        raise e

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_glb(frames, output_dir):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model.glb")

    glb = GLTF2().load(model_path)

    buffer = glb.binary_blob()
    offset = len(buffer)

    times = np.array([i / 12 for i in range(len(frames))], dtype=np.float32)
    times_bytes = times.tobytes()

    animation_data = times_bytes

    time_bv_index = len(glb.bufferViews)
    glb.bufferViews.append(BufferView(
        buffer=0,
        byteOffset=offset,
        byteLength=len(times_bytes)
    ))

    time_accessor_idx = len(glb.accessors)
    glb.accessors.append(Accessor(
        bufferView=time_bv_index,
        byteOffset=0,
        componentType=FLOAT,
        type=SCALAR,
        min=[float(times.min())],
        max=[float(times.max())],
        count=len(times),
    ))

    offset += len(times_bytes) 

    samplers = []
    channels = []

    # animate each joint
    for i in range(52):

        # extracting rotation data from each frame and converting from axis angle to quaternion form
        rotations = []
        for frame in frames:
            theta = math.sqrt(frame[START_INDICES[i]]**2 + frame[START_INDICES[i]+1]**2 + frame[START_INDICES[i]+2]**2)
            try:
                w = math.cos(theta / 2)
                x = frame[START_INDICES[i]] * math.sin(theta / 2)
                y = frame[START_INDICES[i]+1] * math.sin(theta / 2)
                z = frame[START_INDICES[i]+2] * math.sin(theta / 2)

                rotations.append([x, y, z, w])
            except ZeroDivisionError:
                rotations.append[[0,0,0,0]]


        rotations = np.array(rotations, dtype=np.float32)
        rotations_bytes = rotations.tobytes()
        
        animation_data += rotations_bytes

        # Creating buffer views
        rotation_bv_idx = len(glb.bufferViews)
        glb.bufferViews.append(BufferView(
            buffer=0,
            byteOffset=offset,
            byteLength=len(rotations_bytes)
        ))

        rotation_accessor_idx = len(glb.accessors)
        glb.accessors.append(Accessor(
            bufferView=rotation_bv_idx,
            byteOffset=0,
            componentType=FLOAT,
            count=len(rotations),
            type=VEC4
        ))

        # GLTF animation channel
        sampler = AnimationSampler(
            input=time_accessor_idx,
            output=rotation_accessor_idx,
            interpolation=ANIM_LINEAR
        )

        channel = AnimationChannel(
            sampler=i,
            target=AnimationChannelTarget(
                node=i,
                path=ROTATION
            )
        )

        samplers.append(sampler)
        channels.append(channel)
        
        offset += len(rotations_bytes)


    # Appending data
    animation = Animation(
        name="fm_animation",
        samplers=samplers,
        channels=channels
    )

    glb.animations.append(animation)

    buffer = buffer + animation_data

    glb.set_binary_blob(buffer)
    glb.buffers[0].byteLength = len(buffer)

    path = os.path.join(output_dir,  f"{uuid.uuid4().hex}.glb")
    glb.save(path)

    return path

def run_frankmocap(file_path):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    assert torch.cuda.is_available(), "Current version only supports GPU"

    hand_bbox_detector =  HandBboxDetector('third_view', device) 

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    extra_data_dir = os.path.abspath(os.path.join(CURRENT_DIR, "frankmocap", "extra_data"))
    smpl_dir = os.path.abspath(os.path.join(CURRENT_DIR, "frankmocap", "extra_data", "smpl"))

    default_checkpoint_body_smplx = os.path.join(
        os.path.join(extra_data_dir, "body_module", "pretrained_weights"), 
        "smplx-03-28-46060-w_spin_mlc3d_46582-2089_2020_03_28-21_56_16.pt"
    )
    body_mocap = BodyMocap(default_checkpoint_body_smplx, smpl_dir, device = device, use_smplx= True)

    default_checkpoint_hand = os.path.join(
        os.path.join(extra_data_dir, "hand_module", "pretrained_weights"), 
        "pose_shape_best.pth"
    )
    hand_mocap = HandMocap(default_checkpoint_hand, smpl_dir, device = device)

    input_data = cv2.VideoCapture(file_path)

    fm_output = []
    while True:
        _, img_original_bgr = input_data.read()
        if img_original_bgr is None:
            break

        body_bbox_list, _, pred_output_list = run_regress(
            img_original_bgr, 
            hand_bbox_detector,
            body_mocap, 
            hand_mocap
        )

        if len(body_bbox_list) < 1: 
            continue

        pred_output = extract_output(pred_output_list)

        print(pred_output['pred_left_hand_pose'][0])

        output_list = np.concatenate([
            pred_output['pred_body_pose'][0][:3],
            pred_output['pred_body_pose'][0][3:66],
            pred_output['pred_left_hand_pose'][0],
            pred_output['pred_right_hand_pose'][0]
        ])

        fm_output.append(output_list)

    frames = np.array(fm_output)

    return frames
  
def run_regress(
    img_original_bgr, 
    hand_bbox_detector,
    body_mocap, 
    hand_mocap
):
    _, body_bbox_list = hand_bbox_detector.detect_body_bbox(img_original_bgr.copy())

    if len(body_bbox_list) < 1: 
        return list(), list(), list()
    
    # sort the bbox using bbox size 
    # only keep on bbox if args.single_person is set
    hand_bbox_list = [None, ] * len(body_bbox_list)
    body_bbox_list, _ = __filter_bbox_list(
        body_bbox_list, hand_bbox_list, True)

    # body regression first 
    pred_body_list = body_mocap.regress(img_original_bgr, body_bbox_list)
    assert len(body_bbox_list) == len(pred_body_list)

    # get hand bbox from body
    hand_bbox_list = body_mocap.get_hand_bboxes(pred_body_list, img_original_bgr.shape[:2])
    assert len(pred_body_list) == len(hand_bbox_list)

    # hand regression
    pred_hand_list = hand_mocap.regress(
        img_original_bgr, hand_bbox_list, add_margin=True)
    assert len(hand_bbox_list) == len(pred_hand_list) 

    # integration by copy-and-paste
    integral_output_list = integration_copy_paste(
        pred_body_list, pred_hand_list, body_mocap.smpl, img_original_bgr.shape)
    
    return body_bbox_list, hand_bbox_list, integral_output_list

def extract_output(pred_output_list):

    pred_output = pred_output_list[0]
    if pred_output is None:
        return None
    else: 
        saved_pred_output = dict()
        for pred_key in pred_output:
            if pred_key.find("vertices")<0 or pred_key == 'faces' :
                saved_pred_output[pred_key] = pred_output[pred_key]
            else:
                if pred_key != 'faces':
                    saved_pred_output[pred_key] = \
                        pred_output[pred_key].astype(np.float16)
                else:
                    saved_pred_output[pred_key] = pred_output[pred_key]

        return saved_pred_output