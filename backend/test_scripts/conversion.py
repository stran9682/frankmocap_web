import numpy as np
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

# Joint, Node ID, start index
# left_foot 0 30
# left_ankle 1 21
# left_knee 2 12
# left_hip 3 3
# right_foot 4 33
# right_ankle 5 24
# right_knee 6 15
# right_hip 7 6
# head 8 45
# neck 9 36
# left_index3 10 72
# left_index2 11 69
# left_index1 12 66
# left_middle3 13 81
# left_middle2 14 78
# left_middle1 15 75
# left_pinky3 16 90
# left_pinky2 17 87
# left_pinky1 18 84
# left_ring3 19 99
# left_ring2 20 96
# left_ring1 21 93
# left_thumb3 22 108
# left_thumb2 23 105
# left_thumb1 24 102
# left_wrist 25 60
# left_elbow 26 54
# left_shoulder 27 48
# left_collar 28 39
# right_index3 29 117
# right_index2 30 114
# right_index1 31 111
# right_middle3 32 126
# right_middle2 33 123
# right_middle1 34 120
# right_pinky3 35 135
# right_pinky2 36 132
# right_pinky1 37 129
# right_ring3 38 144
# right_ring2 39 141
# right_ring1 40 138
# right_thumb3 41 153
# right_thumb2 42 150
# right_thumb1 43 147
# right_wrist 44 63
# right_elbow 45 57
# right_shoulder 46 51
# right_collar 47 42
# spine3 48 27
# spine2 49 18
# spine1 50 9
# pelvis 51 0

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

fm_data = np.load("animation.npz")
frames = fm_data['poses']

glb = GLTF2().load('model.glb')

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
    
glb.save("fm_animation.glb")