from pygltflib import (
    ANIM_LINEAR,
    GLTF2, 
    BufferView, 
    Accessor, 
    FLOAT, 
    SCALAR, 
    VEC4, 
    AnimationSampler, 
    LINEAR, 
    AnimationChannel,
    AnimationChannelTarget,
    ROTATION,
    Animation,
)

from pygltflib.validator import validate, summary
import numpy as np

gltf = GLTF2().load('model.glb')

times = np.array([0.0, 1.0, 2.0], dtype=np.float32)

rotations = np.array([
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.7071, 0.7071],
    [0.0, 0.0, 0.0, 1.0]
], dtype=np.float32)

time_bytes = times.tobytes()
rotation_bytes = rotations.tobytes()

animation_data = time_bytes + rotation_bytes

# Appending data to buffer
buffer = gltf.binary_blob()
offset = len(buffer)

buffer = buffer + animation_data

gltf.set_binary_blob(buffer)
gltf.buffers[0].byteLength = offset

# Creating buffer views
time_bv_index = len(gltf.bufferViews)
gltf.bufferViews.append(BufferView(
    buffer=0,
    byteOffset=offset,
    byteLength=len(time_bytes)
))

rotation_bv_idx = len(gltf.bufferViews)
gltf.bufferViews.append(BufferView(
    buffer=0,
    byteOffset=len(time_bytes) + offset,
    byteLength=len(rotation_bytes)
))

# Creating accessors
time_accessor_idx = len(gltf.accessors)
gltf.accessors.append(Accessor(
    bufferView=time_bv_index,
    byteOffset=0,
    componentType=FLOAT,
    type=SCALAR,
    min=[float(times.min())],
    max=[float(times.max())],
    count=len(times),
))

rotation_accessor_idx = len(gltf.accessors)
gltf.accessors.append(Accessor(
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
    sampler=0,
    target=AnimationChannelTarget(
        node=51,
        path=ROTATION
    )
)

animation = Animation(
    name="pelvis.quaternion",
    samplers=[sampler],
    channels=[channel]
)

gltf.animations.append(animation)

gltf.save("test.glb")
