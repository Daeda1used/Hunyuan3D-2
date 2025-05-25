# Open Source Model Licensed under the Apache License Version 2.0
# and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited
# ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.

# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software and/or models in this distribution may have been
# modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import torch
import os
from PIL import Image

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FaceReducer, FloaterRemover, DegenerateFaceRemover
from hy3dgen.text2image import HunyuanDiTPipeline


def image_to_3d(image_path, graspmodel_str, view_index):
    rembg = BackgroundRemover()
    model_path = 'tencent/Hunyuan3D-2'

    image = Image.open(image_path)

    if image.mode == 'RGB':
        image = rembg(image)

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)

    mesh = pipeline(image=image, graspmodel_str = graspmodel_str, view_index = view_index, num_inference_steps=30, mc_algo='mc',
                    generator=torch.manual_seed(2025))[0]

    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh)

    mesh_output_dir = "/home/WorkSpace/PointCloud_Reconstruction/Hunyuan3D-2/assets/outputs/GraspNet1b_mesh"
    mesh_filename = f"{graspmodel_str}_view_{view_index}.glb"
    mesh_output_path = os.path.join(mesh_output_dir, mesh_filename)

    mesh.export(mesh_output_path)
    print('Mesh exported to {}'.format(mesh_output_path))

    # try:
    #     from hy3dgen.texgen import Hunyuan3DPaintPipeline
    #     pipeline = Hunyuan3DPaintPipeline.from_pretrained(model_path)
    #     mesh = pipeline(mesh, image=image)
    #     mesh.export('texture.glb')
    # except Exception as e:
    #     print(e)
    #     print('Please try to install requirements by following README.md')


def text_to_3d(prompt='a car'):
    rembg = BackgroundRemover()
    t2i = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled')
    model_path = 'tencent/Hunyuan3D-2'
    i23d = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)

    image = t2i(prompt)
    image = rembg(image)
    mesh = i23d(image, num_inference_steps=30, mc_algo='mc')[0]
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh)
    mesh.export('t2i_demo.glb')


if __name__ == '__main__':
    base_image_folder = '/home/WorkSpace/PointCloud_Reconstruction/Hunyuan3D-2/assets/GraspNet_obj_MultiViewImg'

    for graspmodel_index in range(29,88):  # 遍历 000 到 087
        graspmodel_str = f"{graspmodel_index:03d}"  # 格式化成 3 位数
        graspmodel_path = os.path.join(base_image_folder, graspmodel_str)

        if not os.path.exists(graspmodel_path):
            print(f"目录不存在: {graspmodel_path}，跳过。")
            continue

        for view_index in range(6,20):  # 遍历 view_00.png 到 view_19.png
            two_dig_view_index = f"{view_index:02d}"

            view_str = f"view_{view_index:02d}.png"
            image_path = os.path.join(graspmodel_path, view_str)
            print('Processing:', image_path)

            if not os.path.exists(image_path):
                print(f"⚠文件缺失: {image_path}，跳过。")
                continue

            # 传入 model_index 和 view_index 调用 image_to_3d()
            image_to_3d(image_path, graspmodel_str, two_dig_view_index)
    print('all done')

    # text_to_3d()
