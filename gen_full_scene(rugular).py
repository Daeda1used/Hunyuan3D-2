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
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FaceReducer, FloaterRemover, DegenerateFaceRemover
from hy3dgen.text2image import HunyuanDiTPipeline


def image_to_3d(image_list, graspmodel_str):
    """
    处理一组图片，并生成对应的 3D 模型。

    Args:
        image_list (list): 包含要处理的图片路径的列表。
        graspmodel_str (str): 当前场景的 scene_index。
    """
    rembg = BackgroundRemover()
    model_path = 'tencent/Hunyuan3D-2'
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)

    mesh_output_dir = "/home/WorkSpace/PointCloud_Reconstruction/Hunyuan3D-2/assets/outputs/GraspNet1B_Scene_Mesh"
    os.makedirs(mesh_output_dir, exist_ok=True)

    for image_path, view_index in image_list:
        image = Image.open(image_path)

        if image.mode == 'RGB':
            image = rembg(image)
            image.show()


        # 构建输出文件路径
        mesh_filename = f"{graspmodel_str}_view_{view_index}.glb"
        mesh_output_path = os.path.join(mesh_output_dir, mesh_filename)

        # 检查文件是否已存在
        if os.path.exists(mesh_output_path):
            print(f"✅ 已存在: {mesh_output_path}，跳过生成。")
            continue

        # 生成 3D 网格
        mesh = pipeline(image=image, graspmodel_str=graspmodel_str, view_index=view_index, num_inference_steps=30,
                        mc_algo='mc',
                        generator=torch.manual_seed(2025))[0]

        mesh = FloaterRemover()(mesh)
        mesh = DegenerateFaceRemover()(mesh)
        mesh = FaceReducer()(mesh)

        # 导出网格
        mesh.export(mesh_output_path)
        print(f"Mesh exported to {mesh_output_path}")

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

def process_scene(scene_folder, base_dataset_folder):
    """
    处理单个场景文件夹中的图片。

    Args:
        scene_folder (str): 场景文件夹名称，例如 "scene_0100"。
        base_dataset_folder (str): 数据集的根目录。
    """
    scene_index = int(scene_folder.split("_")[1])
    rgb_folder = os.path.join(base_dataset_folder, scene_folder, "kinect", "rgb")

    if not os.path.exists(rgb_folder):
        print(f"⚠目录不存在: {rgb_folder}，跳过。")
        return

    # 获取该文件夹下所有图片
    image_files = sorted([f for f in os.listdir(rgb_folder) if f.endswith(".png")])

    # 验证图片总数是否为 256
    if len(image_files) != 256:
        print(f"⚠scene_{scene_index} 的图片总数为 {len(image_files)}，而非 256，跳过。")
        return

    # 构建图片路径和对应 view_index 的列表
    image_list = [
        (os.path.join(rgb_folder, image_file), os.path.splitext(image_file)[0])  # (图片路径, view_index)
        for image_file in image_files
    ]

    # 调用 image_to_3d 方法处理整个列表
    print(f"Processing scene_{scene_index} with {len(image_list)} images.")
    image_to_3d(image_list, str(scene_index))

if __name__ == '__main__':
    # 数据集的根目录
    base_dataset_folder = '/home/WorkSpace/datasets/GraspNet-1B/scenes'

    # 获取并排序所有场景文件夹
    scene_folders = sorted(
        [f for f in os.listdir(base_dataset_folder) if f.startswith("scene_")],
        key=lambda x: int(x.split("_")[1])  # 按照 scene_xxxx 中的数字部分排序
    )

    # 限定范围 scene index [100, 189]
    scene_folders = [f for f in scene_folders if 100 <= int(f.split("_")[1]) <= 189]

    print(f"Found {len(scene_folders)} scenes to process.")
    print("Processing sequentially...")  # 提示正在使用顺序处理

    # 顺序处理每个场景
    for scene_folder in scene_folders:
        process_scene(scene_folder, base_dataset_folder)

    print('All done')


# text_to_3d()
