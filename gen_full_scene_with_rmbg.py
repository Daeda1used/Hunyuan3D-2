import torch
import os
from PIL import Image
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FaceReducer, FloaterRemover, DegenerateFaceRemover
from hy3dgen.text2image import HunyuanDiTPipeline


def image_to_3d(image_list, graspmodel_str):
    rembg = BackgroundRemover()
    model_path = 'tencent/Hunyuan3D-2'
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)

    mesh_output_dir = "/home/WorkSpace/PointCloud_Reconstruction/Hunyuan3D-2/assets/outputs/GraspNet1B_Scene_Mesh"
    os.makedirs(mesh_output_dir, exist_ok=True)

    for image_path, scene_index, view_index in image_list:
        image = Image.open(image_path)

        if image.mode == 'RGB':
            image = rembg(image)
            image.show()

        mesh_filename = f"{scene_index}_view_{view_index}.glb"
        mesh_output_path = os.path.join(mesh_output_dir, mesh_filename)

        if os.path.exists(mesh_output_path):
            print(f"✅ 已存在: {mesh_output_path}，跳过生成。")
            continue

        mesh = pipeline(image=image, graspmodel_str=str(scene_index), view_index=view_index, num_inference_steps=30,
                        mc_algo='mc',
                        generator=torch.manual_seed(2025))[0]

        mesh = FloaterRemover()(mesh)
        mesh = DegenerateFaceRemover()(mesh)
        mesh = FaceReducer()(mesh)

        mesh.export(mesh_output_path)
        print(f"Mesh exported to {mesh_output_path}")


def process_all_images(base_dataset_folder, scene_folders):
    image_indices = [f"{i:04d}.png" for i in range(256)]

    for image_index in image_indices:
        image_list = []

        for scene_folder in scene_folders:
            scene_index = int(scene_folder.split("_")[1])
            image_path = os.path.join(base_dataset_folder, scene_folder, image_index)

            if os.path.exists(image_path):
                image_list.append((image_path, scene_index, image_index[:-4]))

        if image_list:
            print(f"Processing {image_index} for all scenes.")
            image_to_3d(image_list, image_index[:-4])


if __name__ == '__main__':
    base_dataset_folder = '/home/WorkSpace/datasets/GraspNet-1B/GraspNet1B_Rgbs_RMBG'

    scene_folders = sorted(
        [f for f in os.listdir(base_dataset_folder) if f.startswith("scene_")],
        key=lambda x: int(x.split("_")[1])
    )

    # 只处理指定的 scene index
    scene_folders = [f for f in scene_folders if int(f.split("_")[1]) in {152}]


    print(f"Found {len(scene_folders)} scenes to process.")
    print("Processing images by index...")

    process_all_images(base_dataset_folder, scene_folders)

    print('All done')
