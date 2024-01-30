import os
import sys
import time

import configargparse
import mcubes
import numpy as np
import torch
import trimesh
from torch import multiprocessing
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import dataio
from torch.utils.data import DataLoader

torch.set_num_threads(8)

p = configargparse.ArgumentParser()

# config file, output directories
p.add(
    "-c", "--config", required=False, is_config_file=True, help="Path to config file."
)
p.add_argument("--logging_root", type=str, default="../logs", help="root for logging")
p.add_argument(
    "--experiment_name",
    type=str,
    required=True,
    help="subdirectory in logging_root for checkpoints, summaries",
)

# general training
p.add_argument("--model_type", type=str, default="mfn", help="options: mfn, siren, ff")
p.add_argument("--hidden_size", type=int, default=128, help="size of hidden layer")
p.add_argument("--hidden_layers", type=int, default=8)
p.add_argument("--lr", type=float, default=1e-4, help="learning rate")
p.add_argument("--num_steps", type=int, default=20000, help="number of training steps")
p.add_argument(
    "--ckpt_step", type=int, default=0, help="step at which to resume training"
)
p.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
p.add_argument(
    "--seed", default=None, help="random seed for experiment reproducibility"
)

# mfn options
p.add_argument(
    "--multiscale", action="store_true", default=False, help="use multiscale"
)
p.add_argument(
    "--max_freq",
    type=int,
    default=512,
    help="The network-equivalent sample rate used to represent the signal."
    + "Should be at least twice the Nyquist frequency.",
)
p.add_argument(
    "--input_scales",
    nargs="*",
    type=float,
    default=None,
    help="fraction of resolution growth at each layer",
)
p.add_argument(
    "--output_layers",
    nargs="*",
    type=int,
    default=None,
    help="layer indices to output, beginning at 1",
)

# mlp options
p.add_argument("--w0", default=30, type=int, help="omega_0 parameter for siren")
p.add_argument("--pe_scale", default=5, type=float, help="positional encoding scale")

# sdf model and sampling
p.add_argument(
    "--num_pts_on",
    type=int,
    default=10000,
    help="number of on-surface points to sample",
)
p.add_argument(
    "--coarse_scale",
    type=float,
    default=1e-1,
    help="laplacian scale factor for coarse samples",
)
p.add_argument(
    "--fine_scale",
    type=float,
    default=1e-3,
    help="laplacian scale factor for fine samples",
)
p.add_argument(
    "--coarse_weight",
    type=float,
    default=1e-2,
    help="weight to apply to coarse loss samples",
)

# data i/o
p.add_argument(
    "--shape", type=str, default="bunny", help="name of point cloud shape in xyz format"
)
p.add_argument(
    "--point_cloud_path",
    type=str,
    default="../data/armadillo.xyz",
    help="path for input point cloud",
)
p.add_argument("--num_workers", default=0, type=int, help="number of workers")

# tensorboard summary
p.add_argument(
    "--steps_til_ckpt",
    type=int,
    default=50000,
    help="epoch frequency to save a checkpoint",
)
p.add_argument(
    "--steps_til_summary",
    type=int,
    default=1000,
    help="epoch frequency to update tensorboard summary",
)
# sdf_sampling_opt_path
p.add_argument(
    "--sdf_sampling_opt_path",
    type=str,
    default="./sdf_sampling_opt.txt",
    help="path to options file for submodels called for sdf sampling during training",
)
# surface_sampling_method
p.add_argument(
    "--surface_sampling_method",
    type=str,
    default="basic",
    help="method to use for surface sampling during training from [basic, sdf_network, mesh_cnn]",
)

opt = p.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)


def init_dataloader(opt, grid_size=128):
    """load sdf dataloader via eikonal equation or fitting sdf directly"""

    sdf_dataset = dataio.MeshSDF(
        opt.point_cloud_path,
        num_samples=opt.num_pts_on,
        coarse_scale=opt.coarse_scale,
        fine_scale=opt.fine_scale,
        opt=opt,
        debug_mcubes=True,
        debug_mcubes_resolution=grid_size,
    )

    dataloader = DataLoader(
        sdf_dataset,
        shuffle=False,
        batch_size=1,
        pin_memory=True,
        num_workers=opt.num_workers,
    )

    return dataloader


def render_mcubes_mesh(grid_size=128):
    dataloader = init_dataloader(opt, grid_size=grid_size)
    sdf_values = None
    train_gen = iter(dataloader)
    if int(grid_size**3 // opt.num_pts_on) != grid_size**3 / opt.num_pts_on:
        raise Warning("grid_size**3 is not divisible by num_pts_on")
    for i in tqdm(range(grid_size**3 // opt.num_pts_on)):
        batch = next(train_gen)
        if i == 0:
            sdf_values = batch[1]["sdf"][0, :, :]
            continue
        sdf_values = np.concatenate((sdf_values, batch[1]["sdf"][0, :, :]), axis=0)
    # Step 3: Reshape the SDF values into a 3D array
    if sdf_values.shape[0] != grid_size**3:
        raise Warning("SDF values shape does not match grid size")
    sdf_values_3d = sdf_values.reshape((grid_size, grid_size, grid_size))
    print(sdf_values_3d.shape)
    print("sampled")
    # Step 4: Pass the 3D array of SDF values to the Marching Cubes algorithm
    vertices, triangles = mcubes.marching_cubes(-sdf_values_3d, 0)
    # Create a trimesh object and adjust the vertices
    # points = (points - np.min(points)) / (np.max(points) - np.min(points)) - 0.5
    mesh = trimesh.Trimesh(vertices=vertices / grid_size - 0.5, faces=triangles)
    # Export the mesh
    os.makedirs("./outputs/direct_meshes", exist_ok=True)
    filename = (
        f"./outputs/direct_meshes/{os.path.basename(opt.point_cloud_path)}_{opt.experiment_name}_"
        f"{opt.surface_sampling_method}_{grid_size}_{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}.obj"
    )
    mesh.export(filename)
    print(f"Exported mesh to {filename}")


if __name__ == "__main__":
    # multiprocessing.set_start_method("spawn")

    for grid_size in [128]:
        render_mcubes_mesh(grid_size=grid_size)
