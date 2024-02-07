import json
import torch
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance, mesh_normal_consistency
import trimesh
import numpy as np
import os

# Path to input mesh
input_mesh = trimesh.load_mesh("df_prediction_networks/MeshCNN/datasets/armadillo_shrec/T54.obj")
input_xyz = torch.from_numpy(input_mesh.vertices)

# Path to reconstructed meshed
reconstructed_mesh_path = "/tmp/pycharm_project_81/experiments/outputs/direct_meshes/"

l2_distances = {}
l1_distances = {}

for mesh in os.listdir(reconstructed_mesh_path):
    try:
        reconstructed_mesh = trimesh.load_mesh("/tmp/pycharm_project_81/experiments/outputs/direct_meshes/" + mesh)
    except:
        print("cannot open mesh {0}".format(mesh))
        continue

    sub_sampled_reconstructed, _ = trimesh.sample.sample_surface(reconstructed_mesh, input_xyz.shape[0]) # making two point clouds of the same size
    sub_sampled_reconstructed = torch.from_numpy(sub_sampled_reconstructed)

    batch_size = 1

    l2_distance, _ = chamfer_distance(input_xyz.unsqueeze(0).float(), sub_sampled_reconstructed.unsqueeze(0).float(),
                                      norm=2)
    l1_distance, _ = chamfer_distance(input_xyz.unsqueeze(0).float(), sub_sampled_reconstructed.unsqueeze(0).float(),
                                      norm=1)

    l2_distances[mesh] = l2_distance.item()
    l1_distances[mesh] = l1_distance.item()

    print("Chamfer Distance:", l2_distance.item(), l1_distance.item())

with open("l2_ablations_distances.txt", "w") as f:
    f.write(json.dumps(l2_distances))

with open("l1_abalations_distances.txt", "w") as f:
    f.write(json.dumps(l1_distances))
