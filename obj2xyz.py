import os

import fire
import numpy as np
import trimesh


def convert(
    path_to_obj: str = "./3d_objects/sponza/sponza.ply",
    num_samples: int = 10000000,
    normalize: bool = False,
    destroy_normals_partil: float = 0.0,
):
    print(f"Sampling {num_samples} points from {path_to_obj} at {os.getcwd()}")
    # np.genfromtxt("./data/gt_sponza_50kk.xyz")
    name = path_to_obj.split("/")[-1].split(".")[0]
    mesh = trimesh.load_mesh(path_to_obj)
    print("Start sampling")
    points, faces = mesh.sample(num_samples, return_index=True)
    print("Sampling finished -> Normals")
    if normalize:
        points = (points - np.min(points)) / (
            np.max(points) - np.min(points)
        ) * 0.9 - 0.45
    normals = mesh.face_normals[faces]
    print("Normals finished -> Destroying")
    # generate random mask to destroy normals in dimension of normals
    inverting_mask = np.random.rand(normals.shape[0]) < destroy_normals_partil
    inverting_mask = inverting_mask[:, np.newaxis].repeat(3, axis=1)
    # invert the normals where the mask is true
    destroyed_normals = np.where(inverting_mask, -normals, normals)

    print("Concatenate")
    xyz_points = np.concatenate((points, destroyed_normals), axis=1)
    print("Saving")
    i = 0
    while num_samples / (1000**i) > 1000:
        i += 1
    filename = f"gt_{name}_{(num_samples / (1000 ** i))}{'k' * i}.xyz"
    # Save the points as a .xyz file
    np.savetxt(filename, xyz_points, delimiter=" ", fmt="%0.6f")
    print(f"Saved {len(xyz_points)} points to {filename}")


if __name__ == "__main__":
    fire.Fire(convert)
