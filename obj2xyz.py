import trimesh
import igl
import numpy as np
mesh = trimesh.load("../data/sponza.obj", force='mesh')
n_points = 100000
points, faces = mesh.sample(n_points, return_index=True)
normals = mesh.face_normals[faces]
points_norms = np.hstack((points, normals))
np.savetxt("../data/gt_sponza.xyz", points_norms, newline="\n", delimiter=" ")

