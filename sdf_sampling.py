import os.path

import numpy as np
import torch
import sys

# is needed to import from the submodules (in pycharm it is not needed and equivalent to "Mark Directory as Sources Root")
sys.path.append(os.path.realpath("../df_prediction_networks"))


def surface_sampling_method_factory(
    method, vectors, normals, kdtree=None, upper_options=None
) -> callable:
    """
    Returns a function that can be used to sample the surface of a mesh from different methods to calc the distance
    to a surface mesh.
    :param method: The method to use for surface sampling string from [basic, sdf_network, mesh_cnn]
    :param vectors: The vectors of the mesh (all vertices indexed by idx later)
    :param normals: The normals of the mesh (all vertices indexed by idx later)
    :param kdtree: The kdtree of the mesh to find the nearest neighbors (their idxs) of a point
    :param upper_options: The options object of the upper level model (in distinction to the sub model options)
    """

    def basic_surface_sampling(points):
        # Use KdTree to find the nearest neighbors
        _, idx = kdtree.query(points, k=3)
        # Average the normals of the neighbors
        avg_normal = np.mean(normals[idx], axis=1)
        # Sum over dot product
        sdf = np.sum((points - vectors[idx][:, 0]) * avg_normal, axis=-1)
        sdf = sdf[..., None]
        return sdf

    if method == "basic" or not method:
        return basic_surface_sampling

    if method == "mesh_cnn":
        sys.path.append(os.path.realpath("../df_prediction_networks/MeshCNN"))
        from df_prediction_networks import inf_options

        sub_model_options = inf_options.InferenceOptions(
            upper_options.sdf_sampling_opt_path
        ).parse()
        from MeshCNN.data.sdf_regression_data import RegressionDataset
        from MeshCNN.models import create_model

        obj_path = os.path.realpath(upper_options.point_cloud_path).replace(
            ".xyz", ".obj"
        )

        if not os.path.exists(obj_path):
            raise FileNotFoundError(
                "No obj file found for the given point cloud. "
                "There has to be a .obj file at the same location as the .xyz file when using MeshCNN."
                f"Expected path: {obj_path}"
            )
        dataset = RegressionDataset(sub_model_options, path=obj_path)
        pos_encoder = dataset.positional_encoder
        from MeshCNN.models.layers.mesh import Mesh

        mesh = Mesh(
            file=obj_path,
            opt=sub_model_options,
            hold_history=False,
            export_folder=None,
        )
        edge_features = dataset.get_edge_features(mesh)
        model = create_model(sub_model_options)

        batched_mesh = np.array([mesh] * upper_options.num_pts_on)
        edge_features_batched = np.repeat(
            np.expand_dims(edge_features, 0), upper_options.num_pts_on, axis=0
        )
        import torch

        edge_features_batched = torch.from_numpy(edge_features_batched).float()

        def mesh_cnn_sampling(points):
            pos_encoded_points = pos_encoder.forward(torch.from_numpy(points)).float()
            expanded_pos_encoded_points = torch.unsqueeze(pos_encoded_points, 2)
            positional_encoded_point_repeated = expanded_pos_encoded_points.repeat(
                1, 1, edge_features.shape[1]
            )
            all_edge_features = torch.cat(
                (edge_features_batched, positional_encoded_point_repeated),
                dim=1,
            )
            batched_meta = {
                "mesh": batched_mesh,
                "edge_features": all_edge_features,
            }
            model.set_input(batched_meta, inference=True)
            with torch.no_grad():
                sdf = model.forward().data.cpu().numpy()
            # for debugging purposes: np.mean(np.abs(sdf, sdf_simple)) should not be significantly different from test mae
            # sdf_simple = basic_surface_sampling(points)
            return sdf

        return mesh_cnn_sampling
    if method == "sdf_network":
        sys.path.append(
            os.path.realpath("../df_prediction_networks/training_sdf_estimators")
        )
        from df_prediction_networks import inf_options
        from training_sdf_estimators.models import model_factory
        import torch

        sub_model_options = inf_options.InferenceOptions(
            upper_options.sdf_sampling_opt_path, "training_sdf_estimators"
        ).parse()

        model = model_factory(sub_model_options.model_name, sub_model_options)
        model.load_weights()
        model = model.eval()

        def sdf_network_surface_sampling(points):
            _, idx = kdtree.query(points, k=sub_model_options.kdtree_num_samples)
            closest_points = vectors[idx]
            relative_batch = closest_points - points[:, None, :]
            model_input = None
            if sub_model_options.model_name == "simplest_regression_model":
                model_input = torch.from_numpy(
                    relative_batch.astype("float32")
                ).flatten(
                    start_dim=1
                )  # shape (batch_size, 3 * kdtree_num_samples)
            elif sub_model_options.model_name == "convo_regression_model":
                normals_batch = normals[idx]
                model_input = np.concatenate(
                    (relative_batch, normals_batch), axis=2
                ).T.astype("float32")
            model_output = model(model_input)
            sdf = model_output.detach().numpy()
            sdf_simple = basic_surface_sampling(points)

            return sdf

        return sdf_network_surface_sampling
    raise NotImplementedError(f"Unknown surface sampling method{method}")


class MeshCNNSampler:
    def __init__(self, pos_encoder, normed_edge_features_batched, batched_mesh, model):
        self.pos_encoder = pos_encoder
        self.normed_edge_features_batched = normed_edge_features_batched
        self.batched_mesh = batched_mesh
        self.model = model

    def __call__(self, points):
        return self.sample(points)

    def sample(self, points):
        pos_encoded_points = self.pos_encoder.forward(torch.from_numpy(points)).float()
        expanded_pos_encoded_points = torch.unsqueeze(pos_encoded_points, 2)
        positional_encoded_point_repeated = expanded_pos_encoded_points.repeat(
            1, 1, self.normed_edge_features_batched.shape[2]
        )
        all_edge_features = torch.cat(
            (self.normed_edge_features_batched, positional_encoded_point_repeated),
            dim=1,
        )
        batched_meta = {
            "mesh": self.batched_mesh,
            "edge_features": all_edge_features,
        }
        self.model.set_input(batched_meta, inference=True)
        with torch.no_grad():
            sdf = self.model.forward().data.cpu().numpy()
        return sdf
