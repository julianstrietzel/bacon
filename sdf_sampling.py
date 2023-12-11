import os.path

import numpy as np
import torch
import sys

# is needed to import from the submodules (in pycharm it is not needed and equivalent to "Mark Directory as Sources Root")
sys.path.append(os.path.realpath("../df_prediction_networks"))


def surface_sampling_method_factory(
    method, vectors, normals, kdtree, upper_options
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

    if method == "basic" or not method:

        def basic_surface_sampling(points):
            # Use KdTree to find the nearest neighbors
            _, idx = kdtree.query(points, k=3)
            # Average the normals of the neighbors
            avg_normal = np.mean(normals[idx], axis=1)
            # Sum over dot product
            sdf = np.sum((points - vectors[idx][:, 0]) * avg_normal, axis=-1)
            sdf = sdf[..., None]
            return sdf

        return basic_surface_sampling

    sys.path.append(
        os.path.realpath("../df_prediction_networks/training_sdf_estimators")
    )
    from options.inf_options import InferenceOptions

    sub_model_options = InferenceOptions(upper_options.sdf_sampling_opt_path).parse()

    if method == "mesh_cnn":
        sys.path.append(os.path.realpath("../df_prediction_networks/MeshCNN"))
        from MeshCNN.data.sdf_regression_data import RegressionDataset
        from MeshCNN.data import collate_fn
        from MeshCNN.models import create_model

        obj_path = upper_options.point_cloud_path.split(".")[0] + ".obj"
        if not os.path.exists(obj_path):
            raise FileNotFoundError(
                "No obj file found for the given point cloud. "
                "There has to be a .obj file at the same location as the .xyz file."
            )
        dataset = RegressionDataset(sub_model_options, path=obj_path)
        pos_encoder = dataset.positional_encoder
        from MeshCNN.models.layers.mesh import Mesh

        mesh = Mesh(
            file=obj_path,
            opt=sub_model_options,
            hold_history=False,
            export_folder=sub_model_options.export_folder,
        )
        normed_edge_features = dataset.get_normed_edge_features(mesh)
        model = create_model(sub_model_options)

        def mesh_cnn_sampling(points):
            # get positional encoding
            import torch

            pos_encoded_points = pos_encoder.forward(torch.from_numpy(points)).float()
            positional_encoded_point_repeated = np.repeat(
                np.expand_dims(pos_encoded_points, 2), 750, axis=1
            )
            batch = [
                {
                    "mesh": mesh,
                    "label": 0,
                    "edge_features": np.concatenate(
                        (
                            normed_edge_features,
                            positional_encoded_point_repeated[i, ...],
                        ),
                        axis=0,
                    ),
                }
                for i in range(points.shape[0])
            ]
            batched_meta = collate_fn(batch)
            model.set_input(batched_meta)
            with torch.no_grad():
                sdf = model.forward().data.cpu().numpy()
            return sdf

        return mesh_cnn_sampling

    if method == "sdf_network":
        sys.path.append(
            os.path.realpath("../df_prediction_networks/training_sdf_estimators")
        )
        from training_sdf_estimators.models import model_factory

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
            sdf = model_output.detach().numpy()[..., None]
            return sdf

        return sdf_network_surface_sampling
    raise NotImplementedError(f"Unknown surface sampling method{method}")
