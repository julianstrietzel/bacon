import os

ablations = os.listdir("../df_prediction_networks/MeshCNN/ablations")
upper_config = " --config ./config/mcubes_debugging/ff_armadillo_T54_meshcnn_extractor.ini --sdf_sampling_opt_path "

for ablation in ablations:
    for run in os.listdir("../df_prediction_networks/MeshCNN/ablations" + "/" + ablation):
        sdf_sampling_opt = "../df_prediction_networks/MeshCNN/ablations/" + ablation + "/" + run + "/opt.txt --experiment_name " + ablation
        os.system("python ./debug_meshcnn_mcubes.py" + upper_config + sdf_sampling_opt)