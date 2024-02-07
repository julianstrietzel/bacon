In this subfolder structure you can find (or need to include) distance prediction networks for training sdf.  
Therefore clone the repositories of the networks you want to use into this folder.
(e.g. [training_sdf_network](https://github.com/julianstrietzel/training_sdf_estimators)
or [MeshCNN](https://github.com/julianstrietzel/MeshCNN])

For correct module resolution in pycharm mark this folder and all its direct childs as source root.
The correct folder structure should look like this:

```
df_prediction_networks
├── MeshCNN
├── training_sdf_estimators
├── opt.txt
└── SubReadMe.md
```

Use opt.txt in this directory as placeholder to configure the submodules, that can be read by the
[Inference Options](inf_options.py). Therefore, you need to replace the placeholders with the opt.txt print out of the
model configuration you want to use and place everything else in the checkpoints_directory of the repo.