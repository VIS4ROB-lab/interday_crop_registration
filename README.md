# Aerial Image-based Inter-day Registration for Precision Agriculture

If you use this code in your academic work, please cite ([PDF](https://www.research-collection.ethz.ch/handle/20.500.11850/662288)):

    @inproceedings{gao2024interday,
      title={Aerial Image-based Inter-day Registration for Precision Agriculture},
      author={Gao, Chen and Daxinger, Franz and Roth, Lukas and Maffra, Fabiola and Beardsley, Paul and Chli, Margarita and Teixeira, Lucas},
      booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
      year={2024}
    }


## Video
<a href="https://youtu.be/RItJI8JfZsQ" target="_blank"><img src="http://img.youtube.com/vi/RItJI8JfZsQ/0.jpg" alt="Mesh" width="480" height="360" border="0" /></a>

# Repo structure
This repo includes code for two tasks: crop alignment and train LoFTR for crop images. For clarity, we gather the relevant codes in different folders.

* [dataset](./dataset/): The dataset (except images and depth images) needed to excute the registration pipeline and train LoFTR. A script needed to run before retraining LoFTR is also included here.

* [crop_alignment](./crop_alignment/): The pipeline to compute 3D models of a crop field and to properly register them. Includes the codes of the experiments done in the paper. This relies on the repository [Hierarchical_Localization](https://github.com/cvg/Hierarchical-Localization) for performing localization between query and reference images.

* [LoFTR](./LoFTR/): Code for training LoFTR. Modified to account for height variations between days. The code is based on the implementation of [original LoFTR](https://github.com/zju3dv/LoFTR).

Please go to each subdirectory for further information.

# Installations
Retrieve the repo ans submodules first. Then please refer to the instructions of each individual tasks.
```bash
git clone --recursive https://github.com/VIS4ROB-lab/interday_crop_registration/

# This repository includes external local features as git submodules, don't forget to pull submodules 
cd interday_crop_registration
git submodule update --init --recursive`
```

For crop alignment, please follow the steps [here](./crop_alignment/README.md#installation).

For LoFTR, please follow the steps [here](./LoFTR/README.md#installation).

# Download datasets
The `dataset` folder in this repo does not contain original images, generated depth images. 
Before starting to run certain tasks, you need to download the corresponding images from the datasets that are not included directly in this repo. 

* To retrain LoFTR, the images and depth images of the Wheat Dataset from 2018 are needed. Download and extract the files below:

  [Wheat_2018_images.xxx](link1) and [Wheat_2018_depth_images.xxx](link2)

* To perform crop alignment, the images of the Wheat Dataset from 2019 are needed. Download and extract the file below:

  [Wheat_2019.xxx](link3)

# How to run
Please refer to each subdirectory for further instruction