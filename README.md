# landmarks_utils

`landmarks_utils` is a Python package for **landmark localization** model training, inference, and annotation, particularly tailored for medical imaging applications.

It builds on [`predict-idlab/landmarker`](https://github.com/predict-idlab/landmarker), with a custom fork at [`ClemensWatzenboeck/landmarker`](https://github.com/ClemensWatzenboeck/landmarker) that includes:

- A **lazy DataLoader** (avoids memory issues on large datasets)
- Other stability and usability fixes for large-scale inference

---

## 🚀 Installation (via conda)

Follow these steps to set up your environment:

```bash
# Step 1: Install CUDA (if using GPU)
# Follow official instructions from: https://developer.nvidia.com/cuda-downloads

# Step 2: Create and activate a conda environment
conda create -n landmark_env python=3.9
conda activate landmark_env

# Step 3: Clone and install the modified landmarker fork
#git clone https://github.com/ClemensWatzenboeck/landmarker.git  /path/to/your/local/code/landmarker 
#cd /path/to/your/local/code/landmarker
#pip install -e .
## Resolved -> Now in pyproject.toml

# Step 4: Install this package (landmarks_utils)
cd /path/to/your/landmarks_utils
pip install -e .
```


#### How to use: 
##### Annotation

Highlights of the provided utility functions are a useful tool to create point-annotations build with `napari`.

After installation and updating the example config files (see `runs/config_/_annotation/config_H.yaml`) it can be stated with

```bash
  landmarks_utils__annotate_landmarks  --config /path/to/runs/config/annotation/config_H.yaml
```

This will start the UI for the annotation. 




<!-----
[🎥 Click here to watch the annotation demo (MP4)](./documentation/images/DEMO_annotation_v1.mkv)
-->

##### Training landmark detection 
Update the config in `./runs/config_landmarks/hands/H_train_landmarks_102p1__updated_data_handler.yaml`

```bash
python ~/code/RA/landmarks_utils/landmarks_utils/training/landmarks/01_train_mlflow.py  \
   --config  `./runs/config_landmarks/hands/H_train_landmarks_102p1__updated_data_handler.yaml
```


##### Inference landmark detection 

Update the config in `./runs/config_landmarks/inference/H_inference_580_cases.yaml`. 
Insert mlflow `runid` of the trained model, path to data, ... 

```bash
python /home/cwatzenboeck/code/RA/landmarks_utils/landmarks_utils/inference/landmarks_inference_v2.py  \
   --config  `./runs/config_landmarks/inference/H_inference_580_cases.yaml
```

##### Inference with updated code: 
I refactored the code. It now uses hydra for the config. 
The same functionality as before can be achieved via: 


```bash 
python /home/cwatzenboeck/code/RA/landmarks_utils/landmarks_utils/inference/landmarks_folder_hydra.py \
  --config-path config_landmarks \
  --config-name F_inference_580_cases_dev \
    output_dst="/home/cwatzenboeck/code/RA/autopix_muw_x_ray_pipeline/dev_dir/output_dir/tmp_landmarks_multi_F_V2/" \
    hydra.run.dir=.  \
    hydra.output_subdir=null 
```

The config files for reloading the model are there. [./landmarks_utils/inference/config_landmarks/](./landmarks_utils/inference/config_landmarks/). 

The goal of the refactor was to allow for a simple inference from just a single image from the command line. So that the whole thing be put into one neat pipeline. 

*** CLI for single image inference ***
```bash 
dcm_file="/path/to/AUTOPIX_000017_20170505_F_R_dp_MTwo.dcm"


python /home/cwatzenboeck/code/RA/landmarks_utils/landmarks_utils/inference/landmarks_single_hydra.py \
  --config-path config_landmarks \
  --config-name config \
  model=model_F_mlflow \
  dcm_file="${dcm_file}"

# Optional: add also   output_dst=/path/to/output/folder
```
This will create `lm_file="/path/to/AUTOPIX_000017_20170505_F_R_dp_MTwo_landmarks_F.csv`

To reload the hands model instead simply give `model=model_H_mlflow` to the CLI. 

**Hint:**  The currently trained models assume that right extremities were mirrored and that all is mapped to MONOCHROME1-> MONOCHROME2. This can be done fully automatically with the code of the pipeline. 
See: [https://gitlab.cir.meduniwien.ac.at:8888/cwatzenboeck/autopix_muw_x_ray_pipeline](autopix_muw_x_ray_pipeline).

```bash 
python /home/cwatzenboeck/code/RA/autopix_muw_x_ray_pipeline/cr_pipeline/preprocessing__x_ray_maybe_flip_and_monochrome.py \
        --image  "${img_F_R_dp}" \
        --output_path  "${out_dir}" \
        --verbose  --link_if_no_change
```

##### Plotting 
This visualizes the landmarks. 
```bash 
python /home/cwatzenboeck/code/RA/landmarks_utils/landmarks_utils/visualization/plot_landmarks.py \
  --landmarks_csv "${lm_file}" \
  #--idx=0   # default anyhow. 
```


---



## 📌 Annotation Examples

<img src="./documentation/images/annoation_F_sample_01.png" width="600"/>  
<img src="./documentation/images/annoation_F_sample_02.png" width="600"/>  
<img src="./documentation/images/annoation_H_sample_01.png" width="600"/>  
<img src="./documentation/images/annoation_H_sample_02.png" width="600"/>  

---

## 🖼️ Demo

- [Annotation Demo 1: Basic point annotations](./documentation/annotation_demo1.md)
- [Annotation Demo 2: Updating predictions of the model](./documentation/annotation_demo2.md)
- [Annotation Demo 3: Training annotators and checking against ground truth](./documentation/annotation_demo3.md)

---

## 🗂️ Patch Extraction Examples

- [Patch Extraction](./documentation/patch_extraction_examples.md)

<!--
<img src="./documentation/images/patch_cropped_F_example01.png" width="600"/>  
<img src="./documentation/images/patch_cropped_H_example01.png" width="600"/>  
--->

---


## 📈 Metrics & Results

See [notebook feet eval](./nb/landmarks_evaluation/eval_landmarker_02a_reload_F_c580.ipynb) and [notebook hands eval](./nb/landmarks_evaluation/eval_landmarker_02a_reload_H_c580.ipynb).


| CPE Curves (F) | CPE Curves (H) |
| :---: | :---: |
| ![CPE F](./documentation/results/CPE_curves_F_580c_2025-08-07.png) | ![CPE H](./documentation/results/CPE_curves_H_580c_2025-08-07.png) |

| SDR (F) | SDR (H) |
| :---: | :---: |
| ![SDR F](./documentation/results/SDR_F_2025-08-07.png) | ![SDR H](./documentation/results/SDR_H_2025-08-07.png) |


---



