# Asset Harvester: Extracting 3D Assets from Autonomous Driving Logs for Simulation



[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://research.nvidia.com/labs/sil/projects/asset-harvester/)
[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv)](https://arxiv.org/abs/2604.18468)
[![License](https://img.shields.io/badge/License-Apache--2.0-orange)](LICENSE.txt)
[![Model](https://img.shields.io/badge/HF-Models-yellow?logo=huggingface&style=flat-square)](https://huggingface.co/nvidia/asset-harvester)[![Live Demo](https://img.shields.io/badge/Live%20Demo!-ff6f00?logoColor=white&style=flat-square)](https://huggingface.co/spaces/nvidia/asset-harvester)[![NCore Data](https://img.shields.io/badge/NCore-0d9488?logo=database&logoColor=white&style=flat-square)](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NCore)[![Benchmark](https://img.shields.io/badge/Benchmark-4f46e5?logoColor=white&style=flat-square)](https://huggingface.co/datasets/nvidia/NuRec-AV-Object-Benchmark)

**NVIDIA**


### Abstract

Closed-loop simulation is a core component of autonomous vehicle (AV) development, enabling scalable testing, training, and safety validation before real-world deployment. Neural scene reconstruction converts driving logs into interactive 3D environments for simulation, but it does not produce complete 3D object assets required for agent manipulation and large-viewpoint novel-view synthesis.
To address this challenge, we present Asset Harvester, an image-to-3D model and end-to-end pipeline that converts sparse, in-the-wild object observations from real driving logs into complete, simulation-ready assets.
Rather than relying on a single model component, we developed a system-level design for real-world AV data that combines large-scale curation of object-centric training tuples, geometry-aware preprocessing across heterogeneous sensors, and a robust training recipe that couples sparse-view-conditioned multiview generation with 3D Gaussian lifting. Within this system, SparseViewDiT is explicitly designed to address limited-angle views and other real-world data challenges.
Together with hybrid data curation, augmentation, and self-distillation, this system enables scalable conversion of sparse AV object observations into reusable 3D assets.


<p align="center">
  <img src="docs/assets/teaser.gif" alt="Asset Harvester teaser" width="100%" style="border: none;">
</p>

**Asset Harvester** turns real-world driving logs into complete, simulation-ready 3D assets — from just one or a few in-the-wild object views. It handles vehicles, pedestrians, riders, and other road objects, even under heavy occlusion, noisy calibration, and extreme viewpoint bias. A multiview diffusion model generates consistent novel viewpoints, and a feed-forward Gaussian reconstructor lifts them to full 3D in seconds. The result: high-fidelity 3D Gaussian splat assets ready for insertion into simulation environments. The pipeline plugs directly into NVIDIA NCore and NuRec for scalable data ingestion and closed-loop simulation.

## Pipeline Overview

NCore V4 Data ─► NCore Parsing ─► Multiview Diffusion + Gaussian Lifting ─► [`metadata.yaml`](docs/end_to_end_example.md#step-3-generate-external-assets-metadata-to-use-with-nvidia-omniverse-nurec-optional) (required for NuRec Object Insertion)

<table>
  <tr>
    <th>Input View</th>
    <th>Multiview Diffusion (2 of 16 views shown) </th>
    <th>3D Gaussian Lifting </th>
  </tr>
  <tr>
    <td><img src="docs/assets/demo_input.jpeg" width="256"></td>
    <td><img src="docs/assets/demo_mv_concat.png" width="512"></td>
    <td><img src="docs/assets/demo_lifting.gif" width="256"></td>
  </tr>
  <tr>
    <td><img src="docs/assets/demo_input_truck.jpeg" width="256"></td>
    <td><img src="docs/assets/demo_mv_truck_concat.png" width="512"></td>
    <td><img src="docs/assets/demo_lifting_truck.gif" width="256"></td>
  </tr>
</table>

## User Guide

For end-to-end asset harvesting from recorded driving sessions, see our  <b>[Full End-to-End Workflow](docs/end_to_end_example.md)</b> :sparkles: !

<details>
<summary><b>Setup</b></summary>

#### Prerequisites

- **conda** (Miniconda or Miniforge)
- **NVIDIA driver** >= 570 (CUDA 12.8 compatible)
- **GCC** 10–13 (tested with GCC 12.3)
- **GPU VRAM** ~16 GB (add `--offload_model_to_cpu` to offload unused models to CPU for lower VRAM)

> **Note:** Initial setup takes ~20 minutes to complete.

```bash
git clone https://github.com/NVIDIA/asset-harvester.git
cd asset-harvester
bash setup.sh
conda activate asset-harvester
```

> **Option note:**  `bash setup.sh --env-name asset-harvester --python 3.10`

The bash script `setup.sh` handles the full environment setup for this repo.

If you need a manual install from a checkout, preinstall the pinned `gsplat` build first, then install the repo with the extras you need:

```bash
pip install --no-cache-dir --no-build-isolation \
    "git+https://github.com/nerfstudio-project/gsplat.git@b60e917c95afc449c5be33a634f1f457e116ff5e"
pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
    -e ".[ncore-parser,multiview_diffusion,tokengs,camera-estimator]"
```

#### Download Model Checkpoints

```bash
pip install huggingface_hub[cli]
hf auth login
hf download nvidia/asset-harvester --local-dir checkpoints
```
or, manually from the [Hugging Face](https://huggingface.co/nvidia/asset-harvester).
This places the following files in `checkpoints/`:

```
checkpoints/
├── AH_multiview_diffusion.safetensors
├── AH_tokengs_lifting.safetensors
├── AH_camera_estimator.safetensors
└── AH_object_seg_jit.pt
```

</details>

<details>
<summary><b>Image-to-3D</b></summary>

Try Asset Harvester on our sample data (Multiview Diffusion + Gaussian Lifting).
Requires ~16GB VRAM. If you run into VRAM OOM issues, add `--offload_model_to_cpu` to offload unused model components to CPU:

```bash
export DATA_ROOT=data_samples/rectified_AV_objects/
export CHECKPOINT_MV=checkpoints/AH_multiview_diffusion.safetensors
export CHECKPOINT_GS=checkpoints/AH_tokengs_lifting.safetensors
export OUTPUT_DIR=outputs/harvesting
python3 run_inference.py \
    --diffusion_checkpoint "${CHECKPOINT_MV}" \
    --data_root "${DATA_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --lifting_checkpoint "${CHECKPOINT_GS}"
```

Or if you have a single-view image with an object in the center and a foreground mask, resize them into 512x512, and place them in a folder with this structure:

```
YOUR_IMAGE_ROOT/
├─── YOUR_IMAGE_NAME_0
│    ├── frame.jpeg
│    └── mask.png
└─── YOUR_IMAGE_NAME_1
...
```

If masks are not available, you can also use our image segmentation model to
 get mask.png from frame.jpeg stored in above structure:

```bash
export CHECKPOINT_SEG=checkpoints/AH_object_seg_jit.pt
export IMAGE_ROOT=data_samples/segmented_images 
python -m asset_harvester.utils.image_segment \
  --checkpoint $CHECKPOINT_SEG \
  --image_folder $IMAGE_ROOT \
  --frame_name frame.jpeg \
  --mask_name mask.png
```

Check the folder `data_samples/OOD_images` for example.

After data preparation, run Asset Harvester with our built-in camera estimator:

```bash
export YOUR_IMAGE_ROOT=data_samples/OOD_images
export CHECKPOINT_MV=checkpoints/AH_multiview_diffusion.safetensors
export CHECKPOINT_GS=checkpoints/AH_tokengs_lifting.safetensors
export CHECKPOINT_CAM=checkpoints/AH_camera_estimator.safetensors
export OUTPUT_DIR=outputs/harvesting_with_camera_estimate
python3 run_inference.py \
    --diffusion_checkpoint "${CHECKPOINT_MV}" \
    --ahc_checkpoint "${CHECKPOINT_CAM}" \
    --image_dir "${YOUR_IMAGE_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --lifting_checkpoint  "${CHECKPOINT_GS}"
```

</details>

<details>
<summary><b>Full End-to-End Workflow</b></summary>

For the complete step-by-step pipeline walkthrough — from raw NCore driving logs through NCore parsing, multiview diffusion, Gaussian lifting, and metadata generation — see the **[End-to-End Guide](docs/end_to_end_example.md)**.

</details>



<a id="benchmark"></a>
<details>
<summary><b>Benchmark</b></summary>
Coming soon
</details>



<details>
<summary><b>Repository Structure</b></summary>

```
asset-harvester/
├── asset_harvester/
│   ├── camera_estimator/            # Camera pose estimator package
│   ├── multiview_diffusion/         # SparseViewDiT package
│   ├── ncore_parser/                # NCore parser package + CLI
│   ├── patches/                     # Compatibility patches
│   ├── tokengs/                     # TokenGS package + main.py training entry point
│   └── utils/                       # Shared utilities and CLI helpers
├── benchmark/                       # Evaluation tools (coming soon)
├── data_samples/                    # Bundled sample data for Quick Start
├── docs/
│   ├── assets/                      # Demo images, GIFs, and videos
│   ├── end_to_end_example.md
│   └── tokengs.md
├── scripts/
│   ├── check_license_headers.py
│   └── run_ncore_parser.sh          # Step 1 wrapper for the parser CLI
├── run_inference.py                 # Main inference entry point
├── run.sh                           # Step 2: multiview diffusion + Gaussian lifting wrapper
├── setup.sh
├── pyproject.toml
├── CONTRIBUTING.MD
├── LICENSE.txt
├── THIRD_PARTY_LICENSE.txt
└── README.md
```

</details>

## License

This project is licensed under the Apache License 2.0. See individual file headers for details.

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{cao2026assetharvester,
  title   = {Asset Harvester: Extracting 3D Assets from Autonomous Driving Logs for Simulation},
  author  = {Cao, Tianshi and Ren, Jiawei and Zhang, Yuxuan and 
             Seo, Jaewoo and Huang, Jiahui and Solanki, Shikhar and 
             Zhang, Haotian and Guo, Mingfei and Turki, Haithem and 
             Li, Muxingzi and Zhu, Yue and Zhang, Sipeng and Gojcic, Zan and
             Fidler, Sanja and Yin, Kangxue},
  year    = {2026},
}
```

## Disclaimer

Asset Harvester is trained for the AV domain, and the results are not guaranteed in other domains.

AI models generate responses and outputs based on complex algorithms and machine learning techniques, and those responses or outputs may be inaccurate or offensive. By downloading a model, you assume the risk of any harm caused by any response or output of the model. By using this software or model, you are agreeing to the terms and conditions of the license, acceptable use policy, and privacy policy as applicable.
