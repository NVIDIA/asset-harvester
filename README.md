# Asset Harvester: Turning In-the-Wild Driving Logs into 3D Assets for Simulation



[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://research.nvidia.com/labs/sil/projects/asset-harvester/)
[![Paper (Coming soon)](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv)](https://research.nvidia.com/labs/sil/projects/asset-harvester/)
[![License](https://img.shields.io/badge/License-Apache--2.0-orange)](LICENSE.txt)
[![Model](https://img.shields.io/badge/HF-Model-yellow?logo=huggingface&style=flat-square)](https://huggingface.co/nvidia/asset-harvester)[![NCore Data](https://img.shields.io/badge/NCore-0d9488?logo=database&logoColor=white&style=flat-square)](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NCore)[![Benchmark](https://img.shields.io/badge/Benchmark-4f46e5?logoColor=white&style=flat-square)](https://huggingface.co/datasets/nvidia/NuRec-AV-Object-Benchmark)

**NVIDIA**


### Abstract

Neural scene reconstruction converts driving logs into interactive environments for closed-loop simulation, but it does not produce complete 3D object assets needed for multi-agent manipulation and dramatic novel-view synthesis.
To address this challenge, we present Asset Harvester, an image-to-3D model and end-to-end pipeline that converts sparse, in-the-wild observations of AV objects from real driving logs into complete, simulation-ready assets. 
At the core of our method is SparseViewDiT, a sparse-view-to-multiview generation model developed in this work. We couple SparseViewDiT with 3D Gaussian lifting and train the model for sparse conditions via hybrid data curation, camera and occlusion augmentation, and self-distillation.
We integrate the pipeline with NVIDIA NCore for data ingestion and with NVIDIA NuRec for asset insertion, replacement, and closed-loop simulation, enabling scalable conversion of driving logs into reusable 3D assets. We also introduce the NuRec AV Object Benchmark and perception-based metrics that are robust to small spatial misalignment for evaluating in-the-wild AV object reconstruction.


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
- **GPU VRAM** ~16 GB (add `--offload` to offload unused models to CPU for lower VRAM)

> **Note:** Initial setup takes ~20 minutes to complete.

```bash
git clone https://github.com/NVIDIA/asset-harvester.git
cd asset-harvester
bash setup.sh
conda activate asset-harvester
```

Options: `bash setup.sh --env-name asset-harvester --python 3.10`

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
python utils/image_segment.py \
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



<details>
<summary><b>Benchmark</b></summary>
Coming soon.
</details>



<details>
<summary><b>Repository Structure</b></summary>

```
asset-harvester/
├── run_inference.py                 # Main inference entry point
├── run.sh                           # Step 2: multiview diffusion + Gaussian lifting wrapper
├── utils/
│   ├── generate_external_assets_metadata.py  # Step 3: metadata.yaml for NuRec asset insertion
│   ├── image_guard.py                        # Optional Llama Guard 3 Vision moderation on inputs
│   └── image_segment.py                      # Object-centric segmentation / masks for frames
├── README.md
├── setup.sh
├── ncore_parser/                    # Step 1: NCore V4 parsing
│   ├── run.sh          
│   ├── pyproject.toml
│   └── src/ncore_parser/            # NCore Data Parser
├── models/
│   ├── multiview_diffusion/         # SparseViewDiT model
│   ├── tokengs/                     # TokenGS (Gaussian lifting)
│   └── camera_estimator/            # Camera pose estimater for object
├── data_samples/                    # Bundled sample data for Quick Start
└── checkpoints/                     # Model weights (download separately)
```

</details>

## License

This project is licensed under the Apache License 2.0. See individual file headers for details.

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{cao2026assetharvester,
  title   = {Asset Harvester: Turning In-the-Wild Driving Logs into 3D Assets for Simulation},
  author  = {Cao, Tianshi and Ren, Jiawei and Zhang, Yuxuan and Seo, Jaewoo and Huang, Jiahui and 
             Solanki, Shikhar and Zhang, Haotian and Guo, Mingfei and Turki, Haithem and Li, Mu and 
             Zhu, Yue and Zhang, Sipeng and Gojcic, Zan and Fidler, Sanja and Yin, Kangxue},
  year    = {2026},
}
```

## Disclaimer

Asset Harvester is trained for the AV domain, and the results are not guaranteed in other domains.

AI models generate responses and outputs based on complex algorithms and machine learning techniques, and those responses or outputs may be inaccurate or offensive. By downloading a model, you assume the risk of any harm caused by any response or output of the model. By using this software or model, you are agreeing to the terms and conditions of the license, acceptable use policy, and privacy policy as applicable.


