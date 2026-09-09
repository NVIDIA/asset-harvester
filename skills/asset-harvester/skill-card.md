## Description: <br>
Use to install and run NVIDIA Asset Harvester (Apache-2.0) to extract per-object 3D Gaussian Splat assets (gaussians.ply) from AV NCore V4 clips or masked single images via SparseViewDiT + TokenGS, optionally producing metadata.yaml for NuRec object insertion. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers who need to extract per-object 3D Gaussian Splat assets from autonomous-vehicle driving-log clips or masked images for simulation in NVIDIA Omniverse NuRec. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Installation Guide](references/installation.md) <br>
- [Workflows](references/workflows.md) <br>
- [End-to-End NCore Walkthrough](references/end-to-end-ncore.md) <br>
- [CLI Reference](references/cli-reference.md) <br>
- [Troubleshooting](references/troubleshooting.md) <br>
- [Upstream GitHub Repository](https://github.com/NVIDIA/asset-harvester) <br>
- [Project Page](https://research.nvidia.com/labs/sil/projects/asset-harvester/) <br>
- [HuggingFace Model](https://huggingface.co/nvidia/asset-harvester) <br>
- [Live Demo](https://huggingface.co/spaces/nvidia/asset-harvester) <br>
- [Paper (arXiv)](https://arxiv.org/abs/2604.18468) <br>
- [NuRec External Assets Guide](https://docs.nvidia.com/nurec/nurec/use-ah-assets.html) <br>


## Skill Output: <br>
**Output Type(s):** [Files, Shell commands] <br>
**Output Format:** [3D Gaussian Splat PLY files, YAML metadata, MP4 video renders] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Per-sample output directory containing gaussians.ply, multiview/, and MP4 renders] <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access. <br>
- Correctness: Checks whether the agent follows the expected workflow and produces the correct final output. <br>
- Discoverability: Checks whether the agent loads the skill when relevant and avoids using it when irrelevant. <br>
- Effectiveness: Checks whether the agent performs measurably better with the skill than without it. <br>
- Efficiency: Checks whether the agent uses fewer tokens and avoids redundant work. <br>



## Skill Version(s): <br>
0.1.1 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
