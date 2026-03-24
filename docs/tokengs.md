#  Object TokenGS

Object TokenGS is an object version of TokenGS. Following TokenGS, it directly regresses 3D mean coordinates using only a self-supervised rendering loss. This formulation allows us to move from the standard encoder-only design to an encoder-decoder architecture with learnable Gaussian tokens, thereby unbinding the number of predicted primitives from input image resolution and number of views.

---

## Installation

From the repo root, install the pinned `gsplat` build first and then the TokenGS extra:

```bash
pip install --no-cache-dir --no-build-isolation \
    "git+https://github.com/nerfstudio-project/gsplat.git@b60e917c95afc449c5be33a634f1f457e116ff5e"
pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
    -e ".[tokengs]"
```

After install, the training entry point is available as `tokengs-train`, backed by
`asset_harvester/tokengs/main.py`.

## Citation

If you use TokenGS in your research, please cite:

```bibtex
@article{tokengs2026,
  title={TokenGS: Decoupling 3D Gaussian Prediction from Pixels with Learnable Tokens},
  author={Jiawei Ren and Michal Tyszkiewicz and Jiahui Huang and Zan Gojcic},
  journal={},
  year={2026}
}
```
