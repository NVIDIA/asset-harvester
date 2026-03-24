#  Object TokenGS

Object TokenGS is an object version of TokenGS. Following TokenGS, it directly regresses 3D mean coordinates using only a self-supervised rendering loss. This formulation allows us to move from the standard encoder-only design to an encoder-decoder architecture with learnable Gaussian tokens, thereby unbinding the number of predicted primitives from input image resolution and number of views.

---

## Installation

Install with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install -e .
```

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

