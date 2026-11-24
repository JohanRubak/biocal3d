# Frozen pretrained models

BioCal3D uses publicly released DINOv3 and Concerto encoders as frozen feature extractors. Obtain the pretrained encoders, checkpoints and upstream usage instructions from their authors. The project does not redistribute those pretrained weights or duplicate their training code.

The six heads supplied in the [final classification run](../analysis/classification-study/archives/README.md) were trained for BioCal3D cleaning-stage prediction. They are separate from the upstream pretrained encoders below.

## DINOv3: 2D encoder

- Paper: [DINOv3](https://arxiv.org/abs/2508.10104).
- Authors' implementation and model instructions: [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3).
- Authors' project/checkpoint access page: [DINOv3 at Meta](https://ai.meta.com/research/dinov3/).
- Model used by the BioCal3D setup: **ViT-S/16**, checkpoint filename `dinov3_vits16_pretrain_lvd1689m-08c60483.pth`.

Follow the official repository's current access and licensing instructions to obtain this checkpoint. The BioCal3D setup helper accepts its local path; it does not bypass the authors' checkpoint access process. Do not substitute a different DINO model or size for an archived experiment.

## Concerto: 3D encoder

- Paper: [Concerto: Joint 2D-3D Self-Supervised Learning Emerges Spatial Representations](https://arxiv.org/abs/2510.23607).
- Authors' pretrained-model/inference repository: [Pointcept/Concerto](https://github.com/Pointcept/Concerto).
- Authors' project page: [Concerto](https://pointcept.github.io/Concerto/).
- Authors' checkpoint repository: [Pointcept/Concerto on Hugging Face](https://huggingface.co/Pointcept/Concerto).
- Authors' training implementation: [Concerto in Pointcept](https://github.com/Pointcept/Pointcept/tree/main/pointcept/models/concerto).
- Model used by the BioCal3D setup: **Concerto Small**, checkpoint filename `concerto_small.pth`.

The BioCal3D helper downloads Concerto Small from the authors' checkpoint repository and verifies its recorded SHA-256. Use the official repositories for upstream model documentation and usage terms.

## Configure the project

Follow [setup](setup.md#encoder-environments-and-checkpoints) for the separate Conda environments and `setup_model_env.py` commands. The helper installs the project adapter dependencies and records local repository/checkpoint paths. The upstream encoders remain frozen; BioCal3D fits its own attention/classification heads.

For exact reproduction, record the upstream Git commit, checkpoint SHA-256, input/render settings and installed package versions. The authors' current main branches are useful for documentation, but are not evidence of which commit was used by a historical run. The supplied final-run ZIP contains trained heads and outputs; it does not record every original encoder checkout or feature setting. Those experiment-specific records remain useful even though the pretrained encoders themselves can be obtained upstream.
