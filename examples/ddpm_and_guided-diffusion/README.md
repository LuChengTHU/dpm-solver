# DPM-Solver for DDPM and Guided-Diffusion

This is the example code for DPM-Solver applying to [DDPM](https://arxiv.org/abs/2006.11239) and [Guided-Diffusion](https://arxiv.org/abs/2105.05233). The code is heavily based on [DDIM codebase](https://github.com/ermongroup/ddim).

## Dataset, Checkpoint and FID Stats
We support various datasets and checkpoints. Please check the config files in `configs/`.

Some checkpoints will be automatically downloaded in `~/ddpm_ckpt/`, please check this code for details: `functions/ckpt_util.py`. Other checkpoints needs to be put in the `model.ckpt_dir` in the config file.

| Config File            | Checkpoint                                                   | FID Stats                                                    |
| ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| cifar10.yml            | Automatically download [DDPM checkpoint on CIFAR-10](https://github.com/pesser/pytorch_diffusion) in `~/ddpm_ckpt/diffusion_models_converted/`. | [Download](https://drive.google.com/drive/folders/1_OpTXVPLffZM8BG-V3Ahsxk99aqxW7C3?usp=sharing) in `./fid_stats/fid_stats_cifar10_train_pytorch.npz` |
| celeba.yml             | Please download [DDIM checkpoint on Celeb-A](https://drive.google.com/file/d/1R_H-fJYXSH79wfSKs9D-fuKQVan5L-GR/view) in `~/ddpm_ckpt/celeba/ckpt.pth`. | [Download](https://drive.google.com/drive/folders/1_OpTXVPLffZM8BG-V3Ahsxk99aqxW7C3?usp=sharing) in `./fid_stats/fid_stats_celeba64_train_50000_ddim.npz` |
| imagenet64.yml         | Please download [improved-DDPM checkpoint on unconditional ImageNet64](https://openaipublic.blob.core.windows.net/diffusion/march-2021/imagenet64_uncond_100M_1500K.pt) in `~/ddpm_ckpt/imagenet64/imagenet64_uncond_100M_1500K.pt`. | [Download](https://drive.google.com/drive/folders/1_OpTXVPLffZM8BG-V3Ahsxk99aqxW7C3?usp=sharing) in `./fid_stats/fid_stats_imagenet64_train.npz` |
| bedroom_guided.yml     | Please download [guided-diffusion checkpoint on unconditional LSUN bedroom](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/lsun_bedroom.pt) in `~/ddpm_ckpt/bedroom/lsun_bedroom.pt`. | [Download](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/lsun/bedroom/VIRTUAL_lsun_bedroom256.npz) in `./fid_stats/VIRTUAL_lsun_bedroom256.npz` |
| imagenet128_guided.yml | Please download [guided-diffusion diffusion checkpoint](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/128x128_diffusion.pt) in `~/ddpm_ckpt/imagenet128/128x128_diffusion.pt`, and [guided-diffusion classifier checkpoint](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/128x128_classifier.pt) in `~/ddpm_ckpt/imagenet128/128x128_classifier.pt`. | [Download](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/128/VIRTUAL_imagenet128_labeled.npz) in `./fid_stats/VIRTUAL_imagenet128_labeled.npz` |
| Imagenet256_guided.yml | Please download [guided-diffusion diffusion checkpoint](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt) in `~/ddpm_ckpt/imagenet256/256x256_diffusion.pt`, and [guided-diffusion classifier checkpoint](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt) in `~/ddpm_ckpt/imagenet256/256x256_classifier.pt`. | [Download](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz) in `./fid_stats/VIRTUAL_imagenet256_labeled.npz` |
| Imagenet512_guided.yml | Please download [guided-diffusion diffusion checkpoint](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/512x512_diffusion.pt) in `~/ddpm_ckpt/imagenet512/512x512_diffusion.pt`, and [guided-diffusion classifier checkpoint](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/512x512_classifier.pt) in `~/ddpm_ckpt/imagenet512/512x512_classifier.pt`. | [Download](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/512/VIRTUAL_imagenet512.npz) in `./fid_stats/VIRTUAL_imagenet512.npz` |

## Example Script
We provide three example scirpt in `sample.sh`, including:

- Unconditional sampling on CIFAR-10 with linear beta schedule.

- Unconditional sampling on ImageNet64 with cosine beta schedule.

- Classifier-guided sampling on ImageNet256 with guidance scale `8.0` and dynamic thresholding. 
