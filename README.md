# DRHPA: Degradation-Routed Hierarchical Prompt Adaptation for Robust Fine-Grained Ship Recognition

**FGSCM-52-C dataset and image-conditioned multi-expert prompting for ship recognition under varied observation conditions.**

## Overview

Cloud cover, imaging artifacts, and sea-surface interference can obscure the image features used to distinguish similar ship categories. This project studies fine-grained ship recognition when the observation condition and its severity are unknown at inference.

The work introduces two components:

- **FGSCM-52-C:** a dataset for condition-robust fine-grained ship recognition, constructed from FGSCM-52 with paired clean and transformed images across twelve conditions and five severity levels.
- **DRHPA:** a classification method that uses image-conditioned routing to combine four prompt experts. Shared expert weights coordinate fine-class and coarse-group scoring, followed by hierarchical score fusion.

## FGSCM-52-C

FGSCM-52-C extends the original [FGSCM-52](https://arxiv.org/abs/2403.08271) images and hierarchical labels with generated observation conditions. FGSCM-52 is the source dataset; FGSCM-52-C is the condition-transformed extension introduced in this work.

![Representative source-paired views from FGSCM-52-C](assets/dataset_examples.png)

Each row shows a clean source image and selected transformed views. The figure illustrates representative conditions; the dataset protocol includes all twelve conditions listed below.

### Dataset design

| Property | Description |
| --- | --- |
| Clean source images | 9,562 |
| Fine-grained ship classes | 52 |
| Coarse ship groups | 10 |
| Generated observation conditions | 12 |
| Severity levels per condition | 5 |
| Source-image split | 6,694 training / 1,434 validation / 1,434 test |
| Transformed validation images | 86,040 |
| Transformed test images | 86,040 |

Source images are split before transformed views are generated. Each transformed view retains its source image's fine-class label, coarse-group label, source identifier, and partition. A clean image and its transformed views remain in the same partition.

### Observation conditions

| Condition | Generation method |
| --- | --- |
| Fog occlusion | Koschmieder-style attenuation with a spatial Perlin opacity mask |
| Gaussian noise | Additive Gaussian noise with clipping |
| Impulse noise | Salt-and-pepper pixel replacement |
| Motion blur | Linear point-spread convolution |
| Defocus blur | Disk point-spread convolution |
| Low resolution | Bicubic downsampling followed by restoration |
| JPEG compression | Controlled-quality JPEG encoding and decoding |
| Cloud cover | Perlin cloud overlays with controlled coverage |
| Sea clutter | Multiscale wave-and-glint overlays |
| Stripe occlusion | Oriented stripe masks |
| SAR-style | CycleGAN translation with SAR-Ship-Dataset as the target domain, followed by speckle injection |
| Thermal-IR-style | CycleGAN translation with DVTOD thermal images as the target domain, followed by appearance remapping |

All twelve conditions participate in the main training and evaluation protocol. The SAR-style and thermal-IR-style views are synthetic appearance translations, not paired measurements acquired by SAR and thermal infrared sensors.

## DRHPA

![DRHPA method overview with paired clean and transformed images](assets/method.png)

### Method

1. **Frozen image encoding.** A pretrained CLIP ViT-B/16 image encoder extracts an image embedding.
2. **Image-conditioned routing.** A soft router assigns weights to four jointly learned prompt experts. Experts are not assigned predefined condition labels.
3. **Hierarchical prompting.** Each expert contains fine-class and coarse-group prompt contexts. Fixed category names and learned contexts are processed by the frozen text encoder.
4. **Shared expert weighting.** The same image-dependent routing weights combine image-text scores at both category levels.
5. **Hierarchical fusion.** Each fine-class score is combined with its parent-group score to obtain the final fine-class prediction.

### Training and inference

Training uses paired clean and transformed images with shared category labels. Each view produces its own routing weights. Classification supervision, prediction consistency, and routing regularization train the classification branch while the pretrained encoders remain frozen.

Inference takes **one cropped ship image** and outputs a fine-grained class prediction. It requires no clean counterpart, condition label, severity label, per-image caption, or test-time parameter update. Category names are fixed class information, not additional annotations required for each test image.

## Evaluation

The main evaluation combines clean-image recognition with recognition across the twelve generated conditions and five severity levels.

| Metric | Definition |
| --- | --- |
| Clean accuracy | Top-1 classification accuracy on clean test images |
| Mean condition accuracy (mCA) | Mean of the twelve condition accuracies, each averaged over its five severity levels |
| Worst-condition mean accuracy (WCA) | Minimum of the twelve severity-averaged condition accuracies |

WCA is the lowest condition mean, not the lowest accuracy at an individual severity level.

The manuscript also examines held-out conditions, cross-severity generalization, base-to-new recognition, component ablations, calibration, and external maritime scenes.

## Availability

This documentation package contains the README and two illustrative figures. It does not include the full FGSCM-52-C dataset, training or evaluation code, pretrained checkpoints, or split manifests. No download or installation instructions are provided for those materials in this package.

## Acknowledgments

The dataset construction and method build on the following resources:

- [FGSCM-52 and hierarchical prompt tuning](https://arxiv.org/abs/2403.08271): source ship images and hierarchical labels.
- [CLIP](https://github.com/openai/CLIP): pretrained image and text encoders.
- [CycleGAN](https://openaccess.thecvf.com/content_iccv_2017/html/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.html): unpaired image-to-image translation.
- [SAR-Ship-Dataset](https://www.mdpi.com/2072-4292/11/7/765): target-domain images for SAR-style translation.
- [DVTOD](https://github.com/VDT-2048/DVTOD): thermal target-domain images for thermal-IR-style translation.
