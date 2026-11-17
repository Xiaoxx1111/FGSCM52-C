# DRHPNet: Degradation-Routed Hierarchical Prompt Network

This repository introduces **DRHPNet** and **FGSCM-52-C** for robust fine-grained ship recognition under diverse observation conditions.

## Overview

Fine-grained ship recognition requires distinguishing vessels with similar appearances. Changes in visibility, image quality, and background conditions can obscure discriminative details and reduce recognition performance. Different conditions affect image features in different ways, making robust adaptation across conditions challenging.

DRHPNet addresses this problem through image-conditioned prompt experts and hierarchical classification. FGSCM-52-C extends FGSCM-52 to support evaluation across a range of observation conditions and severity levels.

## DRHPNet

DRHPNet uses a pretrained vision-language model to support fine-grained ship recognition. Its main components are:

- **Image-conditioned expert routing:** A router selects a combination of prompt experts according to the input image. The routing weights are shared by the fine-class and coarse-group prompt branches.
- **Hierarchical prediction:** Fine-class predictions are combined with coarse-group information to support recognition of visually similar ship categories.
- **Consistency training:** Paired clean and condition-affected images are used during training to encourage consistent predictions across observation conditions.

The pretrained image and text encoders are frozen during training. At inference, DRHPNet takes a single image as input and does not require observation-condition labels or parameter updates.

## FGSCM-52-C Dataset

FGSCM-52-C is an extension of the FGSCM-52 fine-grained ship dataset. It provides images across **12 observation conditions**, each with **five severity levels**, for evaluating robustness under different visual variations.

The dataset includes common image degradations and complex visual conditions, as well as SAR-style and thermal-infrared-style appearance variations. The SAR-style and thermal-infrared-style images are generated from optical images and should not be interpreted as real sensor observations.

The data split is established before condition generation so that an original image and its derived versions do not appear in different splits.

### Dataset Download

[Download FGSCM-52-C-S5.zip (6.42 GB)](https://huggingface.co/datasets/wenaaaaa/FGSCM-52-C-S5/resolve/main/FGSCM-52-C-S5.zip?download=true)
## Evaluation

The evaluation considers recognition performance under clean and varying observation conditions. Mean condition accuracy (mCA) summarizes performance across the 12 conditions, while worst condition accuracy (WCA) reflects performance under the most challenging condition.

The associated paper also studies generalization across observation conditions and classes, along with evaluations on external scenes. See the paper for the complete experimental protocols and results.

## Code

**Coming soon.**

## Citation

Citation information will be added when available.

