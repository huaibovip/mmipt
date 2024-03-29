# Overview

Welcome to MMipt! In this section, you will know about

- [Overview](#overview)
  - [What is MMipt?](#what-is-mmipt)
  - [Why should I use MMipt?](#why-should-i-use-mmipt)
  - [Get started](#get-started)
  - [User guides](#user-guides)
    - [Advanced guides](#advanced-guides)
    - [How to](#how-to)

## What is MMipt?

MMipt (**M**ultimodal **A**dvanced, **G**enerative, and **I**ntelligent **C**reation) is an open-source AIGC toolbox for professional AI researchers and machine learning engineers to explore image and video processing, editing and generation.

MMipt allows researchers and engineers to use pre-trained state-of-the-art models, train and develop new customized models easily.

MMipt supports various foundamental generative models, including:

- Unconditional Generative Adversarial Networks (GANs)
- Conditional Generative Adversarial Networks (GANs)
- Internal Learning
- Diffusion Models
- And many other generative models are coming soon!

MMipt supports various applications, including:

- Text-to-Image
- Image-to-image translation
- 3D-aware generation
- Image super-resolution
- Video super-resolution
- Video frame interpolation
- Image inpainting
- Image matting
- Image restoration
- Image colorization
- Image generation
- And many other applications are coming soon!

<div align=center>
    <video width="100%" controls>
        <source src="https://user-images.githubusercontent.com/49083766/233564593-7d3d48ed-e843-4432-b610-35e3d257765c.mp4" type="video/mp4">
        <object data="https://user-images.githubusercontent.com/49083766/233564593-7d3d48ed-e843-4432-b610-35e3d257765c.mp4" width="100%">
        </object>
    </video>
</div>
</br>

## Why should I use MMipt?

- **State of the Art Models**

  MMipt provides state-of-the-art generative models to process, edit and synthesize images and videos.

- **Powerful and Popular Applications**

  MMipt supports popular and contemporary image restoration, text-to-image, 3D-aware generation, inpainting, matting, super-resolution and generation applications. Specifically, MMipt supports fine-tuning for stable diffusion and many exciting diffusion's application such as ControlNet Animation with SAM. MMipt also supports GAN interpolation, GAN projection, GAN manipulations and many other popular GAN’s applications. It’s time to begin your AIGC exploration journey!

- **Efficient Framework**

  By using MMEngine and MMCV of OpenMMLab 2.0 framework, MMipt decompose the editing framework into different modules and one can easily construct a customized editor framework by combining different modules. We can define the training process just like playing with Legos and provide rich components and strategies. In MMipt, you can complete controls on the training process with different levels of APIs. With the support of [MMSeparateDistributedDataParallel](https://github.com/open-mmlab/mmengine/blob/main/mmengine/model/wrappers/seperate_distributed.py), distributed training for dynamic architectures can be easily implemented.

## Get started

For installation instructions, please see [Installation](install.md).

## User guides

For beginners, we suggest learning the basic usage of MMipt from [user_guides](../user_guides/config.md).

### Advanced guides

For users who are familiar with MMipt, you may want to learn the design of MMipt, as well as how to extend the repo, how to use multiple repos and other advanced usages, please refer to [advanced_guides](../advanced_guides/evaluator.md).

### How to

For users who want to use MMipt to do something, please refer to [How to](../howto/models.md).
