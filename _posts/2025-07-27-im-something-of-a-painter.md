---
layout: page
title: "From DCGAN to CUT: My Journey in the ‘I'm Something of a Painter Myself’ Kaggle Competition"
permalink: /im-something-of-a-painter/
---

_A technical deep dive into training GANs for artistic image generation, inspired by the Kaggle competition “I’m Something of a Painter Myself.”

## 1. Introduction

The Kaggle competition *"I'm Something of a Painter Myself"* focuses on a classic problem in computer vision: translating natural photographs into Monet-style paintings. The challenge is made more interesting by the absence of paired training data — meaning the model has to learn the mapping between domains without explicit photo-painting pairs.

In this blog post, I explore three different GAN-based architectures to tackle this task: **DCGAN**, **CycleGAN**, and **CUT (Contrastive Unpaired Translation)**. Each of these models offers a different approach to generative modeling, and comparing them side-by-side helped me understand their practical trade-offs.

This post covers:
- The core architecture and assumptions behind each model
- A comparison of their performance (both visually and in terms of FID scores)
- Training efficiency and resource requirements
- Observations on where each model works best

If you're working on a similar image translation problem or just looking to understand how different GAN architectures behave in real-world scenarios, this breakdown might be useful.

---

## 2. Dataset and Problem Statement

The competition provides two separate sets of images:

- `photo_jpg/`: Real-world landscape photographs
- `monet_jpg/`: Digitized Monet-style paintings

The objective is to build a model that can translate a given photograph into a Monet-style painting. Since the dataset is **unpaired**, there are no direct mappings between a photo and a corresponding painting — which makes this a good testbed for unpaired image-to-image translation techniques like CycleGAN and CUT.

When I first looked at the data, I noticed a few things:
- The photographs vary a lot in content and lighting, while the Monet paintings have a consistent stylistic theme.
- There’s a strong domain gap in terms of color, texture, and abstraction level.
- No labels, masks, or annotations are provided — just raw images.

### Typical Challenges 
- **Unpaired data** means I couldn’t rely on pixel-wise losses like L1 or SSIM.
- **Style transfer** had to preserve structure while convincingly applying Monet-like brushwork and colors.
- **Evaluation** was based on **FID (Frechet Inception Distance)** — a statistical measure comparing feature distributions of real and generated images.
- There was also **M-FID**, a memorization metric, which penalizes copying training images too closely.

This setup pushed me to think carefully about which GAN architectures can generalize well with minimal supervision — and what kind of inductive biases each model brings to the table.


## Architectures Explored

Three GAN architectures were implemented and compared for this task: **DCGAN**, **CycleGAN**, and **CUT (Contrastive Unpaired Translation)**. Each represents a different approach to generative modeling, particularly in the context of unpaired image translation.

### <span style="color:#135915">DCGAN (FID: ~312.9)</span>  
*[*Unsupervised Representation Learning with Deep Convolutional GANs*, Radford et al., 2016](https://arxiv.org/abs/1511.06434)*

I initially started with DCGANs (Deep Convolutional GANs) — a natural first step for generative modeling. DCGANs are designed for unconditional image generation, where the model learns to generate realistic images from random noise without relying on any input images.

**In short, DCGANs learn how to generate random monet images, not from the input photo, as they are trained only on the moent images dataset.**

As expected, the results weren’t promising for this competition. My DCGAN-based submission scored over 300, which is considered a relatively high (i.e., poor) score in this context.

**Key characteristics:**
- 🧱 Simple generator and discriminator architecture  
- 🎲 Trained without conditioning on input images  
- ❌ No mechanism to preserve input structure  

**Outcome on this task:**
- 🌫️ Produced low-quality, blurry images  
- 🎯 Failed to map photos to meaningful Monet-style outputs  
- 🚫 Not suitable for unpaired translation problems  


### <span style="color:#135915">CycleGAN (FID: ~75.1)</span>  
*[*Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*, Zhu et al., 2017](https://arxiv.org/abs/1703.10593)*

CycleGANs are specifically designed for **unpaired image-to-image translation**. Unlike models that require paired training data (e.g., pix2pix), CycleGANs learn to map between two domains using **cycle consistency** — the idea that if you translate a photo to a painting and then back again, you should recover the original photo. This allows them to learn **bidirectional mappings** between domains without the need for aligned pairs.

In the context of this competition:
- The **photographs** form domain A  
- The **Monet paintings** form domain B  

CycleGAN learns both mappings:  
- `Photo → Painting` (Generator G_AB)  
- `Painting → Photo` (Generator G_BA)  
while also ensuring that:  
- `Photo → Painting → Photo` ≈ G_AB(G_BA) ≈ Original Photo  
- `Painting → Photo → Painting` ≈ G_BA(G_AB) ≈ Original Painting  

This **cycle-consistency loss** is what sets CycleGAN apart and makes it well-suited for the competition's setup. It helps preserve the structure of the original content while transforming its style — crucial for generating artistic yet faithful outputs. In other words, **CycleGAN implicitly learns to separate the "content" (shapes, layout, scene) from the "style" (colors, textures)** of the image, even without explicit labels or supervision.

**Key characteristics:**
- Suitable for unpaired datasets
- Enforces structural preservation through cycle loss
- Balances adversarial, identity, and cycle-consistency losses

**Outcome on this task:**
- Generated more coherent and stylistically aligned images
- Training was computationally intensive
- Required long training schedules to converge (e.g., 50+ epochs)


### <span style="color:#135915">CUT GAN (FID: ~63.0)</span>  
*[*Contrastive Learning for Unpaired Image-to-Image Translation*, Park et al., 2020](https://arxiv.org/abs/2007.15651)*

CUT takes a different approach from CycleGAN by eliminating the need for **cycle-consistency loss** entirely. Instead, it relies on **contrastive learning** to preserve the content of the input image during translation. The key idea is to enforce **feature-level similarity** between patches of the input photo and the corresponding patches in the generated image.

This is done using a **PatchNCE loss** — a contrastive objective that encourages each patch in the generated image to be **closer (in feature space)** to its corresponding input patch than to any other randomly sampled patch. This helps maintain the structure of the source image without needing to reconstruct it.

<div style="border: 1px solid; #d0d7de; background-color: #f9f9f9; padding: 0.5em; border-radius: 8px; margin-top: 1em;">

<details>
  <summary><strong>What is PatchNCE Loss?</strong></summary>
  <p>PatchNCE is a contrastive learning objective introduced in the CUT paper. Instead of enforcing cycle-consistency like CycleGAN, CUT uses feature-level similarity to preserve the content of the source image.</p>

  <p>It works by:</p>
  <ul>
    <li>Sampling a patch from the input image and identifying the corresponding patch in the generated image.</li>
    <li>Using a contrastive loss to <strong>maximize similarity</strong> between the matching patches while <strong>minimizing similarity</strong> with other randomly sampled patches (negatives).</li>
    <li>This encourages the generator to retain the structure of the input image without requiring a reverse mapping.</li>
  </ul>

  <p>In simple terms, PatchNCE tells the model:<br>
  <em>"This patch in the output should look most like this patch in the input — not like other random patches."</em></p>

  <p>This is the core idea that enables CUT to avoid the extra complexity of dual generators and cycle losses.</p>
</details>
</div>

<div style="margin-bottom: 2em;"></div>


Unlike CycleGAN, which requires two generators and two discriminators, CUT uses just:
- A single generator for `Photo → Painting`
- A single discriminator to judge realism
- An auxiliary feature network to compute the contrastive loss

This streamlined setup makes CUT more **computationally efficient**, while still achieving high-quality style transfer with strong content preservation.

**Key characteristics:**
- ⚡ Single generator-discriminator architecture (one-way mapping only)  
- 🧠 Uses PatchNCE contrastive loss to preserve content  
- 🏃‍♀️ More efficient training compared to CycleGAN  

**Outcome on this task:**
- 🎨 Generated high-quality Monet-style images with clear structure  
- 🔁 Did not require backward mapping (Painting → Photo)  
- ⏱️ Converged in fewer epochs with better FID than CycleGAN  


---
## Training Setup

All models were trained on Kaggle’s cloud environments, which offer multiple runtime options. Depending on the hardware selected, compute time and compatibility vary significantly.

### Kaggle Hardware Options

| Environment | Hardware             | Weekly Quota | Framework Suitability     |
|-------------|----------------------|--------------|----------------------------|
| CPU-only    | Standard vCPU        | Unlimited    | For debugging only         |
| GPU (T4×2)  | 2× NVIDIA T4         | 30 hours     | Best for PyTorch training  |
| GPU (P100)  | 1× NVIDIA P100       | 20 hours     | Good for PyTorch training  |
| TPU         | TPU v3-8             | 20 hours     | Optimized for TensorFlow   |

> For PyTorch-based workflows (like this project), the **T4 GPU** and **P100 GPU** environments were the most practical. TPU support for PyTorch is limited and more complex to configure.


### Data Preprocessing

- All images resized to `256 × 256`
- Applied augmentations:
  - Random crop
  - Random horizontal flip
- Pixel values normalized to `[-1, 1]`


### Common Training Hyperparameters

- **Batch size**: `1` (due to limited GPU memory)
- **Optimizer**: Adam (`β₁ = 0.5`, `β₂ = 0.999`)
- **Learning rate**: `2e-4`, with linear decay after halfway point
- **Checkpointing**: every 5 epochs
- **Loss functions**:
  - DCGAN: Adversarial loss (BCE)
  - CycleGAN: Adds cycle-consistency + identity loss
  - CUT: Adds PatchNCE loss for contrastive learning


### Model Training Times

| Model     | Epochs | Avg. Time per Epoch | Total Time (approx.) |
|-----------|--------|----------------------|-----------------------|
| DCGAN     | 50     | ~3 min               | ~2.5 hours            |
| CycleGAN  | 50     | ~50–60 min           | ~40–45 hours          |
| CUT       | 50     | ~20 min              | ~16–17 hours          |

> While both CycleGAN and CUT were trained for 50 epochs, CUT consistently reached competitive FID scores much earlier (within 15–20 epochs).
This setup allowed consistent evaluation across models while staying within Kaggle's free GPU compute quotas.

---

## Evaluation Metrics

The competition evaluates submissions using two key metrics: **Frechet Inception Distance (FID)** and **Memorization FID (M-FID)**. These help assess both the quality and generalizability of the generated Monet-style images.

### Frechet Inception Distance (FID)

FID measures how close the distribution of generated images is to the distribution of real Monet paintings in feature space. It uses a pretrained **Inception-v3** network to extract features from both real and generated images, and then fits a multivariate Gaussian to each set.

The FID score is computed as:

<p align="center"><code>FID = ||μ<sub>r</sub> − μ<sub>g</sub>||² + Tr(Σ<sub>r</sub> + Σ<sub>g</sub> − 2(Σ<sub>r</sub>Σ<sub>g</sub>)<sup>1/2</sup>)</code></p>

Where:
- μ and Σ are the mean and covariance of the features from the real (r) and generated (g) image sets

**Interpretation:**
- Lower FID → generated images are closer in distribution to real Monet paintings
- FID focuses on both **image quality** and **diversity**

### Memorization FID (M-FID)

M-FID is a **memorization penalty** that discourages models from simply copying training images. It does this by checking how similar each generated image is to the **closest** training image using **cosine similarity** in feature space.

**How it works:**
- For each generated image, compute its cosine distance to all Monet training images
- Take the **minimum distance** as a proxy for memorization
- Average over all generated samples

**Interpretation:**
- Lower M-FID → less memorization; better generalization
- High-quality outputs with low memorization are rewarded


### Metric Behavior in Practice

| Metric | Indicates                     | Ideal Value |
|--------|-------------------------------|-------------|
| FID    | Quality + diversity            | As low as possible |
| M-FID  | Memorization penalty           | Also low (but not identical to FID) |

These metrics were used both during offline experimentation and for official Kaggle submissions. Final scores were based on the combined effectiveness across these two axes.
