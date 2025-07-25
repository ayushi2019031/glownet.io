---
layout: page
title: "CycleGANs in Practice: Unpaired Style Transfer for the Monet Painting Task"
description: "A technical deep dive into training CycleGANs for artistic image generation, inspired by the Kaggle competition 'I'm Something of a Painter Myself.'"
date: 2025-07-27
tags: [CycleGAN, GANs, Style Transfer, Deep Learning, Kaggle, Computer Vision]
---

_A technical deep dive into training CycleGANs for artistic image generation, inspired by the Kaggle competition “I’m Something of a Painter Myself.” Based on the [CycleGAN paper (Zhu et al., 2017)](https://arxiv.org/pdf/1703.10593)._


## Trying Cycle GANs amongst different types of GANs

I initially started with **DCGANs (Deep Convolutional GANs)** — a natural first step for generative modeling. DCGANs are designed for *unconditional image generation*, where the model learns to generate realistic images from random noise without relying on any input images.

However, the results weren’t promising for this competition. My DCGAN-based submission scored over **300**, which is considered a relatively high (i.e., poor) score in this context. 

While exploring the competition kernels and past submissions, I noticed several top approaches used **CycleGANs**. This led me to dive deeper into the [CycleGAN paper (Zhu et al., 2017)](https://arxiv.org/pdf/1703.10593), which introduced a compelling method for **unpaired image-to-image translation**.

Unlike DCGANs, CycleGANs are specifically designed for **domain translation tasks**, even when paired training data isn’t available. 

While several types of GAN architectures have been explored for unpaired image-to-image translation — including **CUT (Contrastive Unpaired Translation)**, **MUNIT**, and **DRIT++** — each with its own unique strengths, this article focuses specifically on **CycleGANs**. CycleGANs provide a conceptually intuitive and widely adopted approach for tasks like artistic style transfer, making them a solid foundation to build upon. In the following sections, I’ll dive deep into how CycleGANs work, why they’re effective, and how I applied them to the Kaggle competition _“I’m Something of a Painter Myself.”_

--- 

## Painting Without Pairs: Why CycleGAN Makes Sense

The core objective of the competition was to convert real-world landscape photographs into **Monet-style paintings** — a classic **image-to-image translation** problem. However, the key constraint was that the dataset provided was **unpaired**: we had photographs and Monet paintings, but no one-to-one correspondence between them.

This is where **CycleGANs** shine.

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


Additionally, CycleGANs are more robust to **mode collapse**. Discriminator only needs to check whether the generated image looks like it came from Y.  So the generator could always output the same generic looking image from Y, instead of converting X to Y. The cycle-consistency constraint forces the generator to retain input-dependent structure, discouraging it from collapsing into a single mode of output.


While newer models like **CUT** and **MUNIT** offer alternative approaches with their own advantages, CycleGAN remains a strong baseline that balances interpretability, visual quality, and architectural simplicity — all of which made it a great fit for this challenge.

<div style="border: 1px solid #ddd; border-radius: 10px; padding: 20px; background-color: #fafafa; margin: 1.5rem 0;">
  <h3 style="margin-top: 0;">📊 What Metrics Does Kaggle Use For Evaluation?</h3>
  <p>
    Kaggle evaluates submissions using a combination of <strong>FID</strong> and <strong>M-FID</strong> scores:
  </p>

  <h4>🎯 FID – Fréchet Inception Distance</h4>
  <ul>
    <li>Each image is passed through a pre-trained <a href="https://arxiv.org/pdf/1409.4842" target="_blank">Inception network</a> used as a feature extractor.</li>
    <li>Feature vectors are extracted from an intermediate layer for both real and generated images.</li>
    <li>A multivariate Gaussian distribution is fit to each feature set.</li>
    <li>The <strong>Fréchet distance</strong> is calculated between these two Gaussians to measure realism.</li>
  </ul>
  <p><strong>Lower FID = more realistic and diverse generated images.</strong></p>

  <hr>

  <h4>🧠 M-FID – Memorization Penalty</h4>
  <ul>
    <li>M-FID adds a penalty if your model memorizes the training set.</li>
    <li>It uses <strong>cosine distance</strong> between each generated image and the most similar real image.</li>
    <li>The <strong>minimum distance</strong> is computed for each sample and averaged to get the memorization score.</li>
  </ul>
  <p><strong>Higher memorization = higher penalty. This encourages generalization, not copying.</strong></p>
</div>

## Architecture of my particular Cycle GAN

1. Number of layers: inputs vs outputs - number of layers
2. Kaggle training time - how do I train? 
3. Submission to kaggle 
4. Visualizations created

## Conclusion
