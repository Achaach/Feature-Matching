# Feature-Matching

## Introduction
Local feature matching is a technique used in computer vision and image processing to match similar features between two or more images. It is an essential step in various applications such as object recognition, image stitching, and 3D reconstruction.

The local feature matching algorithm works by first detecting salient features in each image, such as corners, blobs, and edges. These features are then described using descriptors such as SIFT (Scale-Invariant Feature Transform), SURF (Speeded Up Robust Features), ORB (Oriented FAST and Rotated BRIEF), and AKAZE (Accelerated-KAZE). These descriptors are designed to be invariant to scale, rotation, and illumination changes, making them robust to image transformations and distortions.

Local feature matching is a crucial technique in computer vision and has many real-world applications. However, the performance of the algorithm depends on the quality and number of features detected and the choice of descriptors and distance metrics used. Additionally, the algorithm can be computationally expensive, especially when dealing with large datasets, and requires optimization and parallelization techniques to improve its performance.

## Overview
We are going to finish 3 parts of feature matching in this project:

1. Harris Corner Detector
2. SIFT
3. Feature Matching and compute the ratio

## Visualization Result
<img width="421" alt="Screen Shot 2023-02-19 at 7 52 10 PM" src="https://user-images.githubusercontent.com/90078254/219987343-af8f65b7-0e1e-4e7b-81d2-8d5e2f4aacbd.png">
Took in Chongqing, China in 2020
