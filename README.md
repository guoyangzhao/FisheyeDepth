# FisheyeDepth
## FisheyeDepth: A Real Scale Self-Supervised Depth Estimation Model for Fisheye Camera
[![arxiv paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/XXX)
[![demo video](https://img.shields.io/badge/Demo-Video-blue)](https://xxx/XXX)
<br>

### Motivation:
(a) Fisheye cameras offer a wide FOV and advanced geometric properties, making them well-suited for robotic vision tasks. 
(b) The significant distortion in fisheye images poses a major challenge for depth estimation networks in achieving pixel-wise geometric consistency between consecutive frames.
(c) Depth estimation models often neglect the real motion scale during training, limiting their applicability in real-world robotic systems.

### Framework:
**Self-supervised framework for the fisheye camera.** 
(a) We propose a fisheye projection scheme based on the camera model to eliminate distortion in training, which greatly improves the accuracy of depth estimation. 
(b) We incorporate real-scale poses during the training process, which renders the model suitable for real-world robotic interactions and navigation tasks. 
(c) We devise a multi-channel output module that performs adaptive feature fusion across multiple scales, and this ensures robust depth predictions across different scenes.

<img src="https://github.com/guoyangzhao/FisheyeDepth/blob/main/images/framework.png" width="60%" height="auto">

### Depth Estimation Results – KITTI-360:

<img src="https://github.com/guoyangzhao/FisheyeDepth/blob/main/images/result-360.png" width="60%" height="auto">

### Depth Estimation Results – Real Scene:

<img src="https://github.com/guoyangzhao/FisheyeDepth/blob/main/images/result-real.png" width="60%" height="auto">

### Citations:
If you find TSCLIP useful in your research or applications, please consider giving us a star 🌟 and citing it.
