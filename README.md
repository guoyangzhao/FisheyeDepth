# FisheyeDepth
## FisheyeDepth: A Real Scale Self-Supervised Depth Estimation Model for Fisheye Camera
[![arxiv paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/2409.15054)
[![demo video](https://img.shields.io/badge/Demo-Video-blue)](https://www.youtube.com/watch?v=SYICNArRfeI)
<br>

### Motivation:
(a) Fisheye cameras offer a wide FOV and advanced geometric properties, making them well-suited for robotic vision tasks. <br>
(b) The significant distortion in fisheye images poses a major challenge for depth estimation networks in achieving pixel-wise geometric consistency between consecutive frames. <br>
(c) Depth estimation models often neglect the real motion scale during training, limiting their applicability in real-world robotic systems. <br>

### Framework:
**Self-supervised framework for the fisheye camera.** <br>
(a) We propose a fisheye projection scheme based on the camera model to eliminate distortion in training, which greatly improves the accuracy of depth estimation. <br>
(b) We incorporate real-scale poses during the training process, which renders the model suitable for real-world robotic interactions and navigation tasks. <br>
(c) We devise a multi-channel output module that performs adaptive feature fusion across multiple scales, and this ensures robust depth predictions across different scenes. <br>
<img src="https://github.com/guoyangzhao/FisheyeDepth/blob/main/images/framework.png" width="80%" height="auto">

### Results:
**Depth Estimation Results – KITTI-360:** <br>
<img src="https://github.com/guoyangzhao/FisheyeDepth/blob/main/images/result-360.png" width="50%" height="auto">

**Depth Estimation Results – Real Scene:** <br>
<img src="https://github.com/guoyangzhao/FisheyeDepth/blob/main/images/result-real.png" width="50%" height="auto">

More visualization results can be viewed through the **[Demo Video](https://www.youtube.com/watch?v=SYICNArRfeI)**

### Citations:
If you find FisheyeDepth useful in your research or applications, please consider giving us a star 🌟 and citing it.

```bibtex
@misc{zhao2024fisheyedepthrealscaleselfsupervised,
      title={FisheyeDepth: A Real Scale Self-Supervised Depth Estimation Model for Fisheye Camera}, 
      author={Guoyang Zhao and Yuxuan Liu and Weiqing Qi and Fulong Ma and Ming Liu and Jun Ma},
      year={2024},
      eprint={2409.15054},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.15054}, 
}
```
