# Real-data placement

Dataset images are intentionally excluded from this repository.

## MVTec AD

Download MVTec AD from the official dataset page after reviewing and accepting its terms:

https://www.mvtec.com/research-teaching/datasets/mvtec-ad

MVTec states that the dataset is released under CC BY-NC-SA 4.0 and is not permitted for
commercial use. Keep the downloaded dataset outside Git or under `data/mvtec_ad/`, which is
ignored by this repository.

Expected category structure:

```text
data/mvtec_ad/
  metal_nut/
    train/good/*.png
    test/good/*.png
    test/<defect_type>/*.png
    ground_truth/<defect_type>/*_mask.png
```

The supplied `supervised-development` protocol reallocates anomalous images from MVTec's official
test collection into train, validation, and held-out test groups. This is useful for demonstrating
supervised segmentation, but it is not the official unsupervised MVTec AD benchmark protocol.

## Custom photographs

Store your own images and binary masks with matching stems:

```text
data/custom_surface/
  images/panel_001.png
  masks/panel_001.png
```

Masks must match the image dimensions. White/nonzero pixels mean defect; black pixels mean normal
surface. Record the camera, lighting, material, defect source, annotation method, and permission to
use every custom image.
