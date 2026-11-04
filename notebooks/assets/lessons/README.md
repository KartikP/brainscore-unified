# Lesson assets

Small enough to commit (5.2 MB), so the lessons run offline on a CPU.

| File | What | Source |
| --- | --- | --- |
| `squeezenet1_1.pth` | SqueezeNet 1.1 trained on ImageNet: the `state_dict` of `torchvision.models.squeezenet1_1(weights='IMAGENET1K_V1')` | torchvision (BSD-3-Clause); architecture by Iandola et al. 2016 |
| `images/` | 120 test images from CIFAR-10, the first 12 of each category, 32 x 32 PNG | CIFAR-10 test set (Krizhevsky 2009), via `uoft-cs/cifar10` on Hugging Face |
| `images.csv` | `file`, `category` (CIFAR-10 class), `kind` (`animal` or `vehicle`) | made for the lessons |

The network was never trained on CIFAR-10 or on its categories, so what it does with these images is a real test, not a reminder of its training.

Citations: Iandola, F. N. et al. (2016). SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size. arXiv:1602.07360. Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images. Technical report, University of Toronto.
