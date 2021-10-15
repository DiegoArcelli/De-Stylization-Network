# De-Stylization-Network
An implementation of a CNN architecture resistant to adversarial examples obtained by applying Instagram filters to the target image.  The architecture is described in this paper: https://arxiv.org/abs/1912.13000.

# Description
The architecture is a variant of the IBN-Net (https://arxiv.org/abs/1807.09441), where the Instance Normalization (IN) layers are substituted with Adaptive Instance Normalization (AdaIN) layers, and a de-stylization module (https://arxiv.org/abs/1912.13000) is used to produce the inputs for the AdaIN layers. The goal is to provide a network resistant to adversarial examples created by applying Instagram filters to the images.