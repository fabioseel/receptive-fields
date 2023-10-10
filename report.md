# Receptive Fields overview

## "Lindsey Models"
Notes:
- No stride used - just large kernels! (9x9)
- padding used?...
- not implemented (yet)
    - Kernel normalization
    - activity normalization after each layer appears to result in smoother RFs
    - adding gaussian noise to input / after retina (not really used in their programming / analysis though)

BN Channels - 1
| Layer | Grayscale input | Color |
| -- | -- | -- |
| Bottleneck | ![](imgs/lindseydefault-grey-BN.png) | ![](imgs/lindseydefault-color-BN.png) |
| V1 | ![](imgs/lindseydefault-grey-V1.png) | ![](imgs/lindseydefault-color-V1.png) |
| V2 | ![](imgs/lindseydefault-grey-V2.png) | ![](imgs/lindseydefault-color-V2.png) |

BN Channels - 32
| Layer | Grayscale input | Color |
| -- | -- | -- |
| Bottleneck | ![](imgs/lindsey32-grey-BN.png) | ![](imgs/lindsey32-color-BN.png) |
| V1 | ![](imgs/lindsey32-grey-V1.png) | ![](imgs/lindsey32-color-V1.png) |
| V2 | ![](imgs/lindsey32-grey-V2.png) | ![](imgs/lindsey32-color-V2.png) |

Lindsey32

| C | Img|
| -- | -- | -- |
| B | ![](imgs/lindsey32-color-V2-B.png) |
| G | ![](imgs/lindsey32-color-V2-G.png) |
| R | ![](imgs/lindsey32-color-V2-R.png) |