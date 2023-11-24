# Lightweight-Face-Detector-Pruning

Pruning Lightweight Face Detectors **EXTD** and **EResFD** using NNI's `FPGMPruner` and `L1NormPruner`.

To ensure compatibility and proper functioning of the pruning scripts, please install the specific version of NNI using the following pip command:

```bash
pip install nni==3.0rc1
```

## Project Structure

The repository is organized into several key folders:

- `EXTD_Pytorch-master/`: Contains code and resources specific to the EXTD model.
- `EResFD-main/`: Contains code and resources for the EResFD model.
- `Pruned_Models/`: A collection of pre-pruned model weights (`.pth` files) for both EXTD and EResFD.

## Prerequisites

Before running the pruning scripts, users need to prepare the necessary dataset:

### WIDER FACE Dataset

The models are trained and evaluated using the WIDER FACE dataset. To use this dataset:

1. Download the WIDER FACE dataset from [here](https://shuoyang1213.me/WIDERFACE/).
2. Extract and place the `WIDER` folder in the same directory as the `EXTD` and `EResFD` folders.

## Running the Pruning Scripts

To execute the pruning process, use the following commands based on the desired pruning algorithm. E.g., for pruning with the Geometric Median (FPGM) algorithm the EResFD model:
  ```bash
  python fpgm.py --pruning_rate 0.1 --pruned_eres './weights/eres10'
  ```

## License
This code is provided for academic, non-commercial use only. Please also check for any restrictions applied in the code parts and datasets used here from other sources. For the materials not covered by any such restrictions, redistribution and use in source and binary forms, with or without modification, are permitted for academic non-commercial use provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation provided with the distribution. This software is provided by the authors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the authors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.

## Citation

If you our pruning method or pruned models useful in your work, please cite the following publication where this approach was proposed:

K. Gkrispanis, N. Gkalelis, V. Mezaris, "Filter-Pruning of Lightweight Face Detectors Using a Geometric Median Criterion", Proc. IEEE/CVF Winter Conference on Applications of Computer Vision Workshops (WACVW 2024), Waikoloa, Hawaii, USA, Jan. 2024.

BibTex:
```
@inproceedings{Gkrispanis_WACVW2024,
author={Gkrispanis, Konstantinos and Gkalelis, Nikolaos and Mezaris, Vasileios},
title={Filter-Pruning of Lightweight Face Detectors Using a Geometric Median Criterion},
year={2024},
month={Jan.},
booktitle={IEEE/CVF Winter Conference on Applications of Computer Vision Workshops (WACVW 2024)}
}
```

## Acknowledgements

This work was supported by the EU Horizon 2020 programme under grant agreement H2020-951911 AI4Media.
