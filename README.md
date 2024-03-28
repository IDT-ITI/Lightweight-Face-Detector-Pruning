# Lightweight-Face-Detector-Pruning

Pruning Lightweight Face Detectors **EXTD** and **EResFD** using NNI's `FPGMPruner` and `L1NormPruner`.

## Project Structure

The repository is organized into 3 folders:

- `EXTD_Pytorch-master/`: Contains code and resources specific to the EXTD model.
- `EResFD-main/`: Contains code and resources for the EResFD model.
- `Pruned_Models/`: A collection of pre-pruned model weights (`.pth` files) for both EXTD and EResFD. The pruned models that are evaluated in Tables 1 and 2 of our paper are provided, i.e. each of the EXTD, EResFD Face Detectors is pruned using one of the FPGM, L1 pruning techniques, for target pruning rates equal to 10%, 20%, 30%, 40% and 50%. The naming convention we follow for the pruned models is straightforward; for instance, `ERES10_FPGM` refers to the EResFD model pruned with 10% sparsity using the FPGM technique.

## Prerequisites

Before running the pruning scripts, users need to prepare the necessary dataset:

### WIDER FACE Dataset

The models are trained and evaluated using the WIDER FACE dataset. To use this dataset:

1. Download the WIDER FACE dataset from [here](https://shuoyang1213.me/WIDERFACE/).
2. Extract and place the `WIDER` folder in the same directory as the `EXTD` and `EResFD` folders.

## Dependencies

A requirements.txt file is provided with all the necessary python dependencies. Additionaly, the code was developed using Python 3.11.7, CUDA 11.4 and Ubuntu 20.04.06 LTS.

To ensure compatibility and proper functioning of the pruning scripts, please install the specific version of NNI listed in the requirements.txt file, using the following command:

```bash
pip install -r requirements.txt
```

## Running the Scripts for Pruning a Face Detector

The pruning script executes the model pruning process as outlined in Section 4.2 of our paper. It prunes and trains the model iteratively for 200 epochs, following which the pruning is halted and the model is fine-tuned for an additional 10 epochs.

To execute the pruning process, use the following commands based on the desired pruning algorithm. E.g., for pruning with the Geometric Median (FPGM) algorithm the EResFD model:
  ```bash
  python fpgm.py --pruning_rate 0.1 --pruned_eres './weights/eres10'
  ```
Here, `fpgm.py` specifies the pruning methd to be used (in this example, FPGM), the value of `pruning_rate` specifies the sparsity per layer that the user wishes to introduce (`0.1` denotes a 10% target pruning rate), and `pruned_eres` is the prefix of the file where the pruned models will be saved, e.g. `pruned_eres10`.

## License
This code is provided for academic, non-commercial use only. Please also check for any restrictions applied in the code parts and datasets used here from other sources. For the materials not covered by any such restrictions, redistribution and use in source and binary forms, with or without modification, are permitted for academic non-commercial use provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation provided with the distribution. 

This software is provided by the authors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the authors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.

## Citation

If you find our pruning method or pruned models useful in your work, please cite the following publication where this approach was proposed:

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
