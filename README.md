# Lightweight-Face-Detector-Pruning

Pruning Lightweight Face Detectors **EXTD** and **EResFD** using NNI's `FPGMPruner` and `L1NormPruner`
## Project Structure

The repository is organized into several key folders:

- `EXTD_Pytorch-master/`: Contains code and resources specific to the EXTD model.
- `EResFD-main/`: Contains code and resources for the EResFD model.
- `Pruned Models/`: A collection of pre-pruned model weights (`.pth` files) for both EXTD and EResFD.

## Prerequisites

Before running the pruning scripts, users need to prepare the necessary dataset:

### WIDER FACE Dataset

The models are trained and evaluated using the WIDER FACE dataset. To use this dataset:

1. Download the WIDER FACE dataset from [here](https://shuoyang1213.me/WIDERFACE/).
2. Extract and place the `WIDER` folder in the same directory as the `EXTD` and `EResFD` folders.

## Running the Pruning Scripts

To execute the pruning process, use the following commands based on the desired pruning algorithm:

- For pruning with the Geometric Median (FPGM) algorithm in the EResFD model:
  ```bash
  python fpgm.py --pruning_rate 0.1 --pruned_eres './weights/eres10'
