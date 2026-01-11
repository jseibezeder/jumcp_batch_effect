# Bachelor thesis: An Evaluation of Domain Shift-Adaptive Methods on Cellular Microscopy Image Classification

This repository hosts the implementation and experimental setup used for my bachelor's thesis. All scripts required to reproduce the results and figures in the thesis are provided

## Installation

1. Clone the repository and install the environment:
```bash
git clone https://github.com/jseibezeder/jumcp_batch_effect
cd jumcp_batch_effect
```
2. Create the environment
```bash
conda env create -f environment.yml
conda activate <env_name>
```

## Reproduce results
After installation you are set to go to train the models:
```bash
python -u training/main.py \
--train-file="your-data-file" \
--image-path="your-img-path" \
--mapping="your-mapping-path" \
```
The exact commands and experiment configurations for creating the models can be found in the `train_commands.sh` file.

The training applies k-fold cross-validation on the data and trains for n epochs using early stopping. It also evaluates after each epoch on the validation set and saves the best models according to the loss.

## Data and Pretrained models
The pretrained model weights used for evaluation and used data file can be found here: https://huggingface.co/pandahd03/jumpcp_batch_effect
For more information on how to download the JUMP-CP dataset please refer to: https://jump-cellpainting.broadinstitute.org/

## Evaluation and Plots

The necessary files for testing and plotting results are found in the `notebooks` folder.

