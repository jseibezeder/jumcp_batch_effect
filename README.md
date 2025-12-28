# jumcp_batch_effect

This repsoitory hosts the implementation and experimental setup used for my bachelor's thesis. All scripts required to reproduce the results and figures in the thesis are provided

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
The exact commands and experiment configerations for creating the models can be found here: TODO

The training applies k-fold cross-validation on the data and trains for n epcohs using early stopping. It also evaluates afeter each epoch on the validation set and saves the best models according to the loss.

## Data and Pretrained models
The pretrained model weights used for evaluation and used data file can be found here: TODO: add huggingface
For more information on how to download the JUMP-CP dataset please refer to: https://jump-cellpainting.broadinstitute.org/

## Evaluation and Plots

The necessary files for testing and plotting results are found in the `notebooks` folder.
TODO: more information about testing
