# A Bit More Bayesian: Domain-Invariant Learning with Uncertainty
Code for paper "A Bit More Bayesian: Domain-Invariant Learning with Uncertainty" submitted to ICML 2021.

### Prerequisites
 - Python 3.6.9
 - Pytorch 1.1.0

### Components
 - `../kfold/`: Directory of images
 - `../files/`: Directory of the train/validation/test split txt files
 - `pacs_main.py`: script to run classification experiments on PACS
 - `pacs_model.py`: the model used in `pacs_main.py`
 - `pacs_datas.py`: script to load data from PACS for the experiments
 - `pacs_test.py`: script to evaluate the trained model
 - `./logs/`: folder to store the trained model
 - `augs.py`: data augmentation functions for the experiments
 - `utils.py`: assorted functions to support the repository
 


## Setup
The code is for the PACS dataset. Download the datasets from the following link (or the link in the main paper), extract the compressed file, and place the images in `../kfold/` directory and the train/validation/test split txt files in the `../files/` directory.

[[Google Drive](https://drive.google.com/drive/folders/0B6x7gtvErXgfUU1WcGY5SzdwZVk)]

## Training
For training the model run the following:
```
python pacs_main.py --test_domain cartoon --log_dir model_name
```
Change the `cartoon` after `--test_domain` to `art_painting`, `photo` or `sketch` to change the target domain.
Use `--classifier SGP/NO` to choose Bayesian invariant classifier or deterministic classifier. The default value is `--classifier SGP`.
Use `--feature bayes/no` to choose Bayesian invariant feature extractor or deterministic one. The default value is `--feature bayes`.
Use `--classifier NO --feature no` to train the baseline method.
The trained model and logs will be stored in `./logs/model_name/`

## Evaluation
For evaluation of the trained model run the following:
```
python pacs_test.py --test_domain cartoon --log_dir cartoon_model
```
Change the target domain as the training phase.
Change the `cartoon_model` to other names to evaluate other trained models
