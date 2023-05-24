# Audio Classification Repository

This repository contains tools and scripts for audio classification tasks. It provides a framework for training audio classification models and visualizing the results.
## Installation

To clone the repository: 

`git clone https://github.com/Chanumask/AudioClassification.git`

Install the required dependencies using pip:

`pip install -r requirements.txt`

Make sure you have the necessary audio data and set the corresponding dataset path variables in `settings.py` before proceeding.

## Model training
To train an audio classification model, follow these steps:

1. Open `settings.py` and configure the dataset path, training loop, model parameters, and extensions.

2. Execute `main.py` to start the training process.

3. The results will be printed and saved in a JSON file along with the corresponding model parameters.

## Displaying and plotting results
To display and plot the training results, make the following changes in `settings.py`:

- Set `MONITORING` and `UPDATE_INTERVAL` to monitor training progress or only plot the final training metrics using `PLOT_RES`.
- Adjust the `AVG_SEEDS` parameter to accumulate results for different seeds and plot the experiments listed in `BARPLOT_SETTING`.
