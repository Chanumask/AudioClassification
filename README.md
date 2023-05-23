# Audio Classification Repository

## Installation

To clone the repository: 

`git clone https://github.com/Chanumask/AudioClassification.git`

Install the required dependencies. You can use pip to install them:

`pip install -r requirements.txt`

Make sure you have the necessary audio data and set the according dataset path variables in _settings.py_.


## Model training
In _settings.py_:

check the Dataset you want to use, and set its path

set Training Loop and Model parameters and extensions

execute main.py

## Displaying and plotting results
In _settings.py_:

Use `MONITORING` and `UPDATE_INTERVAL` to monitor training or only plot the final training metrics using `PLOT_RES`

`AVG_SEEDS` will return a table of the results accumulating the different seeds for each run and plotting the experiments listed in `BARPLOT_SETTING`