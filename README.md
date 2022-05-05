# NBeats_Time_Series

## N-BEATS: Neural basis expansion analysis for interpretable time series forecasting

N-BEATS is [a type of neural network that was first described in a paper published in the 2019 ICLR conference by Oreshkin et al.](https://arxiv.org/abs/1905.10437) The authors reported that N-BEATS outperformed the M4 forecast competition winner by 3%. The M4 winner was a hybrid between a recurrent neural network and Holt-Winters exponential smoothing — whereas N-BEATS implements a “pure” deep neural architecture.

![nbeats pipeline](https://miro.medium.com/max/875/1*T9XlNOhCqwqVzWMXwkspOA.png)

In this repo, we will use **[nbeats_forecast](https://pepy.tech/project/nbeats-forecast)** which is an end to end library for univariate time series forecasting using N-BEATS. This library uses [nbeats-pytorch](https://github.com/philipperemy/n-beats) as base and simplifies the task of forecasting using N-BEATS by providing a interface similar to scikit-learn and keras.

### Requires: Python >=3.6

### [](https://github.com/amitesh863/nbeats_forecast#installation)Installation

``` python 
$ pip install nbeats_forecast
```

#### [](https://github.com/amitesh863/nbeats_forecast#import)Import
``` python
from nbeats_forecast import NBeats
```
#### Input

numpy array of size nx1

#### [](https://github.com/amitesh863/nbeats_forecast#output)Output

Forecasted values as numpy array of size mx1

Mandatory Parameters for the model:

-   data
-   period_to_forecast


A basic model with only mandatory parameters can be used to get forecasted values as shown below:
``` python
import pandas as pd
from nbeats_forecast import NBeats

data = pd.read_csv('data.csv')   
data = data.values        #univariate time series data of shape nx1 (numpy array)

model = NBeats(data=data, period_to_forecast=12)
model.fit()
forecast = model.predict()
```


The other optional parameters for the object of the NBeats model (as described in the paper) can be tweaked for better performance. If these parameters are not passed, default values as mentioned in the table below are considered.

| Parameter | Type | Default Value| Description|
| ------ | ------ | --------------|------------|
| backcast_length | integer | 3* period_to_forecast |Explained in the paper|
| path | string | '  ' |path to save intermediate training checkpoint |
| checkpoint_file_name | string | 'nbeats-training-checkpoint.th'| name for checkpoint file ending in format  .th |
|mode| string| 'cpu'| Any of the torch.device modes|
| batch_size | integer | len(data)/15 | size of batch|
|  thetas_dims | list of integers | [7, 8] | Explained in the paper|
| nb_blocks_per_stack | integer | 3| Explained in the paper|
| share_weights_in_stack | boolean | False | Explained in the paper|
| train_percent | float(below 1)  | 0.8 | Percentage of data to be used for training |
| save_checkpoint| boolean | False | save intermediate checkpoint files|
| hidden_layer_units | integer | 128 | hissen layer units|
| stack | list of integers | [1,1] | adding stacks in the model as per the paper passed in list as integer. Mapping is as follows -- 1: GENERIC_BLOCK,  2: TREND_BLOCK , 3: SEASONALITY_BLOCK|


## Repository Structure
#### Model

PyTorch implementation of N-BEATS can be found in  `models/nbeats.py`

#### Implementation
The `notebooks` directory contains a notebook with univariate time series analysis using the one of the time series in the provided dataset. The notebook consists of steps such as loading time series data, time series visualization, implementing the **NBeats** model and training it with 75% of the data. It also includes test on the remaining 25% data and evaluation metrics on the prediction along with a forecast plots as well.

#### Iteration
The `iteration` folder consists of a simple code that iterates over all the 50 datasets in the directory and produces NBeats model for each indivdual time series along with its prediction plot.

## How to use the repo

In order to use this basic repo for implementing NBeats model you can follow these basic steps:

 1. Fork or download this repository
 2. Install nbeats_forecast library on your system.
 3. Open the `notebooks/nbeats.ipynb` file and modify the file path as per your dataset.
 4. Run the notebook to get the NBeats model results for your dataset.
 5. Open the `iteration/nbeats_iter.py` file and modify the dataset path according to your directory.
 6. Run the iteration file to create NBeats models for all the time series in your directory.
