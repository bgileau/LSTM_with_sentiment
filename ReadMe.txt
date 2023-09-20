Libraries and how to run the code:

The below is necessary for LSTM. The NLP requirements is a more complex requirements section and requires special hardware to perform (GPU). So, read the NLP section for details regarding that and follow those separate instructions. Otherwise, just use the NLP results detailed below.

The requirements.txt file is a list of all the libraries installed on my working environment. 
However this environment is a general environment with lot's of libararies. These are my specific imports for this project:

import math
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import tensorflow.keras
from tensorflow.keras import layers
import yfinance as yf
from keras.models import Model
from keras.layers import Input, LSTM, Dense, concatenate, Flatten
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error


Notebooks Files:

There are two Notebooks:

LSTM Final.ipynb which run's the LSTM code with no sentiment 
LSTM Sentiment FINAL.ipynb which run's the LSTM code with sentiment

Directory Structure:

In the working directory there will be:
LSTM Final.ipynb
LSTM Sentiment FINAL.ipynb
sentiment_analysis_results.csv
Stock_data.csv - this data is imported using the yfinance function and is not needed to import into the notebook file.


There will also be the two nested directories LSTM and Sentiment where the graphs will be saved.


In order to run you can restart and run all for each notebook. This will run every cell which will include data preparation, training the model, and save the output charts in the correct subfolders.

I have also saved the notebooks as python files as backup as well.
