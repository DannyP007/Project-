# Project-
Indian stocks data prediction using machine learning algorithum

## Project Description 

Nowadays, increasing the trend of stocks market, cryptocurrency, forex, and many more factors everyone want to invest but do have proper knowledge. We all know about that the machine learning model make a big difference in financial sectors because of the find a solution for investors as well as company. In this project stocks analysis and forecasting I will make and use different timeseries model for predicting stocks price based on their historical data which will be useful for new investor and company. In this project i will made three difrent machine learning model LSTM, GRU and Random Forest.

## Exploratory Data Analysis

In this section I download dataset from the Yahoo finance (https://github.com/DannyP007/Project-/blob/main/Dataset.ipynb). This dataset provide the 7 different features for every stocks such as a Open price, Low price, Volume, and so on. In this data there are 20 diffrent company stocks data.
![ Adjusted Close Price for 5 stocks (Nifty50)] (https://github.com/DannyP007/Project-/tree/main/Images/EDA).

## Technical Analysis

The technical analysis use for finding different chart pattern, identify the trend of the market. In this project i used simple moving avarage (SMA), EMA and Relative Strength Index (RSI) technical indicators.
I visuliazation for one stock and apply using group by function through whole dataset.(https://github.com/DannyP007/Project-/blob/main/Images/EDA/download%20(2).png)
(https://github.com/DannyP007/Project-/blob/main/Images/EDA/download%20(3).png).

## Model Building 
### 1. Long Short-Term Memory (LSTM)
LSTM is a type of recurrent neural network (RNN) specifically designed to handle sequential data. There are three main key features first is capture the short and long term dependecies and designed to overcome the vanishing gradient problem in standard RNNs and widely used for time series forecasting and sequence prediction tasks. The model takes sequences of stock prices (60 timesteps) as input sequence length.In this LSTM model using three LSTM layers with 64, 64, and 32 units, respectively. The first two LSTM layers return sequences to pass the data through the next LSTM layers. The final LSTM layer reduces the output to a single vector. Dropout Layers for reduce the risk of overfitting.
In the training Data the model is trained on 80% of the dataset with 20% reserved for testing. the training Process the model is trained for 10 epochs with a batch size of 32, and validation is performed on 10% of the training data.
After training, the model is evaluated using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE). The predictions and actual values are scaled back using a MinMaxScaler for accurate performance assessment.




