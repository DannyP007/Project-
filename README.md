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

Result:
(https://github.com/DannyP007/Project-/blob/main/Images/Result/Rf%20Result.png)

### 2. Gated Recurrent Unit (GRU)

GRU is another type of RNN similar to LSTM but with a simpler architecture. It combines the input gate and forget gate into a single update gate, which makes GRUs computationally more efficient than LSTMs. The main key features is simple architecture compared to LSTM, making it faster to train and effectively captures sequential dependencies in time series data and suitable for scenarios with limited data or computational power. The model takes sequences 60 sequence length of stock prices input. There are two GRU layers with 50 units each. The first GRU layer returns sequences to pass the data through the next GRU layer, while the second layer does not return sequences, outputting a single vector. Using The Adam optimizer with a learning rate of 0.001 is used for training. The model is trained on 80% of the dataset with 20% reserved for testing. The model is trained for 10 epochs with a batch size of 32, and 10% of the training data is used for validation and using same matrix for reach the better model.

Result:
(https://github.com/DannyP007/Project-/blob/main/Images/Result/download(1).png).

### 3. Random Forest
Random Forest is an ensemble learning method based on decision trees. It operates by constructing a multitude of decision trees during training and outputting the mean prediction (for regression tasks) of the individual trees. Unlike LSTM and GRU, which are deep learning models, Random Forest is a traditional machine learning algorithm that doesn’t inherently consider sequential information. which is robust to overfitting due to ensemble learning and performs well with relatively little data preprocessing. Also, useful as a baseline for comparison with more complex models.
The input features for each stock using group by function such as a 7-day SMA, 30-day SMA, 12-day EMA, 26-day EMA, and RSI. Also, used lag Features, stock prices for 1, 2, 5, and 10 days.These features are normalized using MinMaxScaler for consistent scaling. Then,training Data for each company is split into 80% training and 20% testing data. The Random Forest model is trained using 100 decision trees (n_estimators=100) with a fixed random state (random_state=42) for reproducibility.

Result:
(https://github.com/DannyP007/Project-/blob/main/Images/Result/Rf%20Result.png).

## Result And Analysis

After evaluating the performance of all three models—GRU, LSTM, and Random Forest—based on key metrics (MSE, RMSE, and MAE), the results indicate that the GRU model significantly outperformed the others.The GRU model delivered the most accurate results with an MSE of 0.008, RMSE of 0.09, and MAE of 0.06. These metrics suggest that the GRU model closely aligns with the actual stock prices, making it the most reliable model for this prediction task.
The LSTM model also performed well, with evaluation scores close to those of the GRU model. While the LSTM model shows good predictive capabilities, it slightly underperforms compared to GRU, with slightly higher error metrics.
The Random Forest model, while effective in other scenarios, did not perform well with this dataset. Its higher error metrics indicate that it struggled to capture the patterns in the time series data as effectively as the GRU and LSTM models. This model is less suited for sequential data like stock prices in this case.


