Beijing PM2.5 Time Series Forecasting
A deep learning project focused on predicting PM2.5 air pollution levels using LSTM and GRU networks — implemented both from scratch and using PyTorch's built-in modules.

📌 Overview
This project uses the Beijing PM2.5 dataset from the UCI Machine Learning Repository to predict fine particulate matter concentration (PM2.5) based on weather and time variables.
The task is treated as a sequence prediction problem, using models such as:

Custom LSTM (built from scratch)

Custom GRU (built from scratch)

PyTorch nn.LSTM

PyTorch nn.GRU

📂 Dataset Description
The dataset includes the following features:

pm2.5: target variable (µg/m³)

DEWP, TEMP, PRES: weather metrics

cbwd: combined wind direction (categorical)

Iws, Is, Ir: cumulative wind speed, snow hours, and rain hours

year, month, day, hour: used to build the datetime index

📁 Dataset source: UCI Machine Learning Repository

⚙️ Workflow
✅ Task 1: Data Preprocessing
Handled missing values

Created datetime index

One-hot encoded cbwd

Scaled features using MinMaxScaler

Created sequences of length 30

Split into train/test (80/20)

Converted to PyTorch tensors

✅ Task 2: Custom LSTM
Implemented LSTM cell from scratch with forget, input, output gates

Manually iterated over time steps

Trained and tracked loss

✅ Task 3: Custom GRU
Built GRU cell with update and reset gates

Forward pass manually implemented over sequences

Compared performance vs. LSTM

✅ Task 4: Built-in LSTM and GRU
Used PyTorch nn.LSTM and nn.GRU

Connected to a linear layer for prediction

Tracked training and test performance

✅ Task 5: Evaluation
Metrics:

RMSE

MAE

Pseudo-accuracy (±20% tolerance)

Plotted predictions vs actual values

📊 Key Insights
GRU vs LSTM: GRU performed similarly (or better) on this dataset. Simpler architecture = faster convergence and regularization.

Custom vs Built-in: PyTorch built-ins are more efficient and numerically stable. Custom models are great for learning but slower.

🔍 Analytical Questions Answered
Why LSTM is resistant to vanishing gradients

When GRU may outperform LSTM

How gate coupling affects memory control

When even LSTMs can fail

Pros and cons of simplification in recurrent networks

🛠 Tech Stack
Python, Jupyter Notebook

PyTorch

NumPy, Pandas

Matplotlib

Scikit-learn
