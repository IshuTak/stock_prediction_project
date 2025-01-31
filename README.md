# Stock Price Prediction with LSTM and Sentiment Analysis

## Overview

This project predicts stock prices using an LSTM neural network combined with sentiment analysis from financial news articles. It incorporates technical indicators and sentiment scores to improve prediction accuracy.

## Features

-  **Data Collection**: Fetches historical stock data using `yfinance` and performs daily sentiment analysis using NewsAPI and NLTK's VADER.
- **Technical Indicators**: Calculates various technical indicators such as Moving Averages (MA), Moving Average Convergence Divergence (MACD), Relative Strength Index (RSI), and more.
- **Data Preparation**: Preprocesses and scales data, and prepares it for model training.
- **Modeling**: Builds and trains an LSTM neural network with optimized architecture and hyperparameters.
- **Prediction and Evaluation**: Makes future stock price predictions and evaluates model performance using metrics like MSE, RMSE, MAE, and MAPE.
- **Visualization**: Provides visualizations of actual vs. predicted stock prices and training loss graphs.
- **Monte Carlo Simulation**: Uses Monte Carlo simulations to estimate prediction confidence intervals.

  ## Dataset

- **Stock Data**: Historical stock prices for NVIDIA Corporation (NVDA) over the past three years.
- **News Data**: Financial news articles related to NVDA fetched using NewsAPI.
- **Technical Indicators**: Calculated from stock data.
- **Sentiment Scores**: Derived from news articles using NLTK's VADER sentiment analyzer.

## Installation

-pandas==1.4.2
-numpy==1.21.5
-yfinance==0.1.70
-nltk==3.6.7
-scikit-learn==1.0.2
-requests==2.27.1
-tensorflow==2.8.0
-matplotlib==3.5.1

### Prerequisites

- Python 3.6 or higher
- Git
- pip

### Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/IshuTak/stock_prediction_project.git
