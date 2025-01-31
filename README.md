# Stock Price Prediction with LSTM and Sentiment Analysis

## Overview

This project predicts stock prices using an LSTM neural network combined with sentiment analysis from financial news articles. It incorporates technical indicators and sentiment scores to improve prediction accuracy.

## Features

- **Data Collection**: Fetches historical stock data using `yfinance`.
- **Sentiment Analysis**: Analyzes news sentiment using NLTK's VADER.
- **Technical Indicators**: Calculates moving averages, MACD, RSI, etc.
- **Modeling**: Builds and trains an LSTM neural network.
- **Visualization**: Plots actual vs. predicted stock prices.

## Installation

pandas==1.4.2
numpy==1.21.5
yfinance==0.1.70
nltk==3.6.7
scikit-learn==1.0.2
requests==2.27.1
tensorflow==2.8.0
matplotlib==3.5.1

### Prerequisites

- Python 3.6 or higher
- Git

### Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/stock_prediction_project.git