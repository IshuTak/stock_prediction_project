
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import RobustScaler
from sklearn.utils import shuffle
import requests
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')


try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

def get_stock_data(ticker, start_date, end_date):
    
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if stock_data.empty:
            print(f"No data found for {ticker}")
            return None
        print(f"Successfully downloaded stock data for {ticker}")
        return stock_data
    except Exception as e:
        print(f"Error fetching stock data: {str(e)}")
        return None

def get_news_sentiment(ticker, api_key, start_date, end_date):
    
    import numpy as np  # Ensure numpy is imported here
    if not api_key:
        print("Warning: No NewsAPI key provided. Skipping sentiment analysis.")
        return pd.DataFrame()  # Return an empty DataFrame

    base_url = "https://newsapi.org/v2/everything"
    date_range = pd.date_range(start=start_date, end=end_date)
    sentiment_data = []

    for single_date in date_range:
        params = {
            'q': ticker,
            'apiKey': api_key,
            'language': 'en',
            'sortBy': 'publishedAt',
            'from': single_date.strftime('%Y-%m-%d'),
            'to': single_date.strftime('%Y-%m-%d'),
            'pageSize': 100
        }

        try:
            response = requests.get(base_url, params=params)
            news_data = response.json()

            if response.status_code != 200:
                print(f"API Error: {news_data.get('message', 'Unknown error')}")
                continue

            sia = SentimentIntensityAnalyzer()
            sentiments = []

            for article in news_data.get('articles', []):
                title = article.get('title', '') or ''
                description = article.get('description', '') or ''
                text = f"{title} {description}".strip()

                if text:
                    sentiment = sia.polarity_scores(text)
                    sentiments.append(sentiment['compound'])

            avg_sentiment = np.mean(sentiments) if sentiments else 0
            sentiment_data.append({'Date': single_date, 'Sentiment': avg_sentiment})

        except Exception as e:
            print(f"Error in sentiment analysis for date {single_date}: {str(e)}")
            continue

    sentiment_df = pd.DataFrame(sentiment_data).set_index('Date')
    return sentiment_df

def calculate_technical_indicators(data):
    
    df = data.copy()

    
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()

    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))

    
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()

    
    df = df.fillna(method='bfill').fillna(method='ffill')
    return df

def prepare_data(stock_data, sequence_length):
    
    features = ['Close', 'Volume', 'MA5', 'MA20', 'MA50', 'RSI', 'MACD',
                'Signal_Line', 'Returns', 'Volatility', 'Volume_MA', 'Sentiment']

    data = stock_data[features].astype(float)
    data = data.fillna(method='ffill').fillna(method='bfill')

    feature_scaler = RobustScaler()
    data_scaled = feature_scaler.fit_transform(data)

    target_scaler = RobustScaler()
    target_scaled = target_scaler.fit_transform(data[['Close']])

    X, y = [], []
    for i in range(sequence_length, len(data_scaled)):
        X.append(data_scaled[i-sequence_length:i])
        y.append(target_scaled[i, 0])

    X = np.array(X)
    y = np.array(y)

    
    X, y = shuffle(X, y, random_state=42)

    return X, y, feature_scaler, target_scaler

def create_model(sequence_length, n_features):
    
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(sequence_length, n_features)),
        BatchNormalization(),
        Dropout(0.3),

        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),

        Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        BatchNormalization(),
        Dropout(0.2),

        Dense(1, activation='linear')
    ])

    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer,
                  loss='mse',
                  metrics=['mae', 'mse'])

    return model

def main():
    
    ticker = 'NVDA'
    sequence_length = 90  # Increased to capture longer-term trends
    news_api_key = 'YOUR_NEWSAPI_KEY'  # Replace with your NewsAPI key

    try:
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*3)  # 3 years of data
        print(f"Fetching data for {ticker} from {start_date.date()} to {end_date.date()}")

        
        stock_data = get_stock_data(ticker, start_date, end_date)
        if stock_data is None or stock_data.empty:
            return

        
        print("\nFetching news sentiment...")
        sentiment_df = get_news_sentiment(ticker, news_api_key, start_date, end_date)

        
        stock_data = stock_data.merge(sentiment_df, left_index=True, right_index=True, how='left')
        
        stock_data['Sentiment'] = stock_data['Sentiment'].fillna(method='ffill').fillna(0)

        
        print("\nCalculating technical indicators...")
        stock_data = calculate_technical_indicators(stock_data)

        
        print("Preparing data for model...")
        X, y, feature_scaler, target_scaler = prepare_data(stock_data, sequence_length)

       
        split_idx = int(len(X) * 0.9)  # Using 90% for training
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        print(f"Training set size: {X_train.shape}")
        print(f"Test set size: {X_test.shape}")

        
        print("\nCreating and training model...")
        model = create_model(sequence_length, X.shape[2])

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,  # Increased patience
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,  # More aggressive learning rate reduction
                patience=10,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                'best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]

        
        history = model.fit(
            X_train, y_train,
            epochs=200,          # Increased epochs
            batch_size=16,       # Decreased batch size
            validation_split=0.1,
            callbacks=callbacks,
            verbose=1
        )

        
        print("\nMaking predictions...")
        predictions = model.predict(X_test)

        
        predictions_actual = target_scaler.inverse_transform(predictions)
        y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1))

        
        mse = np.mean((predictions_actual - y_test_actual) ** 2)
        mae = np.mean(np.abs(predictions_actual - y_test_actual))
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test_actual - predictions_actual) / y_test_actual)) * 100

        print(f'\nModel Performance:')
        print(f'Mean Squared Error: {mse:.2f}')
        print(f'Root Mean Squared Error: {rmse:.2f}')
        print(f'Mean Absolute Error: {mae:.2f}')
        print(f'Mean Absolute Percentage Error: {mape:.2f}%')

        
        last_sequence = X[-1:]
        predictions = []
        for _ in range(10):  # Monte Carlo simulation
            pred = model.predict(last_sequence)
            predictions.append(pred[0][0])

        next_day_prediction = np.mean(predictions)
        prediction_std = np.std(predictions)

        next_day_prediction_actual = target_scaler.inverse_transform([[next_day_prediction]])[0][0]

        current_price = stock_data['Close'].iloc[-1]
        price_change = ((next_day_prediction_actual - current_price) / current_price) * 100

        print(f'\nPredictions:')
        print(f'Current Price: ${current_price:.2f}')
        print(f'Predicted Next Day: ${next_day_prediction_actual:.2f}')
        print(f'Predicted Change: {price_change:.2f}%')
        print(f'95% Confidence Interval: ${next_day_prediction_actual - 1.96 * prediction_std:.2f} to ${next_day_prediction_actual + 1.96 * prediction_std:.2f}')

        
        plt.figure(figsize=(15, 7))
        plt.plot(y_test_actual, label='Actual', alpha=0.8)
        plt.plot(predictions_actual, label='Predicted', alpha=0.8)
        plt.title(f'Stock Price Prediction for {ticker}')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()

        
        plt.figure(figsize=(15, 7))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
