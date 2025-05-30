from flask import Flask, render_template, request, jsonify
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from textblob import TextBlob
import json
import os
from collections import defaultdict
import time

app = Flask(__name__)

# Configuration
NEWS_API_KEY = "cd4b58d453854540abbd1dd8557fd199"  # Your NewsAPI key
ALPHA_VANTAGE_KEY = "15FNZ28OQ8RSECBF"  # Your Alpha Vantage API key

class SentimentAnalyzer:
    def __init__(self):
        self.cache = {}
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using TextBlob"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                return 'positive', polarity
            elif polarity < -0.1:
                return 'negative', polarity
            else:
                return 'neutral', polarity
        except:
            return 'neutral', 0.0

class NewsDataFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/everything"
    
    def fetch_news(self, query, days_back=7):
        """Fetch news articles from NewsAPI"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            params = {
                'q': query,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'sortBy': 'publishedAt',
                'apiKey': self.api_key,
                'language': 'en',
                'pageSize': 100
            }
            
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            articles = []
            if data.get('status') == 'ok':
                for article in data.get('articles', []):
                    articles.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'content': article.get('content', ''),
                        'published_at': article.get('publishedAt', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'url': article.get('url', '')
                    })
            
            return articles
        except Exception as e:
            print(f"Error fetching news: {e}")
            return []

class StockDataFetcher:
    def __init__(self, alpha_vantage_key=None):
        self.alpha_vantage_key = alpha_vantage_key
        self.av_base_url = "https://www.alphavantage.co/query"
    
    def get_stock_data(self, symbol, period="1mo"):
        """Fetch stock price data using yfinance as primary, Alpha Vantage as backup"""
        try:
            # Primary: Use yfinance
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            data = []
            for date, row in hist.iterrows():
                data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume'])
                })
            
            return data
        except Exception as e:
            print(f"Error fetching stock data from yfinance: {e}")
            # Fallback to Alpha Vantage
            return self.get_stock_data_alpha_vantage(symbol)
    
    def get_stock_data_alpha_vantage(self, symbol):
        """Fetch stock data from Alpha Vantage API"""
        try:
            if not self.alpha_vantage_key:
                print("Alpha Vantage API key not available")
                return []
            
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'apikey': self.alpha_vantage_key,
                'outputsize': 'compact'
            }
            
            response = requests.get(self.av_base_url, params=params)
            data = response.json()
            
            if 'Time Series (Daily)' in data:
                time_series = data['Time Series (Daily)']
                stock_data = []
                
                # Get last 30 days of data
                sorted_dates = sorted(time_series.keys(), reverse=True)[:30]
                
                for date in reversed(sorted_dates):  # Reverse to get chronological order
                    day_data = time_series[date]
                    stock_data.append({
                        'date': date,
                        'open': float(day_data['1. open']),
                        'high': float(day_data['2. high']),
                        'low': float(day_data['3. low']),
                        'close': float(day_data['4. close']),
                        'volume': int(day_data['5. volume'])
                    })
                
                return stock_data
            else:
                print(f"Error in Alpha Vantage response: {data}")
                return []
                
        except Exception as e:
            print(f"Error fetching stock data from Alpha Vantage: {e}")
            return []
    
    def get_real_time_quote(self, symbol):
        """Get real-time quote from Alpha Vantage"""
        try:
            if not self.alpha_vantage_key:
                return {}
            
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.alpha_vantage_key
            }
            
            response = requests.get(self.av_base_url, params=params)
            data = response.json()
            
            if 'Global Quote' in data:
                quote = data['Global Quote']
                return {
                    'symbol': quote.get('01. symbol', ''),
                    'price': float(quote.get('05. price', 0)),
                    'change': float(quote.get('09. change', 0)),
                    'change_percent': quote.get('10. change percent', '0%'),
                    'volume': int(quote.get('06. volume', 0)),
                    'latest_trading_day': quote.get('07. latest trading day', ''),
                    'previous_close': float(quote.get('08. previous close', 0))
                }
            
            return {}
        except Exception as e:
            print(f"Error fetching real-time quote: {e}")
            return {}
    
    def search_symbol(self, keywords):
        """Search for stock symbols using Alpha Vantage"""
        try:
            if not self.alpha_vantage_key:
                return []
            
            params = {
                'function': 'SYMBOL_SEARCH',
                'keywords': keywords,
                'apikey': self.alpha_vantage_key
            }
            
            response = requests.get(self.av_base_url, params=params)
            data = response.json()
            
            if 'bestMatches' in data:
                matches = []
                for match in data['bestMatches'][:10]:  # Limit to top 10 matches
                    matches.append({
                        'symbol': match.get('1. symbol', ''),
                        'name': match.get('2. name', ''),
                        'type': match.get('3. type', ''),
                        'region': match.get('4. region', ''),
                        'currency': match.get('8. currency', '')
                    })
                return matches
            
            return []
        except Exception as e:
            print(f"Error searching symbols: {e}")
            return []

class MarketSentimentAnalyzer:
    def __init__(self):
        self.news_fetcher = NewsDataFetcher(NEWS_API_KEY)
        self.stock_fetcher = StockDataFetcher(ALPHA_VANTAGE_KEY)
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def analyze_market_sentiment(self, asset, symbol, days_back=7):
        """Main function to analyze market sentiment"""
        # Fetch news data
        news_articles = self.news_fetcher.fetch_news(asset, days_back)
        
        # Fetch stock data
        stock_data = self.stock_fetcher.get_stock_data(symbol, f"{days_back}d")
        
        # Get real-time quote
        real_time_quote = self.stock_fetcher.get_real_time_quote(symbol)
        
        # Analyze sentiment for each article
        sentiment_data = []
        daily_sentiment = defaultdict(list)
        
        for article in news_articles:
            text = f"{article['title']} {article['description']}"
            sentiment, score = self.sentiment_analyzer.analyze_sentiment(text)
            
            article_date = article['published_at'][:10]  # Extract date part
            
            sentiment_item = {
                'title': article['title'],
                'sentiment': sentiment,
                'score': score,
                'date': article_date,
                'source': article['source'],
                'url': article['url']
            }
            
            sentiment_data.append(sentiment_item)
            daily_sentiment[article_date].append(score)
        
        # Calculate daily average sentiment
        daily_avg_sentiment = {}
        for date, scores in daily_sentiment.items():
            daily_avg_sentiment[date] = {
                'avg_score': np.mean(scores),
                'count': len(scores),
                'positive': len([s for s in scores if s > 0.1]),
                'negative': len([s for s in scores if s < -0.1]),
                'neutral': len([s for s in scores if -0.1 <= s <= 0.1])
            }
        
        return {
            'sentiment_data': sentiment_data,
            'daily_sentiment': daily_avg_sentiment,
            'stock_data': stock_data,
            'real_time_quote': real_time_quote,
            'summary': {
                'total_articles': len(sentiment_data),
                'positive_count': len([s for s in sentiment_data if s['sentiment'] == 'positive']),
                'negative_count': len([s for s in sentiment_data if s['sentiment'] == 'negative']),
                'neutral_count': len([s for s in sentiment_data if s['sentiment'] == 'neutral'])
            }
        }

# Initialize the analyzer
analyzer = MarketSentimentAnalyzer()

# Sample data for demo purposes (when API keys are not available)
def get_sample_data():
    """Generate sample data for demonstration"""
    dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7, 0, -1)]
    
    sample_sentiment_data = [
        {'title': 'Tesla Reports Strong Q4 Earnings', 'sentiment': 'positive', 'score': 0.8, 'date': dates[0], 'source': 'Reuters', 'url': '#'},
        {'title': 'Bitcoin Price Volatility Concerns Investors', 'sentiment': 'negative', 'score': -0.6, 'date': dates[1], 'source': 'Bloomberg', 'url': '#'},
        {'title': 'Apple Announces New Product Line', 'sentiment': 'positive', 'score': 0.7, 'date': dates[2], 'source': 'TechCrunch', 'url': '#'},
        {'title': 'Market Shows Mixed Signals', 'sentiment': 'neutral', 'score': 0.1, 'date': dates[3], 'source': 'CNBC', 'url': '#'},
        {'title': 'Tesla Stock Reaches New Heights', 'sentiment': 'positive', 'score': 0.9, 'date': dates[4], 'source': 'Yahoo Finance', 'url': '#'},
    ]
    
    daily_sentiment = {}
    for date in dates:
        daily_sentiment[date] = {
            'avg_score': np.random.uniform(-0.5, 0.5),
            'count': np.random.randint(5, 15),
            'positive': np.random.randint(2, 8),
            'negative': np.random.randint(1, 5),
            'neutral': np.random.randint(2, 6)
        }
    
    stock_data = []
    base_price = 150
    for i, date in enumerate(dates):
        price = base_price + np.random.uniform(-10, 10) + i * 2
        stock_data.append({
            'date': date,
            'open': price,
            'high': price + np.random.uniform(1, 5),
            'low': price - np.random.uniform(1, 5),
            'close': price + np.random.uniform(-2, 2),
            'volume': np.random.randint(1000000, 5000000)
        })
    
    real_time_quote = {
        'symbol': 'TSLA',
        'price': 155.50,
        'change': 2.50,
        'change_percent': '1.63%',
        'volume': 15000000,
        'latest_trading_day': datetime.now().strftime('%Y-%m-%d'),
        'previous_close': 153.00
    }
    
    return {
        'sentiment_data': sample_sentiment_data,
        'daily_sentiment': daily_sentiment,
        'stock_data': stock_data,
        'real_time_quote': real_time_quote,
        'summary': {
            'total_articles': len(sample_sentiment_data),
            'positive_count': len([s for s in sample_sentiment_data if s['sentiment'] == 'positive']),
            'negative_count': len([s for s in sample_sentiment_data if s['sentiment'] == 'negative']),
            'neutral_count': len([s for s in sample_sentiment_data if s['sentiment'] == 'neutral'])
        }
    }

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_sentiment():
    """API endpoint to analyze market sentiment"""
    try:
        data = request.get_json()
        asset = data.get('asset', 'Tesla')
        symbol = data.get('symbol', 'TSLA')
        days_back = int(data.get('days_back', 7))
        
        # Use real API data with both keys configured
        result = analyzer.analyze_market_sentiment(asset, symbol, days_back)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/quote/<symbol>')
def get_real_time_quote(symbol):
    """Get real-time quote for a symbol"""
    try:
        quote = analyzer.stock_fetcher.get_real_time_quote(symbol)
        return jsonify(quote)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/search/<keywords>')
def search_symbols(keywords):
    """Search for stock symbols"""
    try:
        results = analyzer.stock_fetcher.search_symbol(keywords)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/assets')
def get_popular_assets():
    """Get list of popular assets"""
    assets = [
        {'name': 'Tesla', 'symbol': 'TSLA'},
        {'name': 'Apple', 'symbol': 'AAPL'},
        {'name': 'Bitcoin', 'symbol': 'BTC-USD'},
        {'name': 'Amazon', 'symbol': 'AMZN'},
        {'name': 'Google', 'symbol': 'GOOGL'},
        {'name': 'Microsoft', 'symbol': 'MSFT'},
        {'name': 'Ethereum', 'symbol': 'ETH-USD'},
        {'name': 'Netflix', 'symbol': 'NFLX'},
        {'name': 'Meta', 'symbol': 'META'},
        {'name': 'NVIDIA', 'symbol': 'NVDA'}
    ]
    return jsonify(assets)

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'news_api': NEWS_API_KEY != "cd4b58d453854540abbd1dd8557fd199",
        'alpha_vantage_api': ALPHA_VANTAGE_KEY != "15FNZ28OQ8RSECBF",
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(debug=True)