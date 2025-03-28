import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np

def analyze_sentiment_momentum():
    # Load the CSV data
    df = pd.read_csv('solana_tweets_analyzed.csv')
    
    # Convert date strings to datetime objects
    df['date'] = pd.to_datetime(df['date'])
    
    # Map sentiment strings to numeric values
    sentiment_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
    df['sentiment_value'] = df['sentiment'].map(sentiment_map)
    
    # Group by date and calculate the average sentiment
    daily_sentiment = df.groupby(df['date'].dt.date)['sentiment_value'].mean().reset_index()
    
    # Convert the sentiment averages back to string categories
    def map_avg_to_sentiment(avg):
        if avg > 0.33:
            return 'Positive'
        elif avg < -0.33:
            return 'Negative'
        else:
            return 'Neutral'
    
    daily_sentiment['sentiment_category'] = daily_sentiment['sentiment_value'].apply(map_avg_to_sentiment)
    
    # Ensure we have data for all 7 days (fill missing days with NaN)
    today = datetime.now().date()
    date_range = [today - timedelta(days=i) for i in range(6, -1, -1)]
    date_df = pd.DataFrame(date_range, columns=['date'])
    
    # Merge with our sentiment data
    complete_daily_sentiment = pd.merge(date_df, daily_sentiment, on='date', how='left')
    
    # Fill any missing days with neutral sentiment
    complete_daily_sentiment['sentiment_value'] = complete_daily_sentiment['sentiment_value'].fillna(0)
    complete_daily_sentiment['sentiment_category'] = complete_daily_sentiment['sentiment_category'].fillna('Neutral')
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Define colors for each sentiment
    colors = {'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'}
    bar_colors = [colors[cat] for cat in complete_daily_sentiment['sentiment_category']]
    
    # Plot bars with colors based on sentiment category
    plt.bar(complete_daily_sentiment['date'], complete_daily_sentiment['sentiment_value'], 
            color=bar_colors, width=0.7)
    
    # Add a horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add horizontal lines at the threshold values
    plt.axhline(y=0.33, color='green', linestyle='--', alpha=0.5)
    plt.axhline(y=-0.33, color='red', linestyle='--', alpha=0.5)
    
    # Add labels and title
    plt.title('Solana Sentiment Momentum (Last 7 Days)', fontsize=16)
    plt.ylabel('Sentiment Score', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    
    # Format the x-axis to show dates nicely
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    # Add a grid for better readability
    plt.grid(axis='y', alpha=0.3)
    
    # Add text labels on top of each bar
    for i, (date, value, category) in enumerate(zip(complete_daily_sentiment['date'], 
                                                   complete_daily_sentiment['sentiment_value'],
                                                   complete_daily_sentiment['sentiment_category'])):
        plt.text(date, value + 0.05 * (1 if value >= 0 else -1), 
                 f"{value:.2f}\n{category}", 
                 ha='center', va='bottom' if value >= 0 else 'top',
                 fontsize=9)
    
    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig('sentiment_momentum_plot.png', dpi=300)
    
    print("Analysis complete! Plot saved as 'sentiment_momentum_plot.png'")
    
    # Return the sentiment data for potential further analysis
    return complete_daily_sentiment

if __name__ == "__main__":
    sentiment_data = analyze_sentiment_momentum()
    print("\nDaily Sentiment Summary:")
    print(sentiment_data[['date', 'sentiment_value', 'sentiment_category']])