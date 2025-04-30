import matplotlib.pyplot as plt
from afinn import Afinn
import pandas as pd
import numpy as np
from datetime import datetime
import random  # For simulating company metrics

def load_companies():
    """Load company data from CSV"""
    try:
        companies = pd.read_csv('Company.csv')
        return companies
    except FileNotFoundError:
        print("Error: Company.csv not found!")
        return None

def analyze_sentiment(text):
    """
    Analyze sentiment of the given text using AFINN
    """
    afinn = Afinn()
    score = afinn.score(text)
    return score

def simulate_company_metrics(sentiment_score):
    """Simulate company metrics based on sentiment score"""
    # These are simulated values - in a real scenario, you'd use actual company data
    base_price = 100.0
    base_volume = 1000000
    
    # Simulate price change based on sentiment
    price_change = sentiment_score * 0.1
    new_price = base_price + price_change
    
    # Simulate volume change based on sentiment
    volume_change = abs(sentiment_score) * 10000
    new_volume = base_volume + volume_change
    
    return new_price, new_volume

def plot_analysis(sentiment_scores, dates, prices, volumes, company_name):
    """Plot sentiment analysis and company metrics"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Sentiment Analysis
    ax1.plot(dates, sentiment_scores, marker='o', linestyle='-', color='blue')
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_title(f'Sentiment Analysis for {company_name}')
    ax1.set_ylabel('Sentiment Score')
    ax1.grid(True)
    
    # Plot 2: Price Impact
    ax2.plot(dates, prices, marker='o', linestyle='-', color='green')
    ax2.set_title('Simulated Price Impact')
    ax2.set_ylabel('Price ($)')
    ax2.grid(True)
    
    # Plot 3: Volume Impact
    ax3.plot(dates, volumes, marker='o', linestyle='-', color='purple')
    ax3.set_title('Simulated Trading Volume Impact')
    ax3.set_ylabel('Volume')
    ax3.grid(True)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    # Load company data
    companies = load_companies()
    if companies is None:
        return
    
    # Display available companies
    print("\nAvailable Companies:")
    for idx, row in companies.iterrows():
        print(f"{idx + 1}. {row['company_name']} ({row['ticker_symbol']})")
    
    # Get company selection
    while True:
        try:
            choice = int(input("\nSelect a company (enter number): ")) - 1
            if 0 <= choice < len(companies):
                selected_company = companies.iloc[choice]
                break
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
    
    print(f"\nSelected Company: {selected_company['company_name']} ({selected_company['ticker_symbol']})")
    
    # Initialize lists to store data
    sentiment_scores = []
    dates = []
    prices = []
    volumes = []
    
    print("\nEnter text for sentiment analysis (type 'quit' to exit):")
    
    while True:
        text = input("\nEnter text to analyze: ")
        if text.lower() == 'quit':
            break
            
        score = analyze_sentiment(text)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Get simulated metrics
        price, volume = simulate_company_metrics(score)
        
        sentiment_scores.append(score)
        dates.append(current_time)
        prices.append(price)
        volumes.append(volume)
        
        print(f"\nSentiment Score: {score}")
        print(f"Interpretation: {'Positive' if score > 0 else 'Negative' if score < 0 else 'Neutral'}")
        print(f"Simulated Price Impact: ${price:.2f}")
        print(f"Simulated Volume Impact: {volume:,.0f}")
    
    if sentiment_scores:
        # Create DataFrame
        df = pd.DataFrame({
            'Date': dates,
            'Sentiment_Score': sentiment_scores,
            'Price': prices,
            'Volume': volumes
        })
        
        # Save results to CSV
        df.to_csv(f'sentiment_results_{selected_company["ticker_symbol"]}.csv', index=False)
        
        # Plot results
        plot_analysis(sentiment_scores, dates, prices, volumes, selected_company['company_name'])
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"Average Sentiment: {np.mean(sentiment_scores):.2f}")
        print(f"Maximum Sentiment: {max(sentiment_scores)}")
        print(f"Minimum Sentiment: {min(sentiment_scores)}")
        print(f"Total Samples: {len(sentiment_scores)}")
        print(f"Final Price: ${prices[-1]:.2f}")
        print(f"Final Volume: {volumes[-1]:,.0f}")

if __name__ == "__main__":
    main() 