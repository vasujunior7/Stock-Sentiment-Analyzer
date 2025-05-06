import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
from afinn import Afinn
import plotly.express as px
from plotly.subplots import make_subplots
import random
import time
from streamlit.components.v1 import html
import re
from textblob import TextBlob

# Set page config with wide layout and custom theme
st.set_page_config(
    page_title="Stock Sentiment Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with animations and 3D effects
st.markdown("""
    <style>
    /* Main background with gradient and animation */
    .main {
        background: linear-gradient(135deg, #0E1117 0%, #1a1a2e 100%);
        animation: gradient 15s ease infinite;
        background-size: 400% 400%;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Card styling with 3D effect */
    .card {
        background: rgba(30, 30, 30, 0.8);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        transition: transform 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
    }
    
    /* Button styling with glow effect */
    .stButton>button {
        background: linear-gradient(45deg, #1E88E5, #00BCD4);
        color: white;
        border-radius: 25px;
        padding: 10px 25px;
        font-weight: bold;
        border: none;
        box-shadow: 0 0 20px rgba(30, 136, 229, 0.5);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        box-shadow: 0 0 30px rgba(30, 136, 229, 0.8);
        transform: scale(1.05);
    }
    
    /* Input field styling */
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea {
        background: rgba(30, 30, 30, 0.8);
        color: white;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 10px;
    }
    
    /* Select box styling */
    .stSelectbox>div>div>select {
        background: rgba(30, 30, 30, 0.8);
        color: white;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Metric styling */
    .stMetric {
        background: rgba(30, 30, 30, 0.8);
        border-radius: 15px;
        padding: 15px;
        margin: 10px;
        box-shadow: 0 4px 16px 0 rgba(31, 38, 135, 0.37);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(30, 30, 30, 0.8);
        border-radius: 15px;
        padding: 20px;
        margin: 10px;
    }
    
    /* Title styling */
    h1 {
        color: #1E88E5;
        text-align: center;
        font-size: 2.5em;
        text-shadow: 0 0 10px rgba(30, 136, 229, 0.5);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { text-shadow: 0 0 10px rgba(30, 136, 229, 0.5); }
        to { text-shadow: 0 0 20px rgba(30, 136, 229, 0.8); }
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(30, 30, 30, 0.8);
        border-radius: 10px;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(30, 136, 229, 0.2);
    }
    
    .stTabs [aria-selected="true"] {
        background: #1E88E5;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize sentiment analyzers
afinn = Afinn()

def load_companies():
    """Load company data from CSV"""
    try:
        companies = pd.read_csv('Company.csv')
        return companies
    except FileNotFoundError:
        st.error("Error: Company.csv not found!")
        return None

def clean_text(text):
    """Clean and preprocess text for better sentiment analysis"""
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user @ references and '#' from tweet
    text = re.sub(r'\@\w+|\#\w+', '', text)
    # Remove punctuations and numbers
    text = re.sub(r'[^\w\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_enhanced_sentiment(text):
    """Get enhanced sentiment score using multiple analyzers"""
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Get AFINN sentiment
    afinn_score = afinn.score(cleaned_text)
    
    # Get TextBlob sentiment
    blob = TextBlob(cleaned_text)
    textblob_score = blob.sentiment.polarity * 10  # Scale up to match AFINN range
    
    # Custom sentiment boosters/reducers
    positive_words = ['excellent', 'amazing', 'great', 'good', 'positive', 'success', 'profit', 'growth', 'innovative']
    negative_words = ['bad', 'poor', 'terrible', 'negative', 'loss', 'decline', 'fail', 'risk', 'crash']
    
    # Count positive and negative words
    positive_count = sum(1 for word in cleaned_text.split() if word in positive_words)
    negative_count = sum(1 for word in cleaned_text.split() if word in negative_words)
    
    # Calculate word impact
    word_impact = (positive_count - negative_count) * 2
    
    # Combine scores with weights
    final_score = (afinn_score * 0.4 + textblob_score * 0.4 + word_impact * 0.2)
    
    # Normalize score to be between -5 and 5
    final_score = max(min(final_score, 5), -5)
    
    return final_score

def get_company_sensitivity(company_name):
    """Get company-specific sensitivity to sentiment"""
    sensitivities = {
        'Apple': 2.5,      # More sensitive to market sentiment
        'Google Inc': 2.0, # Tech companies tend to be more volatile
        'Amazon.com': 2.8, # E-commerce giant, highly sensitive
        'Tesla Inc': 3.0,  # Known for high volatility
        'Microsoft': 1.8,  # Relatively stable but still sensitive
    }
    return sensitivities.get(company_name, 2.0)

def simulate_company_metrics(sentiment_score, company_name, base_price=100.0, base_volume=1000000):
    """Simulate company metrics based on sentiment score with company-specific sensitivity"""
    sensitivity = get_company_sensitivity(company_name)
    
    # Enhanced sentiment impact calculation
    adjusted_score = sentiment_score * sensitivity
    
    # Calculate price change with more significant impact
    price_change_percentage = adjusted_score * (1.5 + random.uniform(-0.3, 0.3))
    
    # Apply the percentage change to the base price
    price_change = (base_price * price_change_percentage) / 100
    new_price = base_price + price_change
    
    # Volume changes are now more dramatic for significant sentiment scores
    volume_multiplier = 1 + (abs(sentiment_score) / 5)
    volume_change = base_volume * (volume_multiplier - 1) * random.uniform(0.8, 1.2)
    new_volume = base_volume + volume_change
    
    return new_price, new_volume

def create_dynamic_chart(prices, dates, sentiment_scores, company_name):
    """Create interactive 3D chart showing price movement and sentiment"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       vertical_spacing=0.1,
                       subplot_titles=(f'Price Movement for {company_name}', 'Sentiment Score'))
    
    # Calculate price changes for color coding
    price_changes = np.diff(prices)
    colors = ['#FF5252' if change < 0 else '#4CAF50' for change in price_changes]
    
    # Add price line with gradient and color-coded segments
    for i in range(len(prices)-1):
        fig.add_trace(
            go.Scatter(x=dates[i:i+2], y=prices[i:i+2], 
                      name='Price',
                      line=dict(color=colors[i], width=3),
                      fill='tozeroy',
                      fillcolor='rgba(255, 82, 82, 0.2)' if price_changes[i] < 0 else 'rgba(76, 175, 80, 0.2)',
                      showlegend=False),
            row=1, col=1
        )
    
    # Add price markers
    fig.add_trace(
        go.Scatter(x=dates, y=prices,
                  mode='markers',
                  marker=dict(
                      size=8,
                      color=['#FF5252' if p < prices[0] else '#4CAF50' for p in prices],
                      line=dict(width=2, color='white')
                  ),
                  name='Price Points',
                  showlegend=False),
        row=1, col=1
    )
    
    # Add sentiment line with gradient
    fig.add_trace(
        go.Scatter(x=dates, y=sentiment_scores, name='Sentiment',
                  line=dict(color='#1E88E5', width=3),
                  fill='tozeroy',
                  fillcolor='rgba(30, 136, 229, 0.2)'),
        row=2, col=1
    )
    
    # Add sentiment markers
    fig.add_trace(
        go.Scatter(x=dates, y=sentiment_scores,
                  mode='markers',
                  marker=dict(
                      size=8,
                      color=['#FF5252' if s < 0 else '#4CAF50' for s in sentiment_scores],
                      line=dict(width=2, color='white')
                  ),
                  name='Sentiment Points',
                  showlegend=False),
        row=2, col=1
    )
    
    # Add annotations for significant price changes
    for i in range(1, len(prices)):
        if abs(prices[i] - prices[i-1]) > 0.5:  # Only show significant changes
            fig.add_annotation(
                x=dates[i],
                y=prices[i],
                text=f"{prices[i]-prices[i-1]:+.2f}",
                showarrow=True,
                arrowhead=1,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='#FF5252' if prices[i] < prices[i-1] else '#4CAF50',
                font=dict(size=12, color='white'),
                bgcolor='rgba(0,0,0,0.8)',
                bordercolor='white',
                borderwidth=1,
                borderpad=4,
                row=1, col=1
            )
    
    # Update layout with 3D effect and enhanced styling
    fig.update_layout(
        height=600,
        showlegend=True,
        template="plotly_dark",
        margin=dict(l=50, r=50, t=50, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='rgba(30, 30, 30, 0.8)',
            font_size=12,
            font_family="Rockwell"
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)',
            zerolinecolor='rgba(255, 255, 255, 0.2)'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)',
            zerolinecolor='rgba(255, 255, 255, 0.2)'
        ),
        xaxis2=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)',
            zerolinecolor='rgba(255, 255, 255, 0.2)'
        ),
        yaxis2=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)',
            zerolinecolor='rgba(255, 255, 255, 0.2)'
        )
    )
    
    # Add shapes for price movement visualization
    for i in range(1, len(prices)):
        fig.add_shape(
            type="rect",
            x0=dates[i-1],
            x1=dates[i],
            y0=min(prices[i-1], prices[i]),
            y1=max(prices[i-1], prices[i]),
            fillcolor='rgba(255, 82, 82, 0.1)' if prices[i] < prices[i-1] else 'rgba(76, 175, 80, 0.1)',
            line=dict(width=0),
            row=1, col=1
        )
    
    return fig

def create_analysis_card(sentiment_score, analysis_text):
    """Create a detailed analysis card with sentiment results"""
    if sentiment_score > 0:
        sentiment_color = '#4CAF50'  # Green for positive
    elif sentiment_score < 0:
        sentiment_color = '#F44336'  # Red for negative
    else:
        sentiment_color = '#9E9E9E'  # Grey for neutral

    sentiment_label = 'Positive' if sentiment_score > 0 else 'Negative' if sentiment_score < 0 else 'Neutral'
    
    return f"""
    <div class='card' style='padding: 20px; margin: 10px 0;'>
        <div style='display: flex; justify-content: space-between; margin-bottom: 15px;'>
            <div>
                <h3 style='color: {sentiment_color};'>Sentiment Score: {sentiment_score:.2f}</h3>
                <p>Overall Sentiment: {sentiment_label}</p>
            </div>
        </div>
        <div style='margin-bottom: 15px;'>
            <h5>Analysis</h5>
            <p>{analysis_text}</p>
        </div>
    </div>
    """

def main():
    # Add animated title
    st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: #1E88E5; text-shadow: 0 0 10px rgba(30, 136, 229, 0.5);'>
            📊 Stock Sentiment Analyzer
        </h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Load companies
    companies = load_companies()
    if companies is None:
        return
    
    # Sidebar with 3D effect
    with st.sidebar:
        st.markdown("""
        <div class='card'>
            <h3 style='color: #1E88E5;'>Company Selection</h3>
        </div>
        """, unsafe_allow_html=True)
        
        selected_companies = st.multiselect(
            "Select Companies",
            options=companies['company_name'].tolist(),
            default=companies['company_name'].tolist()[:2]
        )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class='card'>
            <h3 style='color: #1E88E5;'>Market Analysis Input</h3>
        </div>
        """, unsafe_allow_html=True)
        
        tweet_text = st.text_area(
            "Enter market news or analysis:",
            height=100,
            placeholder="What's happening with these companies? (News, analysis, or market sentiment)"
        )
        
        if st.button("Analyze Impact", key="analyze_button"):
            if tweet_text and selected_companies:
                with st.spinner('Analyzing market sentiment...'):
                    # Get sentiment analysis using TextBlob and AFINN
                    sentiment_score = get_enhanced_sentiment(tweet_text)
                    
                    # Create analysis text
                    blob = TextBlob(tweet_text)
                    analysis_text = f"The text shows {'positive' if sentiment_score > 0 else 'negative' if sentiment_score < 0 else 'neutral'} sentiment with a score of {sentiment_score:.2f}. "
                    analysis_text += f"The text has a polarity of {blob.sentiment.polarity:.2f} and subjectivity of {blob.sentiment.subjectivity:.2f}."
                    
                    # Display detailed analysis card
                    st.markdown(
                        create_analysis_card(sentiment_score, analysis_text),
                        unsafe_allow_html=True
                    )
                    
                    results = {}
                    time_points = [datetime.now() + timedelta(minutes=i) for i in range(10)]
                    
                    for company in selected_companies:
                        company_data = companies[companies['company_name'] == company].iloc[0]
                        
                        prices = [100.0]
                        sentiments = [0.0]
                        cumulative_sentiment = 0
                        
                        for i in range(1, 10):
                            current_score = sentiment_score * (1 + random.uniform(-0.15, 0.15))
                            cumulative_sentiment += current_score * 0.3
                            price, _ = simulate_company_metrics(cumulative_sentiment, company)
                            prices.append(price)
                            sentiments.append(current_score)
                        
                        results[company] = {
                            'final_price': prices[-1],
                            'final_sentiment': sum(sentiments) / len(sentiments),
                            'price_movement': prices,
                            'sentiment_movement': sentiments,
                            'time_points': time_points
                        }
                    
                    st.markdown("""
                    <div class='card'>
                        <h3 style='color: #1E88E5;'>Analysis Results</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    tabs = st.tabs(selected_companies)
                    
                    for tab, company in zip(tabs, selected_companies):
                        with tab:
                            data = results[company]
                            
                            # Create metrics with animation
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown("""
                                <div class='stMetric'>
                                    <h4>Sentiment Score</h4>
                                    <h2>{:.2f}</h2>
                                </div>
                                """.format(data['final_sentiment']), unsafe_allow_html=True)
                            with col2:
                                st.markdown("""
                                <div class='stMetric'>
                                    <h4>Price Impact</h4>
                                    <h2>${:.2f}</h2>
                                </div>
                                """.format(data['final_price']), unsafe_allow_html=True)
                            with col3:
                                st.markdown("""
                                <div class='stMetric'>
                                    <h4>Price Change %</h4>
                                    <h2>{:.2f}%</h2>
                                </div>
                                """.format(((data['final_price'] - 100) / 100 * 100)), unsafe_allow_html=True)
                            
                            st.plotly_chart(create_dynamic_chart(
                                data['price_movement'],
                                data['time_points'],
                                data['sentiment_movement'],
                                company
                            ))
                    
                    # Save results
                    results_df = pd.DataFrame({
                        'Company': selected_companies,
                        'Final_Price': [results[company]['final_price'] for company in selected_companies],
                        'Final_Sentiment': [results[company]['final_sentiment'] for company in selected_companies],
                        'Timestamp': datetime.now()
                    })
                    results_df.to_csv('sentiment_results.csv', index=False)
            else:
                st.warning("Please enter a tweet and select at least one company")
    
    with col2:
        st.markdown("""
        <div class='card'>
            <h3 style='color: #1E88E5;'>Market Overview</h3>
            <p>How it works:</p>
            <ol>
                <li>Select companies from the sidebar</li>
                <li>Enter market news or analysis</li>
                <li>View real-time impact on:
                    <ul>
                        <li>Stock prices</li>
                        <li>Market sentiment</li>
                        <li>Price movement over time</li>
                    </ul>
                </li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='card'>
            <h3 style='color: #1E88E5;'>Selected Companies</h3>
        </div>
        """, unsafe_allow_html=True)
        
        for company in selected_companies:
            company_data = companies[companies['company_name'] == company].iloc[0]
            st.markdown(f"""
            <div class='card'>
                <h4>{company}</h4>
                <p>{company_data['ticker_symbol']}</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 