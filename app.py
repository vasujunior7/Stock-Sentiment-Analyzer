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
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Download NLTK resources (uncomment these lines first time running)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/sentiwordnet')
except LookupError:
    nltk.download('sentiwordnet')

# Set page config with wide layout and custom theme
st.set_page_config(
    page_title="Advanced Stock Sentiment Analyzer",
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
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Initialize VADER
vader = SentimentIntensityAnalyzer()

# Optionally, you can load finBERT or other transformer models here
# tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
# model = AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')

# Enhanced Financial Sentiment Lexicons
financial_positive_terms = {
    'bullish': 3.5, 'rally': 3.0, 'upgrade': 2.8, 'outperform': 2.5, 'buy': 2.5, 'strong buy': 3.0, 
    'overweight': 2.0, 'beat': 2.5, 'exceeded': 2.8, 'surpassed': 2.8, 'surpass': 2.5, 'exceed': 2.5,
    'upside': 2.0, 'growth': 2.2, 'growing': 2.0, 'expanded': 2.0, 'expansion': 2.2, 'profit': 2.5,
    'profitable': 2.5, 'margin': 1.5, 'margins': 1.5, 'dividend': 2.0, 'innovation': 2.5, 'innovative': 2.5,
    'patent': 1.8, 'leadership': 2.0, 'leader': 2.0, 'dominant': 2.2, 'moat': 2.5, 'competitive advantage': 2.8,
    'disruption': 1.5, 'disruptive': 2.0, 'revolutionary': 2.5, 'game-changer': 2.8, 'breakthrough': 2.8,
    'acquisition': 1.5, 'acquire': 1.5, 'merger': 1.5, 'synergy': 2.0, 'synergies': 2.0, 'cost-cutting': 2.0,
    'streamlined': 2.0, 'efficiency': 2.0, 'efficient': 2.0, 'restructuring': 1.0, 'turnaround': 2.0,
    'recovery': 2.5, 'rebound': 2.5, 'accelerate': 2.2, 'acceleration': 2.2, 'momentum': 2.0, 'catalyst': 2.0,
    'uptrend': 2.8, 'breakout': 2.8, 'all-time high': 3.0, 'record high': 3.0, 'multi-year high': 2.8,
    'raised guidance': 3.0, 'guidance raise': 3.0, 'raised forecast': 3.0, 'positive outlook': 2.5,
    'optimistic outlook': 2.5, 'strong demand': 2.8, 'robust demand': 2.8, 'market share gain': 2.5,
    'share gain': 2.5, 'taking share': 2.2, 'customer growth': 2.2, 'user growth': 2.2, 'subscriber growth': 2.2,
    'active user': 1.8, 'retention': 2.0, 'sticky': 2.0, 'engagement': 2.0, 'monetization': 2.0,
    'cash flow': 2.0, 'free cash flow': 2.5, 'cash generation': 2.5, 'cash rich': 2.5, 'balance sheet': 1.0,
    'buyback': 2.5, 'share repurchase': 2.5, 'deleveraging': 2.0, 'debt reduction': 2.0, 'margin expansion': 2.8,
    'operating leverage': 2.2, 'scale': 1.5, 'scalable': 2.0, 'secular growth': 2.5, 'multiple expansion': 2.5,
    'rerating': 2.0, 'undervalued': 2.5, 'bargain': 2.8, 'discount': 2.0, 'attractive valuation': 2.5
}

financial_negative_terms = {
    'bearish': -3.5, 'sell-off': -3.0, 'downgrade': -2.8, 'underperform': -2.5, 'sell': -2.5, 'strong sell': -3.0,
    'underweight': -2.0, 'miss': -2.5, 'disappointed': -2.8, 'disappointing': -2.8, 'shortfall': -2.5, 'below': -2.0,
    'downside': -2.0, 'decline': -2.2, 'declining': -2.0, 'contracted': -2.0, 'contraction': -2.2, 'loss': -2.5,
    'unprofitable': -2.5, 'margin pressure': -2.0, 'margin compression': -2.2, 'dividend cut': -3.0, 'obsolete': -2.5,
    'outdated': -2.0, 'laggard': -2.0, 'behind': -1.5, 'competition': -1.0, 'competitive threat': -2.2,
    'disrupted': -2.0, 'challenged': -2.0, 'headwind': -2.5, 'headwinds': -2.5, 'obstacle': -2.0, 'hurdle': -2.0,
    'litigation': -2.2, 'lawsuit': -2.2, 'legal': -1.0, 'regulatory': -1.0, 'regulation': -1.0, 'fine': -2.0,
    'penalty': -2.2, 'investigation': -2.2, 'probe': -2.0, 'scrutiny': -1.8, 'fraud': -3.0, 'accounting': -1.0,
    'restatement': -2.8, 'write-down': -2.5, 'impairment': -2.5, 'layoff': -2.5, 'layoffs': -2.5, 'downsizing': -2.5,
    'restructuring costs': -2.0, 'closing': -2.0, 'shutdown': -2.5, 'bankruptcy': -3.5, 'chapter 11': -3.5,
    'default': -3.0, 'debt': -1.5, 'leverage': -1.0, 'highly leveraged': -2.2, 'dilution': -2.0, 'dilutive': -2.0,
    'downtrend': -2.8, 'breakdown': -2.8, 'all-time low': -3.0, 'multi-year low': -2.8, 'death cross': -3.0,
    'lowered guidance': -3.0, 'guidance cut': -3.0, 'reduced forecast': -3.0, 'negative outlook': -2.5,
    'uncertain outlook': -2.0, 'weak demand': -2.8, 'soft demand': -2.5, 'market share loss': -2.5,
    'share loss': -2.5, 'losing share': -2.2, 'customer loss': -2.5, 'user decline': -2.5, 'subscriber churn': -2.5,
    'attrition': -2.0, 'disengagement': -2.0, 'cash burn': -2.5, 'cash crunch': -3.0, 'liquidity concern': -3.0,
    'funding need': -2.0, 'capital raise': -1.5, 'equity offering': -2.0, 'margin compression': -2.5,
    'cost pressure': -2.2, 'inflationary': -1.8, 'inflation': -1.5, 'supply chain': -1.0, 'supply constraint': -2.0,
    'overvalued': -2.5, 'expensive': -2.0, 'rich valuation': -2.0, 'bubble': -2.8, 'frothy': -2.5
}

# Industry-specific sentiment terms
tech_positive_terms = {
    'ai': 3.0, 'artificial intelligence': 3.0, 'machine learning': 2.8, 'blockchain': 2.5, 'cloud': 2.5,
    'saas': 2.5, 'subscription': 2.2, 'recurring revenue': 2.5, 'platform': 2.0, 'ecosystem': 2.0,
    'network effect': 2.5, 'data': 1.5, 'big data': 2.0, 'analytics': 2.0, 'digital transformation': 2.5
}

tech_negative_terms = {
    'legacy': -1.5, 'on-premise': -1.0, 'technical debt': -2.0, 'outage': -2.5, 'data breach': -3.0,
    'hack': -2.8, 'vulnerability': -2.5, 'antitrust': -2.5, 'regulation': -1.8, 'privacy': -1.0
}

retail_positive_terms = {
    'foot traffic': 2.0, 'same store sales': 2.0, 'comp': 1.5, 'e-commerce': 2.0, 'omnichannel': 2.0,
    'inventory': 1.0, 'pricing power': 2.5, 'private label': 1.8, 'customer loyalty': 2.2, 'brand': 1.5
}

retail_negative_terms = {
    'inventory glut': -2.5, 'markdown': -2.0, 'discounting': -2.0, 'cannibalization': -2.2,
    'store closure': -2.5, 'mall': -1.0, 'brick and mortar': -1.0, 'amazon effect': -2.0
}

energy_positive_terms = {
    'reserves': 2.0, 'drilling': 1.5, 'production': 1.5, 'refining': 1.5, 'renewable': 2.0, 
    'green energy': 2.0, 'esg': 1.8, 'carbon capture': 2.0, 'energy transition': 1.5
}

energy_negative_terms = {
    'spill': -3.0, 'leak': -2.5, 'emissions': -1.5, 'carbon': -1.0, 'regulation': -1.5,
    'oversupply': -2.5, 'glut': -2.5, 'opec': -1.0, 'price war': -2.5
}

healthcare_positive_terms = {
    'fda approval': 3.0, 'clinical trial': 1.5, 'phase 3': 2.0, 'breakthrough': 2.8, 'patent': 2.0,
    'pipeline': 2.0, 'reimbursement': 2.2, 'medicare': 1.0, 'blockbuster': 2.8, 'specialty': 1.8
}

healthcare_negative_terms = {
    'side effect': -2.5, 'trial failure': -3.0, 'safety concern': -2.8, 'fda rejection': -3.0,
    'patent cliff': -2.5, 'generic competition': -2.2, 'pricing pressure': -2.0, 'litigation': -2.0
}

# Contextual phrases with valence intensities
contextual_phrases = {
    'poised for growth': 3.0, 'positioned to benefit': 2.5, 'set to capitalize': 2.5, 'turned a corner': 2.0,
    'inflection point': 1.5, 'game changing': 2.8, 'transformative': 2.5, 'secular tailwind': 2.8,
    'structural advantage': 2.5, 'significant upside potential': 2.8, 'asymmetric risk reward': 2.5,
    'multi-bagger potential': 3.0, 'under the radar': 1.5, 'hidden gem': 2.5, 'undiscovered': 2.0,
    'short squeeze potential': 2.0, 'fallen angel': 1.5, 'strong catalyst pipeline': 2.5,
    'losing market relevance': -2.5, 'competitive position eroding': -2.5, 'structural headwind': -2.8,
    'deteriorating fundamentals': -2.8, 'challenged business model': -2.5, 'value trap': -2.5,
    'terminal decline': -3.0, 'existential threat': -3.0, 'no clear path to profitability': -2.5,
    'bloated cost structure': -2.2, 'over-earning': -2.0, 'unsustainable dividend': -2.5,
    'excessive leverage': -2.5, 'material weakness': -2.8, 'accounting red flag': -2.8,
    'regulatory overhang': -2.2, 'facing obsolescence': -2.8
}

# Macroeconomic and market environment terms
macro_positive_terms = {
    'rate cut': 2.5, 'dovish': 2.2, 'accommodative': 2.0, 'stimulus': 2.2, 'economic growth': 2.0,
    'gdp growth': 2.0, 'consumer confidence': 2.0, 'strong consumer': 2.2, 'full employment': 2.0,
    'wage growth': 1.8, 'deregulation': 1.8, 'tax cut': 2.0, 'soft landing': 2.5, 'pent-up demand': 2.2
}

macro_negative_terms = {
    'rate hike': -2.0, 'hawkish': -2.0, 'restrictive': -1.8, 'tightening': -1.8, 'recession': -3.0,
    'economic contraction': -2.8, 'stagflation': -2.8, 'inflation': -2.0, 'inflationary': -2.0,
    'deflation': -2.5, 'consumer weakness': -2.5, 'unemployment': -2.0, 'layoffs': -2.2,
    'trade war': -2.5, 'tariff': -2.0, 'geopolitical risk': -2.2, 'crisis': -2.8, 'bubble': -2.5
}

# Combined financial lexicon
financial_lexicon = {**financial_positive_terms, **financial_negative_terms, **tech_positive_terms, 
                     **tech_negative_terms, **retail_positive_terms, **retail_negative_terms,
                     **energy_positive_terms, **energy_negative_terms, **healthcare_positive_terms,
                     **healthcare_negative_terms, **contextual_phrases, **macro_positive_terms,
                     **macro_negative_terms}

# Company-specific terms dictionary
company_specific_terms = {
    'Apple': {
        'iphone': 2.5, 'mac': 2.0, 'ipad': 2.0, 'services': 2.5, 'app store': 2.2, 'airpods': 2.0,
        'apple watch': 2.0, 'tim cook': 1.0, 'ecosystem': 2.5, 'ios': 2.0, 'macos': 2.0,
        'privacy': 2.0, 'security': 2.0, 'china': 0.0, 'supply chain': 0.0, 'hardware': 1.0,
        'wearables': 2.0, 'apple car': 1.5, 'apple silicon': 2.5, 'm1': 2.5, 'm2': 2.5,
        'homepod': 1.0, 'apple tv+': 1.5, 'app tracking transparency': 1.5
    },
    'Google Inc': {
        'search': 2.5, 'android': 2.0, 'youtube': 2.5, 'cloud': 2.5, 'ads': 2.0, 'pixel': 1.5,
        'chrome': 2.0, 'alphabet': 1.0, 'sundar pichai': 1.0, 'waymo': 1.5, 'deepmind': 2.0,
        'antitrust': -2.0, 'privacy': 0.0, 'regulation': -1.5, 'cookies': 0.0, 'tiktok': -1.0,
        'ai': 2.5, 'bard': 1.5, 'gemini': 2.0, 'workspace': 1.5, 'play store': 1.5
    },
    'Amazon.com': {
        'e-commerce': 2.5, 'aws': 3.0, 'prime': 2.5, 'retail': 2.0, 'logistics': 2.0, 'alexa': 1.5,
        'whole foods': 1.0, 'andy jassy': 1.0, 'marketplace': 2.0, '3p sellers': 1.5, 'fulfillment': 2.0,
        'advertising': 2.5, 'streaming': 1.5, 'one-day': 1.5, 'labor': -1.0, 'union': -1.5,
        'margin': 2.0, 'cost-cutting': 1.5, 'online shopping': 2.0, 'grocery': 1.0
    },
    'Tesla Inc': {
        'ev': 2.5, 'model 3': 2.0, 'model y': 2.5, 'cybertruck': 2.0, 'fsd': 2.0, 'autopilot': 1.5,
        'elon musk': 0.0, 'twitter': -1.0, 'giga': 2.0, 'gigafactory': 2.0, 'berlin': 1.5, 'shanghai': 2.0,
        'texas': 1.5, 'battery': 2.0, '4680': 2.0, 'robotaxi': 1.5, 'solar': 1.0, 'powerwall': 1.5,
        'competition': -2.0, 'margin': 2.0, 'demand': 2.0, 'production': 2.0, 'delivery': 2.0,
        'semi': 1.5, 'robotics': 1.5, 'optimus': 1.0, 'recall': -2.5, 'investigation': -2.0
    },
    'Microsoft': {
        'azure': 3.0, 'cloud': 2.5, 'office': 2.0, '365': 2.0, 'teams': 2.5, 'windows': 2.0,
        'gaming': 2.0, 'xbox': 2.0, 'activision': 1.5, 'satya nadella': 1.5, 'surface': 1.0,
        'ai': 2.5, 'openai': 2.5, 'chatgpt': 2.5, 'copilot': 2.5, 'enterprise': 2.0,
        'security': 2.0, 'linkedin': 2.0, 'github': 2.0, 'power platform': 1.5
    }
}

COMPANY_KEYWORDS = {
    'Apple': ['apple', 'aapl', 'iphone', 'mac', 'ipad'],
    'Tesla Inc': ['tesla', 'tsla', 'elon', 'model 3', 'model y', 'cybertruck'],
    'Microsoft': ['microsoft', 'msft', 'windows', 'azure', 'office'],
    'Amazon.com': ['amazon', 'amzn', 'aws', 'prime', 'alexa'],
    'Google Inc': ['google', 'goog', 'googl', 'android', 'youtube', 'search'],
}

# Helper to check if a company is mentioned
def company_mentioned(text, company):
    text = text.lower()
    for kw in COMPANY_KEYWORDS.get(company, []):
        if kw in text:
            return True
    return False

# Main function for financial sentiment analysis

def analyze_financial_sentiment(text, companies):
    """
    Analyze sentiment for each company in companies list based on the input text.
    Returns a dict: {company: {'sentiment': float, 'impact': float, 'details': str}}
    """
    results = {}
    text = text.strip()
    if not text:
        for company in companies:
            results[company] = {'sentiment': 0.0, 'impact': 0.0, 'details': 'No input.'}
        return results

    # VADER sentiment
    vader_scores = vader.polarity_scores(text)
    vader_compound = vader_scores['compound']  # -1 to 1

    # TextBlob sentiment (for extra signal)
    blob = TextBlob(text)
    tb_polarity = blob.sentiment.polarity  # -1 to 1

    # Optionally, add finBERT/transformer-based sentiment here
    # def finbert_sentiment(text):
    #     inputs = tokenizer(text, return_tensors="pt", truncation=True)
    #     outputs = model(**inputs)
    #     scores = torch.nn.functional.softmax(outputs.logits, dim=1)[0].detach().numpy()
    #     # finBERT: [negative, neutral, positive]
    #     return float(scores[2]) - float(scores[0])
    # finbert_score = finbert_sentiment(text)

    # For now, combine VADER and TextBlob
    base_sentiment = 0.7 * vader_compound + 0.3 * tb_polarity

    # Keyword-based impact boost
    for company in companies:
        is_mentioned = company_mentioned(text, company)
        # If company is mentioned, use full sentiment, else reduce impact
        if is_mentioned:
            sentiment = base_sentiment
            impact = min(max(sentiment, -1), 1)  # Clamp between -1 and 1
            details = f"Company mentioned. Sentiment: {sentiment:.2f}, Impact: {impact:.2f}"
        else:
            # Reduce influence for non-mentioned companies
            sentiment = base_sentiment * 0.15
            impact = min(max(sentiment, -0.2), 0.2)  # Clamp between -0.2 and 0.2
            details = f"Company not mentioned. Sentiment: {sentiment:.2f}, Impact: {impact:.2f}"
        results[company] = {'sentiment': sentiment, 'impact': impact, 'details': details}
    return results

def create_mock_company_csv():
    """Create a mock Company.csv file if not available"""
    companies_data = {
        'company_name': ['Apple', 'Google Inc', 'Amazon.com', 'Tesla Inc', 'Microsoft'],
        'ticker_symbol': ['AAPL', 'GOOGL', 'AMZN', 'TSLA', 'MSFT'],
        'sector': ['Technology', 'Technology', 'Consumer Discretionary', 'Automotive', 'Technology']
    }
    companies_df = pd.DataFrame(companies_data)
    companies_df.to_csv('Company.csv', index=False)
    return companies_df

def load_companies():
    """Load company data from CSV or create mock data if file not found"""
    try:
        companies = pd.read_csv('Company.csv')
        return companies
    except FileNotFoundError:
        st.warning("Company.csv not found! Creating mock company data.")
        return create_mock_company_csv()

def clean_text(text):
    """Clean and preprocess text for better sentiment analysis"""
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user @ references and '#' from tweet
    text = re.sub(r'\@\w+|\#\w+', '', text)
    # Remove punctuations and numbers
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove numbers
    text = re.sub(r'\d+', ' ', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_entities(text):
    """Extract entities mentioned in the text"""
    entities = []
    text = text.lower()
    
    # Check for companies
    companies = ['apple', 'google', 'amazon', 'tesla', 'microsoft']
    
    for company in companies:
        if company in text:
            entities.append(company)
    
    # Check for sectors
    sectors = ['tech', 'technology', 'retail', 'automotive', 'healthcare', 'energy', 'financial']
    for sector in sectors:
        if sector in text:
            entities.append(sector)
    
    # Check for products
    products = ['iphone', 'android', 'aws', 'model 3', 'windows', 'office', 'azure', 'prime']
    for product in products:
        if product in text:
            entities.append(product)
    
    # Check for economic indicators
    indicators = ['inflation', 'gdp', 'unemployment', 'interest rate', 'fed', 'recession']
    for indicator in indicators:
        if indicator in text:
            entities.append(indicator)
    
    return list(set(entities))

def get_enhanced_sentiment(text, company_focus=None):
    """Get enhanced sentiment score using multiple analyzers and financial lexicons"""
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Tokenize text
    tokens = word_tokenize(cleaned_text)
    
    # Remove stopwords and lemmatize
    filtered_tokens = []
    for token in tokens:
        if token not in stop_words:
            lemma = lemmatizer.lemmatize(token)
            filtered_tokens.append(lemma)
    
    # Get AFINN base sentiment
    afinn_score = afinn.score(cleaned_text)
    
    # Get TextBlob sentiment
    blob = TextBlob(cleaned_text)
    textblob_score = blob.sentiment.polarity * 10  # Scale up to match AFINN range
    
    # Initialize score components
    lexicon_score = 0
    entity_relevance = 1.0  # Default multiplier
    contextual_score = 0
    
    # Financial terms sentiment
    financial_terms_found = 0
    for word in filtered_tokens:
        if word in financial_lexicon:
            lexicon_score += financial_lexicon[word]
            financial_terms_found += 1
    
    # Check for multi-word financial terms
    for phrase, score in contextual_phrases.items():
        if phrase in cleaned_text:
            contextual_score += score
            financial_terms_found += 1
    
    # Company-specific sentiment if a company is in focus
    company_relevance = 1.0
    if company_focus and company_focus in company_specific_terms:
        company_terms_found = 0
        company_term_score = 0
        
        for term, score in company_specific_terms[company_focus].items():
            if term in cleaned_text:
                company_term_score += score
                company_terms_found += 1
        
        if company_terms_found > 0:
            company_relevance = 1.5  # Boost for highly relevant news
            lexicon_score += company_term_score
            financial_terms_found += company_terms_found
    
    # Get mentioned entities for contextual analysis
    entities = extract_entities(cleaned_text)
    
    # Calculate entity relevance - higher if many specific entities are mentioned
    if len(entities) > 0:
        entity_relevance = 1.0 + (min(len(entities), 5) * 0.1)
    
    # If no financial terms found, fallback to general sentiment
    if financial_terms_found == 0:
        financial_score = (afinn_score * 0.6 + textblob_score * 0.4)
    else:
        # Weighted average of different sentiment sources
        financial_score = (
            afinn_score * 0.3 + 
            textblob_score * 0.2 + 
            (lexicon_score / max(1, financial_terms_found)) * 0.4 +
            contextual_score * 0.1)
    
    # Apply entity and company relevance multipliers
    final_score = financial_score * entity_relevance * company_relevance
    
    # Normalize score to be between -5 and 5
    final_score = max(min(final_score, 5), -5)
    
    # Return detailed sentiment analysis
    return {
        'score': final_score,
        'afinn_score': afinn_score,
        'textblob_score': textblob_score / 10,  # Convert back to -1 to 1 scale
        'financial_terms_found': financial_terms_found,
        'entities_detected': entities,
        'entity_relevance': entity_relevance,
        'company_relevance': company_relevance,
        'subjectivity': blob.sentiment.subjectivity
    }

def get_sentiment_explanation(sentiment_data):
    """Generate a human-readable explanation of the sentiment analysis"""
    score = sentiment_data['score']
    explanation = []
    
    # Overall sentiment classification
    if score > 3:
        explanation.append("The analysis reveals extremely positive sentiment with strong bullish signals.")
    elif score > 1:
        explanation.append("The analysis shows moderately positive sentiment with generally bullish undertones.")
    elif score > 0:
        explanation.append("The analysis indicates slightly positive sentiment with cautiously optimistic signals.")
    elif score == 0:
        explanation.append("The analysis suggests neutral sentiment with balanced positive and negative indicators.")
    elif score > -1:
        explanation.append("The analysis shows slightly negative sentiment with cautious bearish signals.")
    elif score > -3:
        explanation.append("The analysis reveals moderately negative sentiment with generally bearish undertones.")
    else:
        explanation.append("The analysis indicates extremely negative sentiment with strong bearish signals.")
    
    # Financial terms impact
    if sentiment_data['financial_terms_found'] > 5:
        explanation.append(f"Detected {sentiment_data['financial_terms_found']} industry-specific financial terms, indicating deep financial context.")
    elif sentiment_data['financial_terms_found'] > 0:
        explanation.append(f"Detected {sentiment_data['financial_terms_found']} financial terms, providing relevant market context.")
    else:
        explanation.append("No specific financial terminology detected, sentiment based on general language analysis.")
    
    # Entity relevance
    if sentiment_data['entities_detected']:
        explanation.append(f"Entities detected: {', '.join(sentiment_data['entities_detected'])}, increasing contextual relevance.")
    
    # Subjectivity assessment
    if sentiment_data['subjectivity'] > 0.7:
        explanation.append("The text shows high subjectivity, indicating opinion rather than fact-based assessment.")
    elif sentiment_data['subjectivity'] > 0.3:
        explanation.append("The text shows moderate subjectivity, balancing opinions with factual elements.")
    else:
        explanation.append("The text shows low subjectivity, suggesting more factual and objective content.")
    
    return " ".join(explanation)

def get_company_sensitivity(company_name):
    """Get company-specific sensitivity to sentiment based on historical volatility"""
    sensitivities = {
        'Apple': 2.0,      # Stable blue chip, less volatile
        'Google Inc': 2.2, # Tech giant with moderate volatility
        'Amazon.com': 2.5, # E-commerce giant, moderate to high volatility
        'Tesla Inc': 3.5,  # Known for high volatility to news
        'Microsoft': 1.8,  # Stable blue chip, less volatile
    }
    return sensitivities.get(company_name, 2.0)

def get_sector_sensitivity(company_name, companies_df):
    """Get sector-specific sensitivity to market sentiment"""
    try:
        company_data = companies_df[companies_df['company_name'] == company_name].iloc[0]
        sector = company_data['sector']
        
        sector_sensitivities = {
            'Technology': 2.2,
            'Consumer Discretionary': 2.0,
            'Automotive': 2.8,
            'Healthcare': 1.8,
            'Energy': 2.5,
            'Financial': 2.0,
            'Utilities': 1.2,
            'Materials': 2.3,
            'Industrials': 1.9,
            'Real Estate': 1.7,
            'Communication Services': 2.1
        }
        
        return sector_sensitivities.get(sector, 2.0)
    except:
        return 2.0  # Default sensitivity

def get_market_context():
    """Simulate current market context - in a real app, this would pull actual market data"""
    market_contexts = [
        {"trend": "bullish", "volatility": "low", "liquidity": "high", "description": "Bullish market with low volatility"},
        {"trend": "bullish", "volatility": "high", "liquidity": "high", "description": "Bullish but volatile market"},
        {"trend": "bearish", "volatility": "low", "liquidity": "high", "description": "Bearish market with low volatility"},
        {"trend": "bearish", "volatility": "high", "liquidity": "high", "description": "Bearish and volatile market"},
        {"trend": "neutral", "volatility": "low", "liquidity": "high", "description": "Sideways market with low volatility"},
        {"trend": "neutral", "volatility": "high", "liquidity": "high", "description": "Choppy, volatile market"}
    ]
    
    return random.choice(market_contexts)

def simulate_company_metrics(sentiment_result, company_name, companies_df, base_price=100.0, base_volume=1000000):
    company_sensitivity = get_company_sensitivity(company_name)
    sector_sensitivity = get_sector_sensitivity(company_name, companies_df)
    market_context = get_market_context()
    market_trend_factor = 1.1 if market_context["trend"] == "bullish" else 0.9 if market_context["trend"] == "bearish" else 1.0
    volatility_factor = 1.1 if market_context["volatility"] == "high" else 0.95
    sentiment_score = sentiment_result['score']
    adjusted_score = sentiment_score * company_sensitivity * sector_sensitivity * market_trend_factor
    price_change_percentage = np.clip(adjusted_score * 0.3 * volatility_factor, -3, 3)
    price_change = (base_price * price_change_percentage) / 100
    new_price = base_price + price_change
    # More dynamic volume
    sentiment_intensity = abs(sentiment_score)
    volume_multiplier = np.clip(
        1 + (sentiment_intensity / 8) * sentiment_result['subjectivity'] * volatility_factor + np.random.uniform(-0.03, 0.03),
        0.90, 1.15
    )
    volume_change = base_volume * (volume_multiplier - 1)
    new_volume = base_volume + volume_change
    volatility = abs(price_change_percentage) * volatility_factor * 0.05
    bid_ask_spread = 0.05 + (volatility * 0.1)
    return {
        'price': new_price, 
        'volume': new_volume,
        'volatility': volatility,
        'bid_ask_spread': bid_ask_spread,
        'price_change_pct': price_change_percentage,
        'market_context': market_context["description"]
    }

def generate_time_series(initial_sentiment, company_name, companies_df, points=10, session_seed=None):
    prices = [100.0]
    volumes = [1000000]
    volatilities = [0.02]
    sentiments = [initial_sentiment['score']]
    time_points = [datetime.now()]
    momentum = initial_sentiment['score'] * 0.2
    # Use a more unique seed for each company and session
    import hashlib
    if session_seed is None:
        session_seed = np.random.randint(0, 1e9)
    company_hash = int(hashlib.sha256((company_name + str(session_seed)).encode()).hexdigest(), 16) % 1000000
    # Get initial price impact from simulate_company_metrics
    initial_metrics = simulate_company_metrics(initial_sentiment, company_name, companies_df, prices[0], volumes[0])
    initial_price_impact = initial_metrics['price'] - prices[0]
    # First step: apply initial price impact
    prices.append(prices[0] + initial_price_impact)
    volumes.append(initial_metrics['volume'])
    volatilities.append(initial_metrics['volatility'])
    time_points.append(time_points[0] + timedelta(days=1))
    sentiments.append(initial_sentiment['score'])
    # For the next few steps, bias the trend in the same direction as initial sentiment
    for i in range(2, points):
        time_points.append(time_points[-1] + timedelta(days=1))
        np.random.seed(company_hash + i)
        # Bias noise in the direction of initial sentiment for first 3 steps, then allow more randomness
        if i < 5:
            noise = np.random.uniform(0, 0.5) if initial_sentiment['score'] > 0 else np.random.uniform(-0.5, 0)
        else:
            noise = np.random.uniform(-0.7, 0.7)
        sentiment_change = momentum + noise * (1 - abs(momentum) * 0.5)
        new_sentiment = max(min(sentiments[-1] + sentiment_change, 5), -5)
        sentiments.append(new_sentiment)
        momentum = momentum * 0.9 + sentiment_change * 0.1
        mock_sentiment_result = {'score': new_sentiment, 'subjectivity': initial_sentiment['subjectivity']}
        metrics = simulate_company_metrics(mock_sentiment_result, company_name, companies_df, prices[-1], volumes[-1])
        max_step = prices[-1] * 0.01
        price_step = metrics['price'] - prices[-1]
        if abs(price_step) > max_step:
            price_step = max_step if price_step > 0 else -max_step
        prices.append(prices[-1] + price_step)
        volumes.append(metrics['volume'])
        volatilities.append(metrics['volatility'])
    return {
        'time_points': time_points,
        'prices': prices,
        'volumes': volumes,
        'volatilities': volatilities,
        'sentiments': sentiments
    }

def create_dynamic_chart(time_series_data, company_name):
    """Create interactive chart showing price movement and sentiment"""
    # Extract data from time series
    dates = time_series_data['time_points']
    prices = time_series_data['prices']
    volumes = time_series_data['volumes']
    volatilities = time_series_data['volatilities']
    sentiment_scores = time_series_data['sentiments']
    
    # Create subplots: price, volume, sentiment
    fig = make_subplots(rows=3, cols=1, 
                       shared_xaxes=True,
                       vertical_spacing=0.1,
                       row_heights=[0.5, 0.25, 0.25],
                       subplot_titles=(f'Price Movement for {company_name}', 
                                       'Trading Volume', 
                                       'Sentiment Score'))
    
    # Calculate price changes for color coding
    price_changes = np.diff(prices)
    colors = ['#FF5252' if change < 0 else '#4CAF50' for change in price_changes]
    
    # Add candlestick-like visualization for price
    for i in range(len(prices)-1):
        # Calculate "candle" height
        height = abs(prices[i+1] - prices[i])
        width = 0.7  # Width of the candle in time units
        
        # Add price candle
        fig.add_trace(
            go.Bar(
                x=[dates[i]],
                y=[height],
                base=[min(prices[i], prices[i+1])],
                marker_color='#FF5252' if prices[i+1] < prices[i] else '#4CAF50',
                marker_line_width=1,
                marker_line_color='white',
                width=width,
                name='Price',
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Add price line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=prices,
            mode='lines',
            line=dict(color='rgba(255, 255, 255, 0.8)', width=1),
            name='Price Trend',
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Add volume bars
    normalized_volumes = [vol / max(volumes) * 100 for vol in volumes]
    fig.add_trace(
        go.Bar(
            x=dates,
            y=normalized_volumes,
            marker_color=['rgba(255, 82, 82, 0.7)' if s < 0 else 'rgba(76, 175, 80, 0.7)' for s in sentiment_scores],
            name='Volume',
            showlegend=True
        ),
        row=2, col=1
    )
    
    # Add sentiment line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=sentiment_scores,
            mode='lines+markers',
            line=dict(color='#1E88E5', width=2),
            marker=dict(
                size=8,
                color=['#FF5252' if s < 0 else '#4CAF50' for s in sentiment_scores],
                line=dict(width=1, color='white')
            ),
            name='Sentiment',
            showlegend=True
        ),
        row=3, col=1
    )
    
    # Add zero line for sentiment
    fig.add_shape(
        type="line",
        x0=dates[0],
        x1=dates[-1],
        y0=0,
        y1=0,
        line=dict(
            color="rgba(255, 255, 255, 0.5)",
            width=1,
            dash="dash",
        ),
        row=3, col=1
    )
    
    # Update layout with enhanced styling
    fig.update_layout(
        height=700,
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
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update axes
    fig.update_xaxes(
        showgrid=True,
        gridcolor='rgba(255, 255, 255, 0.1)',
        zerolinecolor='rgba(255, 255, 255, 0.2)'
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridcolor='rgba(255, 255, 255, 0.1)',
        zerolinecolor='rgba(255, 255, 255, 0.2)'
    )
    
    return fig

def create_analysis_card(sentiment_result, metrics, company_name):
    sentiment_score = sentiment_result['score']
    if sentiment_score > 0:
        sentiment_color = '#4CAF50'
    elif sentiment_score < 0:
        sentiment_color = '#F44336'
    else:
        sentiment_color = '#9E9E9E'
    sentiment_label = 'Strongly Positive' if sentiment_score > 3 else \
                     'Moderately Positive' if sentiment_score > 1 else \
                     'Slightly Positive' if sentiment_score > 0 else \
                     'Neutral' if sentiment_score == 0 else \
                     'Slightly Negative' if sentiment_score > -1 else \
                     'Moderately Negative' if sentiment_score > -3 else 'Strongly Negative'
    # Remove Expert Analysis section
    price_change = metrics['price_change_pct']
    price_arrow = "↑" if price_change > 0 else "↓" if price_change < 0 else "→"
    price_color = '#4CAF50' if price_change > 0 else '#F44336' if price_change < 0 else '#9E9E9E'
    return f"""
    <div class='card' style='padding: 20px; margin: 10px 0;'>
        <div style='display: flex; justify-content: space-between; margin-bottom: 15px;'>
            <div>
                <h3 style='color: {sentiment_color};'>Sentiment Analysis: {sentiment_label}</h3>
                <p style='font-size: 1.2em;'>Score: <strong>{sentiment_score:.2f}</strong> (Range: -5 to +5)</p>
            </div>
        </div>
        <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-bottom: 15px;'>
            <div style='background: rgba(0,0,0,0.2); padding: 15px; border-radius: 10px;'>
                <h5>Price Impact</h5>
                <p style='font-size: 1.2em; color: {price_color};'>{price_arrow} {price_change:.2f}%</p>
            </div>
            <div style='background: rgba(0,0,0,0.2); padding: 15px; border-radius: 10px;'>
                <h5>Expected Volatility</h5>
                <p style='font-size: 1.2em;'>{metrics['volatility']:.2f}%</p>
            </div>
            <div style='background: rgba(0,0,0,0.2); padding: 15px; border-radius: 10px;'>
                <h5>Market Context</h5>
                <p style='font-size: 1.2em;'>{metrics['market_context']}</p>
            </div>
        </div>
        <div style='margin-bottom: 15px;'>
            <h4>Financial Term Analysis</h4>
            <p>Detected {sentiment_result['financial_terms_found']} financial terms and {len(sentiment_result['entities_detected'])} relevant entities.</p>
            {f"<p>Entities: {', '.join(sentiment_result['entities_detected'])}</p>" if sentiment_result['entities_detected'] else ""}
        </div>
        <div style='background: rgba(0,0,0,0.2); padding: 15px; border-radius: 10px;'>
            <h4>Recommendation</h4>
            <p>{generate_recommendation(sentiment_score, company_name)}</p>
        </div>
    </div>
    """

def generate_recommendation(sentiment_score, company_name):
    """Generate a recommendation based on sentiment score"""
    if sentiment_score > 3:
        return f"The sentiment analysis strongly suggests a potential buying opportunity for {company_name}. The highly positive indicators point to favorable market conditions and potential price appreciation."
    elif sentiment_score > 1:
        return f"Consider a moderate investment in {company_name}. The positive sentiment indicators suggest potential upside, though with typical market risks."
    elif sentiment_score > 0:
        return f"Consider maintaining current positions in {company_name}. The slightly positive sentiment suggests stable performance with moderate upside potential."
    elif sentiment_score == 0:
        return f"Neutral outlook for {company_name}. Consider holding current positions and monitoring for clearer signals before making significant changes."
    elif sentiment_score > -1:
        return f"Exercise caution with {company_name}. The slightly negative sentiment suggests potential headwinds, though not severe enough to warrant immediate selling."
    elif sentiment_score > -3:
        return f"Consider reducing exposure to {company_name}. The moderately negative sentiment indicators suggest possible downside risk in the near term."
    else:
        return f"The sentiment analysis suggests significant caution with {company_name}. The strongly negative indicators point to potential for substantial price declines."

def main():
    # Add animated title
    st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: #1E88E5; text-shadow: 0 0 10px rgba(30, 136, 229, 0.5);'>
            📊 Advanced Stock Sentiment Analyzer
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
            <h3 style='color: #1E88E5;'>Analysis Configuration</h3>
        </div>
        """, unsafe_allow_html=True)
        
        selected_companies = st.multiselect(
            "Select Companies to Analyze",
            options=companies['company_name'].tolist(),
            default=companies['company_name'].tolist()[:2]
        )
        
        analysis_mode = st.radio(
            "Analysis Mode",
            ["Standard", "Deep Analysis"],
            index=1
        )
        
        time_horizon = st.select_slider(
            "Time Horizon",
            options=["Short-term", "Medium-term", "Long-term"],
            value="Medium-term"
        )
        
        st.markdown("""
        <div class='card'>
            <h3 style='color: #1E88E5;'>About</h3>
            <p>This advanced sentiment analyzer uses NLP techniques and financial lexicons to predict market movements based on news and analysis.</p>
            <p>The system combines multiple sentiment analysis methods with financial domain knowledge to provide more accurate predictions.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class='card'>
            <h3 style='color: #1E88E5;'>Market News Input</h3>
            <p>Enter market news, financial reports, or analyst commentary to analyze potential impact on selected stocks.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Only keep the text_area for user input:
        tweet_text = st.session_state.get('tweet_text', "")
        user_input = st.text_area(
            "Enter market news or financial analysis:",
            value=tweet_text,
            height=150,
            placeholder="Example: 'Apple reported strong quarterly earnings with iPhone sales exceeding expectations. The company announced a significant stock buyback program and raised its dividend by 10%."
        )
        st.session_state['tweet_text'] = user_input
        
        if st.button("Analyze Market Impact", key="analyze_button"):
            if tweet_text and selected_companies:
                with st.spinner('Analyzing financial sentiment and market impact...'):
                    results = {}
                    import secrets
                    session_seed = secrets.randbits(32)
                    for company in selected_companies:
                        company_lower = company.lower().replace(' inc', '').replace('.com', '').replace(' ', '')
                        text_lower = tweet_text.lower().replace(' ', '')
                        sentiment_result = get_enhanced_sentiment(tweet_text, company)
                        if company_lower not in text_lower:
                            # Reduce impact for non-mentioned companies and add random walk
                            sentiment_result['score'] = sentiment_result['score'] * 0.01 + np.random.uniform(-0.1, 0.1)
                        metrics = simulate_company_metrics(sentiment_result, company, companies)
                        time_series = generate_time_series(sentiment_result, company, companies, session_seed=session_seed)
                        results[company] = {
                            'sentiment': sentiment_result,
                            'metrics': metrics,
                            'time_series': time_series
                        }
                    
                    st.markdown("""
                    <div class='card'>
                        <h3 style='color: #1E88E5;'>Analysis Results</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create tabs for each company
                    company_tabs = st.tabs(selected_companies)
                    
                    for i, company in enumerate(selected_companies):
                        with company_tabs[i]:
                            data = results[company]
                            
                            # Display sentiment analysis card
                            st.markdown(
                                create_analysis_card(
                                    data['sentiment'], 
                                    data['metrics'], 
                                    company
                                ),
                                unsafe_allow_html=True
                            )
                            
                            # Display interactive charts
                            st.plotly_chart(
                                create_dynamic_chart(
                                    data['time_series'],
                                    company
                                ),
                                use_container_width=True
                            )
                            
                            # If in deep analysis mode, show additional metrics
                            if analysis_mode == "Deep Analysis":
                                st.markdown("""
                                <div class='card'>
                                    <h3 style='color: #1E88E5;'>Advanced Metrics</h3>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric(
                                        "Sentiment Confidence", 
                                        f"{(1 - data['sentiment']['subjectivity']) * 100:.1f}%",
                                        delta=None
                                    )
                                
                                with col2:
                                    st.metric(
                                        "Volatility Impact", 
                                        f"{data['metrics']['volatility'] * 100:.1f}%",
                                        delta=None
                                    )
                                
                                with col3:
                                    st.metric(
                                        "Financial Terms", 
                                        f"{data['sentiment']['financial_terms_found']}",
                                        delta=None
                                    )
                                
                                # Display time series data in tabular format
                                st.markdown("""
                                <div class='card'>
                                    <h4 style='color: #1E88E5;'>Projected Price Movement</h4>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Create a DataFrame for the time series data
                                df = pd.DataFrame({
                                    'Time': data['time_series']['time_points'],
                                    'Price': data['time_series']['prices'],
                                    'Volume': data['time_series']['volumes'],
                                    'Sentiment': data['time_series']['sentiments']
                                })
                                
                                st.dataframe(df)
                    
                    # Save results for historical tracking
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    results_df = pd.DataFrame({
                        'Company': selected_companies,
                        'Sentiment Score': [results[company]['sentiment']['score'] for company in selected_companies],
                        'Price Impact %': [results[company]['metrics']['price_change_pct'] for company in selected_companies],
                        'Financial Terms': [results[company]['sentiment']['financial_terms_found'] for company in selected_companies],
                        'Timestamp': timestamp
                    })
                    
                    try:
                        # Try to load existing results
                        existing_results = pd.read_csv('sentiment_results.csv')
                        updated_results = pd.concat([existing_results, results_df])
                        updated_results.to_csv('sentiment_results.csv', index=False)
                    except:
                        # If file doesn't exist, create new one
                        results_df.to_csv('sentiment_results.csv', index=False)
            else:
                st.warning("Please enter news text and select at least one company")
    
    with col2:
        st.markdown("""
        <div class='card'>
            <h3 style='color: #1E88E5;'>How It Works</h3>
            <p>This advanced sentiment analyzer uses:</p>
            <ul>
                <li>Multi-layered NLP analysis</li>
                <li>Financial-specific lexicons</li>
                <li>Company-specific sensitivity factors</li>
                <li>Market context simulation</li>
                <li>Time series projection</li>
            </ul>
            <p>The system analyzes text for indicators of market sentiment and projects potential price movements.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='card'>
            <h3 style='color: #1E88E5;'>Selected Companies</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Display selected companies with more detailed information
        for company in selected_companies:
            company_data = companies[companies['company_name'] == company].iloc[0]
            
            # Get company sensitivity for display
            sensitivity = get_company_sensitivity(company)
            sensitivity_level = "High" if sensitivity > 2.5 else "Medium" if sensitivity > 1.5 else "Low"
            
            st.markdown(f"""
            <div class='card'>
                <h4>{company} ({company_data['ticker_symbol']})</h4>
                <p><strong>Sector:</strong> {company_data['sector']}</p>
                <p><strong>Sentiment Sensitivity:</strong> {sensitivity_level}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Show sample news for testing if no companies selected
        if not selected_companies:
            st.markdown("""
            <div class='card'>
                <h3 style='color: #1E88E5;'>Sample News</h3>
                <p>Select companies and try these sample headlines:</p>
                <ul>
                    <li>"Apple reports record iPhone sales and raises dividend by 10%"</li>
                    <li>"Tesla faces production challenges with new Model Y"</li>
                    <li>"Microsoft Azure growth accelerates as cloud demand surges"</li>
                    <li>"Google ad revenue declines amid privacy changes"</li>
                    <li>"Amazon announces new layoffs and cost-cutting measures"</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()