import streamlit as st
import torch
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from datetime import datetime, timedelta
import re
import json
import hashlib
from collections import defaultdict, Counter
import warnings
import io
import base64
warnings.filterwarnings('ignore')

# Core libraries for ML and NLP
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    # Set to use GPU if available
    device = 0 if torch.cuda.is_available() else -1
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    device = -1
    TRANSFORMERS_AVAILABLE = False
    st.error("Please install transformers and torch: pip install transformers torch")

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import MiniBatchKMeans, DBSCAN
from sklearn.decomposition import TruncatedSVD

# Try to import optional dependencies
try:
    import tweepy
    TWITTER_AVAILABLE = True
except ImportError:
    TWITTER_AVAILABLE = False
    st.warning("Twitter API not available. Install tweepy: pip install tweepy")

try:
    import praw
    REDDIT_AVAILABLE = True
except ImportError:
    REDDIT_AVAILABLE = False
    st.warning("Reddit API not available. Install praw: pip install praw")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="Anti-India Campaign Detector - Fixed",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Performance configurations
MAX_BATCH_SIZE = 128
MAX_VISUALIZATION_POINTS = 5000
SIMILARITY_THRESHOLD = 0.8
MIN_CLUSTER_SIZE = 3

# Enhanced Data Collector with Reddit API
class DataCollector:
    """Enhanced data collection with Twitter and Reddit APIs"""
    
    def __init__(self):
        self.twitter_client = None
        self.reddit_client = None
        
    def setup_twitter_api(self, bearer_token):
        if TWITTER_AVAILABLE and bearer_token:
            try:
                self.twitter_client = tweepy.Client(bearer_token=bearer_token)
                return True
            except Exception as e:
                st.error(f"Twitter API setup failed: {e}")
                return False
        return False
    
    def setup_reddit_api(self, client_id, client_secret, user_agent):
        """Setup Reddit API connection"""
        if REDDIT_AVAILABLE and all([client_id, client_secret, user_agent]):
            try:
                self.reddit_client = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent=user_agent
                )
                # Test connection
                _ = self.reddit_client.user.me()
                return True
            except Exception as e:
                st.error(f"Reddit API setup failed: {e}")
                return False
        return False
    
    def collect_reddit_data(self, subreddits, keywords, max_results=100):
        """Collect data from Reddit with randomization to avoid cache"""
        if not self.reddit_client:
            return pd.DataFrame()
            
        all_posts = []
        
        # Add randomization to avoid cached results
        import random
        from datetime import datetime, timedelta
        
        # Randomize subreddit order
        subreddits = list(subreddits)
        random.shuffle(subreddits)
        
        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit_client.subreddit(subreddit_name)
                
                # Randomize keywords order
                keywords_shuffled = list(keywords)
                random.shuffle(keywords_shuffled)
                
                for keyword in keywords_shuffled:
                    try:
                        # Use different sort methods randomly to get varied results
                        sort_methods = ['new', 'hot', 'top', 'relevance']
                        sort_method = random.choice(sort_methods)
                        
                        search_results = subreddit.search(
                            keyword, 
                            limit=min(max_results, 100),
                            sort=sort_method,
                            time_filter='week'  # Focus on recent content
                        )
                        
                        collected_count = 0
                        for post in search_results:
                            if collected_count >= max_results:
                                break
                                
                            # Skip if we already have this post
                            if any(existing['post_id'] == str(post.id) for existing in all_posts):
                                continue
                                
                            post_data = {
                                'post_id': str(post.id),
                                'text': f"{post.title} {post.selftext}".strip(),
                                'username': str(post.author) if post.author else 'deleted',
                                'subreddit': subreddit_name,
                                'created_at': datetime.fromtimestamp(post.created_utc),
                                'score': post.score,
                                'upvote_ratio': post.upvote_ratio,
                                'num_comments': post.num_comments,
                                'url': post.url,
                                'search_term': keyword,
                                'sort_method': sort_method  # Track how it was found
                            }
                            all_posts.append(post_data)
                            collected_count += 1
                            
                    except Exception as e:
                        st.warning(f"Error searching for '{keyword}' in r/{subreddit_name}: {e}")
                        
            except Exception as e:
                st.error(f"Error accessing subreddit r/{subreddit_name}: {e}")
        
        df = pd.DataFrame(all_posts)
        
        if not df.empty:
            # Remove duplicates based on content similarity
            df = df.drop_duplicates(subset=['text'], keep='first')
            
            # Standardize Reddit data to match Twitter format
            df['like_count'] = df['score']
            df['retweet_count'] = df['num_comments']
            df['reply_count'] = df['num_comments']
            df['quote_count'] = 0
            
        return df
    
    def collect_twitter_data(self, keywords, max_results=100):
        if not self.twitter_client:
            return pd.DataFrame()
            
        all_tweets = []
        
        # Add randomization and time variance
        import random
        from datetime import datetime, timedelta
        
        # Randomize keywords
        keywords = list(keywords)
        random.shuffle(keywords)
        
        for keyword in keywords:
            try:
                # Add random time component to avoid cached results
                random_minutes = random.randint(1, 60)
                query = f"{keyword} -is:retweet lang:en"
                
                response = self.twitter_client.search_recent_tweets(
                    query=query,
                    max_results=min(max_results, 100),
                    tweet_fields=['created_at', 'public_metrics', 'lang', 'author_id', 'context_annotations'],
                    expansions=['author_id'],
                    user_fields=['username', 'created_at', 'public_metrics', 'verified']
                )
                
                if response.data:
                    users = {}
                    if response.includes and 'users' in response.includes:
                        users = {u.id: u for u in response.includes['users']}
                    
                    for tweet in response.data:
                        user = users.get(tweet.author_id, {})
                        
                        # Skip if we already have this tweet
                        if any(existing['tweet_id'] == str(tweet.id) for existing in all_tweets):
                            continue
                        
                        tweet_data = {
                            'tweet_id': str(tweet.id),
                            'text': tweet.text,
                            'username': getattr(user, 'username', 'unknown'),
                            'user_id': str(tweet.author_id),
                            'created_at': tweet.created_at,
                            'retweet_count': tweet.public_metrics.get('retweet_count', 0),
                            'reply_count': tweet.public_metrics.get('reply_count', 0),
                            'like_count': tweet.public_metrics.get('like_count', 0),
                            'quote_count': tweet.public_metrics.get('quote_count', 0),
                            'lang': tweet.lang,
                            'user_verified': getattr(user, 'verified', False),
                            'user_followers': getattr(user, 'public_metrics', {}).get('followers_count', 0),
                            'search_term': keyword,
                            'collection_timestamp': datetime.now()
                        }
                        all_tweets.append(tweet_data)
                        
            except Exception as e:
                st.error(f"Error collecting tweets for '{keyword}': {e}")
        
        df = pd.DataFrame(all_tweets)
        if not df.empty:
            show_collection_status("Twitter", df)
        return df.drop_duplicates(subset=['tweet_id'], keep='first') if not df.empty else df
        
    
    def load_dataset(self, uploaded_file):
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload CSV, JSON, or Excel files.")
                return pd.DataFrame()
            
            df = self.standardize_columns(df)
            return df
            
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            return pd.DataFrame()
    
    @st.cache_data
    def standardize_columns(_self, df):
        """Optimized column standardization with intelligent text detection"""
        df.columns = df.columns.str.lower().str.strip()
        
        column_mapping = {
            'content': 'text', 'message': 'text', 'post': 'text', 'tweet': 'text',
            'description': 'text', 'body': 'text', 'comment': 'text', 'status': 'text',
            'user': 'username', 'author': 'username', 'user_name': 'username',
            'account': 'username', 'handle': 'username', 'screen_name': 'username',
            'timestamp': 'created_at', 'date': 'created_at', 'time': 'created_at',
            'posted_at': 'created_at', 'publish_date': 'created_at', 'datetime': 'created_at',
            'id': 'tweet_id', 'post_id': 'tweet_id', 'message_id': 'tweet_id', 'uid': 'tweet_id'
        }
    
        # Apply standard mapping
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        # Intelligent text column detection
        if 'text' not in df.columns:
            text_candidates = []
            
            for col in df.columns:
                if col in ['created_at', 'username', 'tweet_id', 'user_id']:
                    continue
                    
                if df[col].dtype == 'object':
                    sample_values = df[col].dropna().head(10)
                    if len(sample_values) > 0:
                        avg_length = sample_values.astype(str).str.len().mean()
                        max_length = sample_values.astype(str).str.len().max()
                        
                        if avg_length > 10 and max_length > 20:
                            text_candidates.append({
                                'column': col,
                                'avg_length': avg_length,
                                'priority_score': avg_length + (max_length * 0.1)
                            })
            
            if text_candidates:
                text_candidates.sort(key=lambda x: x['priority_score'], reverse=True)
                text_column = text_candidates[0]['column']
                df = df.rename(columns={text_column: 'text'})
                st.info(f"Using '{text_column}' as text content (avg: {text_candidates[0]['avg_length']:.0f} chars)")
            else:
                st.error("Could not detect text content column")
                return pd.DataFrame()
        
        # Handle missing required columns
        if 'username' not in df.columns:
            df['username'] = 'user_' + df.index.astype(str)
        if 'created_at' not in df.columns:
            df['created_at'] = datetime.now()
        if 'tweet_id' not in df.columns:
            df['tweet_id'] = df.index.astype(str)
        
        return df

class OptimizedContentAnalyzer:
    """Highly optimized content analyzer with batch processing"""
    
    def __init__(self):
        self.setup_models()
        self.setup_keywords()
        
    @st.cache_resource
    def setup_models(_self):
        """Initialize models with caching - only runs once"""
        models = {}
        
        if TRANSFORMERS_AVAILABLE:
            try:
                models['sentiment'] = pipeline(
                    'sentiment-analysis',
                    model='cardiffnlp/twitter-roberta-base-sentiment-latest',
                    device=device,
                    batch_size=MAX_BATCH_SIZE,
                    return_all_scores=True
                )
                
                models['toxicity'] = pipeline(
                    'text-classification',
                    model='martin-ha/toxic-comment-model',
                    device=device,
                    batch_size=MAX_BATCH_SIZE,
                    return_all_scores=True
                )
                
                st.sidebar.success("✅ GPU-accelerated ML models loaded!")
                
            except Exception as e:
                st.warning(f"GPU models failed, falling back to CPU: {e}")
                models['sentiment'] = None
                models['toxicity'] = None
        else:
            models['sentiment'] = None
            models['toxicity'] = None
            
        return models
    
    def setup_keywords(self):
        """Enhanced keyword detection with compiled regex for speed"""
        self.anti_india_keywords = [
            'anti india', 'break india', 'divide india', 'destroy india',
            'hate india', 'antiindia', 'breakindia', 'balkanize india',
            'separatist', 'terrorist state', 'oppressive regime',
            'kashmir freedom', 'khalistan', 'dravidistan'
        ]
        
        self.suspicious_phrases = [
            'fake democracy', 'hindu terror', 'brahmanical supremacy',
            'fascist regime', 'genocide', 'ethnic cleansing',
            'state terrorism', 'human rights violations'
        ]
        
        # Compile regex patterns for faster matching
        all_patterns = self.anti_india_keywords + self.suspicious_phrases
        self.keyword_pattern = re.compile('|'.join(re.escape(phrase) for phrase in all_patterns), re.IGNORECASE)
        
        # Additional patterns
        self.inflammatory_pattern = re.compile(r'\b(fascist|nazi|terrorist|genocide)\b', re.IGNORECASE)
        self.conspiracy_pattern = re.compile(r'\b(cia|foreign funding|western agenda|divide and rule)\b', re.IGNORECASE)
    
    def analyze_sentiment_batch(self, texts):
        """Vectorized sentiment analysis"""
        models = self.setup_models()
        
        if models['sentiment'] and len(texts) > 0:
            try:
                all_results = []
                for i in range(0, len(texts), MAX_BATCH_SIZE):
                    batch = texts[i:i + MAX_BATCH_SIZE]
                    batch = [text[:512] for text in batch]
                    results = models['sentiment'](batch)
                    all_results.extend(results)
                
                sentiment_labels = []
                sentiment_scores = []
                
                for result in all_results:
                    scores = {r['label']: r['score'] for r in result}
                    if 'NEGATIVE' in scores:
                        sentiment_labels.append('NEGATIVE')
                        sentiment_scores.append(scores['NEGATIVE'])
                    elif 'POSITIVE' in scores:
                        sentiment_labels.append('POSITIVE') 
                        sentiment_scores.append(scores['POSITIVE'])
                    else:
                        sentiment_labels.append('NEUTRAL')
                        sentiment_scores.append(0.5)
                
                return sentiment_labels, sentiment_scores
                
            except Exception as e:
                st.warning(f"ML sentiment analysis failed: {e}")
        
        return self.rule_based_sentiment_batch(texts)
    
    def rule_based_sentiment_batch(self, texts):
        """Vectorized rule-based sentiment analysis"""
        negative_words = ['hate', 'terrible', 'awful', 'bad', 'worse', 'worst', 'destroy', 'kill', 'die']
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'best']
        
        neg_pattern = re.compile('|'.join(negative_words), re.IGNORECASE)
        pos_pattern = re.compile('|'.join(positive_words), re.IGNORECASE)
        
        sentiment_labels = []
        sentiment_scores = []
        
        for text in texts:
            neg_count = len(neg_pattern.findall(text))
            pos_count = len(pos_pattern.findall(text))
            
            if neg_count > pos_count:
                sentiment_labels.append('NEGATIVE')
                sentiment_scores.append(0.7)
            elif pos_count > neg_count:
                sentiment_labels.append('POSITIVE')
                sentiment_scores.append(0.7)
            else:
                sentiment_labels.append('NEUTRAL')
                sentiment_scores.append(0.5)
        
        return sentiment_labels, sentiment_scores
    
    def detect_anti_india_batch(self, texts):
        """Vectorized anti-India content detection"""
        anti_india_flags = []
        anti_india_scores = []
        
        for text in texts:
            text_lower = text.lower()
            
            keyword_matches = len(self.keyword_pattern.findall(text_lower))
            inflammatory_matches = len(self.inflammatory_pattern.findall(text_lower))
            conspiracy_matches = len(self.conspiracy_pattern.findall(text_lower))
            
            total_score = keyword_matches + (inflammatory_matches * 2) + conspiracy_matches
            
            anti_india_flags.append(total_score > 0)
            anti_india_scores.append(total_score)
        
        return anti_india_flags, anti_india_scores
    
    def analyze_toxicity_batch(self, texts):
        """Vectorized toxicity analysis"""
        models = self.setup_models()
        
        if models['toxicity'] and len(texts) > 0:
            try:
                all_results = []
                for i in range(0, len(texts), MAX_BATCH_SIZE):
                    batch = texts[i:i + MAX_BATCH_SIZE]
                    batch = [text[:512] for text in batch]
                    results = models['toxicity'](batch)
                    all_results.extend(results)
                
                toxic_flags = []
                toxic_scores = []
                
                for result in all_results:
                    toxic_score = max([r['score'] for r in result if 'TOXIC' in r['label'].upper()], default=0)
                    toxic_flags.append(toxic_score > 0.5)
                    toxic_scores.append(toxic_score)
                
                return toxic_flags, toxic_scores
                
            except Exception as e:
                st.warning(f"ML toxicity analysis failed: {e}")
        
        return self.rule_based_toxicity_batch(texts)
    
    def rule_based_toxicity_batch(self, texts):
        """Vectorized rule-based toxicity detection"""
        toxic_patterns = [
            r'\b(kill|die|death|murder)\b',
            r'\b(hate|hatred|disgust)\b', 
            r'\b(stupid|idiot|moron|dumb)\b',
            r'[!]{3,}',
            r'[A-Z]{4,}'
        ]
        
        combined_pattern = re.compile('|'.join(toxic_patterns), re.IGNORECASE)
        
        toxic_flags = []
        toxic_scores = []
        
        for text in texts:
            matches = len(combined_pattern.findall(text))
            is_toxic = matches > 1
            score = min(matches * 0.3, 1.0)
            
            toxic_flags.append(is_toxic)
            toxic_scores.append(score)
        
        return toxic_flags, toxic_scores
    
    @st.cache_data
    def analyze_batch(_self, df):
        """Main analysis function with caching - ENHANCED"""
        if df.empty:
            return pd.DataFrame()
        
        texts = df['text'].fillna('').astype(str).tolist()
        
        with st.spinner("Running enhanced batch content analysis..."):
            progress = st.progress(0)
            
            progress.progress(0.20)
            sentiment_labels, sentiment_scores = _self.analyze_sentiment_batch(texts)
            
            progress.progress(0.40)  
            anti_india_flags, anti_india_scores = _self.detect_anti_india_batch(texts)
            
            progress.progress(0.60)
            toxic_flags, toxic_scores = _self.analyze_toxicity_batch(texts)
            
            progress.progress(0.80)
            # Enhanced risk calculation with more factors
            risk_scores = _self.calculate_enhanced_risk_scores(
                sentiment_labels, sentiment_scores,
                anti_india_flags, anti_india_scores,
                toxic_flags, toxic_scores,
                df
            )
            
            progress.progress(1.0)
            progress.empty()
        
        results_df = df.copy()
        results_df['sentiment_label'] = sentiment_labels
        results_df['sentiment_score'] = sentiment_scores
        results_df['is_anti_india'] = anti_india_flags
        results_df['anti_india_score'] = anti_india_scores
        results_df['is_toxic'] = toxic_flags
        results_df['toxicity_score'] = toxic_scores
        results_df['risk_score'] = risk_scores
        
        # Add threat category
        results_df['threat_category'] = results_df['risk_score'].apply(_self.categorize_threat)
        
        return results_df
    
    def calculate_enhanced_risk_scores(self, sentiment_labels, sentiment_scores, 
                                     anti_india_flags, anti_india_scores,
                                     toxic_flags, toxic_scores, df):
        """Enhanced risk score calculation with more factors"""
        risk_scores = np.zeros(len(sentiment_labels))
        
        # Sentiment contribution (weighted higher for negative)
        sentiment_array = np.array(sentiment_scores)
        negative_mask = np.array(sentiment_labels) == 'NEGATIVE'
        risk_scores += np.where(negative_mask, sentiment_array * 4, 0)
        
        # Anti-India content (high weight)
        risk_scores += np.array(anti_india_scores) * 3
        
        # Toxicity contribution
        toxic_array = np.array(toxic_scores)
        toxic_mask = np.array(toxic_flags)
        risk_scores += np.where(toxic_mask, toxic_array * 3, 0)
        
        # Engagement anomaly detection
        engagement_cols = ['like_count', 'retweet_count', 'reply_count', 'quote_count']
        available_cols = [col for col in engagement_cols if col in df.columns]
        
        if available_cols:
            total_engagement = df[available_cols].fillna(0).sum(axis=1).values
            # High engagement with negative content is suspicious
            high_engagement_mask = (total_engagement > 1000) & negative_mask
            risk_scores += np.where(high_engagement_mask, 2, 0)
            
            # Very low engagement might indicate bot activity
            low_engagement_mask = total_engagement < 5
            risk_scores += np.where(low_engagement_mask, 1, 0)
        
        # Text length anomalies (very short or very long posts)
        if 'text' in df.columns:
            text_lengths = df['text'].str.len()
            short_text_mask = text_lengths < 20
            long_text_mask = text_lengths > 500
            risk_scores += np.where(short_text_mask | long_text_mask, 0.5, 0)
        
        return np.minimum(risk_scores, 10)
    
    def categorize_threat(self, risk_score):
        """Categorize threat level based on risk score"""
        if risk_score >= 8:
            return "CRITICAL"
        elif risk_score >= 6:
            return "HIGH"
        elif risk_score >= 4:
            return "MEDIUM"
        elif risk_score >= 2:
            return "LOW"
        else:
            return "MINIMAL"

class OptimizedCoordinationDetector:
    """FIXED: High-performance coordination detection"""
    
    def __init__(self, similarity_threshold=0.8, time_window_minutes=10):
        self.similarity_threshold = similarity_threshold
        self.time_window_minutes = time_window_minutes
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
    @st.cache_data
    def detect_duplicate_content(_self, df):
        """FIXED: Optimized duplicate detection"""
        if df.empty or len(df) < 2:
            return []
        
        texts = df['text'].fillna('').tolist()
        
        try:
            with st.spinner("Analyzing content similarities..."):
                tfidf_matrix = _self.vectorizer.fit_transform(texts)
                
                if len(texts) > 1000:
                    return _self._cluster_based_duplicate_detection(df, tfidf_matrix)
                else:
                    return _self._similarity_based_duplicate_detection(df, tfidf_matrix)
                    
        except Exception as e:
            st.error(f"Error in duplicate detection: {e}")
            return []
    
    def _cluster_based_duplicate_detection(self, df, tfidf_matrix):
        """FIXED: Cluster-based duplicate detection"""
        if tfidf_matrix.shape[1] > 100:
            svd = TruncatedSVD(n_components=100, random_state=42)
            reduced_matrix = svd.fit_transform(tfidf_matrix)
        else:
            reduced_matrix = tfidf_matrix.toarray()
        
        n_clusters = min(50, len(df) // 10)
        if n_clusters < 2:
            return []
            
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000)
        cluster_labels = kmeans.fit_predict(reduced_matrix)
        
        duplicates = []
        
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            
            if len(cluster_indices) < 2:
                continue
                
            cluster_tfidf = tfidf_matrix[cluster_indices]
            similarity_matrix = cosine_similarity(cluster_tfidf)
            
            for i in range(len(similarity_matrix)):
                for j in range(i+1, len(similarity_matrix)):
                    if similarity_matrix[i][j] > self.similarity_threshold:
                        idx_i, idx_j = cluster_indices[i], cluster_indices[j]
                        
                        duplicates.append({
                            'content_1': df.iloc[idx_i]['text'][:100] + '...',
                            'content_2': df.iloc[idx_j]['text'][:100] + '...',
                            'similarity': float(similarity_matrix[i][j]),
                            'accounts': [str(df.iloc[idx_i]['username']), str(df.iloc[idx_j]['username'])],
                            'timestamps': [str(df.iloc[idx_i]['created_at']), str(df.iloc[idx_j]['created_at'])],
                            'cluster_id': int(cluster_id)
                        })
        
        return duplicates[:100]
    
    def _similarity_based_duplicate_detection(self, df, tfidf_matrix):
        """FIXED: Traditional similarity detection"""
        similarity_matrix = cosine_similarity(tfidf_matrix)
        duplicates = []
        processed_pairs = set()
        
        for i in range(len(similarity_matrix)):
            for j in range(i+1, len(similarity_matrix)):
                if similarity_matrix[i][j] > self.similarity_threshold:
                    pair = tuple(sorted([i, j]))
                    if pair not in processed_pairs:
                        processed_pairs.add(pair)
                        
                        duplicates.append({
                            'content_1': df.iloc[i]['text'][:100] + '...',
                            'content_2': df.iloc[j]['text'][:100] + '...',
                            'similarity': float(similarity_matrix[i][j]),
                            'accounts': [str(df.iloc[i]['username']), str(df.iloc[j]['username'])],
                            'timestamps': [str(df.iloc[i]['created_at']), str(df.iloc[j]['created_at'])]
                        })
        
        return duplicates
    
    @st.cache_data  
    def detect_temporal_coordination(_self, df):
        """FIXED: Temporal coordination detection"""
        if df.empty or 'created_at' not in df.columns:
            return []
            
        df_copy = df.copy()
        df_copy['created_at'] = pd.to_datetime(df_copy['created_at'], errors='coerce')
        df_copy = df_copy.dropna(subset=['created_at']).sort_values('created_at')
        
        if len(df_copy) < MIN_CLUSTER_SIZE:
            return []
        
        coordinated_groups = []
        time_delta = timedelta(minutes=_self.time_window_minutes)
        
        df_copy['time_bucket'] = df_copy['created_at'].dt.floor(f'{_self.time_window_minutes}min')
        time_groups = df_copy.groupby('time_bucket')
        
        for time_bucket, group in time_groups:
            unique_users = group['username'].nunique()
            post_count = len(group)
            
            if post_count >= MIN_CLUSTER_SIZE and unique_users >= MIN_CLUSTER_SIZE:
                # Calculate suspicion level
                posts_per_user = post_count / unique_users
                suspicion_level = "HIGH" if posts_per_user > 3 else "MEDIUM"
                
                coordinated_groups.append({
                    'start_time': str(time_bucket),
                    'end_time': str(time_bucket + time_delta),
                    'accounts': group['username'].unique().tolist(),
                    'post_count': int(post_count),
                    'unique_users': int(unique_users),
                    'posts_per_user': round(posts_per_user, 2),
                    'risk_level': suspicion_level
                })
        
        return sorted(coordinated_groups, key=lambda x: x['post_count'], reverse=True)[:50]
    
    @st.cache_data
    def detect_bot_patterns(_self, df):
        """FIXED: Bot detection with better metrics"""
        if df.empty:
            return []
        
        user_metrics = df.groupby('username').agg({
            'text': ['count', 'nunique'],
            'created_at': lambda x: _self._calculate_posting_regularity(x),
            **{col: 'mean' for col in ['like_count', 'retweet_count', 'reply_count', 'quote_count'] 
               if col in df.columns}
        }).round(3)
        
        # Flatten column names properly
        user_metrics.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in user_metrics.columns]
        
        # Rename columns systematically
        rename_mapping = {}
        for col in user_metrics.columns:
            if col.startswith('text_count'):
                rename_mapping[col] = 'post_count'
            elif col.startswith('text_nunique'):
                rename_mapping[col] = 'unique_posts'
            elif col.startswith('created_at_'):
                rename_mapping[col] = 'posting_regularity'
                
        user_metrics = user_metrics.rename(columns=rename_mapping)
        
        # Ensure required columns exist
        required_cols = ['post_count', 'unique_posts', 'posting_regularity']
        for col in required_cols:
            if col not in user_metrics.columns:
                if col == 'post_count':
                    user_metrics[col] = 1
                elif col == 'unique_posts':
                    user_metrics[col] = 1
                elif col == 'posting_regularity':
                    user_metrics[col] = 0.5
        
        # Calculate metrics
        user_metrics['content_diversity'] = user_metrics['unique_posts'] / user_metrics['post_count'].clip(lower=1)
        
        engagement_cols = [col for col in user_metrics.columns 
                          if any(metric in col for metric in ['like', 'retweet', 'reply', 'quote'])]
        if engagement_cols:
            user_metrics['avg_engagement'] = user_metrics[engagement_cols].sum(axis=1)
        else:
            user_metrics['avg_engagement'] = 0
        
        # Enhanced bot scoring
        bot_scores = np.zeros(len(user_metrics))
        
        # High posting frequency
        bot_scores += np.where(user_metrics['post_count'] > 50, 3,
                              np.where(user_metrics['post_count'] > 20, 2, 
                                      np.where(user_metrics['post_count'] > 10, 1, 0)))
        
        # Low engagement relative to posts
        engagement_ratio = user_metrics['avg_engagement'] / user_metrics['post_count'].clip(lower=1)
        bot_scores += np.where(engagement_ratio < 0.5, 3,
                              np.where(engagement_ratio < 2, 2, 
                                      np.where(engagement_ratio < 5, 1, 0)))
        
        # High posting regularity
        bot_scores += np.where(user_metrics['posting_regularity'] > 0.9, 4,
                              np.where(user_metrics['posting_regularity'] > 0.7, 2, 
                                      np.where(user_metrics['posting_regularity'] > 0.5, 1, 0)))
        
        # Low content diversity
        bot_scores += np.where(user_metrics['content_diversity'] < 0.2, 3,
                              np.where(user_metrics['content_diversity'] < 0.4, 2,
                                      np.where(user_metrics['content_diversity'] < 0.6, 1, 0)))
        
        user_metrics['bot_probability'] = np.minimum(bot_scores / 13, 1.0)  # Normalize to max 13 points
        user_metrics['is_likely_bot'] = user_metrics['bot_probability'] > 0.6
        
        # Convert to results
        bot_results = []
        for username, row in user_metrics.iterrows():
            bot_results.append({
                'username': str(username),
                'post_count': int(row.get('post_count', 1)),
                'avg_engagement': float(row.get('avg_engagement', 0)),
                'posting_regularity': float(row.get('posting_regularity', 0.5)),
                'content_diversity': float(row.get('content_diversity', 0.5)),
                'bot_probability': float(row.get('bot_probability', 0)),
                'is_likely_bot': bool(row.get('is_likely_bot', False))
            })
        
        return sorted(bot_results, key=lambda x: x['bot_probability'], reverse=True)
    
    def _calculate_posting_regularity(self, timestamps):
        """Calculate posting regularity"""
        if len(timestamps) < 3:
            return 0
        
        try:
            timestamps = pd.to_datetime(timestamps).sort_values()
            intervals = timestamps.diff().dropna()
            
            if len(intervals) < 2:
                return 0
            
            mean_interval = intervals.mean().total_seconds()
            std_interval = intervals.std().total_seconds()
            
            if mean_interval == 0:
                return 1.0  # Perfect regularity if all same time
            
            cv = std_interval / mean_interval
            regularity = max(0, 1 - min(cv, 2))  # Cap CV at 2
            
            return min(regularity, 1.0)
            
        except Exception:
            return 0

@st.cache_data
def optimize_dataframe_for_visualization(df, max_points=MAX_VISUALIZATION_POINTS):
    """Optimize DataFrame for visualization by sampling if needed"""
    if len(df) <= max_points:
        return df
    
    # Smart sampling: keep high-risk items and sample from the rest
    high_risk = df[df.get('risk_score', 0) > 7] if 'risk_score' in df.columns else pd.DataFrame()
    
    remaining_quota = max_points - len(high_risk)
    if remaining_quota > 0:
        other_items = df[df.get('risk_score', 0) <= 7] if 'risk_score' in df.columns else df
        if len(other_items) > remaining_quota:
            sampled_others = other_items.sample(n=remaining_quota, random_state=42)
            return pd.concat([high_risk, sampled_others]).reset_index(drop=True)
    
    return df.sample(n=max_points, random_state=42).reset_index(drop=True)

def create_fixed_network_visualization(coordination_data, max_nodes=100):
    """FIXED: Network visualization that actually works"""
    if not coordination_data:
        st.info("No coordination patterns detected for network visualization")
        return go.Figure()
    
    G = nx.Graph()
    
    # Build graph from coordination data
    for coord in coordination_data[:10]:
        accounts = coord.get('accounts', [])[:20]
        if len(accounts) < 2:
            continue
            
        # Add all pairs of accounts as edges
        for i, acc1 in enumerate(accounts):
            for acc2 in accounts[i+1:]:
                if G.has_edge(acc1, acc2):
                    G[acc1][acc2]['weight'] += 1
                    G[acc1][acc2]['coordinations'].append(coord)
                else:
                    G.add_edge(acc1, acc2, weight=1, coordinations=[coord])
    
    if not G.edges() or len(G.nodes()) == 0:
        st.info("No network connections found in coordination data")
        return go.Figure()
    
    # Limit nodes for performance
    if len(G.nodes()) > max_nodes:
        degrees = dict(G.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
        G = G.subgraph(top_nodes).copy()
    
    try:
        # Use spring layout with more iterations for better visualization
        pos = nx.spring_layout(G, k=2, iterations=50)
    except:
        pos = nx.random_layout(G)
    
    # Create edge traces
    edge_x, edge_y = [], []
    edge_info = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        weight = G[edge[0]][edge[1]]['weight']
        edge_info.append(f"{edge[0]} → {edge[1]} (weight: {weight})")
    
    # Create node traces
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_text = []
    node_size = []
    node_color = []
    
    for node in G.nodes():
        degree = G.degree(node)
        node_text.append(f"@{node}<br>Connections: {degree}<br>Coordination Events: {degree}")
        node_size.append(min(degree * 8 + 15, 60))  # Size based on connections
        
        # Color based on risk level
        if degree > 5:
            node_color.append('red')
        elif degree > 3:
            node_color.append('orange')
        else:
            node_color.append('yellow')
    
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='rgba(128,128,128,0.5)'),
        hoverinfo='none',
        mode='lines',
        name='Connections'
    ))
    
    # Add nodes  
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        hovertext=node_text,
        text=[node[:8] for node in G.nodes()],  # Show abbreviated names
        textposition="middle center",
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=2, color='darkred'),
            opacity=0.8
        ),
        name='Accounts'
    ))
    
    fig.update_layout(
        title=f"Coordination Network Analysis<br>({len(G.nodes())} accounts, {len(G.edges())} connections)",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=40,l=5,r=5,t=80),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_enhanced_timeline_visualization(df):
    """FIXED: Create diverse and informative timeline visualization"""
    if df.empty or 'created_at' not in df.columns:
        return go.Figure()
    
    # Prepare data
    viz_df = df.copy()
    viz_df['created_at'] = pd.to_datetime(viz_df['created_at'], errors='coerce')
    viz_df = viz_df.dropna(subset=['created_at'])
    
    if len(viz_df) == 0:
        return go.Figure()
    
    # Sample for performance if needed
    if len(viz_df) > MAX_VISUALIZATION_POINTS:
        viz_df = viz_df.sample(n=MAX_VISUALIZATION_POINTS, random_state=42)
    
    # Create time buckets based on data span
    time_span = viz_df['created_at'].max() - viz_df['created_at'].min()
    
    if time_span.days > 30:
        freq = 'D'  # Daily
        title_freq = "Daily"
    elif time_span.days > 7:
        freq = '4H'  # Every 4 hours
        title_freq = "4-Hour"
    elif time_span.days > 1:
        freq = 'H'   # Hourly
        title_freq = "Hourly"
    else:
        freq = '15min'  # 15 minutes
        title_freq = "15-Minute"
    
    viz_df['time_bucket'] = viz_df['created_at'].dt.floor(freq)
    
    # Create multiple data series for richer visualization
    timeline_data = []
    
    # Group by time bucket and sentiment
    sentiment_timeline = viz_df.groupby(['time_bucket', 'sentiment_label']).size().reset_index()
    sentiment_timeline.columns = ['Time', 'Category', 'Count']
    sentiment_timeline['Type'] = 'Sentiment'
    timeline_data.append(sentiment_timeline)
    
    # Group by time bucket and threat category
    if 'threat_category' in viz_df.columns:
        threat_timeline = viz_df.groupby(['time_bucket', 'threat_category']).size().reset_index()
        threat_timeline.columns = ['Time', 'Category', 'Count']
        threat_timeline['Type'] = 'Threat Level'
        timeline_data.append(threat_timeline)
    
    # Group by time bucket and anti-india flag
    antiindia_timeline = viz_df.groupby(['time_bucket', 'is_anti_india']).size().reset_index()
    antiindia_timeline['Category'] = antiindia_timeline['is_anti_india'].map({True: 'Anti-India', False: 'Regular'})
    antiindia_timeline = antiindia_timeline[['time_bucket', 'Category', 'is_anti_india']].rename(columns={'time_bucket': 'Time', 'is_anti_india': 'Count'})
    antiindia_timeline['Type'] = 'Content Type'
    timeline_data.append(antiindia_timeline)
    
    # Combine all timeline data
    if timeline_data:
        combined_timeline = pd.concat(timeline_data, ignore_index=True)
    else:
        # Fallback to basic timeline
        basic_timeline = viz_df.groupby('time_bucket').size().reset_index()
        basic_timeline.columns = ['Time', 'Count']
        basic_timeline['Category'] = 'Total Posts'
        basic_timeline['Type'] = 'Activity'
        combined_timeline = basic_timeline
    
    # Create the visualization
    fig = go.Figure()
    
    # Color mapping for different categories
    color_maps = {
        'NEGATIVE': '#FF4444', 'POSITIVE': '#44FF44', 'NEUTRAL': '#CCCCCC',
        'CRITICAL': '#8B0000', 'HIGH': '#FF4444', 'MEDIUM': '#FFA500', 'LOW': '#FFFF00', 'MINIMAL': '#90EE90',
        'Anti-India': '#FF0000', 'Regular': '#0000FF',
        'Total Posts': '#800080'
    }
    
    # Add traces for each type
    for viz_type in combined_timeline['Type'].unique():
        type_data = combined_timeline[combined_timeline['Type'] == viz_type]
        
        for category in type_data['Category'].unique():
            cat_data = type_data[type_data['Category'] == category]
            
            fig.add_trace(go.Scatter(
                x=cat_data['Time'],
                y=cat_data['Count'],
                mode='lines+markers',
                name=f"{category} ({viz_type})",
                line=dict(color=color_maps.get(category, '#777777'), width=2),
                marker=dict(size=6),
                hovertemplate=f"{category}<br>Time: %{{x}}<br>Count: %{{y}}<extra></extra>"
            ))
    
    fig.update_layout(
        title=f"{title_freq} Activity Timeline - Multi-dimensional Analysis",
        xaxis_title="Time",
        yaxis_title="Post Count",
        hovermode='x unified',
        height=500,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def generate_comprehensive_report(df, analysis_results, coordination_results):
    """FIXED: Generate comprehensive PDF-style report"""
    if df.empty:
        return "No data available for report generation."
    
    report = []
    report.append("=" * 60)
    report.append("#                 ANTI-INDIA CAMPAIGN DETECTION REPORT                 #")
    report.append("=" * 60)

    from datetime import datetime, timezone
    current_time = datetime.now(timezone.utc)
    report.append(f"\nGenerated on     : {current_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    report.append(f"Analysis Period  : {df['created_at'].min()} to {df['created_at'].max()}")
    report.append("\n" + "=" * 60 + "\n")
        
    # Executive Summary
    report.append("## EXECUTIVE SUMMARY")
    total_posts = len(df)
    high_risk_count = len(analysis_results[analysis_results['risk_score'] > 7]) if 'risk_score' in analysis_results.columns else 0
    anti_india_count = analysis_results['is_anti_india'].sum() if 'is_anti_india' in analysis_results.columns else 0
    
    threat_level = "CRITICAL" if high_risk_count > total_posts * 0.1 else "HIGH" if high_risk_count > total_posts * 0.05 else "MODERATE"
    
    report.append(f"- **Dataset Size:** {total_posts:,} posts analyzed")
    report.append(f"- **Overall Threat Level:** {threat_level}")
    report.append(f"- **High-Risk Content:** {high_risk_count:,} posts ({high_risk_count/total_posts*100:.1f}%)")
    report.append(f"- **Anti-India Content:** {anti_india_count:,} posts ({anti_india_count/total_posts*100:.1f}%)")
    report.append("")
    
    # Content Analysis
    report.append("## CONTENT ANALYSIS RESULTS")
    
    if 'sentiment_label' in analysis_results.columns:
        sentiment_dist = analysis_results['sentiment_label'].value_counts()
        report.append("### Sentiment Distribution:")
        for sentiment, count in sentiment_dist.items():
            report.append(f"- {sentiment}: {count:,} posts ({count/total_posts*100:.1f}%)")
        report.append("")
    
    if 'threat_category' in analysis_results.columns:
        threat_dist = analysis_results['threat_category'].value_counts()
        report.append("### Threat Level Distribution:")
        for threat, count in threat_dist.items():
            report.append(f"- {threat}: {count:,} posts ({count/total_posts*100:.1f}%)")
        report.append("")
    
    # Top Risk Content
    # Enhanced Top Risk Content Analysis
    if 'risk_score' in analysis_results.columns:
        high_risk_posts = analysis_results[analysis_results['risk_score'] > 6].sort_values('risk_score', ascending=False)
        
        # Group by content similarity to avoid repetition
        unique_high_risk = high_risk_posts.drop_duplicates(subset=['text'], keep='first')
        
        report.append(f"### TOP {min(10, len(unique_high_risk))} HIGHEST RISK CONTENT (Unique):")
        
        for i, (idx, post) in enumerate(unique_high_risk.head(10).iterrows()):
            # Count how many similar posts exist
            similar_posts = high_risk_posts[high_risk_posts['text'] == post['text']]
            count_suffix = f" (x{len(similar_posts)} similar posts)" if len(similar_posts) > 1 else ""
            
            report.append(f"{i+1}. **Risk Score: {post['risk_score']:.1f}** - @{post['username']}{count_suffix}")
            report.append(f"   Content: {post['text'][:100]}...")
            
            # Enhanced details
            sentiment_info = f"{post.get('sentiment_label', 'N/A')}"
            if 'sentiment_score' in post:
                sentiment_info += f" ({post['sentiment_score']:.2f})"
            
            report.append(f"   Sentiment: {sentiment_info}")
            report.append(f"   Anti-India: {post.get('is_anti_india', False)}")
            report.append(f"   Toxic: {post.get('is_toxic', False)}")
            
            if 'threat_category' in post:
                report.append(f"   Threat Level: {post['threat_category']}")
            
            # Show engagement if available
            engagement_metrics = []
            for metric in ['like_count', 'retweet_count', 'reply_count']:
                if metric in post and pd.notna(post[metric]):
                    engagement_metrics.append(f"{metric.replace('_count', 's')}: {int(post[metric])}")
            
            if engagement_metrics:
                report.append(f"   Engagement: {', '.join(engagement_metrics)}")
            
            report.append("")
        
        # Risk distribution analysis
        risk_ranges = [
            (10, "Maximum Risk"),
            (8, "Critical Risk"), 
            (6, "High Risk"),
            (4, "Medium Risk"),
            (2, "Low Risk")
        ]
        
        report.append("### RISK DISTRIBUTION ANALYSIS:")
        for min_score, label in risk_ranges:
            count = len(analysis_results[analysis_results['risk_score'] >= min_score])
            if count > 0:
                percentage = (count / len(analysis_results)) * 100
                report.append(f"- {label} ({min_score}+): {count:,} posts ({percentage:.1f}%)")
        report.append("")
    
    # Coordination Analysis
    # Enhanced Coordination Analysis
    if coordination_results:
        report.append("## COORDINATION ANALYSIS")
        
        duplicates = coordination_results.get('duplicates', [])
        temporal = coordination_results.get('temporal', [])
        bots = coordination_results.get('bots', [])
        
        # Summary with better metrics
        likely_bots = [b for b in bots if b.get('is_likely_bot', False)]
        
        report.append(f"- **Content Coordination Patterns:** {len(duplicates)} unique patterns detected")
        report.append(f"- **Temporal Coordination Clusters:** {len(temporal)} time-based clusters identified")
        report.append(f"- **Suspected Bot Accounts:** {len(likely_bots)} accounts flagged")
        
        if duplicates or temporal or likely_bots:
            report.append(f"- **Overall Coordination Risk:** {'HIGH' if len(duplicates) > 5 or len(likely_bots) > 10 else 'MODERATE' if len(duplicates) > 2 else 'LOW'}")
        
        report.append("")
        
        # Enhanced duplicate content analysis
        if duplicates:
            # Group by similarity ranges for better analysis
            high_similarity = [d for d in duplicates if d['similarity'] >= 0.95]
            medium_similarity = [d for d in duplicates if 0.8 <= d['similarity'] < 0.95]
            
            report.append("### CONTENT COORDINATION BREAKDOWN:")
            report.append(f"- **Exact/Near-Exact Matches (95%+):** {len(high_similarity)} patterns")
            report.append(f"- **High Similarity (80-94%):** {len(medium_similarity)} patterns")
            report.append("")
            
            report.append("### TOP COORDINATION PATTERNS:")
            
            # Show diverse patterns, avoid repetition
            shown_accounts = set()
            unique_patterns = []
            
            for dup in duplicates[:20]:  # Check more to find unique ones
                account_pair = tuple(sorted(dup['accounts']))
                if account_pair not in shown_accounts:
                    shown_accounts.add(account_pair)
                    unique_patterns.append(dup)
                if len(unique_patterns) >= 8:  # Show top 8 unique patterns
                    break
            
            for i, dup in enumerate(unique_patterns):
                report.append(f"{i+1}. **Similarity: {dup['similarity']:.3f}**")
                report.append(f"   Accounts: {', '.join(dup['accounts'])}")
                
                # Show both content pieces if different enough
                content1_short = dup['content_1'][:60] + "..." if len(dup['content_1']) > 60 else dup['content_1']
                content2_short = dup['content_2'][:60] + "..." if len(dup['content_2']) > 60 else dup['content_2']
                
                if dup['similarity'] < 1.0:  # Not identical
                    report.append(f"   Content A: {content1_short}")
                    report.append(f"   Content B: {content2_short}")
                else:
                    report.append(f"   Identical Content: {content1_short}")
                
                if 'timestamps' in dup and len(dup['timestamps']) == 2:
                    report.append(f"   Posted: {dup['timestamps'][0]} | {dup['timestamps'][1]}")
                
                report.append("")
        
        # Enhanced temporal coordination
        if temporal:
            report.append("### TEMPORAL COORDINATION ANALYSIS:")
            
            high_risk_clusters = [t for t in temporal if t.get('risk_level') == 'HIGH']
            
            report.append(f"- **High-Risk Time Clusters:** {len(high_risk_clusters)}")
            report.append(f"- **Average Posts per Cluster:** {sum(t['post_count'] for t in temporal) / len(temporal):.1f}")
            report.append(f"- **Most Suspicious Cluster:** {max(temporal, key=lambda x: x['post_count'])['post_count']} posts")
            report.append("")
            
            for i, cluster in enumerate(temporal[:5]):
                risk_indicator = "🚨" if cluster['risk_level'] == 'HIGH' else "⚠️"
                report.append(f"{i+1}. {risk_indicator} **{cluster['start_time'][:19]}**")
                report.append(f"   Posts: {cluster['post_count']}, Users: {cluster['unique_users']}")
                report.append(f"   Intensity: {cluster['posts_per_user']:.1f} posts/user")
                report.append(f"   Risk Level: {cluster['risk_level']}")
                report.append("")
        
        # Enhanced bot analysis
        if likely_bots:
            report.append("### SUSPECTED BOT ACCOUNTS:")
            
            # Categorize bots by probability
            high_confidence_bots = [b for b in likely_bots if b['bot_probability'] > 0.8]
            medium_confidence_bots = [b for b in likely_bots if 0.6 <= b['bot_probability'] <= 0.8]
            
            report.append(f"- **High Confidence Bots (80%+):** {len(high_confidence_bots)}")
            report.append(f"- **Medium Confidence Bots (60-80%):** {len(medium_confidence_bots)}")
            report.append("")
            
            for i, bot in enumerate(likely_bots[:10]):
                confidence_level = "HIGH" if bot['bot_probability'] > 0.8 else "MEDIUM"
                report.append(f"{i+1}. **@{bot['username']}** ({confidence_level} - {bot['bot_probability']:.1%})")
                report.append(f"   Posts: {bot['post_count']}, Avg Engagement: {bot['avg_engagement']:.1f}")
                report.append(f"   Posting Regularity: {bot['posting_regularity']:.2f}")
                report.append(f"   Content Diversity: {bot.get('content_diversity', 0):.2f}")
                report.append("")
    else:
        report.append("## COORDINATION ANALYSIS")
        report.append("No coordination analysis performed. Run coordination detection for detailed insights.")
        report.append("")
    
    # Statistical Analysis
    report.append("## STATISTICAL ANALYSIS")
    
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        time_span = df['created_at'].max() - df['created_at'].min()
        report.append(f"- **Time Span:** {time_span.days} days")
        
        # Peak activity analysis
        df['hour'] = df['created_at'].dt.hour
        hourly_dist = df['hour'].value_counts().sort_index()
        peak_hour = hourly_dist.idxmax()
        report.append(f"- **Peak Activity Hour:** {peak_hour}:00 ({hourly_dist[peak_hour]} posts)")
        report.append("")
    
    # User Analysis
    user_stats = df['username'].value_counts()
    report.append(f"- **Total Unique Users:** {len(user_stats):,}")
    report.append(f"- **Most Active User:** @{user_stats.index[0]} ({user_stats.iloc[0]} posts)")
    report.append(f"- **Average Posts per User:** {user_stats.mean():.1f}")
    report.append("")
    
    # Recommendations
    report.append("## RECOMMENDATIONS")
    
    if threat_level == "CRITICAL":
        report.append("🚨 **IMMEDIATE ACTION REQUIRED:**")
        report.append("- Deploy emergency response team")
        report.append("- Implement content filtering measures")
        report.append("- Monitor identified high-risk accounts")
        report.append("- Coordinate with platform security teams")
    elif threat_level == "HIGH":
        report.append("⚠️ **HIGH PRIORITY ACTIONS:**")
        report.append("- Increase monitoring frequency")
        report.append("- Review and flag suspicious accounts")
        report.append("- Implement targeted content moderation")
    else:
        report.append("✅ **STANDARD MONITORING:**")
        report.append("- Continue regular surveillance")
        report.append("- Maintain current security measures")
    
    report.append("")
    report.append("=" * 60)
    report.append("Report generated by Anti-India Campaign Detection System")
    
    return "\n".join(report)

@st.cache_data
@st.cache_data
@st.cache_data
def generate_enhanced_demo_data(size=1000):
    """Generate enhanced demo dataset with realistic threat level distribution"""
    import random
    
    # Critical threat templates (10% of data)
    critical_templates = [
        "Break India movement gaining momentum, time to act #breakindia #separatist",
        "Destroy Indian unity, create divisions among communities #antiindia #chaos",
        "India is fascist state, needs international intervention immediately #genocide #fascist", 
        "Kashmir freedom fighters deserve our support against oppression #kashmir #terrorism",
        "Khalistan movement will succeed, Punjab independence soon #khalistan #separatist",
        "Indian army commits war crimes, world must stop them #warcrime #terrorist",
        "Hindu extremists controlling government, minorities under threat #hindu #extremist",
        "Economic boycott India until they stop human rights violations #boycott #violation",
        "CIA should destabilize Indian government for world peace #cia #destabilize",
        "India biggest threat to world peace, needs regime change #regime #threat"
    ]
    
    # High threat templates (15% of data)
    high_templates = [
        "Indian democracy is failing, authoritarian regime emerging #democracy #authoritarian",
        "Press freedom declining rapidly in India, journalists attacked #press #freedom",
        "Minority rights severely violated across Indian states #minority #rights", 
        "Kashmir situation getting worse, international attention needed #kashmir #attention",
        "Religious extremism rising under current leadership #religious #extremism",
        "Economic policies destroying poor communities nationwide #economic #destruction",
        "Environmental destruction continues unchecked in India #environment #destruction",
        "Judicial system compromised by political interference #judicial #compromise",
        "Student protests brutally suppressed by authorities #student #suppression",
        "Farmers rights ignored by corporate-friendly government #farmers #ignored"
    ]
    
    # Medium threat templates (20% of data)
    medium_templates = [
        "India's foreign policy becoming increasingly aggressive #foreign #aggressive",
        "Social media regulations limiting free speech concerns #social #speech",
        "Border disputes escalating with neighboring countries #border #disputes",
        "Educational system needs major reforms urgently #education #reforms",
        "Healthcare infrastructure struggling with population demands #healthcare #struggling",
        "Urban planning failures creating massive problems #urban #planning",
        "Transportation systems inadequate for growing cities #transport #inadequate",
        "Energy policies favor certain industries unfairly #energy #unfair",
        "Tax policies burdening middle class families #tax #burden",
        "Employment generation not keeping pace with demand #employment #pace"
    ]
    
    # Low threat templates (25% of data)
    low_templates = [
        "India facing challenges but making progress in many areas #challenges #progress",
        "Political debates healthy for democracy, different views important #political #democracy",
        "Regional variations in policies create interesting discussions #regional #policies",
        "Economic growth has both positive and negative impacts #economic #impacts",
        "Social changes happening gradually across the country #social #changes",
        "Cultural diversity sometimes creates minor tensions #cultural #diversity",
        "Infrastructure development progressing at moderate pace #infrastructure #development",
        "International relations require constant diplomatic efforts #international #diplomacy",
        "Climate change affects agriculture and farmer incomes #climate #agriculture",
        "Technology adoption varies across different regions #technology #adoption"
    ]
    
    # Minimal threat templates (30% of data)
    minimal_templates = [
        "India's space program achievements inspiring global community #space #achievements",
        "Celebrating diverse festivals brings people together beautifully #festivals #unity",
        "Traditional arts gaining international recognition and appreciation #arts #recognition",
        "Young entrepreneurs creating innovative solutions nationwide #entrepreneurs #innovation",
        "Educational initiatives improving literacy in rural areas #education #literacy",
        "Healthcare technology reaching remote villages effectively #healthcare #technology",
        "Cultural exchange programs strengthening international bonds #cultural #exchange",
        "Sports achievements making the nation proud globally #sports #achievements",
        "Scientific research contributing to global knowledge base #research #knowledge",
        "Tourism showcasing natural beauty and heritage sites #tourism #heritage",
        "Agricultural innovations helping farmers increase productivity #agriculture #innovation",
        "Renewable energy projects contributing to climate goals #renewable #energy",
        "Digital transformation connecting rural and urban areas #digital #transformation",
        "Community development projects improving local conditions #community #development",
        "International cooperation in various fields expanding #cooperation #expanding"
    ]
    
    # Realistic user categories with engagement patterns
    user_categories = {
        'extremist_bots': {'users': [f'extremist_bot_{i}' for i in range(1, 16)], 'engagement': (0, 5)},
        'activist_critics': {'users': [f'activist_{i}' for i in range(1, 26)], 'engagement': (10, 100)},
        'concerned_citizens': {'users': [f'citizen_{i}' for i in range(1, 51)], 'engagement': (20, 200)},
        'regular_users': {'users': [f'user_{i}' for i in range(1, 101)], 'engagement': (5, 150)},
        'positive_influencers': {'users': [f'influencer_{i}' for i in range(1, 21)], 'engagement': (100, 1000)},
        'journalists': {'users': [f'reporter_{i}' for i in range(1, 16)], 'engagement': (50, 500)},
        'verified_accounts': {'users': [f'verified_{i}' for i in range(1, 11)], 'engagement': (200, 2000)}
    }
    
    # Generate data with proper distribution
    base_time = datetime(2024, 8, 1)
    data = {
        'tweet_id': [f'post_{i:06d}' for i in range(1, size + 1)],
        'text': [],
        'username': [],
        'created_at': [],
        'like_count': [],
        'retweet_count': [],
        'reply_count': [],
        'quote_count': []
    }
    
    # Define distribution (percentages)
    distributions = [
        (0.10, critical_templates, ['extremist_bots', 'activist_critics']),  # 10%
        (0.15, high_templates, ['activist_critics', 'concerned_citizens']),  # 15%
        (0.20, medium_templates, ['concerned_citizens', 'regular_users']),   # 20%
        (0.25, low_templates, ['regular_users', 'journalists']),             # 25%
        (0.30, minimal_templates, ['positive_influencers', 'verified_accounts', 'regular_users'])  # 30%
    ]
    
    post_index = 0
    
    for percentage, templates, allowed_categories in distributions:
        posts_to_generate = int(size * percentage)
        
        for _ in range(posts_to_generate):
            if post_index >= size:
                break
                
            # Select template and add variation
            template = random.choice(templates)
            if random.random() < 0.3:  # 30% chance to add hashtag
                additional_tags = ['#trending', '#viral', '#important', '#awareness', '#discussion']
                template += f" {random.choice(additional_tags)}"
            
            data['text'].append(template)
            
            # Select user category and specific user
            category = random.choice(allowed_categories)
            user = random.choice(user_categories[category]['users'])
            data['username'].append(user)
            
            # Generate engagement based on user category
            min_eng, max_eng = user_categories[category]['engagement']
            data['like_count'].append(random.randint(min_eng, max_eng))
            data['retweet_count'].append(random.randint(min_eng//4, max_eng//4))
            data['reply_count'].append(random.randint(min_eng//3, max_eng//3))
            data['quote_count'].append(random.randint(0, max_eng//10))
            
            # Generate realistic timestamps with clustering
            days_offset = random.randint(0, 30)
            
            # Create posting pattern based on threat level
            if percentage <= 0.15:  # Critical/High threat - more coordinated timing
                hour = random.choices(range(24), weights=[1,1,1,1,1,2,3,5,8,10,12,8,6,5,4,3,2,2,1,1,1,1,1,1])[0]
            else:  # Normal distribution
                hour = random.choices(range(24), weights=[2,1,1,1,2,3,4,6,8,10,12,14,15,14,12,10,8,7,6,5,4,3,2,2])[0]
            
            timestamp = base_time + timedelta(
                days=days_offset,
                hours=hour,
                minutes=random.randint(0, 59),
                seconds=random.randint(0, 59)
            )
            data['created_at'].append(timestamp)
            
            post_index += 1
    
    # Fill remaining slots with minimal threat content
    while post_index < size:
        template = random.choice(minimal_templates)
        data['text'].append(template)
        data['username'].append(random.choice(user_categories['regular_users']['users']))
        
        min_eng, max_eng = user_categories['regular_users']['engagement']
        data['like_count'].append(random.randint(min_eng, max_eng))
        data['retweet_count'].append(random.randint(min_eng//4, max_eng//4))
        data['reply_count'].append(random.randint(min_eng//3, max_eng//3))
        data['quote_count'].append(random.randint(0, max_eng//10))
        
        days_offset = random.randint(0, 30)
        hour = random.randint(0, 23)
        timestamp = base_time + timedelta(days=days_offset, hours=hour, minutes=random.randint(0, 59))
        data['created_at'].append(timestamp)
        
        post_index += 1
    
    return data
def show_collection_status(data_source, df):
    """Show collection status and data freshness"""
    if not df.empty:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success(f"✅ Collected {len(df):,} {data_source} records")
        
        with col2:
            if 'created_at' in df.columns:
                latest_post = pd.to_datetime(df['created_at']).max()
                time_ago = datetime.now() - latest_post.replace(tzinfo=None)
                st.info(f"Latest post: {time_ago.days}d {time_ago.seconds//3600}h ago")
        
        with col3:
            unique_users = df['username'].nunique()
            st.info(f"Unique users: {unique_users:,}")
    else:
        st.warning(f"No {data_source} data collected. Try different keywords or check API credentials.")

def main():
    st.title("🛡️ Anti-India Campaign Detection System")
    st.markdown("**Enhanced analysis with coordination networks, timeline visualization, and report generation**")
    st.markdown("---")
    
    # Performance info
    if torch.cuda.is_available():
        st.sidebar.success(f"🚀 GPU Acceleration: {torch.cuda.get_device_name()}")
    else:
        st.sidebar.info("💻 Running on CPU (install CUDA for GPU acceleration)")
    
    # Initialize components
    if 'data_collector' not in st.session_state:
        st.session_state.data_collector = DataCollector()
    
    if 'content_analyzer' not in st.session_state:
        st.session_state.content_analyzer = OptimizedContentAnalyzer()
    
    if 'coordination_detector' not in st.session_state:
        st.session_state.coordination_detector = OptimizedCoordinationDetector()
    
    # Configuration
    st.sidebar.header("🔧 Configuration")
    
    # Performance settings
    with st.sidebar.expander("⚡ Performance Settings"):
        batch_size = st.slider("ML Batch Size", 16, 256, MAX_BATCH_SIZE, 16)
        viz_limit = st.slider("Visualization Limit", 1000, 10000, MAX_VISUALIZATION_POINTS, 1000)
        
        st.session_state.max_batch_size = batch_size
        st.session_state.max_visualization_points = viz_limit
        
        st.info(f"Current settings optimized for datasets up to {viz_limit:,} records")
    
    # Data source selection with Reddit support
    data_source = st.sidebar.selectbox(
        "Select Data Source",
        ["Upload Dataset", "Twitter API", "Reddit API", "Demo Data"]
    )
    
    df = pd.DataFrame()
    
    # Data loading
    if data_source == "Upload Dataset":
        uploaded_file = st.sidebar.file_uploader(
            "Choose a file",
            type=['csv', 'json', 'xlsx', 'xls'],
            help="Optimized for large files up to 100MB"
        )
        
        if uploaded_file:
            file_size = uploaded_file.size / (1024 * 1024)
            
            if file_size > 100:
                st.warning(f"Large file detected ({file_size:.1f}MB). Processing may take time.")
            
            with st.spinner(f"Loading {file_size:.1f}MB dataset..."):
                df = st.session_state.data_collector.load_dataset(uploaded_file)
                if not df.empty:
                    st.session_state.raw_data = df
                    st.success(f"✅ Loaded {len(df):,} records in {file_size:.1f}MB file")

    elif data_source == "Twitter API":
        bearer_token = st.sidebar.text_input("Bearer Token", type="password")
        keywords = st.sidebar.text_area(
            "Keywords (one per line)",
            value="#antiindia\n#breakindia\nanti india"
        ).split('\n')
        keywords = [k.strip() for k in keywords if k.strip()]
        max_tweets = st.sidebar.slider("Max Tweets/Keyword", 10, 100, 50)
        
        if st.sidebar.button("🔍 Collect Twitter Data"):
            if bearer_token:
                if st.session_state.data_collector.setup_twitter_api(bearer_token):
                    with st.spinner("Collecting tweets..."):
                        df = st.session_state.data_collector.collect_twitter_data(keywords, max_tweets)
                        if not df.empty:
                            st.session_state.raw_data = df
                            st.success(f"✅ Collected {len(df)} tweets")

    elif data_source == "Reddit API":
        with st.sidebar.expander("Reddit API Configuration"):
            client_id = st.text_input("Client ID", type="password")
            client_secret = st.text_input("Client Secret", type="password")
            user_agent = st.text_input("User Agent", value="AntiIndiaDetector/1.0")
            
        subreddits = st.sidebar.text_area(
            "Subreddits (one per line)",
            value="india\nIndiaSpeaks\nindiameme"
        ).split('\n')
        subreddits = [s.strip() for s in subreddits if s.strip()]
        
        keywords = st.sidebar.text_area(
            "Keywords (one per line)",
            value="anti india\nbreak india\nkhalistan"
        ).split('\n')
        keywords = [k.strip() for k in keywords if k.strip()]
        
        max_posts = st.sidebar.slider("Max Posts/Subreddit", 10, 100, 50)
        
        if st.sidebar.button("🔍 Collect Reddit Data"):
            if all([client_id, client_secret, user_agent]):
                if st.session_state.data_collector.setup_reddit_api(client_id, client_secret, user_agent):
                    with st.spinner("Collecting Reddit posts..."):
                        df = st.session_state.data_collector.collect_reddit_data(subreddits, keywords, max_posts)
                        if not df.empty:
                            st.session_state.raw_data = df
                            st.success(f"✅ Collected {len(df)} Reddit posts")
            else:
                st.sidebar.error("Please provide all Reddit API credentials")

    elif data_source == "Demo Data":
        if st.sidebar.button("📊 Load Enhanced Demo"):
            demo_size = st.sidebar.slider("Demo Dataset Size", 100, 5000, 1000)
            
            with st.spinner(f"Generating {demo_size} demo records..."):
                demo_data = generate_enhanced_demo_data(demo_size)
                df = pd.DataFrame(demo_data)
                st.session_state.raw_data = df
                st.success(f"✅ Generated {len(df):,} demo records")
    
    # Load existing data
    if 'raw_data' in st.session_state and not st.session_state.raw_data.empty:
        df = st.session_state.raw_data
    
    # Main interface
    if df.empty:
        st.info("👆 Select a data source and load data to begin enhanced analysis")
        
        # System capabilities
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🧠 ML-Powered Analysis
            - GPU-accelerated transformer models
            - Enhanced risk scoring system
            - Vectorized batch processing
            - Smart threat categorization
            """)
        
        with col2:
            st.markdown("""
            ### 🔍 Fixed Detection Systems
            - Working coordination networks
            - Multi-dimensional timeline analysis
            - Enhanced bot detection
            - Reddit & Twitter integration
            """)
        
        with col3:
            st.markdown("""
            ### 📊 Enhanced Visualization
            - Fixed network graphs
            - Dynamic timeline charts
            - Comprehensive PDF reports
            - Real-time dashboard updates
            """)
        
        return
    
    # Dataset overview
    st.subheader("📋 Dataset Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Unique Users", f"{df['username'].nunique():,}")
    with col3:
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        st.metric("Memory Usage", f"{memory_mb:.1f}MB")
    with col4:
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
            time_span = df['created_at'].max() - df['created_at'].min()
            st.metric("Time Span", f"{time_span.days}d")
        else:
            st.metric("Time Span", "N/A")
    with col5:
        avg_length = df['text'].str.len().mean()
        st.metric("Avg Text Length", f"{avg_length:.0f}")
    
    # Enhanced analysis controls
    st.subheader("⚡ Enhanced Analysis Pipeline")
    
    col1, col2, col3, col4 = st.columns(4)
    
    analyze_content = col1.button("🧠 Enhanced ML Analysis", use_container_width=True)
    detect_coordination = col2.button("🕵️ Fixed Coordination Detection", use_container_width=True)  
    run_all = col3.button("🚀 Full Enhanced Pipeline", use_container_width=True)
    generate_report = col4.button("📄 Generate Complete Report", use_container_width=True)
    
    # Enhanced analysis pipeline
    if run_all or analyze_content:
        start_time = datetime.now()
        
        with st.spinner("Running enhanced content analysis..."):
            analysis_results = st.session_state.content_analyzer.analyze_batch(df)
            st.session_state.analysis_results = analysis_results
            
        processing_time = (datetime.now() - start_time).total_seconds()
        records_per_second = len(df) / processing_time if processing_time > 0 else 0
        
        st.success(f"✅ Enhanced analysis complete! Processed {len(df):,} records in {processing_time:.1f}s ({records_per_second:.0f} records/sec)")
    
    if run_all or detect_coordination:
        if 'analysis_results' not in st.session_state:
            st.warning("Run content analysis first for optimal coordination detection")
        else:
            start_time = datetime.now()
            
            with st.spinner("Running fixed coordination detection..."):
                duplicates = st.session_state.coordination_detector.detect_duplicate_content(df)
                temporal_coord = st.session_state.coordination_detector.detect_temporal_coordination(df)
                bot_signals = st.session_state.coordination_detector.detect_bot_patterns(df)
                
                st.session_state.coordination_results = {
                    'duplicates': duplicates,
                    'temporal': temporal_coord,
                    'bots': bot_signals
                }
            
            processing_time = (datetime.now() - start_time).total_seconds()
            st.success(f"✅ Fixed coordination detection complete in {processing_time:.1f}s")
    
    # Generate comprehensive report
    if generate_report:
        if 'analysis_results' in st.session_state:
            with st.spinner("Generating comprehensive report..."):
                report_content = generate_comprehensive_report(
                    df, 
                    st.session_state.analysis_results,
                    st.session_state.get('coordination_results', {})
                )
                
                # Create download button
                st.download_button(
                    label="📥 Download Complete Report",
                    data=report_content,
                    file_name=f"anti_india_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
                
                # Show preview
                with st.expander("📖 Report Preview"):
                    st.markdown(report_content)
                
                st.success("✅ Report generated successfully!")
        else:
            st.warning("Run analysis first to generate report")
    
    # Enhanced dashboard tabs
    if 'analysis_results' in st.session_state:
        analysis_df = st.session_state.analysis_results
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Enhanced Dashboard", 
            "🚨 High-Risk Detection",
            "🔗 Fixed Coordination Networks", 
            "📈 Advanced Analytics"
        ])
        
        with tab1:
            # Enhanced performance metrics
            col1, col2, col3, col4 = st.columns(4)
            
            total_records = len(analysis_df)
            high_risk_count = len(analysis_df[analysis_df['risk_score'] > 7])
            anti_india_count = analysis_df['is_anti_india'].sum()
            bot_count = len(st.session_state.get('coordination_results', {}).get('bots', []))
            
            with col1:
                st.metric("🚨 High Risk", high_risk_count, 
                         delta=f"{(high_risk_count/total_records*100):.1f}%")
            with col2:
                st.metric("🎯 Anti-India", anti_india_count,
                         delta=f"{(anti_india_count/total_records*100):.1f}%") 
            with col3:
                negative_count = len(analysis_df[analysis_df['sentiment_label'] == 'NEGATIVE'])
                st.metric("😠 Negative", negative_count,
                         delta=f"{(negative_count/total_records*100):.1f}%")
            with col4:
                st.metric("🤖 Likely Bots", bot_count)
            
            # FIXED: Enhanced timeline visualization
            if 'created_at' in analysis_df.columns:
                st.subheader("📈 Enhanced Multi-dimensional Timeline")
                fig = create_enhanced_timeline_visualization(analysis_df)
                st.plotly_chart(fig, use_container_width=True)
            
            # Threat category distribution
            if 'threat_category' in analysis_df.columns:
                st.subheader("⚖️ Threat Category Distribution")
                viz_df = optimize_dataframe_for_visualization(analysis_df)
                threat_dist = viz_df['threat_category'].value_counts()
                
                fig = px.pie(
                    values=threat_dist.values,
                    names=threat_dist.index,
                    title=f'Threat Level Distribution ({len(viz_df):,} samples)',
                    color_discrete_map={
                        'CRITICAL': '#8B0000',
                        'HIGH': '#FF4444', 
                        'MEDIUM': '#FFA500',
                        'LOW': '#FFFF00',
                        'MINIMAL': '#90EE90'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Enhanced high-risk content
            high_risk_posts = analysis_df[analysis_df['risk_score'] > 5].sort_values('risk_score', ascending=False)
            
            if not high_risk_posts.empty:
                st.subheader(f"🚨 {len(high_risk_posts):,} High-Risk Posts")
                
                # Threat filter
                if 'threat_category' in high_risk_posts.columns:
                    threat_filter = st.multiselect(
                        "Filter by Threat Level",
                        options=['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'MINIMAL'],
                        default=['CRITICAL', 'HIGH']
                    )
                    if threat_filter:
                        high_risk_posts = high_risk_posts[high_risk_posts['threat_category'].isin(threat_filter)]
                
                # Pagination
                posts_per_page = 20
                total_pages = (len(high_risk_posts) - 1) // posts_per_page + 1
                
                if total_pages > 1:
                    page = st.selectbox("Page", range(1, total_pages + 1))
                    start_idx = (page - 1) * posts_per_page
                    end_idx = start_idx + posts_per_page
                    posts_to_show = high_risk_posts.iloc[start_idx:end_idx]
                else:
                    posts_to_show = high_risk_posts.head(posts_per_page)
                
                for idx, post in posts_to_show.iterrows():
                    threat_category = post.get('threat_category', 'UNKNOWN')
                    threat_emoji = {
                        'CRITICAL': '🔴',
                        'HIGH': '🟠', 
                        'MEDIUM': '🟡',
                        'LOW': '🟢',
                        'MINIMAL': '⚪'
                    }.get(threat_category, '❓')
                    
                    with st.expander(f"{threat_emoji} {threat_category}: {post['risk_score']:.1f} - @{post['username']}"):
                        st.markdown(f"**Content:** {post['text'][:300]}...")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.write(f"**Sentiment:** {post['sentiment_label']} ({post['sentiment_score']:.2f})")
                        with col2:
                            st.write(f"**Anti-India:** {'Yes' if post['is_anti_india'] else 'No'}")
                        with col3:
                            st.write(f"**Toxic:** {'Yes' if post['is_toxic'] else 'No'}")
                        with col4:
                            st.write(f"**Category:** {threat_category}")
            else:
                st.info("No high-risk content detected")
        
        with tab3:
            # FIXED: Coordination analysis with working network
            if 'coordination_results' in st.session_state:
                coord_results = st.session_state.coordination_results
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🔄 Content Coordination")
                    duplicates = coord_results.get('duplicates', [])
                    
                    if duplicates:
                        st.metric("Suspicious Patterns", len(duplicates))
                        
                        for i, dup in enumerate(duplicates[:10]):
                            with st.expander(f"Pattern {i+1}: {dup['similarity']:.3f} similarity"):
                                st.write(f"**Accounts:** {', '.join(dup['accounts'])}")
                                st.write(f"**Content 1:** {dup['content_1']}")
                                st.write(f"**Content 2:** {dup['content_2']}")
                                if 'cluster_id' in dup:
                                    st.write(f"**Cluster ID:** {dup['cluster_id']}")
                    else:
                        st.info("No coordination patterns detected")
                
                with col2:
                    st.subheader("⏰ Temporal Clusters")
                    temporal = coord_results.get('temporal', [])
                    
                    if temporal:
                        st.metric("Time-based Clusters", len(temporal))
                        
                        for i, cluster in enumerate(temporal[:10]):
                            risk_emoji = "🔴" if cluster['risk_level'] == 'HIGH' else "🟡"
                            
                            with st.expander(f"{risk_emoji} Cluster {i+1}: {cluster['post_count']} posts"):
                                st.write(f"**Risk:** {cluster['risk_level']}")
                                st.write(f"**Unique Users:** {cluster['unique_users']}")
                                st.write(f"**Posts per User:** {cluster['posts_per_user']}")
                                st.write(f"**Time Window:** {cluster['start_time']}")
                    else:
                        st.info("No temporal coordination detected")
                
                # FIXED: Working network visualization
                if temporal:
                    st.subheader("🕸️ Fixed Coordination Network")
                    fig = create_fixed_network_visualization(temporal)
                    if fig.data:  # Only show if there's actual data
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Network visualization requires more coordination data")
        
        with tab4:
            st.subheader("📊 Engagement Analysis")
            
            # Engagement metrics analysis
            engagement_cols = ['like_count', 'retweet_count', 'reply_count', 'quote_count']
            available_engagement_cols = [col for col in engagement_cols if col in analysis_df.columns]
            
            if available_engagement_cols:
                # Calculate total engagement
                analysis_df['total_engagement'] = analysis_df[available_engagement_cols].sum(axis=1)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**High Engagement Posts Analysis**")
                    
                    # Find posts with unusually high engagement
                    engagement_threshold = analysis_df['total_engagement'].quantile(0.9)
                    high_engagement_posts = analysis_df[analysis_df['total_engagement'] > engagement_threshold].copy()
                    
                    if not high_engagement_posts.empty:
                        # Analyze high engagement by threat level
                        if 'threat_category' in high_engagement_posts.columns:
                            threat_engagement = high_engagement_posts.groupby('threat_category').agg({
                                'total_engagement': ['count', 'mean', 'sum'],
                                'risk_score': 'mean'
                            }).round(2)
                            
                            threat_engagement.columns = ['Post Count', 'Avg Engagement', 'Total Engagement', 'Avg Risk Score']
                            st.dataframe(threat_engagement, use_container_width=True)
                        
                        # Show top engaging posts by risk level
                        st.write("**Top Engaging High-Risk Posts:**")
                        high_risk_engaging = high_engagement_posts[
                            high_engagement_posts['risk_score'] > 6
                        ].sort_values('total_engagement', ascending=False).head(5)
                        
                        for idx, post in high_risk_engaging.iterrows():
                            with st.expander(f"Risk: {post['risk_score']:.1f} | Engagement: {post['total_engagement']:,} - @{post['username']}"):
                                st.write(f"**Content:** {post['text'][:200]}...")
                                
                                col_a, col_b, col_c, col_d = st.columns(4)
                                with col_a:
                                    st.metric("Likes", f"{post['like_count']:,}")
                                with col_b:
                                    st.metric("Retweets", f"{post['retweet_count']:,}")  
                                with col_c:
                                    st.metric("Replies", f"{post['reply_count']:,}")
                                with col_d:
                                    st.metric("Quotes", f"{post['quote_count']:,}")
                    else:
                        st.info("No high engagement posts detected")
                
                with col2:
                    # Engagement vs Risk Score correlation
                    fig = px.scatter(
                        analysis_df.sample(min(1000, len(analysis_df))),
                        x='risk_score',
                        y='total_engagement',
                        color='threat_category' if 'threat_category' in analysis_df.columns else None,
                        title='Risk Score vs Total Engagement',
                        labels={'risk_score': 'Risk Score', 'total_engagement': 'Total Engagement'},
                        hover_data=['username']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Engagement anomaly detection
                st.write("**Engagement Anomaly Analysis**")
                
                # Calculate engagement ratio (engagement per follower equivalent)
                analysis_df['engagement_efficiency'] = analysis_df['total_engagement'] / (analysis_df['like_count'] + 1)
                
                # Identify suspicious engagement patterns
                suspicious_patterns = []
                
                # Pattern 1: High engagement with low quality content (short, repetitive)
                analysis_df['text_length'] = analysis_df['text'].str.len()
                short_high_engagement = analysis_df[
                    (analysis_df['text_length'] < 50) & 
                    (analysis_df['total_engagement'] > analysis_df['total_engagement'].quantile(0.8))
                ]
                
                if len(short_high_engagement) > 0:
                    suspicious_patterns.append(f"Short content with high engagement: {len(short_high_engagement)} posts")
                
                # Pattern 2: Low retweet-to-like ratio (bot-like behavior)
                analysis_df['rt_like_ratio'] = analysis_df['retweet_count'] / (analysis_df['like_count'] + 1)
                low_rt_ratio = analysis_df[analysis_df['rt_like_ratio'] < 0.05]
                
                if len(low_rt_ratio) > 0:
                    suspicious_patterns.append(f"Unusual retweet/like patterns: {len(low_rt_ratio)} posts")
                
                # Pattern 3: High engagement during off-hours
                if 'created_at' in analysis_df.columns:
                    analysis_df['hour'] = pd.to_datetime(analysis_df['created_at']).dt.hour
                    off_hours_posts = analysis_df[
                        (analysis_df['hour'].isin([0, 1, 2, 3, 4, 5])) &
                        (analysis_df['total_engagement'] > analysis_df['total_engagement'].median())
                    ]
                    
                    if len(off_hours_posts) > 0:
                        suspicious_patterns.append(f"High engagement during off-hours: {len(off_hours_posts)} posts")
                
                if suspicious_patterns:
                    st.warning("**Suspicious Engagement Patterns Detected:**")
                    for pattern in suspicious_patterns:
                        st.write(f"- {pattern}")
                else:
                    st.success("No suspicious engagement patterns detected")
            
            st.subheader("🕸️ Key Influencers and Network Analysis")
            
            # Identify key influencers based on multiple metrics
            user_influence_metrics = analysis_df.groupby('username').agg({
                'total_engagement': ['sum', 'mean'],
                'text': 'count',
                'risk_score': 'mean',
                'is_anti_india': 'sum',
                'is_toxic': 'sum'
            }).round(2)
            
            # Flatten column names
            user_influence_metrics.columns = [
                'Total_Engagement', 'Avg_Engagement', 'Post_Count', 
                'Avg_Risk_Score', 'Anti_India_Posts', 'Toxic_Posts'
            ]
            
            # Calculate influence score
            user_influence_metrics['Influence_Score'] = (
                (user_influence_metrics['Total_Engagement'] * 0.4) +
                (user_influence_metrics['Avg_Engagement'] * 0.3) +
                (user_influence_metrics['Post_Count'] * 0.2) +
                (user_influence_metrics['Avg_Risk_Score'] * 0.1)
            )
            
            # Normalize influence score to 0-100
            max_influence = user_influence_metrics['Influence_Score'].max()
            if max_influence > 0:
                user_influence_metrics['Influence_Score'] = (
                    user_influence_metrics['Influence_Score'] / max_influence * 100
                ).round(1)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Top Influencers (All Categories)**")
                
                top_influencers = user_influence_metrics.sort_values('Influence_Score', ascending=False).head(15)
                
                # Color-code based on risk level
                def get_risk_color(avg_risk):
                    if avg_risk >= 7:
                        return "🔴"
                    elif avg_risk >= 5:
                        return "🟠"
                    elif avg_risk >= 3:
                        return "🟡"
                    else:
                        return "🟢"
                
                influencer_display = []
                for username, metrics in top_influencers.iterrows():
                    risk_emoji = get_risk_color(metrics['Avg_Risk_Score'])
                    influencer_display.append({
                        'Risk': risk_emoji,
                        'Username': username,
                        'Influence Score': f"{metrics['Influence_Score']:.1f}",
                        'Total Engagement': f"{metrics['Total_Engagement']:,.0f}",
                        'Posts': int(metrics['Post_Count']),
                        'Avg Risk': f"{metrics['Avg_Risk_Score']:.1f}",
                        'Anti-India': int(metrics['Anti_India_Posts']),
                        'Toxic': int(metrics['Toxic_Posts'])
                    })
                
                if influencer_display:
                    st.dataframe(pd.DataFrame(influencer_display), use_container_width=True)
                else:
                    st.info("No influencer data available")
            
            with col2:
                # st.write("**High-Risk Influencers Network**")
                
                # Focus on high-risk influencers
                high_risk_influencers = user_influence_metrics[
                    (user_influence_metrics['Avg_Risk_Score'] > 5) |
                    (user_influence_metrics['Anti_India_Posts'] > 0) |
                    (user_influence_metrics['Toxic_Posts'] > 0)
                ].sort_values('Influence_Score', ascending=False)
                
                if not high_risk_influencers.empty:
                    # Create network visualization data
                    network_data = []
                    for username, metrics in high_risk_influencers.head(10).iterrows():
                        network_data.append({
                            'username': username,
                            'influence_score': metrics['Influence_Score'],
                            'risk_score': metrics['Avg_Risk_Score'],
                            'anti_india_posts': metrics['Anti_India_Posts'],
                            'toxic_posts': metrics['Toxic_Posts'],
                            'total_engagement': metrics['Total_Engagement']
                        })
                    
                    # Display as bubble chart
                    if network_data:
                        network_df = pd.DataFrame(network_data)
                        
                        fig = px.scatter(
                            network_df,
                            x='influence_score',
                            y='risk_score', 
                            size='total_engagement',
                            color='anti_india_posts',
                            hover_name='username',
                            title='High-Risk Influencer Network',
                            labels={
                                'influence_score': 'Influence Score',
                                'risk_score': 'Risk Score',
                                'anti_india_posts': 'Anti-India Posts'
                            },
                            size_max=30
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Network connections analysis
                    st.write("**Network Connection Patterns:**")
                    
                    # Analyze common posting times (potential coordination)
                    if 'created_at' in analysis_df.columns:
                        high_risk_posts = analysis_df[
                            analysis_df['username'].isin(high_risk_influencers.index) &
                            (analysis_df['risk_score'] > 5)
                        ]
                        
                        if not high_risk_posts.empty:
                            high_risk_posts['created_at'] = pd.to_datetime(high_risk_posts['created_at'])
                            high_risk_posts['hour'] = high_risk_posts['created_at'].dt.hour
                            
                            # Find peak activity hours for high-risk users
                            peak_hours = high_risk_posts['hour'].value_counts().head(3)
                            
                            connection_insights = []
                            connection_insights.append(f"High-risk users most active during: {', '.join([f'{h}:00' for h in peak_hours.index])}")
                            
                            # Check for coordinated posting (same hour, multiple users)
                            hourly_users = high_risk_posts.groupby('hour')['username'].nunique()
                            coordinated_hours = hourly_users[hourly_users > 3].index
                            
                            if len(coordinated_hours) > 0:
                                connection_insights.append(f"Potential coordination detected during: {', '.join([f'{h}:00' for h in coordinated_hours])}")
                            
                            # Analyze common hashtags/terms
                            high_risk_texts = ' '.join(high_risk_posts['text'].tolist())
                            common_hashtags = re.findall(r'#\w+', high_risk_texts.lower())
                            if common_hashtags:
                                hashtag_counts = pd.Series(common_hashtags).value_counts().head(5)
                                connection_insights.append(f"Common hashtags: {', '.join(hashtag_counts.index)}")
                            
                            for insight in connection_insights:
                                st.write(f"- {insight}")
                        else:
                            st.info("No high-risk posting patterns detected")
                else:
                    st.info("No high-risk influencers identified")
            
            # Network mapping section
            st.write("**Follower Network Analysis**")
            
            # Simulate follower networks based on engagement patterns
            if 'coordination_results' in st.session_state:
                coord_results = st.session_state.coordination_results
                duplicates = coord_results.get('duplicates', [])
                
                if duplicates:
                    # Extract user connections from coordination data
                    user_connections = {}
                    
                    for dup in duplicates:
                        accounts = dup.get('accounts', [])
                        if len(accounts) >= 2:
                            for i, acc1 in enumerate(accounts):
                                if acc1 not in user_connections:
                                    user_connections[acc1] = set()
                                for acc2 in accounts[i+1:]:
                                    user_connections[acc1].add(acc2)
                                    if acc2 not in user_connections:
                                        user_connections[acc2] = set()
                                    user_connections[acc2].add(acc1)
                    
                    if user_connections:
                        st.write("**Network Connections Detected:**")
                        
                        network_stats = []
                        for user, connections in user_connections.items():
                            if user in user_influence_metrics.index:
                                influence_score = user_influence_metrics.loc[user, 'Influence_Score']
                                risk_score = user_influence_metrics.loc[user, 'Avg_Risk_Score']
                                
                                network_stats.append({
                                    'Username': user,
                                    'Connections': len(connections),
                                    'Influence Score': f"{influence_score:.1f}",
                                    'Risk Score': f"{risk_score:.1f}",
                                    'Connected To': ', '.join(list(connections)[:3]) + ('...' if len(connections) > 3 else '')
                                })
                        
                        if network_stats:
                            network_df = pd.DataFrame(network_stats)
                            network_df = network_df.sort_values('Connections', ascending=False)
                            st.dataframe(network_df, use_container_width=True)
                            
                            # Network density analysis
                            total_users = len(user_connections)
                            total_connections = sum(len(conns) for conns in user_connections.values()) // 2
                            max_possible_connections = total_users * (total_users - 1) // 2
                            network_density = total_connections / max_possible_connections if max_possible_connections > 0 else 0
                            
                            st.metric("Network Density", f"{network_density:.2%}")
                            
                            if network_density > 0.3:
                                st.warning("High network density suggests coordinated behavior")
                            elif network_density > 0.1:
                                st.info("Moderate network connectivity detected")
                            else:
                                st.success("Low network density - appears organic")
                    else:
                        st.info("No significant network connections detected")
                else:
                    st.info("Run coordination detection to analyze network connections")
            else:
                st.info("Network analysis requires coordination detection data")        
    
    # Enhanced alert system
    st.sidebar.markdown("---")
    st.sidebar.subheader("🚨 Enhanced Alert System")
    
    if 'analysis_results' in st.session_state:
        analysis_df = st.session_state.analysis_results
        critical_count = len(analysis_df[analysis_df['risk_score'] > 8])
        high_count = len(analysis_df[analysis_df['risk_score'] > 6])
        
        if critical_count > 0:
            st.sidebar.error(f"🚨 {critical_count} CRITICAL threats detected!")
        elif high_count > 0:
            st.sidebar.warning(f"⚠️ {high_count} HIGH-RISK posts found")
        else:
            st.sidebar.success("✅ No critical threats detected")
        
        # Auto-export for critical cases
        if critical_count > 5:  # Lowered threshold for demo
            if st.sidebar.button("📤 Emergency Export"):
                critical_posts = analysis_df[analysis_df['risk_score'] > 8]
                csv = critical_posts.to_csv(index=False)
                st.sidebar.download_button(
                    "⬇️ Download Critical Cases",
                    csv,
                    f"critical_threats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )

if __name__ == "__main__":
    main()
