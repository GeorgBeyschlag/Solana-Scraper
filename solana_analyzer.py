import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers import BartForSequenceClassification, BartTokenizer
import gc
import time
from tqdm import tqdm
import re
import json

class SolanaSpecializedAnalyzer:
    def __init__(self, batch_size=5):
        """
        Initialize the analyzer with specialized models for each task
        
        Parameters:
            batch_size: Number of tweets to process in one batch (default increased for speed)
        """
        self.batch_size = batch_size
        self.current_model = None
        self.current_task = None
        
        # Check for MPS (Metal Performance Shaders) on Mac
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Define model paths
        self.models = {
            "relevance": "facebook/bart-large-mnli",  # Zero-shot classification for relevance
            "risk": "cardiffnlp/twitter-roberta-base-offensive",  # Risk assessment
            "category": "facebook/bart-large-mnli",  # Zero-shot classification for categories
            "sentiment": "distilbert-base-uncased-finetuned-sst-2-english"  # Sentiment analysis
        }
        
        # Define Solana ecosystem terms for additional keyword-based scoring
        self.solana_ecosystem = {
            "core": ["solana", "sol ", "sol,", "sol.", "$sol", "#sol", "#solana"],
            "projects": ["serum", "raydium", "orca", "magic eden", "phantom", "metaplex", 
                        "pyth", "solend", "mango markets", "step finance", "saber", "sunny"],
            "tokens": ["bonk", "samo", "kin", "ray", "srm", "maps", "cope", "fida", "mer", "sbl"],
            "concepts": ["spl", "token program", "wormhole", "sollet", "break solana", 
                        "solana summer", "solana mobile", "saga phone", "solana pay"]
        }
        
        # Define risk keywords for supplementary analysis
        self.risk_keywords = {
            "high": ["giveaway", "airdrop", "free", "urgent", "hurry", "guaranteed", 
                    "double", "10x", "100x", "pump", "send", "dm me", "opportunity",
                    "limited time", "presale", "whitelist spots", "exclusive access",
                    "only 100 spots", "get rich", "before it's gone", "once in a lifetime"],
            "medium": ["mint", "drop", "address", "DM", "private message", "click", 
                      "link", "claim", "token", "new project", "launching", "listing",
                      "exchange", "low cap", "gem", "moonshot", "early", "seed sale"],
            "low": ["buy", "sell", "trade", "price", "prediction", "soon", "moon",
                   "bullish", "bearish", "dip", "ATH", "all time high", "resistance"]
        }
        
        # Define category patterns for keyword-based fallback
        self.category_patterns = {
            "Price": ["price", "market", "trading", "$", "bullish", "bearish", "chart", "up", "down", "ath"],
            "News": ["announced", "announces", "breaking", "news", "update", "released", "launches", "today"],
            "Project": ["building", "built", "launched", "created", "developing", "project", "protocol", "app"],
            "NFT": ["nft", "mint", "collection", "art", "jpeg", "pfp", "avatar", "generative"],
            "DeFi": ["defi", "yield", "farm", "stake", "lending", "borrowing", "liquidity", "swap"],
            "Scam": ["giveaway", "airdrop", "free", "send", "claim", "urgent", "hurry", "opportunity"],
            "Meme": ["lol", "lmao", "😂", "meme", "funny", "joke", "😭", "wagmi", "ngmi", "gm", "gn"]
        }
    
    def load_model(self, task):
        """Load a specific model for a task, unloading previous model to save memory"""
        if self.current_task == task and self.current_model is not None:
            return  # Already loaded
            
        # Unload current model to free memory
        if self.current_model is not None:
            self.current_model = None
            if hasattr(self, 'tokenizer'):
                self.tokenizer = None
            if hasattr(self, 'classifier'):
                self.classifier = None
            gc.collect()
            if self.device == "mps":
                torch.mps.empty_cache()
            
        print(f"Loading model for {task}...")
        self.current_task = task
        
        try:
            if task == "relevance":
                # Load BART model for zero-shot classification
                self.tokenizer = BartTokenizer.from_pretrained(self.models[task])
                self.current_model = BartForSequenceClassification.from_pretrained(self.models[task])
                self.current_model.to(self.device)
                self.classifier = pipeline("zero-shot-classification", 
                                          model=self.current_model, 
                                          tokenizer=self.tokenizer,
                                          device=0 if self.device == "mps" else -1)
            elif task == "category":
                # Load BART model for zero-shot classification (same as relevance but different labels)
                self.tokenizer = BartTokenizer.from_pretrained(self.models[task])
                self.current_model = BartForSequenceClassification.from_pretrained(self.models[task])
                self.current_model.to(self.device)
                self.classifier = pipeline("zero-shot-classification", 
                                          model=self.current_model, 
                                          tokenizer=self.tokenizer,
                                          device=0 if self.device == "mps" else -1)
            elif task == "risk":
                # Load RoBERTa model for offensive content detection
                self.tokenizer = AutoTokenizer.from_pretrained(self.models[task])
                self.current_model = AutoModelForSequenceClassification.from_pretrained(self.models[task])
                self.current_model.to(self.device)
                self.classifier = pipeline("text-classification", 
                                          model=self.current_model, 
                                          tokenizer=self.tokenizer,
                                          device=0 if self.device == "mps" else -1)
            elif task == "sentiment":
                # Load sentiment analysis model
                self.classifier = pipeline("sentiment-analysis", 
                                          model=self.models[task],
                                          device=0 if self.device == "mps" else -1)
                
            print(f"Successfully loaded model for {task}!")
        except Exception as e:
            print(f"Error loading model for {task}: {e}")
            self.current_model = None
            self.current_task = None

    def analyze_relevance(self, tweet_text):
        """Analyze tweet relevance to Solana using zero-shot classification"""
        try:
            self.load_model("relevance")
            
            hypothesis_template = "This tweet is about {}"
            candidate_labels = ["Solana blockchain", "cryptocurrency unrelated to Solana", "non-cryptocurrency topics"]
            
            result = self.classifier(tweet_text, candidate_labels, hypothesis_template=hypothesis_template)
            
            # Get the probability for Solana
            solana_idx = result["labels"].index("Solana blockchain")
            solana_prob = result["scores"][solana_idx]
            
            # Convert probability to a 1-10 scale
            relevance_score = round(solana_prob * 10)
            
            # If the score is very low but contains Solana keywords, adjust it
            if relevance_score < 3:
                keyword_relevance = self.keyword_relevance(tweet_text)
                if keyword_relevance > relevance_score:
                    relevance_score = min(10, int((relevance_score + keyword_relevance) / 2) + 1)
            
            return relevance_score
        except Exception as e:
            print(f"Error in relevance analysis: {e}")
            return self.keyword_relevance(tweet_text)
    
    def keyword_relevance(self, tweet_text):
        """Backup method for relevance using keyword matching"""
        tweet_lower = tweet_text.lower()
        
        # Check for core Solana mentions
        core_mentions = sum(1 for term in self.solana_ecosystem["core"] if term.lower() in tweet_lower)
        
        # Check for project mentions
        project_mentions = sum(1 for term in self.solana_ecosystem["projects"] if term.lower() in tweet_lower)
        
        # Check for token mentions
        token_mentions = sum(1 for term in self.solana_ecosystem["tokens"] if term.lower() in tweet_lower)
        
        # Check for concept mentions
        concept_mentions = sum(1 for term in self.solana_ecosystem["concepts"] if term.lower() in tweet_lower)
        
        # Calculate weighted score
        relevance_score = (core_mentions * 2.5) + (project_mentions * 1.5) + (token_mentions * 1.0) + (concept_mentions * 1.0)
        
        # Cap at 10
        relevance_score = min(10, relevance_score)
        
        # If no Solana terms found, score should be very low
        if relevance_score == 0 and any(crypto in tweet_lower for crypto in ["crypto", "bitcoin", "eth", "nft"]):
            relevance_score = 1  # At least crypto-related
        
        return relevance_score
    
    def analyze_risk(self, tweet_text):
        """Analyze risk level using offensive content model and keywords"""
        try:
            self.load_model("risk")
            
            # Get offensive content probability
            result = self.classifier(tweet_text)
            
            # Parse result to get risk score
            if result[0]["label"] == "offensive":
                offensive_prob = result[0]["score"]
            else:
                offensive_prob = 1 - result[0]["score"]
            
            # Scale to 1-10 but cap at 7 (we'll add more for specific crypto scam patterns)
            model_risk = min(7, round(offensive_prob * 7) + 1)
            
            # Supplement with crypto-specific risk keywords
            keyword_risk = self.analyze_keyword_risk(tweet_text)
            
            # Take the maximum of the two approaches
            risk_score = max(model_risk, keyword_risk)
            
            return risk_score
        except Exception as e:
            print(f"Error in risk analysis: {e}")
            return self.analyze_keyword_risk(tweet_text)
    
    def analyze_keyword_risk(self, tweet_text):
        """Analyze risk based on crypto-specific keywords"""
        tweet_lower = tweet_text.lower()
        
        # Count occurrences of risk keywords by category
        high_risk_count = sum(1 for word in self.risk_keywords["high"] if word.lower() in tweet_lower)
        medium_risk_count = sum(1 for word in self.risk_keywords["medium"] if word.lower() in tweet_lower)
        low_risk_count = sum(1 for word in self.risk_keywords["low"] if word.lower() in tweet_lower)
        
        # Calculate weighted risk score
        keyword_risk = min(10, (high_risk_count * 2.5) + (medium_risk_count * 1.2) + (low_risk_count * 0.5))
        
        # Check for specific high-risk combinations (improve scam detection)
        if "send" in tweet_lower and any(coin in tweet_lower for coin in ["sol", "solana"]):
            keyword_risk = max(keyword_risk, 8)
        
        if "airdrop" in tweet_lower and any(urgency in tweet_lower for urgency in ["hurry", "limited", "fast", "quick"]):
            keyword_risk = max(keyword_risk, 8)
            
        if "giveaway" in tweet_lower and any(action in tweet_lower for action in ["follow", "retweet", "tag", "dm"]):
            keyword_risk = max(keyword_risk, 7)
        
        return keyword_risk
    
    def analyze_category(self, tweet_text):
        """Categorize the tweet using zero-shot classification"""
        try:
            self.load_model("category")
            
            # Define categories
            categories = [
                "Price discussion", "News announcement", 
                "Project development", "NFT collection", 
                "DeFi protocol", "Scam or suspicious content",
                "Meme or joke", "Opinion"
            ]
            
            result = self.classifier(tweet_text, categories)
            
            # Get the top category
            top_category = result["labels"][0]
            
            # Map to our standard categories
            category_mapping = {
                "Price discussion": "Price",
                "News announcement": "News",
                "Project development": "Project",
                "NFT collection": "NFT",
                "DeFi protocol": "DeFi",
                "Scam or suspicious content": "Scam",
                "Meme or joke": "Meme",
                "Opinion": "Opinion"
            }
            
            category = category_mapping.get(top_category, "Other")
            
            # If it's categorized as Scam, double-check risk score is high
            if category == "Scam":
                risk_score = self.analyze_keyword_risk(tweet_text)
                if risk_score < 6:  # If risk is actually low, recategorize
                    # Get second best category
                    if len(result["labels"]) > 1:
                        second_category = result["labels"][1]
                        category = category_mapping.get(second_category, "Other")
            
            return category
        except Exception as e:
            print(f"Error in category analysis: {e}")
            return self.fallback_categorization(tweet_text)
    
    def fallback_categorization(self, tweet_text):
        """Keyword-based fallback categorization"""
        # Handle non-string values
        if not isinstance(tweet_text, str):
            if pd.isna(tweet_text):
                return "Other"
            try:
                # Try to convert to string if possible
                tweet_text = str(tweet_text)
            except:
                return "Other"
                
        tweet_lower = tweet_text.lower()
        
        # Check each category
        category_scores = {}
        for category, keywords in self.category_patterns.items():
            score = sum(1 for word in keywords if word in tweet_lower)
        
        # Get the highest scoring category
        if max(category_scores.values(), default=0) > 0:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        
        # If no category detected
        return "Other"
    
    def analyze_sentiment(self, tweet_text):
        """Analyze sentiment using specialized sentiment model"""
        try:
            self.load_model("sentiment")
            
            result = self.classifier(tweet_text)
            sentiment_label = result[0]['label']
            sentiment_score = result[0]['score']
            
            # Convert to human-readable format
            if sentiment_label == 'NEGATIVE':
                sentiment = 'Negative'
            elif sentiment_label == 'POSITIVE':
                sentiment = 'Positive'
            else:
                sentiment = sentiment_label
                
            # If score is close to 0.5, set as Neutral
            if 0.4 <= sentiment_score <= 0.6:
                sentiment = 'Neutral'
                
            return sentiment
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            return "Neutral"
    
    def estimate_reliability(self, tweet_text):
        """Estimate reliability of the content (heuristic approach)"""
        reliability_score = 5  # Default middle value
        
        # Adjust based on characteristics
        has_links = "http" in tweet_text.lower()
        has_specific_numbers = any(char.isdigit() for char in tweet_text)
        is_very_short = len(tweet_text) < 40
        has_excessive_punctuation = tweet_text.count('!') > 3 or tweet_text.count('?') > 3
        all_caps = tweet_text.isupper()
        has_emojis = any(char in tweet_text for char in ["😂", "🚀", "💎", "🔥", "💰"])
        
        # Adjust reliability
        if has_links:
            reliability_score += 1  # Links can indicate sources
        if has_specific_numbers:
            reliability_score += 1  # Specific numbers can indicate precision
        if is_very_short:
            reliability_score -= 1  # Very short tweets may lack context
        if has_excessive_punctuation:
            reliability_score -= 2  # Excessive punctuation may indicate hype
        if all_caps:
            reliability_score -= 2  # ALL CAPS can indicate hyperbole
        if has_emojis:
            reliability_score -= 0.5  # Many crypto emojis might indicate hype
            
        # Cap between 1-10
        reliability_score = max(1, min(10, reliability_score))
        
        return reliability_score
    
    def analyze_tweet(self, tweet_text):
        """Comprehensive tweet analysis using specialized models"""
        # Handle non-string inputs
        if not isinstance(tweet_text, str):
            if pd.isna(tweet_text):
                return {
                    "relevance": 0,
                    "risk": 0,
                    "reliability": 0,
                    "sentiment": "Neutral",
                    "category": "Other",
                    "explanation": "Empty tweet"
                }
            try:
                # Try to convert to string if possible
                tweet_text = str(tweet_text)
            except:
                return {
                    "relevance": 0,
                    "risk": 0,
                    "reliability": 0,
                    "sentiment": "Neutral", 
                    "category": "Other",
                    "explanation": "Non-text content"
                }
                
        if not tweet_text or tweet_text.strip() == "":
            return {
                "relevance": 0,
                "risk": 0,
                "reliability": 0,
                "sentiment": "Neutral",
                "category": "Other",
                "explanation": "Empty tweet"
            }
            
        result = {
            "relevance": None,
            "risk": None,
            "reliability": None,
            "sentiment": None,
            "category": None,
            "explanation": ""
        }
        
        # 1. Analyze relevance (should we process this tweet further?)
        result["relevance"] = self.analyze_relevance(tweet_text)
        
        # If very low relevance, skip detailed analysis
        if result["relevance"] <= 2:
            result["risk"] = 1
            result["reliability"] = 5
            result["sentiment"] = "Neutral"
            result["category"] = "Other"
            result["explanation"] = "Low relevance to Solana"
            return result
        
        # 2. Analyze risk
        result["risk"] = self.analyze_risk(tweet_text)
        
        # 3. Analyze category
        result["category"] = self.analyze_category(tweet_text)
        
        # 4. Analyze sentiment
        result["sentiment"] = self.analyze_sentiment(tweet_text)
        
        # 5. Estimate reliability
        result["reliability"] = self.estimate_reliability(tweet_text)
        
        # 6. Generate brief explanation
        explanation_parts = []
        if result["relevance"] > 7:
            explanation_parts.append(f"Highly relevant to Solana")
        elif result["relevance"] < 4:
            explanation_parts.append(f"Somewhat relevant to Solana")
            
        explanation_parts.append(f"{result['sentiment']} sentiment")
            
        if result["risk"] > 7:
            explanation_parts.append(f"High risk indicators")
        elif result["risk"] < 3:
            explanation_parts.append(f"Low risk")
            
        if result["category"] != "Other":
            explanation_parts.append(f"Categorized as {result['category']}")
            
        result["explanation"] = ", ".join(explanation_parts)
        
        return result
    
    def process_dataframe(self, df, output_path="solana_tweets_analyzed.csv"):
        """Process all tweets in the dataframe using a task-based approach for speed"""
        # Create columns for analysis results if they don't exist
        for col in ['relevance', 'risk', 'reliability', 'sentiment', 'category', 'explanation']:
            if col not in df.columns:
                df[col] = None
        
        # Get valid tweets
        valid_indices = []
        valid_texts = []
        
        print("Filtering valid tweets...")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            # Skip if already processed
            if pd.notna(df.at[idx, 'relevance']) and not isinstance(df.at[idx, 'relevance'], str):
                continue
                
            # Get the tweet text
            tweet_text = row.get('text', '')
            
            # Skip empty tweets
            if not isinstance(tweet_text, str) or not tweet_text.strip():
                continue
                
            valid_indices.append(idx)
            valid_texts.append(tweet_text)
        
        print(f"Processing {len(valid_indices)} tweets by task...")
        
        # Process by task instead of by tweet (much faster approach)
        
        # 1. Process all relevance scores
        print("Analyzing relevance for all tweets...")
        self.load_model("relevance")
        relevance_scores = []
        
        # Process in larger batches for relevance
        batch_size = min(10, self.batch_size * 3)  # Increase batch size for speed
        for i in tqdm(range(0, len(valid_texts), batch_size)):
            batch_texts = valid_texts[i:i+batch_size]
            batch_scores = []
            
            for text in batch_texts:
                try:
                    score = self.analyze_relevance(text)
                    batch_scores.append(score)
                except Exception as e:
                    print(f"Error in relevance analysis: {e}")
                    batch_scores.append(self.keyword_relevance(text))
            
            relevance_scores.extend(batch_scores)
            
            # Save intermediate results
            for j, idx in enumerate(valid_indices[i:i+batch_size]):
                if j < len(batch_scores):
                    df.at[idx, 'relevance'] = batch_scores[j]
                    
            # Occasional memory cleanup
            if i % (batch_size * 5) == 0:
                if self.device == "mps":
                    torch.mps.empty_cache()
                gc.collect()
        
        # 2. Analyze only tweets with relevance > 2 for other metrics
        print("Filtering tweets with relevance > 2...")
        relevant_indices = []
        relevant_texts = []
        
        for i, idx in enumerate(valid_indices):
            if i < len(relevance_scores) and relevance_scores[i] > 2:
                relevant_indices.append(idx)
                relevant_texts.append(valid_texts[i])
        
        print(f"Processing {len(relevant_indices)} relevant tweets for remaining metrics...")
        
        # 3. Process risk for relevant tweets
        print("Analyzing risk...")
        self.load_model("risk")
        risk_scores = []
        
        for i in tqdm(range(0, len(relevant_texts), batch_size)):
            batch_texts = relevant_texts[i:i+batch_size]
            batch_scores = []
            
            for text in batch_texts:
                try:
                    score = self.analyze_risk(text)
                    batch_scores.append(score)
                except Exception as e:
                    print(f"Error in risk analysis: {e}")
                    batch_scores.append(self.analyze_keyword_risk(text))
            
            risk_scores.extend(batch_scores)
            
            # Save intermediate results
            for j, idx in enumerate(relevant_indices[i:i+batch_size]):
                if j < len(batch_scores):
                    df.at[idx, 'risk'] = batch_scores[j]
                    
            # Occasional memory cleanup
            if i % (batch_size * 5) == 0:
                if self.device == "mps":
                    torch.mps.empty_cache()
                gc.collect()
        
        # 4. Process categories
        print("Analyzing categories...")
        self.load_model("category")
        categories = []
        
        for i in tqdm(range(0, len(relevant_texts), batch_size)):
            batch_texts = relevant_texts[i:i+batch_size]
            batch_categories = []
            
            for text in batch_texts:
                try:
                    category = self.analyze_category(text)
                    batch_categories.append(category)
                except Exception as e:
                    print(f"Error in category analysis: {e}")
                    batch_categories.append(self.fallback_categorization(text))
            
            categories.extend(batch_categories)
            
            # Save intermediate results
            for j, idx in enumerate(relevant_indices[i:i+batch_size]):
                if j < len(batch_categories):
                    df.at[idx, 'category'] = batch_categories[j]
                    
            # Occasional memory cleanup
            if i % (batch_size * 5) == 0:
                if self.device == "mps":
                    torch.mps.empty_cache()
                gc.collect()
        
        # 5. Process sentiment
        print("Analyzing sentiment...")
        self.load_model("sentiment")
        sentiments = []
        
        for i in tqdm(range(0, len(relevant_texts), batch_size)):
            batch_texts = relevant_texts[i:i+batch_size]
            batch_sentiments = []
            
            for text in batch_texts:
                try:
                    sentiment = self.analyze_sentiment(text)
                    batch_sentiments.append(sentiment)
                except Exception as e:
                    print(f"Error in sentiment analysis: {e}")
                    batch_sentiments.append("Neutral")
            
            sentiments.extend(batch_sentiments)
            
            # Save intermediate results
            for j, idx in enumerate(relevant_indices[i:i+batch_size]):
                if j < len(batch_sentiments):
                    df.at[idx, 'sentiment'] = batch_sentiments[j]
                    
            # Occasional memory cleanup
            if i % (batch_size * 5) == 0:
                if self.device == "mps":
                    torch.mps.empty_cache()
                gc.collect()
        
        # 6. Calculate reliability for all relevant tweets
        print("Calculating reliability scores...")
        for i, idx in enumerate(relevant_indices):
            if i < len(relevant_texts):
                df.at[idx, 'reliability'] = self.estimate_reliability(relevant_texts[i])
        
        # 7. Generate explanations
        print("Generating explanations...")
        for idx in relevant_indices:
            try:
                if pd.notna(df.at[idx, 'relevance']):
                    explanation_parts = []
                    
                    if df.at[idx, 'relevance'] > 7:
                        explanation_parts.append(f"Highly relevant to Solana")
                    elif df.at[idx, 'relevance'] < 4:
                        explanation_parts.append(f"Somewhat relevant to Solana")
                        
                    if pd.notna(df.at[idx, 'sentiment']):
                        explanation_parts.append(f"{df.at[idx, 'sentiment']} sentiment")
                        
                    if pd.notna(df.at[idx, 'risk']) and df.at[idx, 'risk'] > 7:
                        explanation_parts.append(f"High risk indicators")
                    elif pd.notna(df.at[idx, 'risk']) and df.at[idx, 'risk'] < 3:
                        explanation_parts.append(f"Low risk")
                        
                    if pd.notna(df.at[idx, 'category']) and df.at[idx, 'category'] != "Other":
                        explanation_parts.append(f"Categorized as {df.at[idx, 'category']}")
                        
                    df.at[idx, 'explanation'] = ", ".join(explanation_parts)
            except Exception as e:
                print(f"Error generating explanation for tweet {idx}: {e}")
                df.at[idx, 'explanation'] = "Analysis completed"
        
        # Fill in default values for non-relevant tweets
        print("Setting default values for non-relevant tweets...")
        for idx in valid_indices:
            if idx not in relevant_indices and pd.notna(df.at[idx, 'relevance']):
                df.at[idx, 'risk'] = 1
                df.at[idx, 'reliability'] = 5 
                df.at[idx, 'sentiment'] = "Neutral"
                df.at[idx, 'category'] = "Other"
                df.at[idx, 'explanation'] = "Low relevance to Solana"
        
        # Post-process for consistency
        print("Post-processing results...")
        df = self.post_process_results(df)
        
        # Save final results
        df.to_csv(output_path, index=False)
        print(f"Analysis complete! Results saved to {output_path}")
        
        return df
    
    def post_process_results(self, df):
        """Clean and normalize the analysis results"""
        # Ensure numeric columns are numeric
        for col in ['relevance', 'risk', 'reliability']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Fill missing values with reasonable defaults
            if col == 'relevance':
                df[col] = df[col].fillna(1)
            elif col == 'risk':
                df[col] = df[col].fillna(3)
            elif col == 'reliability':
                df[col] = df[col].fillna(5)
        
        # Clean up categories
        valid_categories = ['News', 'Price', 'Project', 'NFT', 'DeFi', 'Scam', 'Meme', 'Opinion', 'Other']
        # Apply fallback categorization where needed
        for idx, row in df.iterrows():
            try:
                if pd.isna(row['category']) or row['category'] not in valid_categories:
                    # Make sure text is a string before processing
                    if isinstance(row['text'], str):
                        df.at[idx, 'category'] = self.fallback_categorization(row['text'])
                    else:
                        df.at[idx, 'category'] = "Other"
            except Exception as e:
                print(f"Error in post-processing row {idx}: {e}")
                df.at[idx, 'category'] = "Other"
        
        # Ensure sentiment is consistent
        df['sentiment'] = df['sentiment'].apply(lambda x: 'Neutral' if pd.isna(x) else x)
        df['sentiment'] = df['sentiment'].apply(lambda x: x.capitalize() if isinstance(x, str) else 'Neutral')
        
        return df
    
    def generate_report(self, df):
        """Generate a summary report of the analysis"""
        print("\n===== ANALYSIS REPORT =====")
        print(f"Total tweets analyzed: {len(df)}")
        
        # Calculate average scores
        print("\n--- Average Scores ---")
        for metric in ['relevance', 'risk', 'reliability']:
            # Convert to numeric, filtering out non-numeric values
            numeric_values = pd.to_numeric(df[metric], errors='coerce')
            avg = numeric_values.mean()
            print(f"Average {metric.capitalize()}: {avg:.2f}/10")
        
        # Sentiment distribution
        print("\n--- Sentiment Distribution ---")
        sentiment_counts = df['sentiment'].value_counts()
        for sentiment, count in sentiment_counts.items():
            if pd.notna(sentiment):
                percentage = (count / len(df)) * 100
                print(f"{sentiment}: {count} tweets ({percentage:.1f}%)")
        
        # Category distribution
        print("\n--- Category Distribution ---")
        category_counts = df['category'].value_counts()
        for category, count in category_counts.items():
            if pd.notna(category):
                percentage = (count / len(df)) * 100
                print(f"{category}: {count} tweets ({percentage:.1f}%)")
        
        # High-risk tweets
        numeric_risk = pd.to_numeric(df['risk'], errors='coerce')
        high_risk = df[numeric_risk > 7].copy()
        print(f"\n--- High Risk Tweets ({len(high_risk)}) ---")
        for _, row in high_risk.head(5).iterrows():
            handle = row.get('handle', 'Unknown')
            text = row.get('text', '')[:70] + "..." if len(row.get('text', '')) > 70 else row.get('text', '')
            risk = row.get('risk', 'N/A')
            print(f"- {handle}: Risk {risk}/10 - {text}")
        
        if len(high_risk) > 5:
            print(f"... and {len(high_risk) - 5} more high-risk tweets")
        
        # Most relevant tweets
        numeric_relevance = pd.to_numeric(df['relevance'], errors='coerce')
        high_relevance = df[numeric_relevance > 7].copy()
        print(f"\n--- Highly Relevant Tweets ({len(high_relevance)}) ---")
        for _, row in high_relevance.head(5).iterrows():
            handle = row.get('handle', 'Unknown')
            text = row.get('text', '')[:70] + "..." if len(row.get('text', '')) > 70 else row.get('text', '')
            relevance = row.get('relevance', 'N/A')
            print(f"- {handle}: Relevance {relevance}/10 - {text}")
        
        if len(high_relevance) > 5:
            print(f"... and {len(high_relevance) - 5} more highly relevant tweets")
        
        # Category examples
        print("\n--- Category Examples ---")
        for category in df['category'].unique():
            if pd.notna(category) and category != "Other":
                category_df = df[df['category'] == category].head(1)
                if not category_df.empty:
                    row = category_df.iloc[0]
                    handle = row.get('handle', 'Unknown')
                    text = row.get('text', '')[:70] + "..." if len(row.get('text', '')) > 70 else row.get('text', '')
                    print(f"{category}: {handle} - {text}")
        
        print("\n===========================")

def main():
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze Solana tweets with specialized models')
    parser.add_argument('--input', type=str, default='solana_search_tweets.csv', help='Input CSV file')
    parser.add_argument('--output', type=str, default='solana_tweets_analyzed.csv', help='Output CSV file')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size for processing')
    args = parser.parse_args()
    
    # Enable MPS (Metal Performance Shaders) for M2 Mac if available
    if torch.backends.mps.is_available():
        print("MPS (Metal Performance Shaders) is available! Using M2 GPU acceleration.")
    else:
        print("MPS not available. Using CPU only.")
    
    # Initialize the analyzer
    analyzer = SolanaSpecializedAnalyzer(batch_size=args.batch_size)
    
    # Load data
    print(f"Loading data from {args.input}")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} tweets")
    
    # Process the data
    result_df = analyzer.process_dataframe(df, output_path=args.output)
    
    # Generate report
    analyzer.generate_report(result_df)
    
    print(f"Analysis complete! Results saved to {args.output}")

if __name__ == "__main__":
    main()