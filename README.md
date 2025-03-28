# Automated Asset Monitoring - Solana

## Approach Description (Overview)

This project implements an automated pipeline for monitoring and analyzing social media discourse around the Solana cryptocurrency using Python. It scrapes Solana-related posts from X (formerly Twitter), processes them through various NLP models, and analyzes sentiment momentum over time. The four main stages of the pipeline are:

---

### 1. Authentication Setup

To access full search results on X, the scraper uses cookie-based authentication. This allows scraping while appearing as a logged-in user.

- A file named `x_cookies.json` contains session cookies for an authenticated account.
- If this file becomes outdated, you can generate a fresh one using the `save_x_cookies.py` script:

#### `save_x_cookies.py`

**Purpose**: Opens a Chrome browser window for manual login and captures cookies.

**Workflow**:
- Launches a Chrome browser (non-headless, maximized)
- Navigates to https://x.com/login
- Waits 90 seconds for manual login
- Saves cookies to `x_cookies.json`

**Usage**:
```bash
python save_x_cookies.py
```

**Login credentials for the test account**:
```
Username: Markus15660171  
Password: vitpaK-jevzod-4heghi
```

---

### 2. Data Collection

The script `scrape_solana_logged_in.py` automates the collection of Solana-related tweets using Selenium.

#### `scrape_solana_logged_in.py`

**Purpose**: Fetch tweets related to Solana from both the "Top" and "Latest" tabs over the past 7 days.

**Search Query**: `"Solana OR $SOL OR #SOL"`

**Features**:
- Cookie-based authentication  
- Date filtering using `since:` and `until:`  
- Tab switching for broader coverage  
- Duplicate tweet filtering  
- Human-like behavior (random scrolling, delays, mouse movements)  
- Resilient to minor UI changes via XPath and fallback logic  
- Extracts engagement metrics (likes count) for analysis  

**Output**:
- `solana_search_tweets.csv`: Combined dataset for all 7 days  
- `solana_tweets_daily/solana_tweets_YYYY-MM-DD.csv`: Daily CSV files  

**Core Techniques**:
- WebDriver configuration with anti-detection flags  
- JavaScript-based smooth scrolling  
- ActionChains for mouse activity  
- Randomized delays between scrolls  
- Early stopping if no new tweets are loaded  
- Engagement data extraction from tweet metadata  

**Key Functions**:
- `setup_driver()`  
- `load_cookies()`  
- `scroll_and_extract_tweets()`  
- `switch_to_tab()`  
- `extract_tweet_content()`  
- `save_to_csv()`  

**Usage**:
```bash
python scrape_solana_logged_in.py
```

**Example Output Row**:
```
author,handle,date,text,likes  
Jane Doe,@janedoe,2025-03-25T10:15:00Z,"Solana just hit a new high! $SOL to the moon.",42
```

---

### 3. Content Analysis

The `solana_analyzer.py` script performs a comprehensive multi-model analysis of the tweets.

#### `solana_analyzer.py`

**Purpose**: Evaluate each tweet on:
- Relevance (incorporating both content and engagement)  
- Risk (e.g., scams)  
- Reliability  
- Sentiment  
- Category (e.g., price, news, NFT)  

**Models Used**:

| Task         | Model                                      | Type                      |
|--------------|--------------------------------------------|---------------------------|
| Relevance    | `facebook/bart-large-mnli`                 | Zero-shot classification |
| Risk         | `cardiffnlp/twitter-roberta-base-offensive`| Text classification       |
| Sentiment    | `distilbert-base-uncased-finetuned-sst-2-english` | Sentiment analysis |
| Categorization | `facebook/bart-large-mnli`               | Zero-shot classification |

**Scoring Logic**:

- **Relevance (1–10)**:  
  Combines Solana-related content analysis with engagement metrics  
  - Content analysis evaluates the tweet's focus on Solana  
  - Engagement metrics (likes) are incorporated using a logarithmic scale  
  - Higher engagement provides a modest boost to relevance (max +3 points)

- **Risk (1–10)**: Model predictions + scam keyword patterns  
- **Reliability (1–10)**: Heuristic (links, stats, ALL CAPS, emojis)  
- **Sentiment**: Positive, Negative, Neutral  
- **Category**: One of `[Price, News, Project, NFT, DeFi, Scam, Meme, Opinion, Other]`

**Performance Optimization**:
- Batch processing with adjustable batch size  
- Sequential model usage to minimize RAM usage  
- Skips analysis for tweets with low relevance  

**Usage**:
```bash
python solana_analyzer.py --input solana_search_tweets.csv --output solana_tweets_analyzed.csv --batch-size 5
```

**Output**:
- `solana_tweets_analyzed.csv`: Enriched data with NLP scores

---

### 4. Trend Analysis

The `sentiment_momentum.py` script analyzes daily sentiment trends.

#### `sentiment_momentum.py`

**Purpose**: Visualize sentiment shifts in the Solana discourse over 7 days.

**Process**:
- Reads `solana_tweets_analyzed.csv`
- Maps sentiment values to numeric scale:
  - Positive: +1  
  - Neutral: 0  
  - Negative: -1
- Averages scores per day
- Converts averages back to sentiment category:
  - `> 0.33` → Positive  
  - `< -0.33` → Negative  
  - Otherwise → Neutral
- Plots and saves `sentiment_momentum_plot.png`

**Usage**:
```bash
python sentiment_momentum.py
```

**Output**:
- `sentiment_momentum_plot.png`: Color-coded sentiment bars over time

---

## How to Run the Code

First, you need to have a Python environment set up with all required dependencies and Chrome installed on your machine.

### Prerequisites

- Python 3.6+  
- Chrome browser installed and available via PATH

**Required Python packages**:
```bash
pip install selenium webdriver-manager pandas tqdm torch transformers matplotlib numpy
```

---

## Pipeline Execution

### Step 1: Authentication Setup  
(only if `x_cookies.json` fails to log you in during execution of the `scrape_solana_logged_in.py` script)
```bash
python save_x_cookies.py
```
Use the test account credentials when the browser opens. Wait for cookies to save.

### Step 2: Data Collection
```bash
python scrape_solana_logged_in.py
```
This collects tweets from both tabs across 7 days and saves them to CSV.

### Step 3: Content Analysis
```bash
python solana_analyzer.py --input solana_search_tweets.csv --output solana_tweets_analyzed.csv --batch-size 5
```
Processes tweets and adds evaluation scores.

### Step 4: Sentiment Trend Visualization
```bash
python sentiment_momentum.py
```
Generates a 7-day sentiment momentum chart.

---

## Expected Output Files

- `x_cookies.json`: Authentication cookie file  
- `solana_search_tweets.csv`: Raw tweet data with content and engagement metrics  
- `solana_tweets_daily/*.csv`: Daily tweet CSVs  
- `solana_tweets_analyzed.csv`: Enriched data with NLP scores  
- `sentiment_momentum_plot.png`: Sentiment trend visualization  

---

## Troubleshooting

- **Cookies expired**: Re-run `save_x_cookies.py`  
- **Memory errors**: Lower `--batch-size` in `solana_analyzer.py`  
- **Scraping fails**: Update XPath selectors if X UI changes, or wait and try again  

---

## Design Rationale

- Logged-in scraping with cookies was chosen over the X API because the free developer tier only allows 100 tweets/month, which is too limited for this project.
- Non-headless Chrome was used because X's interface appeared to detect and throttle headless browsers more aggressively during testing.
- Task-specific NLP models were chosen over lightweight general-purpose LLMs because the latter performed poorly on tweet-level relevance and risk classification.
- Engagement-aware relevance scoring allows the system to prioritize content that resonates with the community, adding a social-proof dimension that pure text analysis cannot provide.
- Storing credentials in the README/script is generally not recommended. However, due to time constraints and the fact that this is a private repository using a disposable test account, this was deemed acceptable for the scope of this project.

---

This project enables automated monitoring of Solana-related discussions on X by combining real-time scraping, NLP analysis, and temporal sentiment tracking in one streamlined pipeline.

