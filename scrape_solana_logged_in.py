import time
import json
import csv
import random
import os
from datetime import datetime, timedelta

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


def load_cookies(driver, cookie_file):
    """Load cookies from file to authenticate with X"""
    with open(cookie_file, "r") as f:
        cookies = json.load(f)
    for cookie in cookies:
        if "sameSite" in cookie and cookie["sameSite"] == "None":
            cookie["sameSite"] = "Strict"
        driver.add_cookie(cookie)


def setup_driver(headless=True):
    """Set up and configure the Chrome webdriver"""
    chrome_options = Options()
    
    # Use a realistic User-Agent (Chrome on macOS)
    user_agent = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
    chrome_options.add_argument(f"user-agent={user_agent}")
    
    if headless:
        chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)


def move_mouse_randomly(driver):
    """Move mouse randomly to avoid detection"""
    actions = ActionChains(driver)
    x = random.randint(0, 500)
    y = random.randint(0, 500)
    actions.move_by_offset(x, y).perform()
    actions.move_by_offset(-x, -y).perform()


def switch_to_tab(driver, tab_name="Top"):
    """Switch to a specific tab (Top or Latest) using URL parameter"""
    try:
        print(f"Switching to {tab_name} tab via URL parameter...")
        
        # Get current URL
        current_url = driver.current_url
        
        # Determine the correct parameter for the desired tab
        if tab_name == "Top":
            # Change f=live to f=top or add f=top if not present
            if "f=live" in current_url:
                new_url = current_url.replace("f=live", "f=top")
            elif "&f=" not in current_url:
                new_url = current_url + "&f=top"
            else:
                # Already has another f= parameter, no change needed
                new_url = current_url
        else:  # Latest tab
            # Change f=top to f=live or add f=live if not present
            if "f=top" in current_url:
                new_url = current_url.replace("f=top", "f=live")
            elif "&f=" not in current_url:
                new_url = current_url + "&f=live"
            else:
                # Already has another f= parameter, no change needed
                new_url = current_url
        
        # Navigate to the new URL if it's different
        if new_url != current_url:
            print(f"Navigating to {tab_name} tab URL: {new_url}")
            driver.get(new_url)
            time.sleep(3)  # Wait for page to load
            return True
        else:
            print(f"Already on {tab_name} tab")
            return True
            
    except Exception as e:
        print(f"⚠️ Failed to switch to {tab_name} tab via URL: {e}")
        
        # Fallback to UI interaction if URL approach fails
        try:
            print(f"Trying UI-based tab switch as fallback...")
            
            # Look for exact tab name in a span element
            try:
                tab = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, f"//span[text()='{tab_name}']"))
                )
                driver.execute_script("arguments[0].scrollIntoView(true);", tab)
                driver.execute_script("arguments[0].click();", tab)
                print(f"🔖 Switched to {tab_name} tweets tab")
                time.sleep(2)
                return True
            except Exception as e:
                print(f"Method 1 failed for {tab_name} tab: {e}")
            
            # Try other methods as before...
            if tab_name == "Top":
                try:
                    tabs = driver.find_elements(By.XPATH, "//nav[@role='tablist']//div[@role='tab'] | //div[@role='tablist']//div[@role='tab']")
                    if tabs and len(tabs) >= 1:
                        driver.execute_script("arguments[0].scrollIntoView(true);", tabs[0])
                        driver.execute_script("arguments[0].click();", tabs[0])
                        print(f"🔖 Switched to first tab which should be {tab_name}")
                        time.sleep(2)
                        return True
                except Exception as e:
                    print(f"Method 2 failed for {tab_name} tab: {e}")
                
            try:
                elements = driver.find_elements(By.XPATH, f"//*[contains(text(), '{tab_name}')]")
                for element in elements:
                    try:
                        if element.is_displayed():
                            driver.execute_script("arguments[0].scrollIntoView(true);", element)
                            driver.execute_script("arguments[0].click();", element)
                            print(f"🔖 Found and clicked element containing '{tab_name}'")
                            time.sleep(2)
                            return True
                    except:
                        continue
            except Exception as e:
                print(f"Method 3 failed for {tab_name} tab: {e}")
                
            print(f"⚠️ Could not find or click {tab_name} tab")
            return False
        except Exception as e:
            print(f"⚠️ Failed to switch to {tab_name} tab: {e}")
            return False


def human_like_scroll(driver):
    """Perform a more variable human-like scroll down the page with occasional longer pauses"""
    # Get viewport height
    viewport_height = driver.execute_script("return window.innerHeight")
    
    # More variable scroll depths - sometimes shorter, sometimes longer
    # Between 40% and 95% of viewport height
    scroll_percentage = random.uniform(0.4, 0.95)
    scroll_amount = int(viewport_height * scroll_percentage)
    
    # More variable scroll speed
    scroll_duration = random.uniform(200, 800)  # milliseconds
    
    # Occasional longer pauses (10% chance)
    should_pause_longer = random.random() < 0.1
    
    # Execute a smooth scroll with custom duration
    driver.execute_script(f"""
        var start = window.pageYOffset;
        var distance = {scroll_amount};
        var duration = {scroll_duration};
        var startTime = null;
        
        function scrollAnimation(currentTime) {{
            if (startTime === null) startTime = currentTime;
            var timeElapsed = currentTime - startTime;
            var progress = Math.min(timeElapsed / duration, 1);
            window.scrollTo(0, start + distance * progress);
            if (timeElapsed < duration) {{
                window.requestAnimationFrame(scrollAnimation);
            }}
        }}
        
        window.requestAnimationFrame(scrollAnimation);
    """)
    
    # Wait a variable amount of time after scrolling
    # Occasionally wait much longer (5-10 seconds) to allow content to fully load
    if should_pause_longer:
        pause_time = random.uniform(5, 10)
        print(f"📊 Taking a longer pause ({pause_time:.1f}s) to allow content to load...")
        time.sleep(pause_time)
    else:
        time.sleep(random.uniform(1, 3))


def jiggle_page(driver):
    """Perform small up and down scrolls to trigger content loading"""
    print("🔄 Jiggling the page to trigger content load...")
    # Scroll up slightly
    driver.execute_script("window.scrollBy(0, -200);")
    time.sleep(1)
    # Scroll down a bit more
    driver.execute_script("window.scrollBy(0, 300);")
    time.sleep(2)


def check_tweet_dates(driver, target_date):
    """Check if tweets on the page match the target date"""
    target_date_str = target_date.strftime("%Y-%m-%d")
    time_elements = driver.find_elements(By.XPATH, './/time')
    
    if not time_elements:
        print("⚠️ No time elements found on page")
        return False
    
    dates_found = []
    for time_elem in time_elements[:5]:  # Check first 5 timestamps
        try:
            dt = time_elem.get_attribute('datetime')
            if dt:
                date_part = dt.split('T')[0]  # Extract date part
                dates_found.append(date_part)
        except:
            continue
    
    print(f"Tweet dates found on page: {dates_found}")
    
    # Check if any tweet has the target date
    target_date_found = any(target_date_str in date for date in dates_found)
    return target_date_found


def extract_tweet_content(article):
    """Extract tweet information from article element with improved error handling"""
    try:
        # Extract author
        try:
            author_elem = article.find_element(By.XPATH, './/div[@data-testid="User-Name"]//span')
            author = author_elem.text
        except:
            author = "Unknown Author"
        
        # Extract handle
        try:
            handle_elem = article.find_element(By.XPATH, './/div[@dir="ltr"]//span[contains(text(), "@")]')
            handle = handle_elem.text
        except:
            handle = "@unknown"
        
        # Extract content
        try:
            content_elem = article.find_element(By.XPATH, './/div[@data-testid="tweetText"]')
            content = content_elem.text.replace("\n", " ").strip()
        except:
            content = "No content available"
        
        # Extract timestamp
        try:
            time_elem = article.find_element(By.XPATH, './/time')
            timestamp = time_elem.get_attribute('datetime')
        except:
            timestamp = datetime.now().isoformat()
        
        # Extract likes count - UPDATED METHOD
        try:
            # First try to find button with data-testid="like"
            try:
                like_button = article.find_element(By.XPATH, './/div[@data-testid="like"]')
                
                # Try to find the aria-label which contains the likes count
                aria_label = like_button.get_attribute('aria-label')
                
                # If aria-label is found, extract the number
                if aria_label and ('like' in aria_label.lower() or 'Like' in aria_label):
                    # Parse the number from the aria-label text
                    import re
                    likes_match = re.search(r'(\d+,?\d*)\s+[Ll]ikes?', aria_label)
                    if likes_match:
                        # Remove commas from numbers like "1,234"
                        likes = int(likes_match.group(1).replace(',', ''))
                    else:
                        # If specific pattern doesn't match, try to find any number
                        likes_match = re.search(r'(\d+)', aria_label)
                        if likes_match:
                            likes = int(likes_match.group(1))
                        else:
                            likes = 0
                else:
                    # Fallback: try to get it directly from the button text
                    likes_text = like_button.text.strip()
                    if likes_text and likes_text.isdigit():
                        likes = int(likes_text)
                    else:
                        likes = 0
            except:
                # Second method: try to find by searching for button with "Likes" in aria-label
                like_buttons = article.find_elements(By.XPATH, './/button[contains(@aria-label, "Like")]')
                
                if like_buttons:
                    for btn in like_buttons:
                        aria_label = btn.get_attribute('aria-label')
                        if aria_label and ('like' in aria_label.lower() or 'Like' in aria_label):
                            import re
                            likes_match = re.search(r'(\d+,?\d*)\s+[Ll]ikes?', aria_label)
                            if likes_match:
                                # Remove commas from numbers like "1,234"
                                likes = int(likes_match.group(1).replace(',', ''))
                                break
                    else:  # No match found in the loop
                        likes = 0
                else:
                    # Third method: try to find a span with a like count
                    try:
                        # Look for spans inside containers that might have like counts
                        metric_spans = article.find_elements(By.XPATH, './/div[contains(@role, "group")]//span')
                        for span in metric_spans:
                            if span.text and span.text.strip().isdigit():
                                likes = int(span.text.strip())
                                break
                        else:
                            likes = 0
                    except:
                        likes = 0
        except Exception as e:
            print(f"Error extracting likes count: {e}")
            likes = 0
            
        return {
            "author": author,
            "handle": handle,
            "date": timestamp,
            "text": content,
            "likes": likes
        }
    except Exception as e:
        print(f"Error extracting tweet content: {e}")
        return None


def scroll_and_extract_tweets(driver, max_scrolls=25, max_tweets=100):
    """Scroll through search results and extract tweets with improved error handling"""
    tweets_data = []
    seen_texts = set()
    same_count = 0
    last_seen = 0

    for i in range(max_scrolls):
        print(f"🔁 Scrolling... ({i+1}/{max_scrolls})")
        move_mouse_randomly(driver)
        
        # Use human-like scrolling for more natural behavior
        human_like_scroll(driver)
        
        # Find articles - first try data-testid, then fallback to role
        articles = driver.find_elements(By.XPATH, '//article[@data-testid="tweet"]')
        if not articles:
            articles = driver.find_elements(By.XPATH, '//article[@role="article"]')
        
        for article in articles:
            try:
                tweet_info = extract_tweet_content(article)
                
                if tweet_info and tweet_info["text"] != "No content available" and tweet_info["text"] not in seen_texts:
                    seen_texts.add(tweet_info["text"])
                    tweets_data.append(tweet_info)
                    content_preview = tweet_info["text"][:80] + "..." if len(tweet_info["text"]) > 80 else tweet_info["text"]
                    # Include likes count in the log
                    print(f"📄 {tweet_info['author']} – {content_preview} [❤️ {tweet_info['likes']}]")
                    
                    if len(tweets_data) >= max_tweets:
                        print(f"✅ Reached max_tweets limit ({max_tweets})")
                        return tweets_data
            except Exception as e:
                print(f"Error processing an article: {e}")
                continue

        if len(tweets_data) == last_seen:
            same_count += 1
            print("⚠️ No new tweets loaded.")
        else:
            same_count = 0
            last_seen = len(tweets_data)

        if same_count >= 3:
            print("🔄 Jiggling page due to tweet load stagnation...")
            jiggle_page(driver)
            same_count = 0

    return tweets_data


def scrape_day(driver, current_date, next_date, max_tweets_per_tab=50, max_scrolls_per_tab=25):
    """Scrape tweets for a specific day using both Top and Latest tabs"""
    current_date_str = current_date.strftime("%Y-%m-%d")
    next_date_str = next_date.strftime("%Y-%m-%d")
    
    # Format dates in the style Twitter/X expects (YYYY-MM-DD → YY-MM-DD)
    formatted_current = current_date.strftime("%y-%m-%d")
    formatted_next = next_date.strftime("%y-%m-%d")
    
    # Create search query with properly formatted dates
    today = datetime.now().date()
    
    # Properly encode hashtag symbol # as %23 for URL
    query = "Solana OR $SOL OR #SOL"
    encoded_query = query.replace("#", "%23").replace(" ", "%20")
    
    # For today, don't specify the date range (to get most recent tweets)
    if current_date == today:
        search_url = f"https://x.com/search?q={encoded_query}&src=typed_query"
    else:
        # For past days, explicitly specify date range with properly formatted dates
        since_date = current_date.strftime('%y-%m-%d')
        until_date = next_date.strftime('%y-%m-%d')
        search_url = f"https://x.com/search?q={encoded_query}%20until%3A{until_date}%20since%3A{since_date}&src=typed_query"
    
    print(f"\n📅 Searching for day: {current_date_str}")
    print(f"🔍 Query: {query}")
    print(f"🔗 Encoded URL: {search_url}")
    
    all_day_tweets = []
    
    try:
        # Navigate to search URL and ensure it loads properly
        print(f"Navigating to: {search_url}")
        driver.get(search_url)
        
        # Wait longer for the page to fully load
        time.sleep(8)
        
        # Print the current URL to verify we're on the right page
        current_url = driver.current_url
        print(f"Current browser URL: {current_url}")
        
        # Wait for page to load
        try:
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.TAG_NAME, "article"))
            )
            print("🔍 Search results loaded")
            
            # Check if we need to verify we're getting results from the right day
            if current_date != today:
                # Try to find any date indicators on the page
                time.sleep(2)  # Give a moment for content to render
                
                # Print the first few tweet timestamps we find to debug
                try:
                    time_elements = driver.find_elements(By.XPATH, './/time')[:3]
                    if time_elements:
                        print("Sample tweet timestamps:")
                        for i, time_elem in enumerate(time_elements):
                            dt = time_elem.get_attribute('datetime')
                            print(f"  Tweet {i+1}: {dt}")
                    
                    # If we don't see results from the correct day, try an alternative URL format
                    correct_date_found = False
                    
                    if time_elements and len(time_elements) > 0:
                        for time_elem in time_elements:
                            dt = time_elem.get_attribute('datetime')
                            if dt and current_date_str in dt:
                                correct_date_found = True
                                break
                    
                    if not correct_date_found:
                        print("⚠️ Did not find tweets from the correct date. Trying alternative URL format...")
                        
                        # Try alternative URL format with explicit encoding for the hashtag
                        query = "Solana OR $SOL OR #SOL"
                        encoded_query = query.replace("#", "%23").replace(" ", "%20")
                        since_date = current_date.strftime('%Y-%m-%d')
                        until_date = next_date.strftime('%Y-%m-%d')
                        
                        alternative_url = (
                            f"https://x.com/search?q={encoded_query}"
                            f"%20since%3A{since_date}"
                            f"%20until%3A{until_date}&src=typed_query"
                        )
                        print(f"Trying alternative URL: {alternative_url}")
                        driver.get(alternative_url)
                        time.sleep(8)
                except Exception as e:
                    print(f"Error during date verification: {e}")
                    
        except Exception as e:
            print(f"⚠️ Timeout waiting for search results: {e}")
            print("Continuing anyway - will check if content appears later")
        
        # First try Top tab
        print("Checking Top tab for tweets...")
        if switch_to_tab(driver, "Top"):
            print("Starting extraction from Top tab...")
            top_tweets = scroll_and_extract_tweets(
                driver, 
                max_scrolls=max_scrolls_per_tab, 
                max_tweets=max_tweets_per_tab
            )
            all_day_tweets.extend(top_tweets)
            print(f"✅ Collected {len(top_tweets)} tweets from Top tab")
        
        # Then switch to Latest tab
        print("Checking Latest tab for tweets...")
        if switch_to_tab(driver, "Latest"):
            print("Starting extraction from Latest tab...")
            latest_tweets = scroll_and_extract_tweets(
                driver, 
                max_scrolls=max_scrolls_per_tab, 
                max_tweets=max_tweets_per_tab
            )
            
            # Filter out duplicates when adding latest tweets
            seen_texts = {tweet["text"] for tweet in all_day_tweets}
            unique_latest_tweets = [
                tweet for tweet in latest_tweets 
                if tweet["text"] not in seen_texts
            ]
            
            all_day_tweets.extend(unique_latest_tweets)
            print(f"✅ Collected {len(unique_latest_tweets)} unique tweets from Latest tab")
        
    except Exception as e:
        print(f"❌ Error during search and extraction for {current_date_str}: {e}")
    
    # Add day identifier to each tweet
    for tweet in all_day_tweets:
        tweet["search_date"] = current_date_str
    
    print(f"📊 Final collection: {len(all_day_tweets)} tweets for {current_date_str}")
    return all_day_tweets


def remove_duplicates(all_tweets):
    """Remove duplicate tweets from the collection based on tweet text"""
    texts_seen = set()
    unique_tweets = []
    
    for tweet in all_tweets:
        if tweet["text"] not in texts_seen and tweet["text"] != "No content available":
            texts_seen.add(tweet["text"])
            unique_tweets.append(tweet)
    
    print(f"Removed {len(all_tweets) - len(unique_tweets)} duplicates")
    return unique_tweets


def save_to_csv(data, filename):
    """Save tweets data to CSV file"""
    # Updated keys to include likes
    keys = ["author", "handle", "date", "text", "likes"]
    
    # Filter out any additional fields we added
    filtered_data = []
    for tweet in data:
        filtered_tweet = {k: tweet.get(k, "") for k in keys}
        filtered_data.append(filtered_tweet)
    
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(filtered_data)
    print(f"✅ Saved {len(filtered_data)} tweets to {filename}")


def create_output_directory():
    """Create an output directory for daily CSV files"""
    output_dir = "solana_tweets_daily"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def main():
    # Maximum total tweets to collect
    MAX_TOTAL_TWEETS = 700
    
    # Create output directory for daily files
    output_dir = create_output_directory()
    
    # Calculate the date range (last 7 days, starting from today)
    today = datetime.now().date()
    
    # Create a list of days to scrape (starting from today, going backwards)
    days_to_scrape = []
    for day_offset in range(0, 7):
        current_date = today - timedelta(days=day_offset)
        next_date = current_date + timedelta(days=1)
        days_to_scrape.append((current_date, next_date))
        
    print(f"Today is: {today}")
    print("Days to scrape:")
    for i, (start, end) in enumerate(days_to_scrape):
        print(f"  Day {i+1}: {start} to {end}")
    
    # Initialize collection for all tweets
    all_tweets = []
    
    # Initialize driver
    print("🌐 Initializing browser...")
    try:
        driver = setup_driver(headless=False)
        driver.get("https://x.com/")
        time.sleep(3)
        
        # Log in using cookies
        load_cookies(driver, "x_cookies.json")
        driver.refresh()
        time.sleep(5)
        print("🔑 Logged in using cookies")
        
        # Process each day
        for current_date, next_date in days_to_scrape:
            current_date_str = current_date.strftime("%Y-%m-%d")
            
            print(f"\n{'='*80}")
            print(f"📅 Processing day: {current_date_str}")
            print(f"{'='*80}")
            
            try:
                max_per_tab = 100  # Set a high enough max_tweets value to ensure we're limited by scrolls, not tweet count
                
                print(f"🎯 Using fixed scroll limit: 25 scrolls per tab (Top and Latest)")
                
                # Scrape this day using both Top and Latest tabs
                day_tweets = scrape_day(
                    driver, 
                    current_date, 
                    next_date, 
                    max_tweets_per_tab=max_per_tab,
                    max_scrolls_per_tab=25
                )
                
                # Save daily tweets to individual file
                if day_tweets:
                    daily_filename = os.path.join(output_dir, f"solana_tweets_{current_date_str}.csv")
                    save_to_csv(day_tweets, daily_filename)
                
                # Add to collection of all tweets
                all_tweets.extend(day_tweets)
                
                # Check if we've reached our total limit
                if len(all_tweets) >= MAX_TOTAL_TWEETS:
                    print(f"✅ Reached overall maximum of {MAX_TOTAL_TWEETS} tweets")
                    break
                    
                print(f"Progress: {len(all_tweets)}/{MAX_TOTAL_TWEETS} total tweets")
                
            except Exception as e:
                print(f"❌ Error processing day {current_date_str}: {e}")
                print("Continuing to next day...")
            
            # Brief pause between days
            time.sleep(random.uniform(2, 4))
    
    except Exception as e:
        print(f"❌ Error in main scraping loop: {e}")
    
    finally:
        # Always ensure we close the driver
        try:
            if 'driver' in locals():
                driver.quit()
                print("✅ Browser closed successfully")
        except Exception as e:
            print(f"Error closing browser: {e}")
    
    # Ensure no duplicates in the final collection
    all_tweets = remove_duplicates(all_tweets)
    
    # Save all tweets to the combined CSV file
    save_to_csv(all_tweets, "solana_search_tweets.csv")
    
    tweet_count = len(all_tweets)
    print(f"\n🎉 Completed! Scraped {tweet_count} unique tweets across 7 days")
    
    # Add warning if we didn't meet the minimum goal
    if tweet_count < 100:
        print(f"⚠️ Warning: Only collected {tweet_count} tweets, which is below the target minimum of 100")
    # Add success message if we got a good number
    elif tweet_count <= 700:
        print(f"✅ Successfully collected {tweet_count} tweets, which is within the target range (100-700)")
    else:
        print(f"⚠️ Collected {tweet_count} tweets, which exceeds the maximum of 700")
    
    print(f"📊 Output saved to solana_search_tweets.csv")


if __name__ == "__main__":
    main()