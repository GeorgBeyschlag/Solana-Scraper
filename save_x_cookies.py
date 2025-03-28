from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import json
import os

# Test account credentials: 
# Username: Markus15660171
# Password: vitpaK-jevzod-4heghi

# Launch non-headless Chrome so you can log in
options = Options()
options.add_argument("--start-maximized")  # optional

driver = webdriver.Chrome(options=options)

# Step 1: Go to Twitter login page
driver.get("https://x.com/login")

print("🔐 Please log in to X manually in the opened Chrome window.")
print("⏳ Waiting 90 seconds for login...")

# Step 2: Wait for you to log in manually
time.sleep(90)

# Step 3: Save cookies
cookies = driver.get_cookies()

with open("x_cookies.json", "w") as f:
    json.dump(cookies, f)

print("✅ Cookies saved to x_cookies.json")

driver.quit()
