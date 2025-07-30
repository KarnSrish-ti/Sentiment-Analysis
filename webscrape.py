from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import csv
import random
from urllib.parse import urljoin

# Configure browser
options = Options()
options.add_argument('--headless')
options.add_argument('--disable-gpu')
options.add_argument('--no-sandbox')
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Expanded list of Nepali news sources with multiple pages/archives
news_sources = {
    "Online Khabar": {
        "base_url": "https://www.onlinekhabar.com",
        "pages": ["/content/news"] + [f"/content/news/page/{i}" for i in range(2, 21)],
        "archive": [f"/content/news/archive?year={year}" for year in range(2020, 2025)]
    },
    "Ekantipur": {
        "base_url": "https://ekantipur.com",
        "pages": ["/news"] + [f"/news?page={i}" for i in range(2, 21)],
        "archive": [f"/news/archive/{year}-{month:02d}" 
                   for year in range(2020, 2025) 
                   for month in range(1, 13)]
    },
    "Ratopati": {
        "base_url": "https://www.ratopati.com",
        "pages": ["/category/headline-news"] + [f"/category/headline-news?page={i}" for i in range(2, 21)],
        "archive": [f"/archive/{year}" for year in range(2020, 2025)]
    },
    "Setopati": {
        "base_url": "https://www.setopati.com",
        "pages": ["/news"] + [f"/news?page={i}" for i in range(2, 21)],
        "archive": [f"/archive/{year}" for year in range(2020, 2025)]
    },
    "Nagarik News": {
        "base_url": "https://nagariknews.nagariknetwork.com",
        "pages": ["/news"] + [f"/news?page={i}" for i in range(2, 21)],
        "archive": [f"/news/archive/{year}-{month:02d}" 
                   for year in range(2020, 2025) 
                   for month in range(1, 13)]
    },
    "Himalayan Times": {
        "base_url": "https://thehimalayantimes.com",
        "pages": ["/nepal"] + [f"/nepal?page={i}" for i in range(2, 11)],
        "archive": [f"/archive/{year}-{month:02d}" 
                   for year in range(2020, 2025) 
                   for month in range(1, 13)]
    }
}

def scrape_headlines():
    headlines = set()
    target_count = 1000
    
    for source, config in news_sources.items():
        if len(headlines) >= target_count:
            break
            
        # Scrape regular pages
        for page in config['pages']:
            if len(headlines) >= target_count:
                break
                
            url = urljoin(config['base_url'], page)
            try:
                print(f"Scraping {source}: {url}")
                driver.get(url)
                time.sleep(random.uniform(1, 3))
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                
                # Site-specific selectors
                if source == "Online Khabar":
                    elements = soup.select('h2 a, .ok-news-post h2')
                elif source == "Ekantipur":
                    elements = soup.select('article h1, article h2')
                elif source == "Ratopati":
                    elements = soup.select('.news-list h2, .news-box h2')
                elif source == "Setopati":
                    elements = soup.select('.article-title a, .news-title a')
                elif source == "Nagarik News":
                    elements = soup.select('.title a, h2 a')
                elif source == "Himalayan Times":
                    elements = soup.select('.post-title a, h2 a')
                
                for el in elements:
                    text = el.get_text().strip()
                    if text and len(text.split()) > 3:
                        headlines.add(text)
                        if len(headlines) % 100 == 0:
                            print(f"Collected {len(headlines)} headlines so far...")
                        
            except Exception as e:
                print(f"Error scraping {url}: {str(e)}")
                continue
    
    return sorted(headlines)[:target_count]  # Return up to target count

def save_to_csv(headlines, filename="nepali_headlines1.csv"):
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Headline"])
        for hl in headlines:
            writer.writerow([hl])

if __name__ == "__main__":
    print("Starting large-scale Nepali headline scraping...")
    all_headlines = scrape_headlines()
    save_to_csv(all_headlines)
    print(f"Successfully saved {len(all_headlines)} headlines to CSV")
    driver.quit()
