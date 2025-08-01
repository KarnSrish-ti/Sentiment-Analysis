#import necessary libraries
#for browser automation
from selenium import webdriver
#chrome browser options
from selenium.webdriver.chrome.options import Options
#Chrome driver service
from selenium.webdriver.chrome.service import Service
#automatic chrome driver management
from webdriver_manager.chrome import ChromeDriverManager
#HTML parsing
from bs4 import BeautifulSoup
#for adding delays
import time
#for CSV file operations
import csv
#for random delays
import random
#for url joining operations
from urllib.parse import urljoin

# Configure browser options for headless chro,e
options = Options()
options.add_argument('--headless') #run browser in headless mode no GUI
options.add_argument('--disable-gpu') # disable gpu hardware acceleration
options.add_argument('--no-sandbox') #disable dandboxing for linus systems
#innitializing chrome webdriver with automatic driver management
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# dictionaly containing configuration for multiple nepali news sources
#each sourse has:
# - base_url: the main website url
# - pages: list of regular news pages 
# - archived: list of archived urls for historical news
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
    """scrape headlines from multiple nepali news sources until reaching target 
    cout
    returns a list of unique headlines
    """
    headlines = set() #using set to avoid duplicates
    target_count = 1000 #target number of headlines to collect
    
    #iterate through each news source
    for source, config in news_sources.items():
        if len(headlines) >= target_count:
            break #stop if we've reached our target
            
        # Scrape regular pages
        for page in config['pages']:
            if len(headlines) >= target_count:
                break #stop if we've reached our target
            
            #construct full url
            url = urljoin(config['base_url'], page)
            try:
                print(f"Scraping {source}: {url}")
                driver.get(url) #load the page
                time.sleep(random.uniform(1, 3)) #random delay to avoid detection
                soup = BeautifulSoup(driver.page_source, 'html.parser') #parse HTML
                
                # Site-specific selectors for finding headlines
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
                
                #extract text from each matched element
                for el in elements:
                    text = el.get_text().strip() #clean and normaplized text
                    if text and len(text.split()) > 3: #only keep meaningful headlines
                        headlines.add(text)
                        if len(headlines) % 100 == 0: #progress reporting
                            print(f"Collected {len(headlines)} headlines so far...")
                        
            except Exception as e:
                print(f"Error scraping {url}: {str(e)}")
                continue
    
    return sorted(headlines)[:target_count]  # Return sorted lidt up to target count

def save_to_csv(headlines, filename="nepali_headlines1.csv"):
    """
    save collected headlines to a CSV file. 
    args: 
        headlines: list of headlines to save
        filename: output CSV filename
    """
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Headline"]) #write header row
        for hl in headlines:
            writer.writerow([hl]) #write each headline as a roe

if __name__ == "__main__":
    #main execution block
    print("Starting large-scale Nepali headline scraping...")
    all_headlines = scrape_headlines() #scrape headlines
    save_to_csv(all_headlines) #save to csv
    print(f"Successfully saved {len(all_headlines)} headlines to CSV")
    driver.quit() #close the browser
#pushh