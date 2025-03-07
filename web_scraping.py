import os
import requests
from bs4 import BeautifulSoup

# List of websites to scrape
websites = [
    "https://news.ycombinator.com",
    "https://reddit.com",
    "https://nytimes.com",
    "https://github.com",
    "https://stackoverflow.com"
]

# Directory to save scraped HTML files
output_dir = "scraped_html"
os.makedirs(output_dir, exist_ok=True)

# Function to scrape and save HTML content
def scrape_website(url, output_dir):
    try:
        # Send a GET request to the website
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an error for bad status codes (e.g., 404)

        # Parse the HTML content
        soup = BeautifulSoup(response.text, "html.parser")

        # Clean up the URL to create a valid filename
        filename = url.replace("https://", "").replace("http://", "").replace("/", "_") + ".html"
        filepath = os.path.join(output_dir, filename)

        # Save the HTML content to a file
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(soup.prettify())

        print(f"Successfully scraped and saved: {url} -> {filepath}")

    except Exception as e:
        print(f"Failed to scrape {url}: {e}")

# Scrape each website in the list
for website in websites:
    scrape_website(website, output_dir)
