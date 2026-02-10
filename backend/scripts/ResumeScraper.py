import requests
from bs4 import BeautifulSoup
import csv
import time

# Limit for testing - set to None for no limit
SCRAPE_LIMIT = 10

url = "https://www.postjobfree.com/resumes?q=&n=&t=python+developer&d=&l=United+States&radius=25&r=100"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')
title_tags = soup.find_all('h3', attrs={'class': 'itemTitle'})
links = ["https://www.postjobfree.com" + title_tag.a['href'] for title_tag in title_tags]

# Limit the number of links if SCRAPE_LIMIT is set
if SCRAPE_LIMIT:
    links = links[:SCRAPE_LIMIT]
    print(f"Limiting scrape to {len(links)} resumes")

results = []
for link in links:
    res = requests.get(link)
    print(res.status_code, link)
    content = BeautifulSoup(res.content, 'html.parser')
    results.append({
        'job_title': content.find('div', attrs={'class': 'innercontent'}).find('h1').get_text(),
        'resume': content.find('div', attrs={'class': 'normalText'}).get_text()[:-23]
    })
    time.sleep(3)
with open('resumes.csv', 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=results[0].keys())
    writer.writeheader()
    for row in results:
        writer.writerow(row)
