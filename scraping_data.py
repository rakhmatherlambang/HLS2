import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

base_url = "https://sinta.kemdikbud.go.id/affiliations/profile/398/?page="
view_param = "&view=googlescholar"
start_page = 1669
end_page = 1670
data = []

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
}

for page in range(start_page, end_page + 1):
    url = base_url + str(page) + view_param
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")

        articles = soup.find_all("div", class_="ar-list-item mb-5")

        for item in articles:
            title_tag = item.find("div", class_="ar-title").find("a")
            title = title_tag.text.strip() if title_tag else "Tidak ditemukan"
            link = title_tag["href"] if title_tag else "Tidak ditemukan"

            author_tag = item.find("div", class_="ar-meta").find("a", href="#!")
            authors = author_tag.text.strip() if author_tag else "Tidak ditemukan"

            journal_tag = item.find("a", class_="ar-pub")
            journal_info = journal_tag.text.strip() if journal_tag else "Tidak ditemukan"

            year_tag = item.find("a", class_="ar-year")
            year = year_tag.text.strip() if year_tag else "Tidak ditemukan"

            cited_tag = item.find("a", class_="ar-cited")
            citations = cited_tag.text.strip() if cited_tag else "Tidak ditemukan"

            data.append([title, link, authors, journal_info, year, citations])
        
        print(f"Scraped page {page}")
    
    time.sleep(1) 

df = pd.DataFrame(data, columns=["Title", "URL", "Authors", "Journal", "Year", "Citations"])
df.to_csv("sinta_scraped_data.csv", index=False, encoding='utf-8')

print("Scraping selesai, data disimpan ke sinta_unila.csv")
