from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import time
import re


def webdriver_config():
    options = Options()
    # tanpa GUI, bisa dihapus kalau mau lihat browser
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    return options


def sinta_login(username, password):
    driver = webdriver.Chrome(options=webdriver_config())
    driver.get("https://sinta.kemdikbud.go.id/logins")

    try:
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.NAME, "username"))
        )
        driver.find_element(By.NAME, "username").send_keys(username)
        driver.find_element(By.NAME, "password").send_keys(password)
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[@type='submit']"))
        ).click()

        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located(
                (By.XPATH, "//a[contains(text(), 'Logout')]"))
        )
        print("✅ Login berhasil!")
        return driver

    except Exception as e:
        print(f"❌ Login gagal: {e}")
        driver.quit()
        return None


def scrape_articles(driver, start_page, end_page):
    data = []
    for page in range(start_page, end_page + 1):
        url = f"https://sinta.kemdikbud.go.id/affiliations/profile/398/?page={page}&view=googlescholar"
        driver.get(url)
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "ar-list-item"))
            )
            soup = BeautifulSoup(driver.page_source, "html.parser")
            articles = soup.find_all("div", class_="ar-list-item mb-5")

            if not articles:
                print(f"[{page}] Tidak ada artikel ditemukan.")
            else:
                for item in articles:
                    title_tag = item.find("div", class_="ar-title").find("a")
                    title = title_tag.text.strip() if title_tag else "Tidak ditemukan"
                    link = title_tag["href"] if title_tag else "Tidak ditemukan"

                    creator_tag = item.find(
                        "a", string=re.compile(r"Creator\s*:", re.I))
                    creator = "Tidak ditemukan"
                    if creator_tag:
                        creator = creator_tag.text.replace(
                            "Creator :", "").strip()

                    journal_tag = item.find("a", class_="ar-pub")
                    journal_name = journal_tag.text.strip() if journal_tag else "Tidak ditemukan"

                    year_tag = item.find("a", class_="ar-year")
                    year = year_tag.text.strip() if year_tag else "Tidak ditemukan"

                    cited_tag = item.find("a", class_="ar-cited")
                    citations = "0"
                    if cited_tag:
                        cited_text = cited_tag.text.strip()
                        match = re.search(r"(\d+)", cited_text)
                        if match:
                            citations = match.group(1)

                    data.append(
                        [title, link, creator, journal_name, year, citations])

                print(f"[{page}] {len(articles)} artikel diproses.")
        except Exception as e:
            print(f"[{page}] ERROR: {e}")

        time.sleep(5)  # istirahat biar gak kebanyakan request

    return data


if __name__ == "__main__":
    USERNAME = "igit.sabda@fmipa.unila.ac.id"
    PASSWORD = "1G1t01011996"
    START_PAGE = 1669
    END_PAGE = 2502  # contoh dulu 3 halaman

    driver = sinta_login(USERNAME, PASSWORD)
    if driver:
        all_data = scrape_articles(driver, START_PAGE, END_PAGE)
        driver.quit()

        df = pd.DataFrame(all_data, columns=[
                          "Title", "URL", "Creator", "Journal", "Year", "Citations"])
        df.to_csv("sinta_scraped_data.csv", index=False)
        print("✅ Scraping selesai, data disimpan di sinta_scraped_data.csv")
    else:
        print("Gagal login, scraping dibatalkan.")
