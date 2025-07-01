import asyncio
from playwright.async_api import async_playwright
import csv
import os

CHROME_PATH = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"

async def scrape_explore_beauty(num_scrolls=100):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, slow_mo=50, executable_path=CHROME_PATH)
        context = await browser.new_context()
        page = await context.new_page()

        await page.goto("https://www.tiktok.com/explore", timeout=60000)

        # ✅ Click vào mục "Chăm sóc sắc đẹp"
        try:
            await page.wait_for_selector('span:has-text("Beauty Care")', timeout=10000)
            button = await page.query_selector('span:has-text("Beauty Care")')
            await button.click()
            await page.wait_for_timeout(3000)
        except Exception as e:
            print("❌ Không tìm thấy nút chuyên mục 'Sắc đẹp':", e)
            await browser.close()
            return []

        results = []

        for i in range(num_scrolls):
            print(f"\n🔄 Scroll #{i+1}")
            await page.mouse.wheel(0, 1000)
            await page.wait_for_timeout(3000)

            cards = await page.query_selector_all('div[data-e2e="explore-card-desc"]')
            print(f"🔍 Found {len(cards)} cards.")

            for card in cards:
                try:
                    link_elem = await card.query_selector('a[href*="/@"]')
                    if not link_elem:
                        continue

                    href = await link_elem.get_attribute("href")
                    if href:
                        username = href.split("/@")[-1]
                        results.append({
                            "user_name": username,
                            "topic": "Beauty"
                        })
                except Exception as e:
                    print("⚠️ Lỗi khi xử lý 1 card:", e)

        await browser.close()
        return results

def save_to_csv(data, filename='D:/Downloads/beauty_users_crawl2.csv'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["user_name", "topic"])
        writer.writeheader()
        writer.writerows(data)
    print(f"✅ Đã lưu {len(data)} users vào {filename}")

if __name__ == "__main__":
    try:
        data = asyncio.run(scrape_explore_beauty(num_scrolls=150))
        if data:
            unique_users = {item['user_name']: item for item in data}.values()
            save_to_csv(list(unique_users))
            print(f"🧮 Tổng số user: {len(data)} | Unique: {len(unique_users)}")
        else:
            print("⚠️ Không có dữ liệu.")
    except Exception as e:
        print("❗Lỗi toàn cục:", e)

