import asyncio
import pandas as pd
from playwright.async_api import async_playwright
from transformers import AutoTokenizer, AutoModel
import torch

CHROME_PATH = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"

BEAUTY_KEYWORDS = [
    "makeup", "skincare", "dưỡng da", "trang điểm", "làm đẹp", "chăm sóc da",
    "routine", "serum", "tẩy trang", "kem chống nắng", "foundation", "lipstick",
    "phấn", "má hồng", "mascara", "make up", "beauty tips", "skincare routine",
    "dưỡng ẩm", "tẩy tế bào chết", "face mask", "beauty", "chống lão hóa", "retinol",
    "son", "mặt nạ", "kem dưỡng", "collagen", "nước hoa hồng", "mụn"
]

EXCLUDE_KEYWORDS = [
    "barber", "tiệm tóc", "cắt tóc", "hair salon", "salon tóc", "mua mỹ phẩm",
    "shop mỹ phẩm", "nhập hàng", "bán sỉ", "đại lý", "chuyên sỉ", "salon", "bán hàng",
    "dịch vụ", "tuyển sỉ", "nhập khẩu", "order mỹ phẩm"
]

def contains_keywords(text, keywords):
    text = text.lower()
    return any(kw in text for kw in keywords)

def is_about_beauty(text):
    if contains_keywords(text, EXCLUDE_KEYWORDS):
        return False
    return contains_keywords(text, BEAUTY_KEYWORDS)

async def close_blocking_modals(page):
    try:
        agree_btn = await page.query_selector('[data-e2e="confirm-button"]')
        if agree_btn:
            await agree_btn.click()
            await page.wait_for_timeout(1000)
    except:
        pass
    try:
        close_interest = await page.query_selector('button[aria-label="Close"]')
        if close_interest:
            await close_interest.click()
            await page.wait_for_timeout(1000)
    except:
        pass
    try:
        overlay = await page.query_selector('.css-1o4zb36-DivModalMask')
        if overlay:
            await overlay.click()
            await page.wait_for_timeout(500)
    except:
        pass

async def get_user_video_captions(page, username, max_videos=10):
    try:
        await page.goto(f"https://www.tiktok.com/@{username}", timeout=60000)

        for attempt in range(2):
            try:
                await page.wait_for_selector('[data-e2e="user-post-item-list"]', timeout=10000)
                break
            except:
                if attempt == 1:
                    raise Exception("Không tìm thấy danh sách video.")
                await page.reload()
                await page.wait_for_timeout(2000)

        await page.mouse.wheel(0, 5000)
        await page.wait_for_timeout(3000)

        cards = await page.query_selector_all('[data-e2e="user-post-item"]')
        captions = []

        for i in range(min(max_videos, len(cards))):
            try:
                await close_blocking_modals(page)
                await cards[i].click()
                await page.wait_for_selector('[data-e2e="browse-video-desc"]', timeout=10000)

                desc = await page.query_selector('[data-e2e="browse-video-desc"] span')
                if desc:
                    text = await desc.inner_text()
                    if text.strip():
                        captions.append(text.strip())
                        print(f"📄 Caption {i+1}: {text.strip()}")

                await page.go_back()
                await page.wait_for_selector('[data-e2e="user-post-item-list"]', timeout=10000)
                await page.wait_for_timeout(1000)
                cards = await page.query_selector_all('[data-e2e="user-post-item"]')

            except Exception as e:
                print("⚠️ Lỗi lấy caption:", e)

        return captions

    except Exception as e:
        print(f"❌ Lỗi với @{username}: {e}")
        return []

async def verify_beauty_users(csv_path):
    df = pd.read_csv(csv_path)
    df['is_beauty'] = False
    df['reason'] = ""

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # headless=True nếu bạn muốn ẩn trình duyệt
        context = await browser.new_context()
        page = await context.new_page()

        for idx, row in df.iterrows():
            username = row['user_name']
            print(f"\n🔍 Đang kiểm tra @{username}...")

            captions = await get_user_video_captions(page, username)

            if any(contains_keywords(c, EXCLUDE_KEYWORDS) for c in captions):
                df.at[idx, 'is_beauty'] = False
                df.at[idx, 'reason'] = "Contains exclude keywords"
            else:
                match_count = sum(contains_keywords(c, BEAUTY_KEYWORDS) for c in captions)
                df.at[idx, 'is_beauty'] = match_count >= 2
                df.at[idx, 'reason'] = f"{match_count} beauty-related captions" if match_count >= 2 else "Too few beauty captions"

            print(f"✅ @{username}: {df.at[idx, 'is_beauty']} → {df.at[idx, 'reason']}")

        await browser.close()

    output_path = csv_path.replace(".csv", "_verified_beauty.csv")
    df.to_csv(output_path, index=False)
    print(f"\n✅ Đã lưu kết quả xác minh vào: {output_path}")

if __name__ == "__main__":
    asyncio.run(verify_beauty_users("D:/Downloads/tiktok_beauty_users_merged.csv"))