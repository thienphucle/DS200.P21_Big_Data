import asyncio
import pandas as pd
from playwright.async_api import async_playwright
import datetime

CHROME_PATH = "C:\Program Files\Google\Chrome\Application\chrome.exe"

def safe_strip(value):
    return value.strip() if value else ""

async def get_duration_from_video(context, video_url):
    page = await context.new_page()
    duration = ""
    try:
        await page.goto(video_url, timeout=30000)

        # Nếu là photo post thì trả về "photo"
        if "/photo/" in video_url:
            print("Detected photo post.")
            return 0

        # Chờ thanh thời gian chỉ dành cho video
        await page.wait_for_selector('div[class*="DivSeekBarTimeContainer"]', timeout=10000)
        duration_elem = await page.query_selector('div[class*="DivSeekBarTimeContainer"]')
        if duration_elem:
            duration_text = await duration_elem.text_content()
            duration = duration_text.split('/')[-1].strip() if "/" in duration_text else duration_text.strip()

    except Exception as e:
        print(f"Failed to get duration from {video_url}: {e}")
    finally:
        await page.close()
    return duration

async def fill_missing_durations():
    df = pd.read_csv(r"D:\UIT\DS200\DS2000 Project\Raw Data\Merged Data\streaming_data.csv")
    missing_df = df[df["vid_duration"].isna() | (df["vid_duration"].astype(str).str.strip() == "")]

    if missing_df.empty:
        print("No missing durations found.")
        return

    print(f"Found {len(missing_df)} videos with missing duration.")

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            slow_mo=50,
            executable_path=CHROME_PATH
        )
        context = await browser.new_context()

        for idx in missing_df.index:
            row = df.loc[idx]
            print(f"\nRetrieving duration for video {idx + 1}/{len(df)}")
            print(f"URL: {row['vid_url']}")

            duration = await get_duration_from_video(context, row['vid_url'])
            df.at[idx, "vid_duration"] = duration
            print(f"Duration: {duration}")

        await browser.close()

    df.to_csv(r"D:\UIT\DS200\DS2000 Project\Preprocessed Data\streaming_data.csv", index=False, encoding="utf-8-sig")
    print("Missing durations filled and saved.csv'.")

if __name__ == "__main__":
    try:
        asyncio.run(fill_missing_durations())
    except Exception as e:
        print(f"Fatal error: {e}")
