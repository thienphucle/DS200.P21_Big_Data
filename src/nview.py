import asyncio
import pandas as pd
from playwright.async_api import async_playwright

CHROME_PATH = "C:\Program Files\Google\Chrome\Application\chrome.exe"
MAX_RETRIES = 3
MAX_SCROLLS = 100
SAVE_EVERY = 10  # save every N successful rows

def safe_strip(value):
    return value.strip() if value else ""

async def get_video_views(context, username, target_video_id, max_scroll=MAX_SCROLLS):
    page = await context.new_page()
    try:
        await page.goto(f"https://www.tiktok.com/@{username}", timeout=30000)
        await page.wait_for_selector('div[data-e2e="user-post-item"]', timeout=20000)

        for scroll_attempt in range(max_scroll):
            video_cards = await page.query_selector_all('div[data-e2e="user-post-item"]')
            for card in video_cards:
                link_tag = await card.query_selector("a")
                video_url = await link_tag.get_attribute("href") if link_tag else ""
                if target_video_id in video_url:
                    views_elem = await card.query_selector('strong[data-e2e="video-views"]')
                    views = safe_strip(await views_elem.text_content()) if views_elem else ""
                    await page.close()
                    return views

            # Scroll to load more videos
            await page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
            await page.wait_for_timeout(2000)

        print(f"  [Warning] Video ID {target_video_id} not found after {max_scroll} scrolls.")
    except Exception as e:
        print(f"[ERROR] Failed to get views for @{username}: {e}")
    finally:
        await page.close()
    return ""

async def main():
    df = pd.read_csv(r"D:\UIT\DS200\DS2000 Project\Preprocessed Data\streaming_data.csv")
    failed_rows = []
    updated_count = 0

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            slow_mo=50,
            executable_path=CHROME_PATH
        )
        context = await browser.new_context()

        for idx, row in df.iterrows():
            if pd.isna(row["vid_nview"]) or str(row["vid_nview"]).strip() == "":
                username = row["user_name"]
                video_id = str(row["vid_id"])
                print(f"\n[{idx + 1}] Refetching views for @{username} - {video_id}")

                views = ""
                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        print(f"  Attempt {attempt}...")
                        views = await get_video_views(context, username, video_id)
                        if views:
                            break
                    except Exception as e:
                        print(f"  Retry error: {e}")

                if views:
                    df.at[idx, "vid_nview"] = views
                    print(f"Views: {views}")
                    updated_count += 1

                    # Save periodically
                    if updated_count % SAVE_EVERY == 0:
                        df.to_csv("videos.csv", index=False, encoding="utf-8-sig")
                        print(f"Saved progress after {updated_count} updates.")
                else:
                    print("Failed to retrieve views after retries.")
                    failed_rows.append(idx)

        await browser.close()

    # Final save
    df.to_csv(r"D:\UIT\DS200\DS2000 Project\Preprocessed Data\streaming_data.csv", index=False, encoding="utf-8-sig")
    print(f"\nDone. Saved all data to videoss.csv")

    if failed_rows:
        print(f"\nFailed to fetch views for {len(failed_rows)} videos. Saving to 'failed_videos.csv'")
        df.iloc[failed_rows].to_csv("failed_videos.csv", index=False, encoding="utf-8-sig")
    else:
        print("All views fetched successfully!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Fatal error: {e}")
