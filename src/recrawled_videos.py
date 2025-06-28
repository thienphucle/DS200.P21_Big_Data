import asyncio
import pandas as pd
import datetime
import re
from playwright.async_api import async_playwright

CHROME_PATH = "C:\Program Files\Google\Chrome\Application\chrome.exe"

def safe_strip(value):
    return value.strip() if value else ""

async def get_video_views(context, username, target_video_id):
    page = await context.new_page()
    try:
        await page.goto(f"https://www.tiktok.com/@{username}", timeout=30000)
        await page.wait_for_selector('div[data-e2e="user-post-item"]', timeout=20000)
        
        video_cards = await page.query_selector_all('div[data-e2e="user-post-item"]')
        for card in video_cards:
            link_tag = await card.query_selector("a")
            video_url = await link_tag.get_attribute("href") if link_tag else ""
            if target_video_id in video_url:
                views_elem = await card.query_selector('strong[data-e2e="video-views"]')
                views = safe_strip(await views_elem.text_content()) if views_elem else ""
                return views
    except Exception as e:
        print(f"Could not get views for @{username}: {e}")
    finally:
        await page.close()
    return ""

async def get_video_metadata(context, video_url, username, post_time):
    page = await context.new_page()
    try:
        await page.goto(video_url, timeout=30000)
        await page.wait_for_selector('strong[data-e2e="like-count"]', timeout=20000)

        # Extract video elements
        likes_elem = await page.query_selector('strong[data-e2e="like-count"]')
        comments_elem = await page.query_selector('strong[data-e2e="comment-count"]')
        shares_elem = await page.query_selector('strong[data-e2e="share-count"]')
        saves_elem = await page.query_selector('strong[data-e2e="undefined-count"]')
        caption_elem = await page.query_selector('span[data-e2e="new-desc-span"]')

        # Hashtags
        hashtags = []
        hashtag_links = await page.query_selector_all('a[data-e2e="search-common-link"]')
        for tag in hashtag_links:
            href = await tag.get_attribute("href")
            if href and "/tag/" in href:
                hashtags.append(href.split("/tag/")[-1])

        # Music
        music_href = await page.query_selector('a[data-e2e="video-music"]') or \
                     await page.query_selector('h4[data-e2e="video-music"] a') or \
                     await page.query_selector('h4[data-e2e="browse-music"] a')

        sound_id = sound_title = uses_sound_count = music_author = music_originality = ""
        if music_href:
            href = await music_href.get_attribute("href")
            if href:
                music_url = f"https://www.tiktok.com{href}"
                sound_page = await context.new_page()
                try:
                    await sound_page.goto(music_url, timeout=20000)
                    await sound_page.wait_for_timeout(2000)

                    match = re.search(r'(\d{10,})/?$', href)
                    if match:
                        sound_id = match.group(1)

                    music_title_elem = await sound_page.query_selector('h1[data-e2e="music-title"]')
                    uses_elem = await sound_page.query_selector('h2[data-e2e="music-video-count"] strong')
                    music_author_elems = await sound_page.query_selector_all('h2[data-e2e="music-creator"] a')

                    if music_title_elem:
                        sound_title = safe_strip(await music_title_elem.text_content())
                    if uses_elem:
                        uses_sound_count = safe_strip(await uses_elem.text_content())

                    music_authors = []
                    music_usernames = []
                    for author_elem in music_author_elems:
                        display_name = safe_strip(await author_elem.text_content())
                        author_href = await author_elem.get_attribute("href")
                        if author_href and author_href.startswith("/@"):
                            username_only = author_href.split("/")[-1].lstrip("@").lower()
                            music_usernames.append(username_only)
                        music_authors.append(display_name)

                    music_author = "|".join(music_authors)
                    music_originality = "true" if username.lower() in music_usernames else "false"
                finally:
                    await sound_page.close()

        # Follower count + video views
        profile_page = await context.new_page()
        followers =""
        try:
            await profile_page.goto(f"https://www.tiktok.com/@{username}", timeout=30000)
            await profile_page.wait_for_selector('strong[data-e2e="followers-count"]', timeout=20000)
            follower_elem = await profile_page.query_selector('strong[data-e2e="followers-count"]')
            if follower_elem:
                followers = safe_strip(await follower_elem.text_content())
        finally:
            await profile_page.close()

        video_id = video_url.split("/")[-1]
        views = await get_video_views(context, username, video_id)

        duration = ""
        duration_elem = await page.query_selector('div[class*="DivSeekBarTimeContainer"]')
        if duration_elem:
            duration_text = await duration_elem.text_content()
            duration = duration_text.split('/')[-1].strip() if "/" in duration_text else duration_text.strip()


        # Return result dict with postTime next to scrapeTime
        return {
            "user_name": username,
            "user_nfollower": followers,
            "vid_id": video_url.split("/")[-1],
            "vid_caption": safe_strip(await caption_elem.text_content() if caption_elem else ""),
            "vid_scrapeTime": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "vid_postTime": post_time,
            "vid_duration": duration,
            "vid_nview": views,
            "vid_nlike": safe_strip(await likes_elem.text_content() if likes_elem else ""),
            "vid_ncomment": safe_strip(await comments_elem.text_content() if comments_elem else ""),
            "vid_nshare": safe_strip(await shares_elem.text_content() if shares_elem else ""),
            "vid_nsave": safe_strip(await saves_elem.text_content() if saves_elem else ""),
            "vid_hashtags": ", ".join(hashtags),
            "vid_url": video_url,
            "music_id": sound_id,
            "music_title": sound_title,
            "music_nused": uses_sound_count,
            "music_authorName": music_author,
            "music_originality": music_originality
        }

    except Exception as e:
        print(f"Error scraping video {video_url}: {e}")
        return None
    finally:
        await page.close()

async def main():
    df = pd.read_csv(r"IE403.P21_Social_Media_Data_Mining/Raw data/raw_1.csv")
    results = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            slow_mo=50,
            executable_path=CHROME_PATH
        )
        context = await browser.new_context()

        for idx, row in df.iterrows():
            print(f"\nProcessing video {idx + 1}/{len(df)}")
            print(f"URL: {row['vid_url']}")

            try:
                metadata = await get_video_metadata(
                    context,
                    row['vid_url'],
                    row['user_name'],
                    row['vid_postTime'], 
                )
                if metadata:
                    results.append(metadata)

                    # Save temp result
                    temp_df = pd.DataFrame(results)
                    temp_df.to_csv(r"IE403.P21_Social_Media_Data_Mining/Raw data/raw_1_recraled_1.csv", index=False, encoding='utf-8-sig')
            except Exception as e:
                print(f"Failed to process video: {e}")

        await browser.close()

    if results:
        df_final = pd.DataFrame(results)
        df_final.to_csv(r"IE403.P21_Social_Media_Data_Mining/Raw data/raw_1_recraled_1.csv", index=False, encoding='utf-8-sig')
        print("\nCrawling completed successfully!")
    else:
        print("\nNo data was collected!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Fatal error: {e}")
