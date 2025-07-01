import asyncio
import csv
import time
import re
import datetime
import pandas as pd
import os
from playwright.async_api import async_playwright

CHROME_PATH = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
INPUT_CSV = r"D:\UIT\DS200\DS2000 Project\Raw Data\User name\recrawl.csv"
OUTPUT_CSV = r"D:\UIT\DS200\DS2000 Project\Raw Data\User Videos\user_video_4_add.csv"

def safe_strip(value):
    return value.strip() if value else ""

def get_post_time(video_id: str) -> str:
    try:
        video_id_int = int(video_id)
        binary = format(video_id_int, '064b')
        timestamp_bin = binary[:32]
        timestamp_int = int(timestamp_bin, 2)
        post_time = datetime.datetime.utcfromtimestamp(timestamp_int)
        return post_time.strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        return f"Error: {e}"

async def get_user_followers_totalLike(context, username):
    page = await context.new_page()
    try:
        await page.goto(f"https://www.tiktok.com/@{username}", timeout=30000)
        await page.wait_for_selector('strong[data-e2e="followers-count"]', timeout=20000)

        follower_elem = await page.query_selector('strong[data-e2e="followers-count"]')
        like_elem = await page.query_selector('strong[data-e2e="likes-count"]')

        followers = safe_strip(await follower_elem.text_content()) if follower_elem else ""
        total_likes = safe_strip(await like_elem.text_content()) if like_elem else ""

        return followers, total_likes
    except Exception as e:
        print(f"Error getting followers/likes for @{username}: {e}")
        return "", ""
    finally:
        await page.close()

async def get_video_views_from_profile(context, username, target_video_id):
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
        print(f"Could not get views from profile for @{username}: {e}")
    finally:
        await page.close()
    return ""

async def extract_video_details_enhanced(context, video_url, username, followers="", total_likes=""):
    page = await context.new_page()
    try:
        await page.goto(video_url, timeout=30000)
        await page.wait_for_selector('strong[data-e2e="like-count"]', timeout=15000)

        video_id = video_url.split("/")[-1]
        post_time = get_post_time(video_id)
        scrape_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        caption_elem = await page.query_selector('span[data-e2e="new-desc-span"]')
        likes_elem = await page.query_selector('strong[data-e2e="like-count"]')
        comments_elem = await page.query_selector('strong[data-e2e="comment-count"]')
        shares_elem = await page.query_selector('strong[data-e2e="share-count"]')
        saves_elem = await page.query_selector('strong[data-e2e="undefined-count"]')
        views_elem = await page.query_selector('strong[data-e2e="video-views"]')

        views = safe_strip(await views_elem.text_content()) if views_elem else ""
        if not views:
            views = await get_video_views_from_profile(context, username, video_id)

        hashtags = []
        hashtag_links = await page.query_selector_all('a[data-e2e="search-common-link"]')
        for tag in hashtag_links:
            href = await tag.get_attribute("href")
            if href and "/tag/" in href:
                hashtags.append(href.split("/tag/")[-1])

        duration = ""
        duration_elem = await page.query_selector('div[class*="DivSeekBarTimeContainer"]') or \
                        await page.query_selector('p[class*="StyledTimeDisplayText"]')
        if duration_elem:
            duration_text = await duration_elem.text_content()
            if "/" in duration_text:
                duration = duration_text.split('/')[-1].strip()
            else:
                duration = duration_text.strip()

        sound_id = sound_title = uses_sound_count = music_author = music_originality = ""

        music_href = await page.query_selector('a[data-e2e="video-music"]') or \
                     await page.query_selector('h4[data-e2e="video-music"] a') or \
                     await page.query_selector('h4[data-e2e="browse-music"] a')
        if music_href:
            try:
                href = await music_href.get_attribute("href")
                if href:
                    music_url = f"https://www.tiktok.com{href}"
                    sound_page = await context.new_page()
                    await sound_page.goto(music_url, timeout=20000)
                    await sound_page.wait_for_timeout(2000)

                    match = re.search(r'(\d{10,})/?$', href)
                    if match:
                        sound_id = match.group(1)

                    music_title_elem = await sound_page.query_selector('h1[data-e2e="music-title"]')
                    if music_title_elem:
                        sound_title = safe_strip(await music_title_elem.text_content())

                    uses_elem = await sound_page.query_selector('h2[data-e2e="music-video-count"] strong')
                    if uses_elem:
                        uses_sound_count = safe_strip(await uses_elem.text_content())

                    music_author_elems = await sound_page.query_selector_all('h2[data-e2e="music-creator"] a')
                    authors = []
                    usernames = []

                    for author_elem in music_author_elems:
                        display_name = safe_strip(await author_elem.text_content())
                        author_href = await author_elem.get_attribute("href")
                        if author_href and author_href.startswith("/@"):
                            username_only = author_href.split("/")[-1].lstrip("@").lower()
                            usernames.append(username_only)
                        authors.append(display_name)

                    music_author = "|".join(authors)
                    music_originality = "true" if username.lower() in usernames else "false"

                    await sound_page.close()
            except Exception as e:
                print(f"Error extracting music info: {e}")
        else:
            music_text_div = await page.query_selector('div[class*="DivMusicText"]')
            if music_text_div:
                sound_title = safe_strip(await music_text_div.text_content())

        return {
            "user_name": username,
            "user_nfollower": followers,
            "user_total_likes": total_likes,
            "vid_id": video_id,
            "vid_caption": safe_strip(await caption_elem.text_content() if caption_elem else ""),
            "vid_postTime": post_time,
            "vid_scrapeTime": scrape_time,
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
        print(f"Error extracting video details: {e}")
        return None
    finally:
        await page.close()

def save_video_data(video_data, writer, output_file):
    try:
        writer.writerow(video_data)
        output_file.flush()
        print(f"✓ Saved video {video_data['vid_id']} from @{video_data['user_name']}")
        return True
    except Exception as e:
        print(f"Error saving video: {e}")
        return False

async def scrape_user_videos(context, username, writer, output_file):
    page = await context.new_page()
    successful_saves = 0
    failed_saves = 0

    try:
        print(f"Starting to scrape videos from @{username}")
        followers, total_likes = await get_user_followers_totalLike(context, username)
        print(f"@{username} has {followers} followers and {total_likes} total likes")

        await page.goto(f"https://www.tiktok.com/@{username}", timeout=60000)
        await page.wait_for_selector('div[data-e2e="user-post-item-list"]', timeout=20000)

        video_info = []
        seen_urls = set()

        scroll_attempts = 0
        max_scroll_attempts = 70
        required_videos = 40

        while scroll_attempts < max_scroll_attempts:
            cards = await page.query_selector_all('div[data-e2e="user-post-item"]')
            new_videos_found = False

            for card in cards:
                link_tag = await card.query_selector("a")
                if link_tag:
                    video_url = await link_tag.get_attribute("href")
                    if video_url and video_url not in seen_urls:
                        seen_urls.add(video_url)
                        video_id = video_url.split("/")[-1]
                        post_time = get_post_time(video_id)

                        try:
                            timestamp = datetime.datetime.strptime(post_time.split()[0], "%Y-%m-%d") if "Error" not in post_time else datetime.datetime.now()
                        except:
                            timestamp = datetime.datetime.now()

                        video_info.append({
                            "url": video_url,
                            "id": video_id,
                            "post_time": post_time,
                            "timestamp": timestamp
                        })
                        new_videos_found = True

            if not new_videos_found:
                scroll_attempts += 1
                print(f"Scrolling to load more videos... (attempt {scroll_attempts})")
                await page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
                await page.wait_for_timeout(3000)
            else:
                scroll_attempts = 0

            if len(video_info) >= required_videos:
                break

        await page.close()

        video_info.sort(key=lambda x: x["timestamp"], reverse=True)  # Descending: latest first
        latest_videos = video_info[:required_videos]

        if latest_videos:
            print(f"Date range: {latest_videos[-1]['post_time'].split()[0]} to {latest_videos[0]['post_time'].split()[0]}")

        for i, video in enumerate(latest_videos, 1):
            try:
                print(f"\nProcessing video {i}/{len(latest_videos)}: {video['id']}")
                print(f"Posted: {video['post_time']}")
                video_data = await extract_video_details_enhanced(context, video['url'], username, followers, total_likes)

                if video_data:
                    if save_video_data(video_data, writer, output_file):
                        successful_saves += 1
                    else:
                        failed_saves += 1
                else:
                    print(f"Failed to extract data for video {video['id']}")
                    failed_saves += 1

                print(f"Progress for @{username}: {successful_saves} saved, {failed_saves} failed")
                await asyncio.sleep(2)

            except Exception as e:
                print(f"Error processing video {video['id']}: {e}")
                failed_saves += 1
                continue

        print(f"\nCompleted @{username}: {successful_saves} saved, {failed_saves} failed")

    except Exception as e:
        print(f"\nError scraping @{username}: {e}")

    return successful_saves, failed_saves

async def run_scraper(input_path, output_path):
    users = pd.read_csv(input_path)["username"].dropna().unique().tolist()
    usernames = [u.strip().lstrip("@") for u in users if str(u).strip()]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            slow_mo=50,
            executable_path=CHROME_PATH
        )
        context = await browser.new_context()

        with open(output_path, mode="w", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "user_name", "user_nfollower", "user_total_likes", "vid_id", "vid_caption", "vid_postTime", "vid_scrapeTime", 
                "vid_duration", "vid_nview", "vid_nlike", "vid_ncomment", "vid_nshare", "vid_nsave",
                "vid_hashtags", "vid_url", "music_id", "music_title", "music_nused", "music_authorName", "music_originality"
            ])
            writer.writeheader()
            f.flush()

            for i, username in enumerate(usernames, 1):
                print(f"\n--- [{i}/{len(usernames)}] Scraping: @{username}")
                await scrape_user_videos(context, username, writer, f)
                await asyncio.sleep(3)

        await browser.close()

if __name__ == "__main__":
    try:
        asyncio.run(run_scraper(INPUT_CSV, OUTPUT_CSV))
    except KeyboardInterrupt:
        print("Stopped by user.")
    except Exception as e:
        print(f"Fatal error: {e}")