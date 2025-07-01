# ----------- tiki_crawler_producer.py -----------
from confluent_kafka import Producer
import json
import requests
import time
import random
import pandas as pd
from tqdm import tqdm
from config import KAFKA_CONFIG
from processor import VietnamesePreprocessor  # Import preprocessor

class TikiProducer:
    def __init__(self):
        self.producer = Producer({
            'bootstrap.servers': KAFKA_CONFIG['bootstrap.servers'],
            'client.id': 'tiki-crawler-producer',
            'message.max.bytes': 10485760  # Tăng kích thước message tối đa lên 10MB
        })
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'x-guest-token': self._refresh_guest_token()
        }
        
        # Khởi tạo preprocessor với cấu hình tùy chỉnh
        self.preprocessor = VietnamesePreprocessor(
            remove_stopwords=True,
            abbreviation_dict={
                'ko': 'không',
                'j': 'gì',
                'dc': 'được',
                'ntn': 'như thế nào'
            }
        )

    def _refresh_guest_token(self):
        """Làm mới guest token định kỳ"""
        try:
            response = requests.get('https://tiki.vn', timeout=5)
            return response.cookies.get('TIKI_GUEST_TOKEN', '')
        except Exception as e:
            print(f"Error refreshing token: {e}")
            return ''

    def _delivery_report(self, err, msg):
        """Callback xử lý kết quả gửi message"""
        if err is not None:
            print(f'❌ Delivery failed: {err}')
        else:
            print(f'✅ Delivered to {msg.topic()}[{msg.partition()}] @ offset {msg.offset()}')

    def _preprocess_comment(self, comment):
        """Pipeline tiền xử lý comment"""
        try:
            # Xử lý với preprocessor
            processed_text = self.preprocessor.transform([comment])[0]
            
            # Validate kết quả
            if len(processed_text) < 5:  # Bình luận quá ngắn
                return None
                
            if any(word in processed_text for word in ['spam', 'quảng cáo']):
                return None
                
            return processed_text
            
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None

    def produce_comment(self, comment_data):
        """Gửi comment đã xử lý đến Kafka"""
        processed_content = self._preprocess_comment(comment_data['content'])
        
        if processed_content:
            message = {
                'product_id': comment_data['product_id'],
                'original_content': comment_data['content'],
                'processed_content': processed_content,
                'rating': comment_data.get('rating', 0),
                'timestamp': comment_data.get('created_at', int(time.time()))
            }
            
            self.producer.produce(
                topic=KAFKA_CONFIG['input_topic'],
                value=json.dumps(message, ensure_ascii=False).encode('utf-8'),
                callback=self._delivery_report,
                headers={'source': 'crawler'}
            )
            self.producer.poll(0)

    def crawl_product_comments(self, product_id, max_retries=3):
        """Crawl comments cho một sản phẩm với retry mechanism"""
        comments = []
        for page in range(1, 4):
            params = {
                'product_id': product_id,
                'page': page,
                'limit': 10,
                'sort': 'score|desc'
            }
            
            for attempt in range(max_retries):
                try:
                    response = requests.get(
                        'https://tiki.vn/api/v2/reviews',
                        headers=self.headers,
                        params=params,
                        timeout=15
                    )
                    
                    if response.status_code == 200:
                        comments.extend(response.json().get('data', []))
                        break
                    elif response.status_code == 429:
                        print(f"Rate limited, retrying in {2**attempt} seconds...")
                        time.sleep(2**attempt)
                        self.headers['x-guest-token'] = self._refresh_guest_token()
                    else:
                        print(f"Unexpected status code: {response.status_code}")
                        break
                        
                except Exception as e:
                    print(f"Attempt {attempt+1} failed: {e}")
                    if attempt == max_retries - 1:
                        return []
                    time.sleep(1)
        
        return comments

    def run(self):
        """Main execution flow"""
        df = pd.read_csv('product_id_ncds.csv')
        product_ids = df.id.astype(str).tolist()
        
        for pid in tqdm(product_ids, desc="Processing products"):
            try:
                comments = self.crawl_product_comments(pid)
                
                for comment in comments:
                    # Lọc comment hợp lệ
                    if comment.get('content') and len(comment['content'].strip()) >= 10:
                        self.produce_comment({
                            'content': comment['content'],
                            'product_id': pid,
                            'rating': comment.get('rating', 0),
                            'created_at': comment.get('created_at', int(time.time()))
                        })
                
                time.sleep(random.uniform(1.2, 2.8))
                
            except Exception as e:
                print(f"Error processing product {pid}: {e}")
                continue
                
        # Cleanup
        self.producer.flush()
        print("Crawling completed!")

if __name__ == "__main__":
    producer = TikiProducer()
    producer.run()