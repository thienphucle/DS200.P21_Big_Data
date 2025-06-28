import re
from sklearn.base import BaseEstimator, TransformerMixin
from pyvi import ViTokenizer
from langdetect import detect

class VietnamesePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 remove_stopwords=True,
                 punctuation_chars='.,!?;:()[]{}<>"\'',
                 abbreviation_dict={'ko': 'không', 'j': 'gì', 'dc': 'được'}):
        self.remove_stopwords = remove_stopwords
        self.stopwords = self._load_stopwords()
        self.punctuation_chars = punctuation_chars
        self.abbreviation_dict = abbreviation_dict
        self.allcase_string = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠ-ỹ'
        
        # Danh sách thay thế Unicode
        self.unicode_replacements = {       
            "òa": "oà", "óa": "oá", "ỏa": "oả", "õa": "oã", "ọa": "oạ",
            "òe": "oè", "óe": "oé", "ỏe": "oẻ", "õe": "oẽ", "ọe": "oẹ",
            "ùy": "uỳ", "úy": "uý", "ủy": "uỷ", "ũy": "uỹ", "ụy": "uỵ",
            "Ủy": "Uỷ", "\n": ".", "\t": "."
        }

    def _load_stopwords(self):
        """Tải danh sách stopwords tùy chỉnh"""
        return {
            'và', 'của', 'các', 'là', 'cho', 'từ', 'nhưng', 'hoặc',
            'nếu', 'thì', 'mà', 'để', 'ở', 'trên', 'dưới', 'trong',
            'với', 'về', 'theo', 'có', 'được', 'không', 'này', 'khi'
        }

    def transform(self, texts):
        """Xử lý batch văn bản"""
        processed_texts = []
        for text in texts:
            text = self._full_pipeline(text)
            processed_texts.append(text)
        return processed_texts

    def _full_pipeline(self, text):
        """Quy trình xử lý đầy đủ cho một văn bản"""
        text = self.unicodeReplace(text)
        text = self.remove_emojis_url(text)
        text = self.abbreviationReplace(text)
        text = self.stickyPreprocess(text)
        text = self.remove_punctuation_and_numbers(text)
        text = self._normalize(text)
        text = self._tokenize(text)
        if self.remove_stopwords:
            text = self._remove_stopwords(text)
        return text

    def unicodeReplace(self, text):
        """Chuẩn hóa Unicode"""
        for old, new in self.unicode_replacements.items():
            text = text.replace(old, new)
        return text

    def remove_emojis_url(self, text):
        """Loại bỏ emoji và URL"""
        # Pattern cho emoji
        emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F700-\U0001F77F"  # alchemical symbols
                           u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                           u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                           u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                           u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                           u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                           u"\U00002702-\U000027B0"  # Dingbats
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
        
        # Loại bỏ URL và emoji
        text = emoji_pattern.sub(r'', text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        return text.strip()

    def stickyPreprocess(self, text):
        """Xử lý dấu câu dính vào chữ"""
        result = []
        # Tìm các vị trí cần chèn khoảng trắng
        for i in range(len(text) - 2):
            prev, curr, next_char = text[i], text[i+1], text[i+2]
            if curr in self.punctuation_chars:
                if prev in self.allcase_string:
                    result.append(i)
                if next_char in self.allcase_string:
                    result.append(i+1)
        
        # Chèn khoảng trắng từ cuối về đầu
        for pos in sorted(set(result), reverse=True):
            text = text[:pos] + ' ' + text[pos:]
        return text

    def abbreviationReplace(self, text):
        """Thay thế từ viết tắt"""
        for abbr, full in self.abbreviation_dict.items():
            text = re.sub(r'\b' + re.escape(abbr) + r'\b', full, text, flags=re.IGNORECASE)
        return text

    def remove_punctuation_and_numbers(self, text):
        """Loại bỏ dấu câu và số"""
        text = re.sub(r'[!\"#$%&\'()*+,\-./:;<=>?@\[\\\]^`{|}~\“\”₫]', '', text)
        text = re.sub(r'\d+', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def _normalize(self, text):
        """Chuẩn hóa văn bản"""
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()

    def _tokenize(self, text):
        """Tokenize tiếng Việt"""
        return ViTokenizer.tokenize(text)

    def _remove_stopwords(self, text):
        """Lọc stopwords"""
        tokens = text.split()
        return ' '.join([word for word in tokens if word not in self.stopwords])

    def clean_data(self, data):
        """Làm sạch DataFrame"""
        # Loại bỏ trùng lặp và NaN
        data = data.drop_duplicates(subset=['comment'], keep='first')
        data = data.dropna(subset=["comment"])
        
        # Lọc comment không phải tiếng Việt
        def is_vietnamese(text):
            try:
                return detect(text) == 'vi'
            except:
                return False
            
        data = data[data['comment'].apply(is_vietnamese)]
        return data.reset_index(drop=True)