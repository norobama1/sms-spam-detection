import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Make sure these are downloaded at least once in your environment:
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text: str):
    """
    Clean + tokenize + lemmatize an input text.
    Returns a LIST of tokens (for use as a custom tokenizer).
    """
    # Lowercase
    text = text.lower()

    # Replace money and numbers with tokens (keep the 'spammy' info)
    text = re.sub(r'\$', ' money ', text)
    text = re.sub(r'\d+', ' number ', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', ' ', text)

    # Remove non-letters (keep spaces)
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords, very short tokens, and lemmatize
    cleaned_tokens = []
    for token in tokens:
        if token not in stop_words and len(token) > 3:
            lemma = lemmatizer.lemmatize(token)
            cleaned_tokens.append(lemma)

    return cleaned_tokens
