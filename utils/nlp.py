import re
import emoji
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import nltk
from nltk.tokenize import word_tokenize

nltk.download('words')
nltk.download('punkt')

words = set(nltk.corpus.words.words())
stop_words = set(stopwords.words('english'))

def stemming(words):
    ps=PorterStemmer()
    return [ps.stem(word) for word in words]

def remove_stop_words(words):
    return [c.lower() for c in words if c not in stop_words]

def lemmatizing(words):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in words]


def cleaner(tweet):
    tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweet).split())
    tweet = " ".join(tweet.split())
    tweet = ''.join(c for c in tweet if c not in emoji.UNICODE_EMOJI) #Remove Emojis
    return tweet

def final_text(words):
     return ' '.join(words)

def preprocess_text(
    text: str,
    tokenize=True
):
    """ Helper method for parametrizing text preprocessing """
    if tokenize:
        return word_tokenize(text)