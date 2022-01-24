# In new env:
# Download fasttext embeddings and save them in data/embeddings/fasttext
# https://fasttext.cc/docs/en/english-vectors.html
# uncomment
# nltk.download('stopwords')
# nltk.download('words')
# nltk.download('punkt')
# nltk.download('wordnet')

import re
import emoji
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
import pickle

INPUT_DATA_LOCATION = '../data/labeled_datasets/'
OUTPUT_DATA_LOCATION = '../data/embeddings/'

def load_fasttext(limit):
    cache_loc = f'../data/embeddings/fasttext/cache-{limit}'
    try:
        with open(cache_loc, 'rb') as f:
            print("Loaded fasttext from cache!", cache_loc)
            return pickle.load(f)
    except Exception as e:
        print("Failed to load fasttext from cache", cache_loc)
        fasttext = KeyedVectors.load_word2vec_format('../data/embeddings/fasttext/wiki-news-300d-1M.vec', limit=limit)
        with open(cache_loc, 'wb') as f:
            print("Writing fasttext to cache", cache_loc)
            pickle.dump(fasttext, f)
        return fasttext

custom_stopwords = {
    "rt"
}

words = set(nltk.corpus.words.words())
stop_words = set(stopwords.words('english')).union(custom_stopwords)

def stemming(words):
    ps=PorterStemmer()
    return [ps.stem(word) for word in words]

def remove_stop_words(words):
    return [c.lower() for c in words if c not in stop_words]

def lemmatizing(words):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in words]

def keep_noun_verb(words):
    nouns = {'NN', 'NNS', 'NNP', 'NNPS'}
    verbs = {'VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ'}
    pos = set([t[1] for t in nltk.pos_tag(words)])
    if len(nouns & pos) > 0 and len(verbs & pos) > 0 and len(words) > 3:
        return words
    return []

def lower_text(words):
    return [word.lower() for word in words]

def cleaner(tweet):
    # remove hashtags and mentions

    remove_urls = lambda x: re.sub("http(.+)?(\W|$)", ' ', x)
    normalize_spaces = lambda x: re.sub("[\n\r\t ]+", ' ', x)
    
    remove_mentions = lambda x: re.sub("@(\w+)", "", x)
    remove_hashtags = lambda x: re.sub("#(\w+)", "", x)

    tweet = remove_mentions(remove_hashtags(normalize_spaces(remove_urls(tweet))))

    tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweet).split())
    tweet = " ".join(tweet.split())
    tweet = ''.join(c for c in tweet if c not in emoji.UNICODE_EMOJI) #Remove Emojis
    return tweet

def preprocess_text(
    text: str,
    tokenize=True,
    lower=True,
    clean=True,
    lemmatize=False,
    stem=False,
    stop_words=True,
    ensure_unique_tokens=True,
    noun_verb=False
):
    """ Helper method for parametrizing text preprocessing """
    if clean:
        text = cleaner(text)
    tokens = word_tokenize(text)
    if noun_verb:
        tokens = keep_noun_verb(tokens)
    if lower:
        tokens = lower_text(tokens)
    if lemmatize:
        tokens = lemmatizing(tokens)
    if stem:
        tokens = stemming(tokens)
    if stop_words:
        tokens = remove_stop_words(tokens)
    if ensure_unique_tokens:
        tokens = list(set(tokens))
    if tokenize:
        return tokens
    return ' '.join(tokens)
    
def construct_vocabulary_encoding(
    tokenized_strings,
    fasttext
):
    """ Helper method for constructing vocabulary encoding from tokenized strings """
    vocabulary = {}
    fasttext_subset = {}
    tokenized_string_idxs = []
    tokenized_strings = tokenized_strings
    for tokens in tokenized_strings:
        idxs = []
        tokenized_string_idxs.append(idxs)
        for token in tokens:
            if token not in vocabulary:
                vocabulary[token] = {
                    "occurrences": 1,
                    "idx": len(vocabulary),
                    "fasttext_idx": None
                }
                if token in fasttext:
                    vocabulary[token]["fasttext_idx"] = fasttext.key_to_index[token]
                    fasttext_subset[token] = fasttext[token].tolist()
            else:
                vocabulary[token]["occurrences"] += 1
            idxs.append(vocabulary[token]["idx"])
    return vocabulary, tokenized_string_idxs, fasttext_subset
    