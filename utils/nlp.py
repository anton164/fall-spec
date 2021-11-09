import re
import emoji
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import nltk
from nltk.tokenize import word_tokenize

# uncomment in new env
# nltk.download('stopwords')
# nltk.download('words')
# nltk.download('punkt')
# nltk.download('wordnet')

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

def lower_text(words):
    return [word.lower() for word in words]

def cleaner(tweet):
    tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweet).split())
    tweet = " ".join(tweet.split())
    tweet = ''.join(c for c in tweet if c not in emoji.UNICODE_EMOJI) #Remove Emojis
    return tweet

def final_text(words):
     return ' '.join(words)

def preprocess_text(
    text: str,
    tokenize=True,
    lower=True,
    clean=True,
    lemmatize=True,
    stem=True,
    stop_words=True
):
    """ Helper method for parametrizing text preprocessing """
    if clean:
        text = cleaner(text)
    text = word_tokenize(text)
    if lower:
        text = lower_text(text)
    if lemmatize:
        text = lemmatizing(text)
    if stem:
        text = stemming(text)
    if stop_words:
        text = remove_stop_words(text)
    if tokenize:
        return text
    return final_text(text)
    

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



