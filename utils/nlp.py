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
    ensure_unique_tokens=True
):
    """ Helper method for parametrizing text preprocessing """
    if clean:
        text = cleaner(text)
    tokens = word_tokenize(text)
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


def exclude_retweet_text(seen_tweets=set()):
    def inner_fn(tweet):
        """ Exclude retweet text, but ensure that an original tweet's text 
            is returned once (i.e. when we process the first original tweet/retweet)
        """
        retweeted_id = "retweeted" in tweet and tweet["retweeted"]
        if (retweeted_id and retweeted_id in seen_tweets):
            return ""
        else:
            if (retweeted_id):
                seen_tweets.add(retweeted_id)
            else:
                seen_tweets.add(tweet.name)
            return tweet.text
    return inner_fn