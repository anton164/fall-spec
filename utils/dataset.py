from utils import flatten
import pandas as pd
from .dtypes import tweet_dtypes
from collections import Counter
import numpy as np

def get_with_default(obj, keys, default):
    for key in keys:
        if key in obj:
            obj = obj[key]
        else:
            return default
    return obj

def read_columns(columns_names_file):
    with open(columns_names_file) as f:
        first_line = f.readline()
    return first_line.split(",")

def create_tweet_df(raw_dataset_tweets):
    """ Create tweet dataset dataframe from raw tweet objects response """
    tweets = []
    for raw_tweet in raw_dataset_tweets["data"]:
        parsed_tweet = {
            "id": raw_tweet["id"],
            "text": raw_tweet["text"],
            "created_at": pd.to_datetime(raw_tweet["created_at"]),
            "hashtags": [hashtag["tag"] for hashtag in get_with_default(raw_tweet, ["entities", "hashtags"], [])],
            "mentions": [mention["username"] for mention in get_with_default(raw_tweet, ["entities", "mentions"], [])],
            "in_reply_to_user_id": raw_tweet["in_reply_to_user_id"] if "in_reply_to_user_id" in raw_tweet else None,
            "user_id": raw_tweet["author_id"],
            "retweet_count": raw_tweet["public_metrics"]["retweet_count"],
            "quote_count": raw_tweet["public_metrics"]["quote_count"],
            "reply_count": raw_tweet["public_metrics"]["reply_count"],
            "like_count": raw_tweet["public_metrics"]["like_count"],
            # TODO
            # to get user screen name we must look at raw_dataset_tweets["users"]
            
        }
        if "referenced_tweets" in raw_tweet:
            for reference in raw_tweet["referenced_tweets"]:
                parsed_tweet[reference["type"]] = reference["id"]
        tweets.append(parsed_tweet)

    df_tweets = pd.DataFrame(tweets)
    return df_tweets.sort_values("created_at")

def load_tweet_dataset(location):
    df_tweets = pd.read_json(location, dtype=tweet_dtypes)
    df_tweets["created_at"] = pd.to_datetime(df_tweets.created_at)

    df_tweets["retweeted"] = df_tweets.retweeted.apply(lambda x: None if x == "None" else x)
    df_tweets["quoted"] = df_tweets.quoted.apply(lambda x: None if x == "None" else x)
    df_tweets["replied_to"] = df_tweets.replied_to.apply(lambda x: None if x == "None" else x)

    # lower entities
    df_tweets["hashtags"] = df_tweets.hashtags.apply(lambda xs: [x.lower() for x in xs])
    df_tweets["mentions"] = df_tweets.mentions.apply(lambda xs: [x.lower() for x in xs])

    # Set retweet count 0 for retweets
    df_tweets["retweet_count"] = df_tweets.apply(
        lambda x: x.retweet_count if x.retweeted is None else 0, 
        axis=1
    )

    return df_tweets

def count_array_column(df_column):
    import streamlit as st
    column_count = Counter([x.lower() for x in flatten(df_column)])
    df_counts = pd.DataFrame(
        column_count.items(), 
        columns=["value", "count"]
    ).sort_values("count", ascending=False)
    df_counts["pct"] = df_counts["count"] / len(df_column)
    return df_counts