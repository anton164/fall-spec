import pandas as pd

def get_with_default(obj, keys, default):
    for key in keys:
        if key in obj:
            obj = obj[key]
        else:
            return default
    return obj

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