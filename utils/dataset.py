import pandas as pd

def create_tweet_df(raw_dataset_tweets):
    """ Create tweet dataset dataframe from raw tweet objects response """
    tweets = []
    for raw_tweet in raw_dataset_tweets["data"]:
        parsed_tweet = {
            "id": raw_tweet["id"],
            "text": raw_tweet["text"],
            "created_at": pd.to_datetime(raw_tweet["created_at"]),
            "hashtags": [hashtag["tag"] for hashtag in raw_tweet["entities"]["hashtags"]] if "hashtags" in raw_tweet["entities"] else [],
            "mentions": [mention["username"] for mention in raw_tweet["entities"]["mentions"]] if "mentions" in raw_tweet["entities"] else [],
            "in_reply_to_user_id": raw_tweet["in_reply_to_user_id"] if "in_reply_to_user_id" in raw_tweet else None,
        }
        if "referenced_tweets" in raw_tweet:
            for reference in raw_tweet["referenced_tweets"]:
                parsed_tweet[reference["type"]] = reference["id"]
        tweets.append(parsed_tweet)

    df_tweets = pd.DataFrame(tweets)
    return df_tweets.sort_values("created_at")