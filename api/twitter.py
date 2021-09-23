from typing import Any, List, TypedDict, Union
import requests
import logging
from config import twitter_keys
import streamlit as st
import time

bearer_token = twitter_keys["bearer_token"]

def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """

    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "FallSpecProject"
    return r


def authenticated_request(url, params, type="GET", next_token=None):
    if next_token is not None:
        url += f"?next_token={next_token}"
    response = requests.request(
        "GET", 
        url, 
        auth=bearer_oauth, 
        params=params
    )
    logging.info(f"{type} request to {url}, response {response.status_code}")
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()

# Optional params: start_time,end_time,since_id,until_id,next_token,granularity
# Documentation: https://developer.twitter.com/en/docs/twitter-api/tweets/counts/api-reference/get-tweets-counts-all
class CountQueryParams(TypedDict):
    query: str
    granularity: str
    start_time: str # isoformat
    end_time: str # isoformat

CountQueryData = TypedDict("data", {
    "start": str,
    "end": str,
    "tweet_count": int
})
CountQueryResponse = TypedDict("CountQueryResponse", {
    "meta": TypedDict("meta", {
        "total_tweet_count": int,
        "next_token": Union[str, None]
    }),
    "data": List[CountQueryData]
})

@st.cache(allow_output_mutation=True)
def fetch_historical_counts(query_params: CountQueryParams, paginated=False) -> CountQueryResponse:
    response: CountQueryResponse = authenticated_request(
        "https://api.twitter.com/2/tweets/counts/all",
        query_params
    )
    while response["meta"]["next_token"] is not None and not paginated:
        next_response = authenticated_request(
            "https://api.twitter.com/2/tweets/counts/all",
            query_params,
            next_token=response["meta"]["next_token"]
        )
        response["data"] += next_response["data"]
        response["meta"]["total_tweet_count"] += next_response["meta"]["total_tweet_count"]

        if "next_token" not in next_response["meta"]:
            response["meta"]["next_token"] = None
            break
        else:
            response["meta"]["next_token"] = next_response["meta"]["next_token"]
    
    return response



# Documentation: https://developer.twitter.com/en/docs/twitter-api/tweets/search/api-reference/get-tweets-search-all
class HistoricalTweetsParams(TypedDict):
    query: str
    start_time: str # isoformat
    end_time: str # isoformat

HistoricalTweetsResponse = TypedDict("CountQueryResponse", {
    "meta": TypedDict("meta", {
        "result_count": int,
        "newest_id": str,
        "oldest_id": str,
        "next_token": Union[str, None]
    }),
    "data": List[Any],
    "includes": Any,
    "errors": List[Any]
})

def fetch_historical_tweets(
    query_params: HistoricalTweetsParams,
    include_context_annotations, 
    n_pages=1
) -> HistoricalTweetsResponse:
    tweet_fields = [
        "entities",
        "geo",
        "id",
        "public_metrics",
        "lang",
        "reply_settings",
        "possibly_sensitive",
        "text",
        "created_at",
        "author_id",
        "in_reply_to_user_id",
        "referenced_tweets"
    ]
    progress_bar = st.progress(0)

    if include_context_annotations:
        tweet_fields.append("context_annotations")
    query_params_with_defaults: dict[str, Any] = {
        "max_results": 100 if include_context_annotations else 500,
        "expansions": ",".join([
            "author_id",
            "referenced_tweets.id",
            "referenced_tweets.id.author_id",
            # "geo.place_id"
        ]),
        "tweet.fields": ",".join(tweet_fields),
        "user.fields": ",".join([
            "created_at",
            "description",
            "entities",
            "id",
            "location",
            "name",
            "verified",
            "username"
        ]),
    }
    query_params_with_defaults.update(query_params)
    response: HistoricalTweetsResponse = authenticated_request(
        "https://api.twitter.com/2/tweets/search/all",
        query_params_with_defaults
    )
    fetched_pages = 1
    while response["meta"]["next_token"] is not None and fetched_pages < n_pages:
        progress_bar.progress(fetched_pages / n_pages)
        # avoid rate-limit (1 per second)
        time.sleep(1)
        next_response = authenticated_request(
            "https://api.twitter.com/2/tweets/search/all",
            query_params_with_defaults,
            next_token=response["meta"]["next_token"]
        )
        fetched_pages += 1
        response["data"] += next_response["data"]
        response["includes"]["users"] += next_response["includes"]["users"]
        response["includes"]["tweets"] += next_response["includes"]["tweets"]
        if "errors" in next_response:
            response["errors"] += next_response["errors"]
        response["meta"]["result_count"] += next_response["meta"]["result_count"]

        if "next_token" not in next_response["meta"]:
            response["meta"]["next_token"] = None
            break
        else:
            response["meta"]["next_token"] = next_response["meta"]["next_token"]
    progress_bar.empty()
    return response