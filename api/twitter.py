from typing import List, TypedDict, Union
import requests
import logging
from config import twitter_keys
import streamlit as st

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
query_params = {
    "query": "from:twitterdev",
    "granularity": "day",
    "start_time": "2021-01-01T00:00:00Z",
}
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
