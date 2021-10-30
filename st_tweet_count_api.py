from api.twitter import CountQueryParams, st_fetch_historical_counts
import streamlit as st
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
import pandas as pd
import plotly.express as px
import json

st.header("Explore Tweet Counts API")
query = st.text_input("Query", value="@ParisHilton OR from:ParisHilton OR to:ParisHilton")
col1, col2 = st.columns(2)
default_end_time = datetime.strptime(
    "2021-04-01", 
    '%Y-%m-%d'
)
start_time = col1.date_input(
    "Start time", 
    value=default_end_time - relativedelta(months=3)
)
end_time = col2.date_input(
    "End time", 
    value=default_end_time
)
    
granularity = st.selectbox("Granularity", options=[
#    "minute", 
    "hour",
    "day"
])

query_params: CountQueryParams = {
    "query": query,
    "granularity": granularity,
    "start_time": datetime.combine(
        start_time,
        datetime.min.time()
    ).replace(tzinfo=timezone.utc).isoformat(),
     "end_time": datetime.combine(
        end_time,
        datetime.min.time()
    ).replace(tzinfo=timezone.utc).isoformat()
}

query_counts = st_fetch_historical_counts(query_params)

st.write(f"**Total tweet count:** {query_counts['meta']['total_tweet_count']:,}")
with st.expander("Show API Response"):
    st.write(query_counts)

df_timeseries = pd.DataFrame([
    {"time_end": timestep["end"], "tweet_count": timestep["tweet_count"]}
    for timestep in query_counts["data"]
]).sort_values("time_end", ascending=True)

st.write(px.line(df_timeseries, x="time_end", y="tweet_count"))

filename = st.text_input("Filename").replace(".json", "")
if st.button("Export dataframe to json"):
    if filename != "":
        out_file = f"data/tweet_counts/{filename}.json"
        with open(out_file, "w") as f:
            json.dump({
                "query_params": query_params,
                "data": df_timeseries.to_dict(orient="records")
            }, f)
        st.success(f"Wrote to {out_file}!")
    else:
        st.error("Specify a filename!")