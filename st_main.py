import streamlit as st
from st_featurize_tweets import render_featurize_tweets
from st_mstream_results import render_mstream_results

PAGES = {
    "Featurize tweets": render_featurize_tweets,
    "MStream Results": render_mstream_results
}


PAGE_OPTIONS = list(PAGES.keys())

page_selection = st.sidebar.radio("Pages", PAGE_OPTIONS)
page = PAGES[page_selection]
page()