# fall-spec


## Setup
```
conda create python=3.8 -n spec-project
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```
#### Merlion dependencies
```
conda install -c conda-forge lightgbm 
brew install libomp
``` 

#### Set up API keys

1. Add Twitter API keys to `config/twitter.json` file.


# Pipeline Documentation

## 1) Explore tweet count API
```
streamlit run st_tweet_count_api.py
```

- Reads from Twitter API
- Saves to `data/tweet_objects`
## 2) Explore saved tweet counts & fetch tweet objects 
```
streamlit run st_explore_tweet_counts.py
```

- Reads from `data/tweet_counts` & Twitter API
- Saves to `data/tweet_objects`

## 3) Explore tweet object datasets
```
streamlit run st_explore_tweet_objects.py
```

- Reads from `data/tweet_objects` 

## 4) Label tweet object datasets
```
juptyer_notebooks/label_tweet_dataset.ipynb
```

- Reads from `data/tweet_objects` 
- Writes to `data/labeled_datasets`


## 5) Explore labeled tweet object datasets
```
streamlit run st_featurize_tweets.py
```

- Reads from `data/tweet_objects` 

## 6) Convert labeled dataset to mstream dataset 
```
python prepare_mstream_data.py <input_filename> <output_filename>
```

- Reads from `data/labeled_datasets` 
- Writes to `MStream/data`

## 7) Run MStream
```
cd MStream
./run.sh <input_dataset_name>
```

- Reads from `MStream/data`
- Writes to `MStream/data`

## 8) Inspect MStream results
```
streamlit run st_featurize_tweets.py
```

- Reads from `data/tweet_objects` 
- Reads from `MStream/data`