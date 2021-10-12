import pandas as pd
import argparse
from utils.nlp import preprocess_text
from utils.dataset import load_tweet_dataset
import time
import datetime
import random

INPUT_DATA_LOCATION = './data/labeled_datasets/'
OUTPUT_DATA_LOCATION = './MStream/data/'

parser = argparse.ArgumentParser()

parser.add_argument(
    'input_file',  
    help='Input file'
)
parser.add_argument(
    'output_name', 
    help='Outuput name'
)
parser.add_argument(
    '--anomaly_threshold',
    required=False,
    default=1,
    type=float
)

args = parser.parse_args()

df = load_tweet_dataset(
    INPUT_DATA_LOCATION + args.input_file
).set_index("id")
# TODO this is ugly don't kill me

def create_unix(x):
    return int(time.mktime((x).timetuple()))

# Load labels from merlion
df["is_anomaly"] = df["merlion_anomaly_total_count"].apply(lambda x: x > args.anomaly_threshold)
df["is_anomaly_hashtag1"] = df["merlion_anomaly_top1_hashtag_count"].apply(lambda x: x > args.anomaly_threshold)
df["is_anomaly_hashtag2"] = df["merlion_anomaly_top2_hashtag_count"].apply(lambda x: x > args.anomaly_threshold)
df["is_anomaly_hashtag3"] = df["merlion_anomaly_top3_hashtag_count"].apply(lambda x: x > args.anomaly_threshold)


df['text'] = df['text'].apply(lambda x: preprocess_text(x))
#df['hashtags'] = df['hashtags'].apply(lambda xs: [x for x in xs if x.lower() == "unitedairlines"])

#df = df.explode('hashtags').explode('mentions').explode('text')
df = df.explode('hashtags')
continuous_index = []
#symbolic_index = ['hashtags', 'user_id', 'mentions', 'text']
symbolic_index = ['hashtags']

df_continuous = df.loc[:, continuous_index]
df_symbolic = df.loc[:, symbolic_index]
df_label = df.loc[:, ['is_anomaly']]


df_symbolic.loc[:,'hashtags'].unique()
#df_symbolic.loc[:,'user_id'].unique()
#df_symbolic.loc[:,'mentions'].unique()
#df_symbolic.loc[:,'text'].unique()


hashtags_dict = {}
for i, entry in enumerate(df_symbolic.loc[:,'hashtags'].unique()):
    hashtags_dict[entry] = i
    
'''user_id_dict = {}
for i, entry in enumerate(df_symbolic.loc[:,'user_id'].unique()):
    user_id_dict[entry] = i

mentions_dict = {}
for i, entry in enumerate(df_symbolic.loc[:,'mentions'].unique()):
    mentions_dict[entry] = i
    
text_dict = {}
for i, entry in enumerate(df_symbolic.loc[:,'text'].unique()):
    text_dict[entry] = i'''

df_symbolic.loc[:,'hashtags'] =  df_symbolic.loc[:,'hashtags'].map(hashtags_dict)
#df_symbolic.loc[:,'user_id'] =  df_symbolic.loc[:,'user_id'].map(user_id_dict)
#df_symbolic.loc[:,'mentions'] =  df_symbolic.loc[:,'mentions'].map(mentions_dict)
#df_symbolic.loc[:,'text'] =  df_symbolic.loc[:,'text'].map(text_dict)

df_continuous.to_csv(f"{OUTPUT_DATA_LOCATION}{args.output_name}_numeric.txt", index=False, header=False)
df_symbolic.to_csv(f"{OUTPUT_DATA_LOCATION}{args.output_name}_categ.txt", index=False, header=False)
df_label.to_csv(f"{OUTPUT_DATA_LOCATION}{args.output_name}_label.txt", index=False, header=False)

df_label.reset_index()[["id"]].to_csv(f"{OUTPUT_DATA_LOCATION}{args.output_name}_tweet_id.txt", index=False, header=False)
df.loc[:,'created_at'] = pd.to_datetime(df['created_at']).dt.floor('60T')
df.loc[:,'created_at'].map(create_unix).to_csv(f"{OUTPUT_DATA_LOCATION}{args.output_name}_time.txt", index=False, header=False)
df_label.reset_index()[["id"]].duplicated().astype(int).to_csv(f"{OUTPUT_DATA_LOCATION}{args.output_name}_ignore_record_score.txt", index=False, header=False)

text_file = open(f"{OUTPUT_DATA_LOCATION}{args.output_name}_numeric.txt", "w")
df_symbolic.shape[0]
n = text_file.write('\n'*df_symbolic.shape[0])
text_file.close()