"""

This script runs the full MStream 
pipeline for a labeled dataset

"""


import argparse
import os

from create_embeddings import tokenize_dataframe_fasttext
from utils.dataset import load_tweet_dataset

CUR_DIR = os.getcwd()
INPUT_DATA_LOCATION = './data/labeled_datasets/'
OUTPUT_DATA_LOCATION = './data/embeddings/'

parser = argparse.ArgumentParser()

parser.add_argument(
    'input_file',  
    help='Input file'
)

parser.add_argument(
    '--include_text', 
    required=False,
    default=False,
    help='Output name'
)
parser.add_argument(
    '--merlion_anomaly_threshold', 
    required=False,
    default=1,
    type=float,
    help='Anomaly threshold for merlion labels (passed to prepare_mstream_data)'
)
parser.add_argument(
    '--mstream_alpha', 
    required=False,
    default=0.8,
    type=float,
    help='MStream decay factor'
)
args = parser.parse_args()

output_name = args.input_file.replace(".json", "")

print("Loading tweet dataset...")
df = load_tweet_dataset(
    INPUT_DATA_LOCATION + args.input_file
).set_index("id")

# Prepare embeddings
if (args.include_text):
    print("Tokenizing dataframe...")
    vocabulary, tokenized_string_idxs, fasttext_lookup = tokenize_dataframe_fasttext(
        df
    )

# Convert labeled dataset to mstream dataset
print("Preparing Mstream data...")
os.system(
    f"python prepare_mstream_data.py {args.input_file} {output_name} --anomaly_threshold {args.merlion_anomaly_threshold}"
)

# Run MStream
os.chdir("mstream/mstream")
print("Compiling MStream...")
os.system("make clean")
os.system("make")
os.chdir("..")
cmd = " ".join([
    "mstream/mstream",
    f"-t 'data/{output_name}_time.txt'",
    f"-n 'data/{output_name}_numeric.txt'",
    f"-c 'data/{output_name}_categ.txt'",
    f"-o 'data/{output_name}_score.txt'",
    f"-a {args.mstream_alpha}"
])
print("Running MStream...", cmd)
os.system(cmd)
os.system(
    " ".join([
        "python results.py",
        f"--label 'data/{output_name}_label.txt'",
        f"--scores 'data/{output_name}_score.txt'"
    ])
)
os.chdir(CUR_DIR)