"""

This script runs the full MStream 
pipeline for a labeled dataset

"""


import os
from utils.dataset import load_tweet_dataset
import subprocess
from prepare_mstream_data import parser
import time

global_start_time = time.time()
class Timer(object):
    def __init__(self, message: str):
        self.message = message
      
    def __enter__(self):
        self.start_time = time.time()
        
  
    def __exit__(self, etype, value, traceback):
        elapsed_time = time.time() - self.start_time
        print(f"[Timer]: {self.message} took {elapsed_time}s")
  

def run_command(cmd: str):
    print("[Command]:", cmd)
    return subprocess.run(cmd.split(" "), check=True)


CUR_DIR = os.getcwd()
INPUT_DATA_LOCATION = './data/labeled_datasets/'
OUTPUT_DATA_LOCATION = './data/embeddings/'

parser.add_argument(
    'input_file',  
    help='Input file'
)

parser.add_argument(
    '--mstream_alpha', 
    required=False,
    default=0.8,
    type=float,
    help='MStream decay factor'
)
parser.add_argument(
    '--mstream_beta', 
    required=False,
    default=0,
    type=int,
    help='MStream smoothing term'
)
parser.add_argument(
    '--mstream_buckets', 
    required=False,
    default=1024,
    type=int,
    help='MStream number of buckets'
)
parser.add_argument(
    '--abs_min_max', 
    type=int,
    required=False,
    default=1,
    help='Abs min max, should the max and min value should be searched in stream fashion or a priori. Default is 1'
)
args = parser.parse_args()

print("Args", args)
output_name = args.input_file.replace(".json", "")

with Timer("Load data"):
    print("Loading tweet dataset...")
    df = load_tweet_dataset(
        INPUT_DATA_LOCATION + args.input_file
    ).set_index("id")

# Convert labeled dataset to mstream dataset
with Timer("prepare mstream data"):
    run_command(" ".join([
        "python prepare_mstream_data.py",
        f"{args.input_file} {output_name}",
        f"--merlion_anomaly_threshold {args.merlion_anomaly_threshold}",
        f"--text_encoding {args.text_encoding}",
        f"--text_synthetic {args.text_synthetic}",
        f"--text_exclude_retweets {args.text_exclude_retweets}",
        f"--hashtag_encoding {args.hashtag_encoding}",
        f"--hashtag_filter {args.hashtag_filter}",
        f"--unix_timestamp {args.unix_timestamp}",
        f"--fasttext_limit {args.fasttext_limit}",
    ]))

# Run MStream
with Timer("Run & compile MSTREAM"):
    os.chdir("mstream/mstream")
    print("Compiling MStream...")
    run_command("make clean")
    run_command("make")
    os.chdir("..")
    cmd = " ".join([
        "mstream/mstream",
        f"-t 'data/{output_name}_time.txt'",
        f"-n 'data/{output_name}_numeric.txt'",
        f"-c 'data/{output_name}_categ.txt'",
        f"-o 'data/{output_name}_score.txt'",
        f"-d 'data/{output_name}_decomposed.txt'",
        f"-i 'data/{output_name}_ignore_score_record.txt'",
        f"-dp 'data/{output_name}_decomposed_percentage.txt'",
        f"-a {args.mstream_alpha}",
        f"-b {args.mstream_buckets}",
        f"-beta {args.mstream_beta}",
        f"-absminmax {args.abs_min_max}",
        f"-tb 'data/{output_name}_token_buckets.txt'"
    ])
    print("Running MSTREAM...")
    os.system(cmd)

with Timer("Write results"):
    os.system(
        " ".join([
            "python results.py",
            f"--label 'data/{output_name}_label.txt'",
            f"--scores 'data/{output_name}_score.txt'",
            f"--name '{output_name}'"
        ])
    )
    os.chdir(CUR_DIR)
    print(f"Wrote final results to data/{output_name}")

print(f"[Timer]: full pipeline took {time.time() - global_start_time}")