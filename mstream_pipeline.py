"""

This script runs the full MStream 
pipeline for a labeled dataset

"""


import os
from utils.dataset import load_tweet_dataset
import subprocess
from prepare_mstream_data import parser

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
args = parser.parse_args()

print("Args", args)
output_name = args.input_file.replace(".json", "")

print("Loading tweet dataset...")
df = load_tweet_dataset(
    INPUT_DATA_LOCATION + args.input_file
).set_index("id")

# Convert labeled dataset to mstream dataset
run_command(" ".join([
    "python prepare_mstream_data.py",
    f"{args.input_file} {output_name}",
    f"--merlion_anomaly_threshold {args.merlion_anomaly_threshold}",
    f"--text_encoding {args.text_encoding}",
    f"--text_synthetic {args.text_synthetic}",
    f"--hashtag_encoding {args.hashtag_encoding}",
]))

# Run MStream
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
    f"-a {args.mstream_alpha}"
])
print("Running MSTREAM...")
os.system(cmd)
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