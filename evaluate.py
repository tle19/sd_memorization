from utils import euclidean_func, manhattan_func
import argparse
import os
import pandas as pd

def str_to_metric(metric_str):
    metric_functions = {
        "euclidean": euclidean_func,
        "manhattan": manhattan_func,
    }
    
    if metric_str in metric_functions:
        return metric_functions[metric_str]
    else:
        raise argparse.ArgumentTypeError("Invalid metric provided")
    
def parse_args():
    parser = argparse.ArgumentParser(description="Memorization Metrics")
    parser.add_argument('--metric', type=str_to_metric, default="euclidean")
    parser.add_argument('--prompt', type=str, default="popular_actors")
    args = parser.parse_args()
    return args

args = parse_args()

eval_metric = args.metric
prompt_type = args.prompt

csv_file_path = os.path.join('output/', prompt_type, 'prompts.tsv')

generated_prompts_df = pd.read_csv(csv_file_path)
generated_prompts = generated_prompts_df.iloc[:, 3].values.tolist()

def metric():
    
    for prompt in generated_prompts:
        loc = os.path.join('output/', prompt_type)
        x = os.path.join(loc, 'images1', prompt + '.png')
        y = os.path.join(loc, 'images2', prompt + '.png')
        print(prompt, ':')
        euclidean_func(x, y)

metric()