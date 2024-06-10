from utils import euclidean_func, manhattan_func
import argparse
import os
import pandas as pd

def metric(func):
    metric_functions = {
        "euclidean": euclidean_func,
        "manhattan": manhattan_func,
    }
    
    if func in metric_functions:
        return metric_functions[func]
    else:
        raise argparse.ArgumentTypeError("Invalid metric provided")
    
def parse_args():
    parser = argparse.ArgumentParser(description="Memorization Metrics")
    parser.add_argument('--metric', type=metric, default="euclidean")
    parser.add_argument('--prompt', type=str, default="popular_actors")
    args = parser.parse_args()
    return args

args = parse_args()

eval_metric = args.metric
prompt_type = args.prompt


csv_path = os.path.join('output/', prompt_type, 'prompts.csv')
prompts_df = pd.read_csv(csv_path)
generated_prompts = prompts_df['Name'].tolist()

distances = []
for prompt in generated_prompts:
    path = os.path.join('output/', prompt_type)
    x = os.path.join(path, 'images1', prompt + '.png')
    y = os.path.join(path, 'images2', prompt + '.png')
    
    dist = eval_metric(x, y)
    distances.append(dist)

    print(prompt, ':')
    print(dist)

print('Closest', min(distances)) #sanity check

prompts_df['Metric'] = distances
prompts_df.to_csv(csv_path)