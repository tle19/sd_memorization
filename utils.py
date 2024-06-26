import os
import pandas as pd
from torch_fidelity import calculate_metrics
import matplotlib.pyplot as plt

def print_title(typ, name, index):
    counter = '{:0{width}d}'.format(index, width=8)
    print(f'\033[1m{typ}{counter}-{name}\033[0m')

def punc_splice(punc, text):
    pos = text.find(punc)
    if pos != -1:
        return text[:pos]
    else:
        return text

def folder_size(path):
    files = os.listdir(path)
    return len(files) - 2

def calculate_fid(base_images_path, generated_images_path):
    fid_score = calculate_metrics(
        input1=base_images_path,
        input2=generated_images_path,
        cuda=True,
        isc=True,
        fid=True,
        verbose=False
    )
    return fid_score

def bar_graph(file, metric):
    df = pd.read_csv(file)

    scores = df[metric][df[metric] != -1]
    column_name = scores.name

    plt.bar(scores, bins=20, color='blue', width=0.4)
    plt.xlabel(f'{column_name} Score')
    plt.ylabel(f'Number of People')
    plt.title(f'{column_name} Score Distribution')
    plt.show()