import os
import pandas as pd
import shutil
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

def make_temp_dir(src, dst, cond=None):
    os.makedirs(dst)
    if any(cond):
        for name in cond:
            src_img = os.path.join(src, name + '.png')
            dst_img = os.path.join(dst, name + '.png')
            shutil.copy(src_img, dst_img)
    else:
        shutil.copytree(src, dst)

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

    scores = df['Cosine Avg'][df['is_human']]
    column_name = scores.name

    plt.bar(scores, bins=20, color='blue', width=0.4)
    plt.xlabel(f'{column_name} Score')
    plt.ylabel(f'Number of People')
    plt.title(f'{column_name} Score Distribution')
    plt.show()