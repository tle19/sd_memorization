import numpy as np

def euclidean_distance(x, y):
    return np.linalg.norm(x - y)

def manhattan_distance(x, y):
    return np.sum(np.abs(x - y))

def cosine_similarity(x, y):
    return (np.dot(x, y.T) / (np.linalg.norm(x) * np.linalg.norm(y)))[0][0]

def fid_score(x, y):
    pass
    #implement FID

def inception_score(x, y):
    pass
    #implement IS

def print_title(typ, name, index):
    counter = '{:0{width}d}'.format(index, width=8)
    print('\033[1m' + typ, counter, '-', name + '\033[0m')

def punc_splice(punc, text):
    pos = text.find(punc)
    if pos != -1:
        return text[:pos]
    else:
        return text
    
def generate_graph():
    pass