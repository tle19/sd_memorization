import numpy as np

def euclidean_distance(x, y):
    return np.linalg.norm(x - y)

def manhattan_distance(x, y):
    return np.sum(np.abs(x - y))

def cosine_similarity(x, y):
    return (np.dot(x, y.T) / (np.linalg.norm(x) * np.linalg.norm(y)))[0][0]

def fid_distance(x, y):
    pass
    #implement FID

def is_distance(x, y):
    pass
    #implement IS
    
def generate_graph():
    pass