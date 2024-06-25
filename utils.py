import torch_fidelity

def calculate_fid(image1, image2):
    fid_score = torch_fidelity.calculate_metrics(
        input1=image1,
        input2=image2,
        cuda=True,
        isc=True,
        fid=True,
        verbose=False
    )
    return fid_score

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