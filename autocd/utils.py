import os
import json
import pickle


def check_extension(filename):
    if os.path.splitext(filename)[1] != ".json":
        return filename + ".json"
    return filename
    
def save_json(dataset, filename):
    filedir = os.path.split(filename)[0]

    if not os.path.isdir(filedir):
        os.makedirs(filedir)

    with open(check_extension(filename), 'w') as f:
        json.dump(dataset, f)

def load_json(filename):
    with open(check_extension(filename), 'r') as f:
        return json.load(f)
    
def save_pickle(dataset, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
