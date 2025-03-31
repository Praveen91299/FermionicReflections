import pickle as pkl

def get_pkl_object(filename: str, directory: str, verbose: bool = False):
    with open(file = directory + filename, mode = "rb") as file:
        if verbose: print('\nLoading file {}, from {}'.format(filename, directory))
        return pkl.load(file)

def save_pkl_object(object, filename: str, directory: str):
    with open(file = directory + filename, mode = "wb") as file:
        pkl.dump(object, file=file, protocol=pkl.HIGHEST_PROTOCOL)
    return