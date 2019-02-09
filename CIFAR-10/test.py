def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding = 'bytes')
    return dict
batch = unpickle('cifar-10-batches-py/test_batch')
print(len(batch[b'data']))