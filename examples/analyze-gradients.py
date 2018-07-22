import pickle
import torch

hyperparams = [  # (chunk size, batch chunks)
    # (64, 4),
    (64, 8),
    (64, 16),
    (64, 32),
    (128, 2),
    (128, 4),
    (128, 8),
    (128, 16),
    (256, 1),
    (256, 2),
    (256, 4),
    (256, 8),
    (512, 1),
    (512, 2),
    (512, 4),
    (1024, 1),
    (1024, 2),
    (2048, 1),
]

for chunk_size, batch_chunks in hyperparams:
    with open('data/avg-{}-{}.dat'.format(chunk_size, batch_chunks),
              'rb') as f:
        ag, agsqr = pickle.load(f)
        variance = torch.sum(agsqr) - torch.sum(ag**2)
        stddev = torch.sqrt(variance).item()
        print(chunk_size, batch_chunks, stddev)
