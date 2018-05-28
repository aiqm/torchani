import torchani.data
import pickle

data_path = '../ANI-1x_complete.h5'
chunk_size = 64
dataset = torchani.data.load_dataset(data_path)
chunks = len(torchani.data.BatchSampler(dataset, chunk_size, 1))
print(chunks, 'chunks')


training_size = int(chunks*0.8)
validation_size = int(chunks*0.1)
testing_size = chunks - training_size - validation_size
training, validation, testing = torchani.data.random_split(dataset, [training_size, validation_size, testing_size], chunk_size)

with open('dataset.dat', 'wb') as f:
    pickle.dump((training, validation, testing), f)