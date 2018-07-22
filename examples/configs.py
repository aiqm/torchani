import torch
data_path = 'data/ANI-1x_complete.h5'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
