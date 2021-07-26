# type: ignore
"""Training with low memory usage"""
import torch
import torchani
import os
import math
from pathlib import Path

import torch.utils.tensorboard
import tqdm
import pkbar  # noqa

from torchani.transforms import AtomicNumbersToIndices, SubtractSAE
from torchani.units import hartree2kcalmol

# Explanation of the Batched Dataset API for ANI, which is a dataset that
# consumes minimal memory since it lives on disk, and batches are fetched on
# the fly
# This example is meant for internal use of Roitberg's Group

# device to run the training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --Starting here this is different from the usual nnp_training.py--
h5_path = '../dataset/ani-1x/sample.h5'
batched_dataset_path = './batched_dataset_1x'

# We prebatch the dataset to train with memory efficiency, keeping a good performance.
folds = True
if not Path(batched_dataset_path).resolve().is_dir():

    # In this example we use training = 80%, validation = 20%, we don't use folds
    if not folds:
        torchani.datasets.create_batched_dataset(h5_path,
                                                 dest_path=batched_dataset_path,
                                                 batch_size=2560,
                                                 splits={'training': 0.8, 'validation': 0.2})

    else:
        # We can prebatch the dataset using folds, this is useful for ensemble training
        # or cross validation.
        # for example, if the dataset has 10 conformations, we can split into 5
        # folds, each fold will have 8 conformations for a training set and 2
        # conformations for the validation set. All validation sets will be
        # disjoint.
        #
        # schematically:
        # total set: 0 1 2 3 4 5 6 7 8 9
        #
        # fold 0:   {0 1 } [2 3 4 5 6 7 8 9 ]
        # fold 1:   [0 1] {2 3} [4 5 6 7 8 9]
        # fold 2:   [0 1 2 3] {4 5} [6 7 8 9]
        # fold 3:   [0 1 2 3 4 5] {6 7} [8 9]
        # fold 4:   [0 1 2 3 4 5 6 7 ] {8 9 }
        # where {} encloses validation and [] encloses training conformations
        #
        # the fold names will be training0, validation0, training1, validation1,
        # ... etc.
        #
        torchani.datasets.create_batched_dataset(h5_path,
                                                 dest_path=batched_dataset_path,
                                                 batch_size=2560,
                                                 folds=5)

# This batched datasets can be directly iterated upon, but it may be more
# practical to wrap it with a torch DataLoader
if not folds:
    training = torchani.datasets.ANIBatchedDataset(batched_dataset_path, split='training')
    validation = torchani.datasets.ANIBatchedDataset(batched_dataset_path, split='validation')

else:
    training = torchani.datasets.ANIBatchedDataset(batched_dataset_path, split='training0')
    validation = torchani.datasets.ANIBatchedDataset(batched_dataset_path, split='validation0')

cache = False
if not cache:
    # If we decide not to cache the dataset it is a good idea to use
    # multiprocessing. Here we use some default useful arguments for
    # num_workers (extra cores for training) and prefetch_factor (data units
    # each worker buffers), but you should probably experiment depending on
    # your batch size and system to get the best performance. Performance can
    # be made in general almost the same as what you get caching the dataset
    # for pure python, but it is a bit slower than cacheing if using cuaev
    # (this is because cuaev is very fast).
    #
    # We also use shuffle=True, to shuffle batches every epoch (takes no time at all)
    # and pin_memory=True, to speed up transfer of memory to the GPU.
    #
    # If you can afford it in terms of memory you can sometimes get a bit of a
    # speedup by cacheing the validation set and setting persistent_workers = True
    # for the training set.
    #
    # NOTE: it is very important here to pass batch_size = None since the dataset is
    # already batched!
    #
    # NOTE: for more info about the DataLoader and multiprocessing read
    # https://pytorch.org/docs/stable/data.html
    training = torch.utils.data.DataLoader(training,
                                           shuffle=True,
                                           num_workers=2,
                                           prefetch_factor=2,
                                           pin_memory=True,
                                           batch_size=None)

    validation = torch.utils.data.DataLoader(validation,
                                             shuffle=False,
                                             num_workers=2,
                                             prefetch_factor=2,
                                             pin_memory=True,
                                             batch_size=None)
elif cache:
    # If need some extra speedup you can cache the dataset before passing it to
    # the DataLoader or iterating on it, but this may occupy a lot of memory,
    # so be careful!!!
    #
    # this is basically what the previous dataset api (data.load) did always so
    # you should get the same speed as with the previous api, but without the
    # initial memory peak the previous api had.
    #
    # Note: it is very important to **not** pass pin_memory=True here, since
    # cacheing automatically pins the memory of the whole dataset
    training = torch.utils.data.DataLoader(training.cache(),
                                           shuffle=True,
                                           batch_size=None)

    validation = torch.utils.data.DataLoader(validation.cache(),
                                             shuffle=False,
                                             batch_size=None)

# We use a transform on the dataset to perform transformations on the fly, the
# API for transforms is very similar to torchvision
# https://pytorch.org/vision/stable/transforms.html with the difference that
# the transforms are applied to both target and inputs in all cases.
#
# A transform can be passed to the "transform" argument of ANIBatchedDataset to
# perform the transforms on CPU after fetching the batches. If you do this and
# cache the dataset afterwards, the transform is applied only once, so there is
# no overhead at all.
#
# Another option is to apply the transform directly during training. We will do
# this in the example, since it allows us to cast the transform to GPU, which
# is very fast, and it avoids transforming the dataset in-place, which is error
# prone.
#
# Finally, a transform can be passed to create_batched_dataset using the
# argument "inplace_transform", but this is only really recommended if your
# transforms take a lot of time, since this will modify the dataset and may
# introduce hard to track discrepancies and reproducibility issues.
# This last thing is basically what the previous dataset api (data.load) did
# always so you should get the same speed as with the previous api by doing
# this.
#
elements = ('H', 'C', 'N', 'O')
# here we use the GSAEs for self energies
self_energies = [-0.499321200000, -37.83383340000, -54.57328250000, -75.04245190000]
transform = torchani.transforms.Compose([AtomicNumbersToIndices(elements), SubtractSAE(elements, self_energies)]).to(device)

estimate_saes = False
if estimate_saes:
    # NOTE: Take into account that in general now we are training with GSAEs
    # instead of SAEs, so the step that follows is largely not necessary
    #
    # If you would like to use SAEs in place of GSAEs you have the option to
    # estimate SAEs using SGD over the whole training set (it is important not
    # to include the validation set since the model never has to see it).
    #
    # This allows calculating the linear fit without loading the whole dataset
    # into memory, it is accurate, and it is very fast, even in CPU. Also, it
    # doesn't matter what transform your dataset had, this function doesn't
    # take the transform into account, and works well unless you performed some
    # inplace operations in your dataset
    from torchani.transforms import calculate_saes  # noqa
    saes, _ = calculate_saes(training, elements, mode='sgd')
    print(saes)
    # now we build the transform using the new self energies
    transform = torchani.transforms.Compose([AtomicNumbersToIndices(elements), SubtractSAE(elements, saes)]).to(device)
    # If we really want to, we can also calculate the saes exactly by passing
    # mode = exact, but this will take up a lot of memory because it uses the
    # whole dataset We can also pass a fraction of the dataset, for example
    # with fraction=0.01, 1% already gives a pretty good estimate
    # my tests:
    # all batches of 1x training:
    # tensor([ -0.6013, -38.0832, -54.7081, -75.1927])
    # 1% of 1x training:
    # tensor([ -0.5997, -38.0840, -54.7085, -75.1936])
    # 5% of 1x training:
    # tensor([ -0.5999, -38.0838, -54.7085, -75.1938])

# --Differences largely end here, besides application of transform in training/validation loops--
###############################################################################
# First lets define an aev computer like the one in the 1x model
aev_computer = torchani.AEVComputer.like_1x(use_cuda_extension=False)
# Now let's define atomic neural networks.
aev_dim = aev_computer.aev_length

H_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 160),
    torch.nn.CELU(0.1),
    torch.nn.Linear(160, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)

C_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 144),
    torch.nn.CELU(0.1),
    torch.nn.Linear(144, 112),
    torch.nn.CELU(0.1),
    torch.nn.Linear(112, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)

N_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 112),
    torch.nn.CELU(0.1),
    torch.nn.Linear(112, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)

O_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 112),
    torch.nn.CELU(0.1),
    torch.nn.Linear(112, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)

nn = torchani.ANIModel([H_network, C_network, N_network, O_network])
print(nn)

###############################################################################
# Initialize the weights and biases.
#
# .. note::
#   Pytorch default initialization for the weights and biases in linear layers
#   is Kaiming uniform. See: `TORCH.NN.MODULES.LINEAR`_
#   We initialize the weights similarly but from the normal distribution.
#   The biases were initialized to zero.
#
# .. _TORCH.NN.MODULES.LINEAR:
#   https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear


def init_params(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, a=1.0)
        torch.nn.init.zeros_(m.bias)


nn.apply(init_params)

###############################################################################
# Let's now create a pipeline of AEV Computer --> Neural Networks.
model = torchani.nn.Sequential(aev_computer, nn).to(device)

###############################################################################
# Now let's setup the optimizers. NeuroChem uses Adam with decoupled weight decay
# to updates the weights and Stochastic Gradient Descent (SGD) to update the biases.
# Moreover, we need to specify different weight decay rate for different layes.
#
# .. note::
#
#   The weight decay in `inputtrain.ipt`_ is named "l2", but it is actually not
#   L2 regularization. The confusion between L2 and weight decay is a common
#   mistake in deep learning.  See: `Decoupled Weight Decay Regularization`_
#   Also note that the weight decay only applies to weight in the training
#   of ANI models, not bias.
#
# .. _Decoupled Weight Decay Regularization:
#   https://arxiv.org/abs/1711.05101

AdamW = torch.optim.AdamW([
    # H networks
    {'params': [H_network[0].weight]},
    {'params': [H_network[2].weight], 'weight_decay': 0.00001},
    {'params': [H_network[4].weight], 'weight_decay': 0.000001},
    {'params': [H_network[6].weight]},
    # C networks
    {'params': [C_network[0].weight]},
    {'params': [C_network[2].weight], 'weight_decay': 0.00001},
    {'params': [C_network[4].weight], 'weight_decay': 0.000001},
    {'params': [C_network[6].weight]},
    # N networks
    {'params': [N_network[0].weight]},
    {'params': [N_network[2].weight], 'weight_decay': 0.00001},
    {'params': [N_network[4].weight], 'weight_decay': 0.000001},
    {'params': [N_network[6].weight]},
    # O networks
    {'params': [O_network[0].weight]},
    {'params': [O_network[2].weight], 'weight_decay': 0.00001},
    {'params': [O_network[4].weight], 'weight_decay': 0.000001},
    {'params': [O_network[6].weight]},
])

SGD = torch.optim.SGD([
    # H networks
    {'params': [H_network[0].bias]},
    {'params': [H_network[2].bias]},
    {'params': [H_network[4].bias]},
    {'params': [H_network[6].bias]},
    # C networks
    {'params': [C_network[0].bias]},
    {'params': [C_network[2].bias]},
    {'params': [C_network[4].bias]},
    {'params': [C_network[6].bias]},
    # N networks
    {'params': [N_network[0].bias]},
    {'params': [N_network[2].bias]},
    {'params': [N_network[4].bias]},
    {'params': [N_network[6].bias]},
    # O networks
    {'params': [O_network[0].bias]},
    {'params': [O_network[2].bias]},
    {'params': [O_network[4].bias]},
    {'params': [O_network[6].bias]},
], lr=1e-3)

###############################################################################
# Setting up a learning rate scheduler to do learning rate decay
AdamW_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(AdamW, factor=0.5, patience=100, threshold=0)
SGD_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(SGD, factor=0.5, patience=100, threshold=0)

###############################################################################
# Train the model by minimizing the MSE loss, until validation RMSE no longer
# improves during a certain number of steps, decay the learning rate and repeat
# the same process, stop until the learning rate is smaller than a threshold.
#
# We first read the checkpoint files to restart training. We use `latest.pt`
# to store current training state.
latest_checkpoint = 'latest.pt'

###############################################################################
# Resume training from previously saved checkpoints:
if os.path.isfile(latest_checkpoint):
    checkpoint = torch.load(latest_checkpoint)
    nn.load_state_dict(checkpoint['nn'])
    AdamW.load_state_dict(checkpoint['AdamW'])
    SGD.load_state_dict(checkpoint['SGD'])
    AdamW_scheduler.load_state_dict(checkpoint['AdamW_scheduler'])
    SGD_scheduler.load_state_dict(checkpoint['SGD_scheduler'])

###############################################################################
# During training, we need to validate on validation set and if validation error
# is better than the best, then save the new best model to a checkpoint


def validate():
    # run validation
    mse_sum = torch.nn.MSELoss(reduction='sum')
    total_mse = 0.0
    count = 0
    model.train(False)
    with torch.no_grad():
        for properties in validation:
            properties = {k: v.to(device, non_blocking=True) for k, v in properties.items()}
            properties = transform(properties)
            species = properties['species']
            coordinates = properties['coordinates'].float()
            true_energies = properties['energies'].float()
            _, predicted_energies = model((species, coordinates))
            total_mse += mse_sum(predicted_energies, true_energies).item()
            count += predicted_energies.shape[0]
    model.train(True)
    return hartree2kcalmol(math.sqrt(total_mse / count))


###############################################################################
# We will also use TensorBoard to visualize our training process
tensorboard = torch.utils.tensorboard.SummaryWriter()

###############################################################################
# Finally, we come to the training loop.
#
# In this tutorial, we are setting the maximum epoch to a very small number,
# only to make this demo terminate fast. For serious training, this should be
# set to a much larger value
mse = torch.nn.MSELoss(reduction='none')

print("training starting from epoch", AdamW_scheduler.last_epoch + 1)
max_epochs = 100
early_stopping_learning_rate = 1.0E-5
best_model_checkpoint = 'best.pt'

for _ in range(AdamW_scheduler.last_epoch + 1, max_epochs):
    rmse = validate()
    print('RMSE:', rmse, 'at epoch', AdamW_scheduler.last_epoch + 1)

    learning_rate = AdamW.param_groups[0]['lr']

    if learning_rate < early_stopping_learning_rate:
        break

    # checkpoint
    if AdamW_scheduler.is_better(rmse, AdamW_scheduler.best):
        torch.save(nn.state_dict(), best_model_checkpoint)

    AdamW_scheduler.step(rmse)
    SGD_scheduler.step(rmse)

    tensorboard.add_scalar('validation_rmse', rmse, AdamW_scheduler.last_epoch)
    tensorboard.add_scalar('best_validation_rmse', AdamW_scheduler.best, AdamW_scheduler.last_epoch)
    tensorboard.add_scalar('learning_rate', learning_rate, AdamW_scheduler.last_epoch)

    for i, properties in tqdm.tqdm(
        enumerate(training),
        total=len(training),
        desc="epoch {}".format(AdamW_scheduler.last_epoch)
    ):
        properties = {k: v.to(device, non_blocking=True) for k, v in properties.items()}
        properties = transform(properties)
        species = properties['species']
        coordinates = properties['coordinates'].float()
        true_energies = properties['energies'].float()
        num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
        _, predicted_energies = model((species, coordinates))

        loss = (mse(predicted_energies, true_energies) / num_atoms.sqrt()).mean()

        AdamW.zero_grad()
        SGD.zero_grad()
        loss.backward()
        AdamW.step()
        SGD.step()

        # write current batch loss to TensorBoard
        tensorboard.add_scalar('batch_loss', loss, AdamW_scheduler.last_epoch * len(training) + i)

    torch.save({
        'nn': nn.state_dict(),
        'AdamW': AdamW.state_dict(),
        'SGD': SGD.state_dict(),
        'AdamW_scheduler': AdamW_scheduler.state_dict(),
        'SGD_scheduler': SGD_scheduler.state_dict(),
    }, latest_checkpoint)
