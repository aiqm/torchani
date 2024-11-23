r"""
Training an ANI network using a custom script
=============================================

This example shows how to use TorchANI to train a neural network potential.
"""
# %% To begin with, let's first import the modules and setup devices we will use
import math
from pathlib import Path

import torch
import torch.utils.tensorboard
from tqdm import tqdm

import torchani
from torchani.arch import ANI, simple_ani
from torchani.datasets import ANIDataset, ANIBatchedDataset, BatchedDataset
from torchani.units import hartree2kcalpermol
from torchani.grad import forces_for_training
# %%
# Device and dataset to run the training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ds = ANIDataset("../dataset/ani-1x/sample.h5")
# %%
# We prebatch the dataset to train with memory efficiency, keeping a good
# performance.
batched_dataset_path = Path("./batched_dataset").resolve()
if not batched_dataset_path.exists():
    torchani.datasets.create_batched_dataset(
        ds,
        dest_path=batched_dataset_path,
        batch_size=2560,
        splits={"training": 0.8, "validation": 0.2},
    )

train_ds: BatchedDataset = ANIBatchedDataset(batched_dataset_path, split="training")
valid_ds: BatchedDataset = ANIBatchedDataset(batched_dataset_path, split="validation")
# %%
# We use the pytorch DataLoader with multiprocessing to load the batches while we train
#
# For more info about the DataLoader and multiprocessing read
# https://pytorch.org/docs/stable/data.html
#
# CACHE saves all data in memory. It is very memory intensive but faster.
# Also, pin_memory is automatically performed by ANIBatchedDataset in the CACHE
# case, so it should be set to False for the DataLoader.
CACHE: bool = True
if CACHE:
    train_ds = train_ds.cache()
    valid_ds = valid_ds.cache()

training = train_ds.as_dataloader(num_workers=0)
validation = valid_ds.as_dataloader(num_workers=0)
# %%
# We can use the transforms module to modify the batches, the API for transforms is very
# similar to `torchvision's API <https://pytorch.org/vision/stable/transforms.html>`_
# with the difference that the transforms are applied to both target and inputs in all
# cases.
#
# Transform can be passed to the "transform" argument of ANIBatchedDataset to
# to be performed on-the-fly on CPU (slow if no CACHE)
#
# Transform can also be applied directly when training on GPU
#
# Transform can also be applied to a dataset when batching it, by using the
# inplace_transform argument of create_batched_dataset (Be careful, this may be
# error prone)
#
# In this case we wont apply any transform
#
# Lets generate a model from scratch. For simplicity we use PyTorch's default random
# initialization for the weights.
model = simple_ani(("H", "C", "N", "O"), lot="wb97x-631gd", repulsion=True)
# %%
# Set up of optimizer and lr-scheduler
optimizer = torch.optim.AdamW(
    params=model.neural_networks.parameters(),
    lr=0.5e-3,
    weight_decay=1e-6,
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=0.5,
    patience=100,
    threshold=0,
)
# %%
# We first read the checkpoint files to restart training. We use ``latest_traininig.pt``
# to store current training state.
latest_training_state_checkpoint_path = Path("./latest_training_state.pt").resolve()
best_model_state_checkpoint_path = Path("./best_model_state.pt").resolve()
if latest_training_state_checkpoint_path.exists():
    checkpoint = torch.load(latest_training_state_checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    optimizer.load_state_dict(checkpoint["optimizer"])

model.to(dtype=torch.float32, device=device)
# %%
# During training, we need to validate on validation set and if validation error
# is better than the best, then save the new best model to a checkpoint


def validate(model: ANI, validation: torch.utils.data.DataLoader) -> float:
    squared_error = 0.0
    count = 0
    model.train(False)
    with torch.no_grad():
        for properties in validation:
            properties = {
                k: v.to(device, non_blocking=True) for k, v in properties.items()
            }
            species = properties["species"]
            coordinates = properties["coordinates"].float()
            target_energies = properties["energies"].float()
            output = model((species, coordinates))
            predicted_energies = output.energies
            squared_error += (predicted_energies - target_energies).pow(2).sum().item()
            count += predicted_energies.shape[0]
    model.train(True)
    rmse = math.sqrt(squared_error / count)
    return hartree2kcalpermol(rmse)


# %%
# We will also use TensorBoard to visualize our training process
tensorboard = torch.utils.tensorboard.SummaryWriter()
# %%
# Criteria for stopping training
max_epochs = 5
min_learning_rate = 1.0e-10
# %%
# Epoch 0 is right before training starts
if scheduler.last_epoch == 0:
    rmse = validate(model, validation)
    print(f"Before training starts: Validation RMSE (kcal/mol) {rmse}")
    scheduler.step(rmse)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        latest_training_state_checkpoint_path,
    )
# %%
# Finally, we come to the training loop.
mse = torch.nn.MSELoss(reduction="none")
force_training = False
force_coefficient = 0.1
for epoch in range(scheduler.last_epoch, max_epochs + 1):
    # Stop training if the lr is below a given threshold
    if optimizer.param_groups[0]["lr"] < min_learning_rate:
        break

    # Loop over batches
    for batch in tqdm(
        training,
        total=len(training),
        desc=f"Epoch {epoch}",
    ):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        species = batch["species"]
        coordinates = batch["coordinates"].float()
        target_energies = batch["energies"].float()
        num_atoms = (species >= 0).sum(dim=1, dtype=target_energies.dtype)
        output = model((species, coordinates))
        predicted_energies = output.energies
        if force_training:
            target_forces = batch["forces"].float()
            predicted_forces = forces_for_training(predicted_energies, coordinates)
            energy_loss = (
                mse(predicted_energies, target_energies) / num_atoms.sqrt()
            ).mean()
            force_loss = (
                mse(predicted_forces, target_forces).sum(dim=(1, 2)) / num_atoms
            ).mean()
            loss = energy_loss + force_coefficient * force_loss
        else:
            loss = (mse(predicted_energies, target_energies) / num_atoms.sqrt()).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validate
    rmse = validate(model, validation)
    print(f"After epoch {epoch}: Validation RMSE (kcal/mol) {rmse}")

    # Checkpoint the model if the RMSE; improved
    if scheduler.is_better(rmse, scheduler.best):
        torch.save(model.state_dict(), best_model_state_checkpoint_path)

    # Step the epoch-scheduler
    scheduler.step(rmse)

    # Checkpoint the training state
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        latest_training_state_checkpoint_path,
    )

    # Log scalars
    tensorboard.add_scalar("validation_rmse_kcalpermol", rmse, epoch)
    tensorboard.add_scalar("best_validation_rmse_kcalpermol", scheduler.best, epoch)
    tensorboard.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)
    tensorboard.add_scalar("epoch_loss_square_ha", loss, epoch)
