import torch
import torchani
from ANIv import ANIv
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np


device = "cuda"


def validate(model, val_data_loader):
    """
    Computes RMSE for predicted volumes using a validation data set, 
    does not update gradients in the model.
    """
    mses = []
    count = 0
    mse_fn = torch.nn.MSELoss(reduction="mean")
    model.eval()  # deactivates training behaviours like droput, etc
    with torch.no_grad():
        for batch in val_data_loader:
            data = {  # each batch gets loaded to GPU from disk
                k: v.to(device, non_blocking=True) for k, v in batch.items()
            }
            pred_volumes = model(data["species"], data["coordinates"])
            mse = mse_fn(pred_volumes, data["atomic_volumes_mbis"]).item()
            mses.append(mse)
            count += 1
    model.train()
    return np.sqrt(sum(mses) / count)

############################## First create the model #####################################


symbols = torchani.utils.SYMBOLS_2X

species_converter = torchani.nn.SpeciesConverter(symbols)

aev_computer = torchani.aev.AEVComputer.like_2x(strategy="cuaev")

volume_networks = torchani.nn.ANINetworks.build(
    symbols=symbols,
    in_dim=1008,
    dims={
        "H": (256, 192, 160),
        "C": (224, 192, 160),
        "N": (192, 160, 128),
        "O": (192, 160, 128),
        "S": (160, 128, 96),
        "F": (160, 128, 96),
        "Cl": (160, 128, 96),
    },
    out_dim=1,
    bias=True,
)


model = ANIv(
    species_converter,
    aev_computer,
    volume_networks,
).to(dtype=torch.float32, device=device)

model.train()

###################################### Create GPU DataLoaders  ################

train_ds = torchani.datasets.ANIBatchedDataset(
    store_dir="/home/jolmos/ani_training/datasets/batched_dataset/",
    split="training",
    properties=["coordinates", "species", "atomic_volumes_mbis"],
)

validate_ds = torchani.datasets.ANIBatchedDataset(
    store_dir="/home/jolmos/ani_training/datasets/batched_dataset/",
    split="validation",
    properties=["coordinates", "species", "atomic_volumes_mbis"],
)

training = train_ds.as_dataloader(
    pin_memory=True,
    shuffle=True,
)

validation = validate_ds.as_dataloader(
    pin_memory=True,
    shuffle=False,
)

##################################### Prepare training ###########################

optimizer = torch.optim.AdamW(
    params=model.parameters(),
    lr=0.5e-3,
    weight_decay=1e-8,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=0.5,
    patience=5,
    threshold=0,
)


latest_training_state_ckpt_path = Path("./latest_training_state.pt").resolve()
best_model_state_ckpt_path = Path("./best_model_state.pt").resolve()


train_rmse = []
validate_rmse = []
max_epochs = 500
mse = torch.nn.MSELoss(reduction="mean")

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
        latest_training_state_ckpt_path,
    )


for epoch in range(scheduler.last_epoch, max_epochs + 1):
    for batch in training:
        batch = {
            k: v.to(device, non_blocking=True) for k, v in batch.items()
        }
        species = batch["species"]
        coords = batch["coordinates"]
        vols = batch["atomic_volumes_mbis"]

        optimizer.zero_grad()
        pred_vols = model(species, coords)
        loss = mse(pred_vols, vols)
        loss.backward()
        optimizer.step()

    train_rmse.append(np.sqrt(loss.item()))
    rmse = validate(model, validation)
    validate_rmse.append(rmse)

    if scheduler.is_better(rmse, scheduler.best):
        torch.save({"model": model.state_dict()}, best_model_state_ckpt_path)

    scheduler.step(rmse)

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        latest_training_state_ckpt_path,
    )

np.savetxt("train_rmse.dat", np.array(train_rmse))
np.savetxt("validate_rmse.dat", np.array(validate_rmse))

plt.plot(train_rmse, label="Training error")
plt.plot(validate_rmse, label="Validation error")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.savefig("training_results.png", format="png")
