r"""Command line interface entrypoints"""

from typing import Annotated
import matplotlib.pyplot as plt
import pandas as pd
from enum import Enum
import jinja2
import sys
import typing_extensions as tpx
import json
import tempfile
import uuid
import subprocess
from copy import deepcopy
import hashlib
import shutil
import typing as tp
from typing import Optional
from pathlib import Path

from typer import Argument, Option, Typer, Abort

from torchani.train.paths import MODELS_PATH, DataKind, select_subdirs
from torchani.train._lit_training import train_lit_model
from torchani.train.config import (
    load_state_dict,
    FinetuneConfig,
    TrainConfig,
    DatasetConfig,
    AccelConfig,
    ModelConfig,
    LossConfig,
    OptimizerConfig,
    SchedulerConfig,
    SrcConfig,
)
from torchani.train.display import ls
from torchani.train.defaults import (
    resolve_options,
    parse_scheduler_str,
    parse_optimizer_str,
)
from rich.console import Console

console = Console()

app = Typer(
    rich_markup_mode="markdown",
    help=r"""
    ## ANI

    Utility for generating a fine-tuned models from pre trained ANI style
    models, given a set of reference structures
    """,
)


class DTypeKind(Enum):
    F32 = "f32"
    F64 = "f64"


class DeviceKind(Enum):
    CUDA = "cuda"
    CPU = "cpu"


@app.command()
def save(
    name: Annotated[
        str,
        Option(
            "-n",
            "--ens-name",
            help="Name of ensemble or saved model. CamelCase recommended",
        ),
    ] = "Ensemble",
    desc: Annotated[
        str,
        Option(
            "-d",
            "--description",
            help="Description of the model",
        ),
    ] = "Custom ANI model",
    ftune_names_or_idxs: Annotated[
        Optional[tp.List[str]],
        Option("-f", "--ftune-run", help="Name|idx of train run"),
    ] = None,
    ptrain_names_or_idxs: Annotated[
        Optional[tp.List[str]],
        Option("-t", "--train-run", help="Name|idx of ftune run"),
    ] = None,
) -> None:
    r"""Extract and save a model or an ensemble an ensemble from a set of models"""
    if ptrain_names_or_idxs is None:
        ptrain_names_or_idxs = []
    if ftune_names_or_idxs is None:
        ftune_names_or_idxs = []
    ptrain_paths = select_subdirs(
        ptrain_names_or_idxs,
        kind=DataKind.TRAIN,
    )
    ftune_paths = select_subdirs(
        ftune_names_or_idxs,
        kind=DataKind.FTUNE,
    )
    paths = deepcopy(ptrain_paths)
    paths.extend(ftune_paths)
    hasher = hashlib.shake_128()
    for p in paths:
        hasher.update(p.name.encode("utf-8"))
    ckpt_paths = [(p / "best-model") / "best.ckpt" for p in paths]

    import torch
    from torchani.utils import merge_state_dicts

    state_dict = merge_state_dicts(ckpt_paths)

    _hash = hasher.hexdigest(4)
    config = TrainConfig.from_json_file(paths[0] / "config.json")
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(Path(__file__).parent / "templates/"),
        undefined=jinja2.StrictUndefined,
        autoescape=jinja2.select_autoescape(),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    tmpl = env.get_template("custom.py.jinja").render(
        name=name,
        desc=desc,
        ensemble_size=len(paths),
        lot=config.model.lot,
        symbols=config.model.symbols,
        arch_fn=config.model.arch_fn,
        arch_opts=config.model.options,
    )
    path = MODELS_PATH / f"{name}-{_hash}"
    path.mkdir(exist_ok=True, parents=True)
    src_config = SrcConfig(
        train_src=list(p.name for p in ptrain_paths),
        ftune_src=list(p.name for p in ftune_paths),
    )
    src_config.to_json_file(path / "src_config.json")
    (path / "model.py").write_text(tmpl)
    torch.save(state_dict, path / "model.pt")


@app.command()
def restart(
    ftune_name_or_idx: Annotated[
        str, Option("-f", "--ftune-run", help="Name or idx of ftune run")
    ] = "",
    ptrain_name_or_idx: Annotated[
        str, Option("-t", "--train-run", help="Name or idx of train run")
    ] = "",
    slurm: tpx.Annotated[
        str,
        Option("--slurm"),
    ] = "",
    slurm_gpu: tpx.Annotated[
        str,
        Option("--slurm-gpu"),
    ] = "",
    num_workers: tpx.Annotated[
        int,
        Option("--num-workers"),
    ] = 1,
    max_epochs: Annotated[
        Optional[int],
        Option("--max-epochs", help="Max epochs to train"),
    ] = None,
    # Loss config
    xc: Annotated[
        bool, Option("--xc/ ", help="Train to XC energies", rich_help_panel="Loss")
    ] = False,
    no_sqrt_atoms: Annotated[
        bool,
        Option(
            "--no-sqrt-atoms/ ",
            help="Divide energy loss by atoms instead of sqrt(atoms)",
            rich_help_panel="Loss",
        ),
    ] = False,
    energies: Annotated[
        float, Option("-e", "--energies", help="Energy factor", rich_help_panel="Loss")
    ] = 0.0,
    forces: Annotated[
        float, Option("-f", "--forces", help="Force factor", rich_help_panel="Loss")
    ] = 0.0,
    dipoles: Annotated[
        float, Option("-m", "--dipoles", help="Dipole factor", rich_help_panel="Loss")
    ] = 0.0,
    atomic_volumes: Annotated[
        float,
        Option(
            "-V",
            "--atomic-volumes",
            help="Atomic volumes factor",
            rich_help_panel="Loss",
        ),
    ] = 0.0,
    atomic_charges: Annotated[
        float,
        Option(
            "-q",
            "--atomic-charges",
            help="Atomic charges factor",
            rich_help_panel="Loss",
        ),
    ] = 0.0,
    total_charge: Annotated[
        float,
        Option("--total-q", help="Total charge factor", rich_help_panel="Loss"),
    ] = 0.0,
    swa: tpx.Annotated[
        bool,
        Option(
            "--swa/--no-swa",
            help="Perform SWA for the remaining epochs instead of normal training",
        ),
    ] = False,
    verbose: Annotated[bool, Option("-v/ ", "--verbose/ ")] = False,
) -> None:
    r"""Continue a checkpointed run"""
    if (
        ftune_name_or_idx
        and ptrain_name_or_idx
        or not (ftune_name_or_idx or ptrain_name_or_idx)
    ):
        console.print("One and only one of -f and -t should be specified", style="red")
        raise Abort()
    name_or_idx = ftune_name_or_idx or ptrain_name_or_idx
    kind = DataKind.FTUNE if ftune_name_or_idx else DataKind.TRAIN

    terms_and_factors: tp.Optional[tp.Dict[str, float]] = {}
    assert isinstance(terms_and_factors, dict)
    if energies > 0.0:
        label = "EnergiesXC" if xc else "Energies"
        terms_and_factors[label if no_sqrt_atoms else f"{label}SqrtAtoms"] = energies
    if forces > 0.0:
        terms_and_factors["Forces"] = forces
    if dipoles > 0.0:
        terms_and_factors["Dipoles"] = dipoles
    if atomic_charges > 0.0:
        terms_and_factors["AtomicCharges"] = atomic_charges
    if atomic_volumes > 0.0:
        terms_and_factors["AtomicVolumes"] = atomic_volumes
    if total_charge > 0.0:
        terms_and_factors["TotalCharge"] = total_charge
    if not terms_and_factors:
        terms_and_factors = None

    path = select_subdirs((name_or_idx,), kind=kind)[0] / "config.json"
    if not path.is_file():
        console.print(f"{path} is not a file dir", style="red")
        raise Abort()

    config = TrainConfig.from_json_file(path)
    # TODO: Remove duplicated code
    if slurm:
        if slurm == "moria":
            assert slurm_gpu in ["v100", "gp100", "titanv", "gtx1080ti", ""]
        elif slurm == "hpg":
            assert slurm_gpu in ["b200", "l4", ""]
        else:
            console.print(f"Unknown cluster {slurm}", style="red")
            raise Abort()
        slurm_gpu = f"{slurm_gpu}:1" if slurm_gpu else "1"

        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(Path(__file__).parent / "templates/"),
            undefined=jinja2.StrictUndefined,
            autoescape=jinja2.select_autoescape(),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        arg_list = sys.argv[1:]
        for j, arg in enumerate(deepcopy(arg_list)):
            # re-introduce quotes in strings
            if arg in ["--prof", "--ftune-from", "--monitor", "--lot"]:
                arg_list[j + 1] = f"'{arg_list[j + 1]}'"
            if arg == "--slurm":
                arg_list[j] = ""
                arg_list[j + 1] = ""
            if arg == "--slurm-gpu":
                arg_list[j] = ""
                arg_list[j + 1] = ""
        args = " ".join(arg_list)
        tmpl = env.get_template(f"{slurm}.slurm.sh.jinja").render(
            name=str(config.path.name),
            num_workers=num_workers,
            gpu=slurm_gpu,
            args=args,
        )
        unique_id = config.path.name.split("-")[-1]
        j = 0
        input_dir = Path(Path.home(), "IO", "ani", f"{unique_id}_v{j}")
        while input_dir.is_dir():
            j += 1
            input_dir = Path(Path.home(), "IO", "ani", f"{unique_id}_v{j}")
        input_dir.mkdir(exist_ok=False, parents=True)
        input_fpath = input_dir / f"{slurm}.slurm.sh"
        input_fpath.write_text(tmpl)
        console.print("Launching slurm script ...")
        subprocess.run(["sbatch", str(input_fpath)], cwd=input_dir, check=True)
        sys.exit(0)
    config.accel.num_workers = num_workers
    if max_epochs is not None:
        config.accel.max_epochs = max_epochs
    train_lit_model(config, restart=True, verbose=verbose)


ls = app.command(help="Display training and finetuning runs")(ls)


@app.command()
def rm(
    ftune_id: Annotated[
        Optional[tp.List[str]],
        Option("-f", "--ftune-run", help="Name|idx of ftune run"),
    ] = None,
    train_id: Annotated[
        Optional[tp.List[str]],
        Option("-t", "--train-run", help="Name|idx of train run"),
    ] = None,
    batch_id: Annotated[
        Optional[tp.List[str]], Option("-b", help="Name|idx of batched dataset")
    ] = None,
    ensemble_id: Annotated[
        Optional[tp.List[str]], Option("-e", help="Name|idx of ensemble")
    ] = None,
) -> None:
    r"""Delete one or more batched datasets, training, or finetuning run"""
    for selectors, dkind in zip(
        (ftune_id, train_id, batch_id, ensemble_id),
        (DataKind.FTUNE, DataKind.TRAIN, DataKind.BATCH, DataKind.MODELS),
    ):
        if selectors is not None:
            paths = select_subdirs(selectors, kind=dkind)
            for p in paths:
                shutil.rmtree(p)
                console.print(f"Removed {p.name}")
            console.print()


@app.command(help="Compare params of two models")
def compare(
    ftune_id: Annotated[
        str, Option("-f", "--ftune-run", help="Name|idx of ftune run")
    ] = "",
    train_id: Annotated[
        str, Option("-t", "--train-run", help="Name|idx of train run")
    ] = "",
) -> None:
    if (not (ftune_id or train_id)) or (ftune_id and train_id):
        console.print("One and only one of -t or -f has to be specified", style="red")
        raise Abort()
    kind = DataKind.FTUNE if ftune_id else DataKind.TRAIN
    root = select_subdirs(
        (train_id or ftune_id,),
        kind=kind,
    )[0]
    trained_path = root / "best-model"
    trained_state_dict = load_state_dict(trained_path / "best.ckpt")
    init_path = root / "init-model"
    init_state_dict = load_state_dict(init_path / "init.ckpt")

    for k in init_state_dict:
        if "weight" in k or "bias" in k:
            pretrained_param = init_state_dict[k]
            ftuned_param = trained_state_dict[k]
            diff = pretrained_param - ftuned_param
            if (diff == 0.0).all():
                console.print(f"No difference found for param {k}")
            else:
                diff = diff.abs()
                console.print(f"Difference found for param {k}")
                console.print(f"Min abs diff: {diff.min()}")
                console.print(f"Mean abs diff: {diff.mean()}")
                console.print(f"Max abs diff: {diff.max()}")
            console.print()


@app.command(help="Train from scratch or finetune an ANI-style model")
def train(
    batch_id: Annotated[str, Argument(help="Name|idx of the batched dataset")],
    fold_idx: Annotated[
        Optional[int],
        Option(
            "-i",
            "--fold-idx",
            help="Idx to use if training from folds",
            show_default=False,
        ),
    ] = None,
    name: Annotated[str, Option("-n", "--run-name", help="Name of run")] = "",
    slurm: tpx.Annotated[
        str,
        Option("--slurm"),
    ] = "",
    slurm_gpu: tpx.Annotated[
        str,
        Option("--slurm-gpu"),
    ] = "",
    num_workers: tpx.Annotated[
        int,
        Option("-n", "--num-workers"),
    ] = 1,
    allow_lot_mismatch: tpx.Annotated[
        bool,
        Option(
            "--allow-ds-model-lot-mismatch/ ",
            help="Allow model lot to differ from ds lot. Useful for transfer learning.",
        ),
    ] = False,
    auto_restart: Annotated[
        bool,
        Option("--auto-restart/ ", help="Auto restart runs that match a prev run"),
    ] = False,
    max_epochs: Annotated[
        int, Option("--max-epochs", help="Max epochs to train")
    ] = 1000,
    early_stop_patience: Annotated[
        int,
        Option(
            "--early-stop-patience",
            help="Max epochs without improving monitor metric before early stopping",
        ),
    ] = 50,
    # From-scratch specific config
    symbols: Annotated[
        str,
        Option(
            "--symbols",
            help="Chemical symbols the model will support."
            " The default is 'all present in the dataset'."
            " If specified, it should be a single string"
            " with symbols separated by commas. e.g. '--symbols H,C,N,O,F,S'",
            show_default=False,
            rich_help_panel="Arch",
        ),
    ] = "",
    lot: Annotated[
        str,
        Option(
            "--lot",
            help="LoT of the model. Default is 'dataset lot'.",
            show_default=False,
            rich_help_panel="Arch",
        ),
    ] = "",
    arch_fn: Annotated[
        str,
        Option(
            "-a",
            "--arch",
            help="Callable that creates the model",
            rich_help_panel="Arch",
        ),
    ] = "simple_ani",
    arch_options: Annotated[
        Optional[tp.List[str]],
        Option(
            "--ao",
            "--arch-opt",
            help="Options for arch fn, key=val fmt",
            rich_help_panel="Arch",
            show_default=False,
        ),
    ] = None,
    # LrSched config
    lrsched: Annotated[
        str,
        Option(
            "-s", "--sched", help="Type of lr-scheduler", rich_help_panel="LR scheduler"
        ),
    ] = "Plateau",
    lrsched_opts: Annotated[
        Optional[tp.List[str]],
        Option(
            "--so",
            "--sched-opt",
            help="Options for lr-scheduler, key=val fmt",
            rich_help_panel="LR scheduler",
            show_default=False,
        ),
    ] = None,
    # Optimizer config
    optim: Annotated[
        str,
        Option("-o", "--optim", help="Type of optimizer", rich_help_panel="Optimizer"),
    ] = "AdamW",
    optim_opts: Annotated[
        Optional[tp.List[str]],
        Option(
            "--oo",
            "--optim-opt",
            rich_help_panel="Optimizer",
            help="Options for optim, key=val fmt (lr, wd are separate)",
            show_default=False,
        ),
    ] = None,
    wd: Annotated[
        float,
        Option("--wd", help="Weight decay for optim", rich_help_panel="Optimizer"),
    ] = 1e-7,
    lr: Annotated[
        float,
        Option(
            "--lr",
            help="Initial lr. If ftune, used for the 'head'",
            rich_help_panel="Optimizer",
        ),
    ] = 5e-4,
    # Loss config
    xc: Annotated[
        bool, Option("--xc/ ", help="Train to XC energies", rich_help_panel="Loss")
    ] = False,
    no_sqrt_atoms: Annotated[
        bool,
        Option(
            "--no-sqrt-atoms/ ",
            help="Divide energy loss by atoms instead of sqrt(atoms)",
            rich_help_panel="Loss",
        ),
    ] = False,
    energies: Annotated[
        float, Option("-e", "--energies", help="Energy factor", rich_help_panel="Loss")
    ] = 1.0,
    forces: Annotated[
        float, Option("-f", "--forces", help="Force factor", rich_help_panel="Loss")
    ] = 0.0,
    dipoles: Annotated[
        float, Option("-m", "--dipoles", help="Dipole factor", rich_help_panel="Loss")
    ] = 0.0,
    atomic_volumes: Annotated[
        float,
        Option(
            "-V",
            "--atomic-volumes",
            help="Atomic volumes factor",
            rich_help_panel="Loss",
        ),
    ] = 0.0,
    atomic_charges: Annotated[
        float,
        Option(
            "-q",
            "--atomic-charges",
            help="Atomic charges factor",
            rich_help_panel="Loss",
        ),
    ] = 0.0,
    total_charge: Annotated[
        float,
        Option("--total-q", help="Total charge factor", rich_help_panel="Loss"),
    ] = 0.0,
    monitor: Annotated[
        str,
        Option(
            "--monitor",
            help="Loss label to monitor during training."
            " Format is 'valid/rmse_energies', 'train/rmse_forces', etc."
            " If a single loss term is present, it is the valid/rmse_'loss-term'."
            " Otherwise, if 'forces' is a loss term, it is valid/rmse_forces."
            " Otherwise it must be explicitly specified.",
            rich_help_panel="Loss",
            show_default=False,
        ),
    ] = "valid/rmse_default",
    # Finetuning specific config
    dummy_ftune: tpx.Annotated[
        bool,
        Option("--dummy-ftune/--no-dummy-ftune"),
    ] = False,
    ftune_from: Annotated[
        str,
        Option(
            "--ftune-from",
            help="Name|idx of pretrain run. ani1x:idx, ... also supported",
            rich_help_panel="Finetuning",
            show_default=False,
        ),
    ] = "",
    num_head_layers: Annotated[
        Optional[int],
        Option(
            "--num-head",
            help="If fine-tuning, num. of head layers. Defaults to 1",
            rich_help_panel="Finetuning",
            show_default=False,
        ),
    ] = None,
    backbone_lr: Annotated[
        Optional[float],
        Option(
            "--backbone-lr",
            help="If fine-tuning, lr for backbone. Defaults to 0",
            rich_help_panel="Finetuning",
            show_default=False,
        ),
    ] = None,
    # Debug and profiling specific config
    debug: Annotated[
        bool,
        Option(
            "-g/ ", "--debug/ ", help="Run in debug config", rich_help_panel="Debug"
        ),
    ] = False,
    profiler: Annotated[
        Optional[str],
        Option(
            "--prof",
            help="Profiler, 'simple', 'advanced', or 'pytorch'",
            rich_help_panel="Debug",
            show_default=False,
        ),
    ] = None,
    limit: Annotated[
        Optional[int],
        Option(
            "--lim",
            help="Limit num batches or percent",
            rich_help_panel="Debug",
            show_default=False,
        ),
    ] = None,
    deterministic: Annotated[
        bool,
        Option(
            "--deterministic/ ",
            help="Deterministic training",
            rich_help_panel="Debug",
        ),
    ] = False,
    detect_anomaly: Annotated[
        bool,
        Option(
            "--detect-anomaly/ ",
            help="Detect anomalies during training",
            rich_help_panel="Debug",
        ),
    ] = False,
    device: tpx.Annotated[
        tp.Optional[DeviceKind],
        Option("-d", "--device", case_sensitive=False),
    ] = None,
    verbose: Annotated[
        bool, Option("-v/ ", "--verbose/ ", rich_help_panel="Debug")
    ] = False,
) -> None:

    import torch

    if device is None:
        device = DeviceKind.CUDA if torch.cuda.is_available() else DeviceKind.CPU

    batched_dataset_path = select_subdirs((batch_id,), kind=DataKind.BATCH)[0]
    ds_config_path = batched_dataset_path / "ds_config.json"
    ds_config = DatasetConfig.from_json_file(ds_config_path)
    ds_config.fold_idx = "train" if fold_idx is None else fold_idx

    if fold_idx is not None:
        if not name:
            name = "train" if not ftune_from else "ftune"
        name = f"{str(fold_idx).zfill(2)}-{name}"

    with open(ds_config.path / "creation_log.json", mode="rt") as f:
        ds_symbols = json.load(f)["symbols"]

    if debug:
        console.print("Debugging enabled:")
        if name == "train":
            _uuid = uuid.uuid4().hex[:8]
            console.print(f"    - Name set to 'debug-{_uuid}'")
            name = f"debug-{_uuid}"
        if max_epochs == 1000:
            max_epochs = 3
            console.print(f"    - Max epochs set to {max_epochs}")
        if limit is None:
            limit = 3
            console.print(f"    - Batch limit set to {limit}")
        console.print("    - Verbosity increased")
        verbose = True
        console.print("    - Deterministic mode set")
        deterministic = True
        console.print("    - Anomaly detection mode set")
        detect_anomaly = True

    terms_and_factors: tp.Dict[str, float] = {}
    if energies > 0.0:
        label = "EnergiesXC" if xc else "Energies"
        terms_and_factors[label if no_sqrt_atoms else f"{label}SqrtAtoms"] = energies
    if forces > 0.0:
        terms_and_factors["Forces"] = forces
    if dipoles > 0.0:
        terms_and_factors["Dipoles"] = dipoles
    if atomic_charges > 0.0:
        terms_and_factors["AtomicCharges"] = atomic_charges
    if atomic_volumes > 0.0:
        terms_and_factors["AtomicVolumes"] = atomic_volumes
    if total_charge > 0.0:
        terms_and_factors["TotalCharge"] = total_charge

    lrsched = parse_scheduler_str(lrsched)
    optim = parse_optimizer_str(optim)
    lrsched_opts = lrsched_opts or []
    optim_opts = (optim_opts or []) + [f"lr={lr}", f"weight_decay={wd}"]

    if lr <= 0.0:
        console.print("lr must be strictly positive", style="red")
        raise Abort()

    # Finetune config
    if ftune_from:
        if arch_fn != "simple_ani" or arch_options or symbols:
            console.print(
                "Don't specify 'arch', 'arch-opts' or 'symbols' for ftune", style="red"
            )
            raise Abort()
        backbone_lr = backbone_lr or 0.0
        num_head_layers = num_head_layers or 1
        # Validation
        if backbone_lr < 0.0:
            console.print("backbone lr must be >= 0", style="red")
            raise Abort()
        if backbone_lr > lr:
            console.print("Backbone lr must be <= head lr", style="red")
            raise Abort()
        if num_head_layers < 1:
            console.print("There must be at least one head layer", style="red")
            raise Abort()
        # Create finetune and model configs
        if ftune_from.split(":")[0] in ("ani1x", "ani2x", "ani1ccx", "anidr", "aniala"):
            ptrain_name = ftune_from
            model_config = ModelConfig.from_builtin(ftune_from)
            raw_ptrain_state_dict_path = ""
        else:
            try:
                _path = select_subdirs((ftune_from,), kind=DataKind.TRAIN)[0]
            except RuntimeError:
                _path = select_subdirs((ftune_from,), kind=DataKind.FTUNE)[0]

            ptrain_name = _path.name
            model_config = TrainConfig.from_json_file(_path / "config.json").model
            raw_ptrain_state_dict_path = str(Path(_path, "best-model", "best.ckpt"))

            if not Path(raw_ptrain_state_dict_path).is_file():
                console.print(
                    f"{raw_ptrain_state_dict_path} is not a valid ckpt", style="red"
                )
                raise Abort()
        ftune_config = FinetuneConfig(
            pretrained_name=ptrain_name,
            raw_state_dict_path=raw_ptrain_state_dict_path,
            num_head_layers=num_head_layers,
            backbone_lr=backbone_lr,
            dummy_ftune=dummy_ftune,
        )
    else:
        ftune_config = None
        model_config = ModelConfig(
            lot=lot or ds_config.lot,
            symbols=symbols.split(",") if symbols else ds_symbols,
            arch_fn=arch_fn,
            options=resolve_options(arch_options or (), arch_fn),
        )
    if not allow_lot_mismatch and model_config.lot != ds_config.lot:
        console.print(
            "Model LoT must match dataset LoT unless --allow-ds-model-lot-mismatch",
            style="red",
        )
        raise Abort()

    if not set(model_config.symbols).issubset(ds_symbols):
        console.print(
            f"Not all ds symbols {ds_symbols} are supported by the model."
            f"Model supports {model_config.symbols}",
            style="red",
        )
        raise Abort()

    config = TrainConfig(
        name=name,
        debug=debug,
        ds=ds_config,
        monitor_label=monitor,
        ftune=ftune_config,
        model=model_config,
        loss=LossConfig(terms_and_factors=terms_and_factors),
        optim=OptimizerConfig(resolve_options(optim_opts, optim), optim),
        scheduler=SchedulerConfig(resolve_options(lrsched_opts, lrsched), lrsched),
        accel=AccelConfig(
            device=device.value,
            max_batches_per_packet=100,
            limit=limit,
            deterministic=deterministic,
            detect_anomaly=detect_anomaly,
            max_epochs=max_epochs,
            early_stop_patience=early_stop_patience,
            profiler=profiler,
            num_workers=num_workers,
        ),
    )

    # Re-run everything after the train config has been set up, to prevent potential
    # issues
    if slurm:
        if slurm == "moria":
            assert slurm_gpu in ["v100", "gp100", "titanv", "gtx1080ti", ""]
        elif slurm == "hpg":
            assert slurm_gpu in ["b200", "l4", ""]
        else:
            console.print(f"Unknown cluster {slurm}", style="red")
            raise Abort()
        slurm_gpu = f"{slurm_gpu}:1" if slurm_gpu else "1"

        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(Path(__file__).parent / "templates/"),
            undefined=jinja2.StrictUndefined,
            autoescape=jinja2.select_autoescape(),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        arg_list = sys.argv[1:]
        for j, arg in enumerate(deepcopy(arg_list)):
            # re-introduce quotes in strings
            if arg in ["--prof", "--ftune-from", "--monitor", "--lot"]:
                arg_list[j + 1] = f"'{arg_list[j + 1]}'"
            if arg == "--slurm":
                arg_list[j] = ""
                arg_list[j + 1] = ""
            if arg == "--slurm-gpu":
                arg_list[j] = ""
                arg_list[j + 1] = ""
        args = " ".join(arg_list)
        tmpl = env.get_template(f"{slurm}.slurm.sh.jinja").render(
            num_workers=num_workers,
            name=str(config.path.name),
            gpu=slurm_gpu,
            args=args,
        )
        unique_id = config.path.name.split("-")[-1]
        j = 0
        input_dir = Path(Path.home(), "IO", "ani", f"{unique_id}_v{j}")
        while input_dir.is_dir():
            j += 1
            input_dir = Path(Path.home(), "IO", "ani", f"{unique_id}_v{j}")
        input_dir.mkdir(exist_ok=False, parents=True)
        input_fpath = input_dir / f"{slurm}.slurm.sh"
        input_fpath.write_text(tmpl)
        console.print("Launching slurm script ...")
        subprocess.run(["sbatch", str(input_fpath)], cwd=input_dir, check=True)
        sys.exit(0)
    train_lit_model(config, allow_restart=auto_restart, verbose=verbose)


@app.command()
def bench(
    builtin: Annotated[
        str,
        Argument(
            help="Built-in ANI ds name to benchmark on. Format is 'name:lot'",
        ),
    ],
    model_name: tpx.Annotated[
        str,
        Option("-m", "--model-name"),
    ] = "ANI2x",
    device: tpx.Annotated[
        tp.Optional[DeviceKind],
        Option("-d", "--device", case_sensitive=False),
    ] = None,
    dtype: tpx.Annotated[
        tp.Optional[DTypeKind],
        Option("-d", "--dtype", case_sensitive=False),
    ] = None,
    forces: tpx.Annotated[
        bool,
        Option("-f/-F", "--forces/--no-forces", help="Also benchmark forces"),
    ] = True,
    chunk_size: tpx.Annotated[
        int,
        Option("-c", "--chunk-size"),
    ] = 2500,
    convert: Annotated[
        bool,
        Option("--convert/--no-convert"),
    ] = True,
    convert_model_to_ev: Annotated[
        bool,
        Option("--convert-model-to-ev/--no-convert-model-to-ev"),
    ] = False,
) -> None:
    r"""Benchmark model on a given dataset"""
    import math
    import dataclasses
    import torch
    import torchani
    from torchani.annotations import Device, DType
    from torchani.units import HARTREE_TO_KCALPERMOL, HARTREE_TO_EV

    if not convert or convert_model_to_ev:
        factor = 1.
    else:
        factor = HARTREE_TO_KCALPERMOL

    from tqdm import tqdm

    def parse_device_and_dtype(
        device: tp.Optional[DeviceKind] = None,
        dtype: tp.Optional[DTypeKind] = None,
    ) -> tp.Tuple[Device, DType]:
        if dtype is None:
            dtype = DTypeKind.F32

        if dtype is DTypeKind.F32:
            _dtype = torch.float32
        elif dtype is DTypeKind.F64:
            _dtype = torch.float64

        if device is DeviceKind.CUDA:
            _device = "cuda"
        elif device is DeviceKind.CPU:
            _device = "cpu"
        else:
            _device = "cuda" if torch.cuda.is_available() else "cpu"
        return _device, _dtype

    _device, _dtype = parse_device_and_dtype(device, dtype)
    if Path(builtin).exists():
        if Path(builtin).is_dir():
            ds = torchani.datasets.ANIDataset.from_dir(builtin)
        else:
            ds = torchani.datasets.ANIDataset(builtin)
        bench_set = "Custom"
        lot = "unk"
    else:
        bench_set, lot = builtin.split(":")
        ds = getattr(torchani.datasets, bench_set)(lot=lot)
    ds_key = f"{bench_set}-{lot}"
    model = getattr(torchani.models, model_name)(device=_device, dtype=_dtype)

    @dataclasses.dataclass
    class Metrics:
        force_rmse: float = 0.0
        force_mae: float = 0.0
        rmse: float = 0.0
        mae: float = 0.0
        num: int = 0

    _results: tp.Dict[str, Metrics] = {s: Metrics() for s in ds.store_names}
    _results["total"] = Metrics()
    props = ["coordinates", "energies", "species"]
    if forces:
        props.append("forces")
    for k, idx, v in tqdm(
        ds.chunked_items(max_size=chunk_size, properties=props),
        total=ds.num_chunks(chunk_size),
    ):
        parts = k.split("/")
        if len(parts) > 1:
            store_name = parts[0]
        else:
            store_name = ds.store_names[0]
        v["species"] = v["species"].to(device=_device)
        v["coordinates"] = v["coordinates"].to(device=_device, dtype=_dtype)
        v["energies"] = v["energies"].to(device=_device, dtype=_dtype).view(-1)
        if forces:
            num_atoms = (v["species"] != -1).sum(-1)
            v["forces"] = v["forces"].to(device=_device, dtype=_dtype)
            v["coordinates"].requires_grad_(True)

        energies = model((v["species"], v["coordinates"])).energies.view(-1)
        if convert_model_to_ev:
            energies = energies * HARTREE_TO_EV
        delta = torch.abs(v["energies"] - energies) * factor
        sum_sq_err = (delta**2).sum().item()
        sum_abs_err = delta.sum().item()

        _results[store_name].rmse += sum_sq_err
        _results["total"].rmse += sum_sq_err
        _results[store_name].mae += sum_abs_err
        _results["total"].mae += sum_abs_err
        _results[store_name].num += len(delta)
        _results["total"].num += len(delta)

        if forces:
            _forces = -torch.autograd.grad(energies.sum(), v["coordinates"])[0]
            delta_f = torch.abs(v["forces"] - _forces) * factor
            sum_sq_err_f = ((delta_f**2).sum((-1, -2)) / num_atoms / 3).sum().item()
            sum_abs_err_f = (delta_f.sum((-1, -2)) / num_atoms / 3).sum().item()

            _results[store_name].force_rmse += sum_sq_err_f
            _results["total"].force_rmse += sum_sq_err_f
            _results[store_name].force_mae += sum_abs_err_f
            _results["total"].force_mae += sum_abs_err_f
    _results = {
        k: Metrics(
            force_rmse=math.sqrt(v.force_rmse / v.num),
            force_mae=v.force_mae / v.num,
            rmse=math.sqrt(v.rmse / v.num),
            mae=v.mae / v.num,
            num=v.num,
        )
        for k, v in _results.items()
    }

    with open(Path(f"{ds_key}.{model_name}.json"), mode="wt", encoding="utf-8") as f:
        output = {}
        for k, v in _results.items():
            _dict = dataclasses.asdict(v)
            if not forces:
                _dict.pop("force_rmse")
                _dict.pop("force_mae")
            output[k] = _dict
        json.dump(output, f, indent=4)


@app.command()
def tb(
    ftune_name_or_idx: Annotated[
        Optional[tp.List[str]],
        Option("-f", "--ftune-run", help="Name|idx of ftune run"),
    ] = None,
    ptrain_name_or_idx: Annotated[
        Optional[tp.List[str]],
        Option("-t", "--train-run", help="Name|idx of train run"),
    ] = None,
) -> None:
    r"""Visualize train|ftune process with tensorboard"""
    paths = []
    for selectors, dkind in zip(
        (
            ftune_name_or_idx,
            ptrain_name_or_idx,
        ),
        (
            DataKind.FTUNE,
            DataKind.TRAIN,
        ),
    ):
        if selectors is not None:
            paths.extend(select_subdirs(selectors, kind=dkind))
    with tempfile.TemporaryDirectory() as d:
        for path in paths:
            run_subdir = Path(d, path.name)
            run_subdir.mkdir()
            tb_path = path / "tb-logs"
            for version_dir in sorted(tb_path.glob("version_*")):
                for events_file in sorted(version_dir.iterdir()):
                    if not events_file.name.startswith("events.out.tfevents"):
                        continue
                    tmp_events_symlink = run_subdir / events_file.name
                    tmp_events_symlink.symlink_to(events_file)
        subprocess.run(["tensorboard", "--logdir", d])


@app.command()
def plot(
    ftune_name_or_idx: Annotated[
        Optional[tp.List[str]],
        Option("-f", "--ftune-run", help="Name|idx of ftune run"),
    ] = None,
    ptrain_name_or_idx: Annotated[
        Optional[tp.List[str]],
        Option("-t", "--train-run", help="Name|idx of train run"),
    ] = None,
    labels: tpx.Annotated[
        tp.Optional[tp.List[str]],
        Option("-l", "--label"),
    ] = None,
    limits: tpx.Annotated[
        tp.Optional[tp.List[str]],
        Option("--lim"),
    ] = None,
    validation: tpx.Annotated[
        bool,
        Option("--val/--train"),
    ] = True,
) -> None:
    prefix = "train" if not validation else "valid"
    r"""Plot a specific metric"""
    if labels is None:
        labels = [
            "mae_energies_kcal|mol",
            "mae_forces_kcal|mol|ang",
            "rmse_energies_kcal|mol",
            "rmse_forces_kcal|mol|ang",
        ]
    if limits is None:
        limit_tuples = [(0.8, 4), (0.8, 2), (1.3, 4), (2.75, 4)]
    else:
        limit_tuples = tp.cast(
            tp.List[tp.Tuple[float, int]],
            [tuple(map(int, lim.split(","))) for lim in limits],
        )
    if len(limit_tuples) != len(labels):
        raise ValueError("Limit tuples and labels must have the same length")
    paths = []
    for selectors, dkind in zip(
        (
            ftune_name_or_idx,
            ptrain_name_or_idx,
        ),
        (
            DataKind.FTUNE,
            DataKind.TRAIN,
        ),
    ):
        if selectors is not None:
            paths.extend(select_subdirs(selectors, kind=dkind))

        dfs: tp.Dict[str, pd.DataFrame] = {}
        for path in paths:
            csv_path = path / "csv-logs"
            _df = []
            for version_dir in sorted(csv_path.glob("version_*")):
                metrics = version_dir / "metrics.csv"
                if metrics.is_file():
                    _df.append(pd.read_csv(metrics))
            dfs[path.name] = pd.concat(_df)

        if dfs:
            for label, lim in zip(labels, limit_tuples):
                fig, ax = plt.subplots()
                for j, (name, df) in enumerate(dfs.items()):
                    ax.plot(df["epoch"], df[f"{prefix}/{label}"], label=f"Model {j}")
                label = label.replace(
                    "mae_energies_kcal|mol", r"$E_{\text{MAE}}$ (kcal/mol)"
                )
                label = label.replace(
                    "mae_forces_kcal|mol|ang", r"$F_{\text{MAE}}$ (kcal/mol/\AA{})"
                )
                label = label.replace(
                    "rmse_energies_kcal|mol", r"$E_{\text{RMSE}}$ (kcal/mol)"
                )
                label = label.replace(
                    "rmse_forces_kcal|mol|ang", r"$F_{\text{RMSE}}$ (kcal/mol/\AA{})"
                )
                ax.set_ylabel(f"{label}")
                ax.set_xlabel(r"Epoch")
                ax.set_ylim(lim[0], lim[1])
                ax.legend()
                plt.show()
