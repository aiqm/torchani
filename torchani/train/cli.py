r"""Command line interface entrypoints"""

from typing import Annotated
import matplotlib.pyplot as plt
import pandas as pd
from enum import Enum
import jinja2
import sys
import typing_extensions as tpx
import subprocess
from copy import deepcopy
import hashlib
import shutil
import typing as tp
from typing import Optional
from pathlib import Path

from typer import Option, Typer, Abort

from torchani.paths import DataKind, select_subdirs

from torchani.train._lit_training import train_lit_model
from torchani.train.config import load_state_dict, TrainConfig, SrcConfig
from torchani.train.display import ls
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
            "--name",
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
    from torchani.paths import custom_models_dir

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
    path = custom_models_dir() / f"{name}-{_hash}"
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
