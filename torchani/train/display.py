import json
import typing_extensions as tpx

from typer import Option
from rich.table import Table
from rich.console import Console

from torchani.train.config import DatasetConfig, TrainConfig, SrcConfig
from torchani.paths import train_dir, ftune_dir, batched_data_dir, custom_models_dir

console = Console()


def simplify_metric(arg: str) -> str:
    if arg.endswith("rmse_forces"):
        return "F_ha/ang"
    if arg.endswith("mae_forces"):
        return "Fmae_ha/ang"
    if arg.endswith("rmse_energies"):
        return "E_ha"
    if arg.endswith("mae_energies"):
        return "Emae_ha"
    return (
        arg.replace("valid/", "")
        .replace("train/", "")
        .replace("mae_energies", "Emae")
        .replace("rmse_energies", "E")
        .replace("mae_forces", "Fmae")
        .replace("rmse_forces", "F")
        .replace("|ang", "/ang")
        .replace("|mol", "/mol")
        .replace("rmse_atomic_charges", "Q")
        .replace("mae_atomic_charges", "Qmae")
        .replace("rmse_atomic_volumes", "V")
        .replace("mae_atomic_volumes", "Vmae")
        .replace("rmse_dipoles", "M")
        .replace("mae_dipoles", "Mmae")
        # bw compat
        .replace("train_", "")
        .replace("valid_", "")
    )


def ls(
    metric_fmt: tpx.Annotated[
        str,
        Option(
            "--fmt",
            help="Format to use for displaying metrics using python formatting lang",
        ),
    ] = ".2f",
    hparams: tpx.Annotated[
        bool,
        Option("-x/-X", "--hparams/--no-hparams", help="Equivalent to -a -s -o"),
    ] = False,
    sizes: tpx.Annotated[
        bool,
        Option(
            "-g/-G",
            "--sizes/--no-sizes",
            help="Show file sizes",
        ),
    ] = False,
    best: tpx.Annotated[
        bool,
        Option(
            "-k/-K",
            "--best/--no-best",
            help="Show best metrics",
        ),
    ] = True,
    latest: tpx.Annotated[
        bool,
        Option(
            "-c/-C",
            "--current/--no-current",
            help="Show current metrics",
        ),
    ] = False,
    mae: tpx.Annotated[
        bool,
        Option(
            "-m/-M",
            "--mae/--no-mae",
            help="Show MAE metrics",
        ),
    ] = False,
    hartree: tpx.Annotated[
        bool,
        Option(
            "--hartree/--no-hartree",
            help="Show metrics in hartrees",
        ),
    ] = False,
    arch_detail: tpx.Annotated[
        bool,
        Option(
            "-a/-A",
            "--arch/--no-arch",
            help="Show architecture options",
        ),
    ] = False,
    optim_detail: tpx.Annotated[
        bool,
        Option(
            "-o/-O",
            "--optim/--no-optim",
            help="Show optimizer options",
        ),
    ] = False,
    scheduler_detail: tpx.Annotated[
        bool,
        Option(
            "-s/-S",
            "--scheduler/--no-scheduler",
            help="Show scheduler options",
        ),
    ] = False,
) -> None:
    if hparams:
        arch_detail = True
        optim_detail = True
        scheduler_detail = True
    batch = sorted(batched_data_dir().iterdir())
    train = sorted(train_dir().iterdir())
    ftune = sorted(ftune_dir().iterdir())
    ensemble = sorted(custom_models_dir().iterdir())
    if batch:
        table = Table(title="Batched datasets", box=None)
        table.add_column("", style="magenta")
        table.add_column("data-name", style="magenta")
        table.add_column("divs", style="magenta")
        table.add_column("builtin-src")
        table.add_column("other-src")
        table.add_column("lot")
        table.add_column("conformers")
        table.add_column("symbols")
        table.add_column("properties")
        table.add_column("batch-size")
        table.add_column("batch-seed")
        table.add_column("divs-seed")
        if sizes:
            table.add_column("size (GB)")
        for j, p in enumerate(batch):
            try:
                ds_config = DatasetConfig.from_json_file(p / "ds_config.json")
                with open(p / "creation_log.json", mode="rt") as ft:
                    ds_log = json.load(ft)

                row_args = [
                    f"[bold]{j}[/bold]",
                    p.name,
                    (
                        f"{ds_config.folds}-folds"
                        if ds_config.folds is not None
                        else f"train:{ds_config.train_frac} valid:{ds_config.validation_frac}"  # noqa:E501
                    ),
                    " ".join(ds_config.data_names) or "--",
                    " ".join((p.stem for p in ds_config.src_paths)) or "--",
                    ds_config.lot,
                    str(ds_log["num_conformers"]),
                    " ".join(ds_log["symbols"]),
                    " ".join(ds_log["properties"]),
                    str(ds_config.batch_size),
                    str(ds_config.batch_seed),
                    str(ds_config.divs_seed),
                ]
                if sizes:
                    size = sum(f.stat().st_size for f in p.glob("**/*") if f.is_file())
                    row_args.append(format(size / 1024**3, ".1f"))
            except Exception:
                row_args = [f"[bold]{j}[/bold]", p.name, "???"]
            table.add_row(*row_args)
        console.print(table)
    else:
        console.print("(No batched datasets found)")

    if train:
        table = Table(title="Training runs", box=None)
        table.add_column("", style="green")
        table.add_column("run-name", style="green")
        table.add_column("data|div", style="magenta")
        if scheduler_detail:
            table.add_column("sched")
            table.add_column("sched-options")
        if optim_detail:
            table.add_column("optim")
            table.add_column("optim-options")
        table.add_column("lot")
        table.add_column("arch")
        if arch_detail:
            table.add_column("arch-options")
        table.add_column("wd")
        table.add_column("lr")
        table.add_column("epoch(best)")
        if best:
            table.add_column("best-valid")
            table.add_column("best-train")
        if latest:
            table.add_column("curr-valid")
            table.add_column("curr-train")
        for j, p in enumerate(train):
            try:
                config = TrainConfig.from_json_file(p / "config.json")
                with open(
                    (p / "best-model") / "metrics.json", mode="rt", encoding="utf-8"
                ) as ft:
                    metrics = json.load(ft)
                    best_epoch = metrics.pop("epoch")
                with open(
                    (p / "latest-model") / "metrics.json", mode="rt", encoding="utf-8"
                ) as ft:
                    latest_metrics = json.load(ft)
                    epoch = latest_metrics.pop("epoch")
                row_args = [
                    f"[bold]{j}[/bold]",
                    p.name,
                    f"{config.ds.path.name}|{config.ds.fold_idx}",
                    config.model.lot,
                    config.model.arch_fn,
                    f"{config.optim.weight_decay:.0e}",
                    f"{config.optim.lr:.0e}",
                    f"{epoch}({best_epoch})",
                ]
                if arch_detail:
                    options = " ".join(
                        sorted(f"{k}={v}" for k, v in config.model.options.items())
                    )
                    row_args.insert(4, options)
                if optim_detail:
                    options = " ".join(
                        sorted(
                            f"{k}={v}"
                            for k, v in config.optim.options.items()
                            if k not in ["lr", "weight_decay"]
                        )
                    )
                    row_args.insert(3, options or "--")
                    row_args.insert(3, config.optim.cls)
                if scheduler_detail:
                    options = " ".join(
                        sorted(f"{k}={v}" for k, v in config.scheduler.options.items())
                    )
                    row_args.insert(3, options or "--")
                    row_args.insert(3, config.scheduler.cls)
                if best:
                    if not mae:
                        metrics = {k: v for k, v in metrics.items() if "mae" not in k}
                    if not hartree:
                        metrics = {
                            k: v
                            for k, v in metrics.items()
                            if not (
                                ("energies" in k and "kcal" not in k)
                                or ("forces" in k and "kcal" not in k)
                            )
                        }
                    else:
                        metrics = {
                            k: v
                            for k, v in metrics.items()
                            if not (
                                ("energies" in k and "kcal" in k)
                                or ("forces" in k and "kcal" in k)
                            )
                        }
                    row_args.extend(
                        [
                            " ".join(
                                f"{simplify_metric(k)}={format(v, metric_fmt)}"
                                for k, v in metrics.items()
                                if "valid" in k
                            ),
                            " ".join(
                                f"{simplify_metric(k)}={format(v, metric_fmt)}"
                                for k, v in metrics.items()
                                if "train" in k
                            ),
                        ]
                    )
                if latest:
                    if not mae:
                        latest_metrics = {
                            k: v for k, v in latest_metrics.items() if "mae" not in k
                        }
                    if not hartree:
                        latest_metrics = {
                            k: v
                            for k, v in latest_metrics.items()
                            if not (
                                ("energies" in k and "kcal" not in k)
                                or ("forces" in k and "kcal" not in k)
                            )
                        }
                    else:
                        latest_metrics = {
                            k: v
                            for k, v in latest_metrics.items()
                            if not (
                                ("energies" in k and "kcal" in k)
                                or ("forces" in k and "kcal" in k)
                            )
                        }
                    row_args.extend(
                        [
                            " ".join(
                                f"{simplify_metric(k)}={format(v, metric_fmt)}"
                                for k, v in latest_metrics.items()
                                if "valid" in k
                            ),
                            " ".join(
                                f"{simplify_metric(k)}={format(v, metric_fmt)}"
                                for k, v in latest_metrics.items()
                                if "train" in k
                            ),
                        ]
                    )
            except Exception:
                row_args = [f"[bold]{j}[/bold]", p.name, "???"]
            table.add_row(*row_args)
        console.print(table)
    else:
        console.print("(No training runs found)")
    console.print()
    if ftune:
        table = Table(title="Finetuning runs", box=None)
        table.add_column("", style="blue")
        table.add_column("run-name", style="blue")
        table.add_column("data|div", style="magenta")
        table.add_column("from", style="green")
        if scheduler_detail:
            table.add_column("sched")
            table.add_column("sched-options")
        if optim_detail:
            table.add_column("optim")
            table.add_column("optim-options")
        table.add_column("head")
        table.add_column("wd")
        table.add_column("head|bbone-lr")
        table.add_column("epoch(best)")
        if best:
            table.add_column("best-valid")
            table.add_column("best-train")
        if latest:
            table.add_column("curr-valid")
            table.add_column("curr-train")
        for j, p in enumerate(ftune):
            try:
                config = TrainConfig.from_json_file(p / "config.json")
                with open(
                    (p / "best-model") / "metrics.json", mode="rt", encoding="utf-8"
                ) as ft:
                    metrics = json.load(ft)
                    best_epoch = metrics.pop("epoch")
                with open(
                    (p / "latest-model") / "metrics.json", mode="rt", encoding="utf-8"
                ) as ft:
                    latest_metrics = json.load(ft)
                    epoch = latest_metrics.pop("epoch")
                row_args = [
                    f"[bold]{j}[/bold]",
                    p.name,
                    f"{config.ds.path.name}|{config.ds.fold_idx}",
                    (
                        config.ftune.pretrained_name
                        if config.ftune is not None
                        else "error"
                    ),
                    (
                        str(config.ftune.num_head_layers)
                        if config.ftune is not None
                        else "error"
                    ),
                    f"{config.optim.weight_decay:.0e}",
                    f"{config.optim.lr:.0e}|{(config.ftune.backbone_lr if config.ftune is not None else 0.0):.0e}",  # noqa
                    f"{epoch}({best_epoch})",
                ]
                if optim_detail:
                    options = " ".join(
                        sorted(
                            f"{k}={v}"
                            for k, v in config.optim.options.items()
                            if k not in ["lr", "weight_decay"]
                        )
                    )
                    row_args.insert(4, options or "--")
                    row_args.insert(4, config.optim.cls)
                if scheduler_detail:
                    options = " ".join(
                        sorted(f"{k}={v}" for k, v in config.scheduler.options.items())
                    )
                    row_args.insert(4, options or "--")
                    row_args.insert(4, config.scheduler.cls)
                if best:
                    if not mae:
                        metrics = {k: v for k, v in metrics.items() if "mae" not in k}
                    if not hartree:
                        metrics = {
                            k: v
                            for k, v in metrics.items()
                            if not (
                                ("energies" in k and "kcal" not in k)
                                or ("forces" in k and "kcal" not in k)
                            )
                        }
                    else:
                        metrics = {
                            k: v
                            for k, v in metrics.items()
                            if not (
                                ("energies" in k and "kcal" in k)
                                or ("forces" in k and "kcal" in k)
                            )
                        }
                    row_args.extend(
                        [
                            " ".join(
                                f"{simplify_metric(k)}={format(v, metric_fmt)}"
                                for k, v in metrics.items()
                                if "valid" in k
                            ),
                            " ".join(
                                f"{simplify_metric(k)}={format(v, metric_fmt)}"
                                for k, v in metrics.items()
                                if "train" in k
                            ),
                        ]
                    )
                if latest:
                    if not mae:
                        latest_metrics = {
                            k: v for k, v in latest_metrics.items() if "mae" not in k
                        }
                    if not hartree:
                        latest_metrics = {
                            k: v
                            for k, v in latest_metrics.items()
                            if not (
                                ("energies" in k and "kcal" not in k)
                                or ("forces" in k and "kcal" not in k)
                            )
                        }
                    else:
                        latest_metrics = {
                            k: v
                            for k, v in latest_metrics.items()
                            if not (
                                ("energies" in k and "kcal" in k)
                                or ("forces" in k and "kcal" in k)
                            )
                        }
                    row_args.extend(
                        [
                            " ".join(
                                f"{simplify_metric(k)}={format(v, metric_fmt)}"
                                for k, v in latest_metrics.items()
                                if "valid" in k
                            ),
                            " ".join(
                                f"{simplify_metric(k)}={format(v, metric_fmt)}"
                                for k, v in latest_metrics.items()
                                if "train" in k
                            ),
                        ]
                    )
            except Exception:
                row_args = [f"[bold]{j}[/bold]", p.name, "???"]
            table.add_row(*row_args)
        console.print(table)
    else:
        console.print("(No finetuning runs found)")
    console.print()
    if ensemble:
        table = Table(title="Saved models and ensembles", box=None)
        table.add_column("", style="cyan")
        table.add_column("name", style="cyan")
        table.add_column("ftune-src", style="blue")
        table.add_column("train-src", style="green")
        table.add_column("num-networks")
        for j, p in enumerate(ensemble):
            src_config = SrcConfig.from_json_file(p / "src_config.json")
            row_args = [
                f"[bold]{j}[/bold]",
                p.name,
                " ".join(src_config.ftune_src).strip() or "--",
                " ".join(src_config.train_src).strip() or "--",
                str(src_config.num),
            ]
            table.add_row(*row_args)
        console.print(table)
    else:
        console.print("(No saved models or ensembles found)")
    console.print()
