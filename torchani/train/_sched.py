from copy import deepcopy
import sys
import subprocess
from pathlib import Path

from typer import Abort
from rich.console import Console
import jinja2


def send_to_scheduler(
    slurm_gpu: str, cluster: str, num_cpu: int, filename: str
) -> None:
    console = Console()
    if cluster == "moria":
        assert slurm_gpu in ["v100", "gp100", "titanv", "gtx1080ti", ""]
    elif cluster == "hpg":
        assert slurm_gpu in ["b200", "l4", ""]
    else:
        console.print(f"Unknown cluster {cluster}", style="red")
        raise Abort()
    slurm_gpu = f"{slurm_gpu}:1" if slurm_gpu else "1"

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(Path(__file__).parent / "train" / "templates/"),
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
    tmpl = env.get_template(f"{cluster}.slurm.sh.jinja").render(
        num_workers=num_cpu,
        name=filename,
        gpu=slurm_gpu,
        args=args,
    )
    unique_id = filename.split("-")[-1]
    j = 0
    input_dir = Path(Path.home(), "IO", "ani", f"{unique_id}_v{j}")
    while input_dir.is_dir():
        j += 1
        input_dir = Path(Path.home(), "IO", "ani", f"{unique_id}_v{j}")
    input_dir.mkdir(exist_ok=False, parents=True)
    input_fpath = input_dir / f"{cluster}.slurm.sh"
    input_fpath.write_text(tmpl)
    console.print("Launching slurm script ...")
    subprocess.run(["sbatch", str(input_fpath)], cwd=input_dir, check=True)
