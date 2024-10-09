r"""
The torchani.datasets.builtin module is automatically generated using this code,
it should be run before a PR, every time a new builtin dataset is added
"""

from pathlib import Path
import jinja2
import json

datasets_dir = Path(Path(__file__).parent.parent, "torchani", "datasets")

_DATASETS_JSON_PATH = datasets_dir / "builtin_datasets.json"
with open(_DATASETS_JSON_PATH, mode="rt", encoding="utf-8") as f:
    _DATASETS_SPEC = json.load(f)

env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(datasets_dir),
    undefined=jinja2.StrictUndefined,
    autoescape=jinja2.select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
)

for fname in ("builtin", "__init__"):
    template = env.get_template(f"{fname}.jinja")
    template_kwargs = {
        "datasets": [
            {"name": k, "default_lot": ds["default-lot"]}
            for k, ds in _DATASETS_SPEC.items()
        ],
        "lots": sorted(
            set().union(*[set(ds["lot"].keys()) for ds in _DATASETS_SPEC.values()])
        ),
    }
    string = template.render(**template_kwargs)
    builtin_ds_file = datasets_dir / f"{fname}.py"
    builtin_ds_file.touch()
    builtin_ds_file.write_text(string)
print("Regenerated necessary torchani.datasets files jinja templates")
