r"""
The torchani.datasets.builtin module is automatically generated using this
code, it should be run before every push (pre-commit can be used with the
provided configuration to ensure this), ad every time a new builtin dataset is
added, or any of the jinja files is modified.
"""

import textwrap
import typing as tp
from pathlib import Path
import jinja2
import json

datasets_dir = Path(Path(__file__).parent.parent, "torchani", "datasets")

_DATASETS_JSON_PATH = datasets_dir / "builtin_datasets.json"
with open(_DATASETS_JSON_PATH, mode="rt", encoding="utf-8") as f:
    _DATASETS_SPEC = json.load(f)

    # Update keys using the metadata-only keys, which have "-meta" suffix,
    # and remove them afterwards
    meta_map = {}
    for k in _DATASETS_SPEC.copy():
        if k.endswith("-meta"):
            meta = _DATASETS_SPEC.pop(k)
            meta_map[k.replace("-meta", "")] = meta

    for k in _DATASETS_SPEC.copy():
        meta = _DATASETS_SPEC[k].pop("meta", "")
        if meta in meta_map:
            _DATASETS_SPEC[k].update(meta_map[meta])

env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(datasets_dir),
    undefined=jinja2.StrictUndefined,
    autoescape=jinja2.select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
)

for fname in ("builtin", "__init__"):
    template = env.get_template(f"{fname}.jinja")
    template_kwargs: tp.Dict[str, tp.List[tp.Any]] = {"datasets": []}
    for k, ds in _DATASETS_SPEC.items():
        docstr_parts = []
        if "info" in ds:
            docstr_parts.append(ds["info"])
        if "article" in ds:
            docstr_parts.append(f"Originally published in *{ds['article']}*")
        if "doi" in ds:
            doi_key = "doi"
            docstr_parts.append(f"DOI: '{ds[doi_key]}'")

        template_kwargs["datasets"].append(
            {
                "name": k,
                "default_lot": ds["default-lot"],
                "docstr": "\n".join(
                    textwrap.wrap(
                        ". ".join(docstr_parts),
                        width=88,
                        tabsize=4,
                        initial_indent="    ",
                        subsequent_indent="    ",
                    )
                ),
            }
        )

    template_kwargs["lots"] = sorted(
        set().union(*[set(ds["lot"].keys()) for ds in _DATASETS_SPEC.values()])
    )
    string = template.render(**template_kwargs)
    builtin_ds_file = datasets_dir / f"{fname}.py"
    builtin_ds_file.touch()
    builtin_ds_file.write_text(string)
print("Regenerated necessary torchani.datasets files from jinja templates")
