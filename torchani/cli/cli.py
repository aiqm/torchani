import argparse

from torchani.cli.utils import h5info, h5pack
from torchani.datasets import (
    download_builtin_dataset,
    _BUILTIN_DATASETS,
    _BUILTIN_DATASETS_LOT,
)


def build_parser():
    # main parser
    main_parser = argparse.ArgumentParser(prog='torchani', description="TorchANI Command Line Interface")
    subparsers = main_parser.add_subparsers(dest='workflow', help='select workflow', required=True)

    # dataset download parser
    parser_download = subparsers.add_parser('download', help='download datasets',
                                            description="Download dataset from Moria (needs to be within UF network).\n"
                                            "Check avaiable dataset at: \n"
                                            "https://github.com/roitberg-group/torchani_sandbox/blob/master/torchani/datasets/builtin.py",
                                            formatter_class=argparse.RawTextHelpFormatter)
    parser_download.add_argument('dataset', type=str, choices=_BUILTIN_DATASETS, help='the dataset to download')
    parser_download.add_argument('lot', type=str, choices=_BUILTIN_DATASETS_LOT, help='level of theory')
    parser_download.add_argument('--root', type=str, default=None,
                                 help='Optional root directory to save the dataset, default folder is set as datasets/{dataset}-{lot}')

    # dataset download parser
    parser_h5info = subparsers.add_parser('h5info', help='show h5file informations',
                                            description="Show h5file informations.")
    parser_h5info.add_argument('path', type=str, help='path to a h5 dataset file or a directory')

    parser_h5pack = subparsers.add_parser(
        'h5pack',
        help='Package h5 files into a dataset',
        description="Package h5 files into a dataset"
    )
    parser_h5pack.add_argument(
        'path',
        type=str,
        help='Path to a h5 dataset directory'
    )
    parser_h5pack.add_argument(
        '--internal',
        action="store_true",
        help='Append the dataset to the internal datasets'
    )
    parser_h5pack.add_argument(
        '--interactive',
        action="store_true",
    )
    parser_h5pack.add_argument(
        '--no-interactive',
        action="store_false",
        dest="interactive",
    )
    parser_h5pack.add_argument('--functional', type=str, default="")
    parser_h5pack.add_argument('--basis-set', type=str, default="")
    parser_h5pack.add_argument('--name', type=str, default="")
    parser_h5pack.set_defaults(
        interactive=True,
        force_renaming=True,
    )
    # parse args
    args = main_parser.parse_args()
    return args


def main():
    args = build_parser()

    if args.workflow == "download":
        download_builtin_dataset(args.dataset, args.lot, args.root)

    if args.workflow == "h5info":
        h5info(args.path)

    if args.workflow == "h5pack":
        if not args.interactive:
            force_renaming = False
        else:
            force_renaming = True
        h5pack(
            args.path,
            args.internal,
            force_renaming=force_renaming,
            interactive=args.interactive,
            functional=args.functional,
            basis_set=args.basis_set,
            name=args.name
        )


if __name__ == '__main__':
    main()
