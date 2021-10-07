import argparse
from .datasets._builtin_datasets import download_builtin_dataset, _BUILTIN_DATASETS, _BUILTIN_DATASETS_LOT


def build_parser():
    # main parser
    main_parser = argparse.ArgumentParser(prog='torchani', description="TorchANI Command Line Interface")
    subparsers = main_parser.add_subparsers(dest='workflow', help='select workflow', required=True)

    # dataset download parser
    parser_download = subparsers.add_parser('download', help='Download dataset from Moria (needs UF VPN to be able to work).')
    parser_download.add_argument('dataset', type=str, choices=_BUILTIN_DATASETS, help='the dataset to download')
    parser_download.add_argument('lot', type=str, choices=_BUILTIN_DATASETS_LOT, help='level of theory')
    parser_download.add_argument('--root', type=str, default=None, help='Optional root directory to save the dataset, default folder is set as datasets/{dataset}-{lot}')

    # parse args
    args = main_parser.parse_args()
    return args


def main():
    args = build_parser()

    if args.workflow == "download":
        download_builtin_dataset(args.dataset, args.lot, args.root)


if __name__ == '__main__':
    main()
