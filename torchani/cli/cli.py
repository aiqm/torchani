import argparse
from .utils import h5info
from ..datasets._builtin_datasets import download_builtin_dataset, _BUILTIN_DATASETS, _BUILTIN_DATASETS_LOT


def build_parser():
    # main parser
    main_parser = argparse.ArgumentParser(prog='torchani', description="TorchANI Command Line Interface")
    subparsers = main_parser.add_subparsers(dest='workflow', help='select workflow', required=True)

    # dataset download parser
    parser_download = subparsers.add_parser('download', help='download datasets',
                                            description="Download dataset from Moria (needs to be within UF network).\n"
                                            "Check avaiable dataset at: \n"
                                            "https://github.com/roitberg-group/torchani_sandbox/blob/master/torchani/datasets/_builtin_datasets.py",
                                            formatter_class=argparse.RawTextHelpFormatter)
    parser_download.add_argument('dataset', type=str, choices=_BUILTIN_DATASETS, help='the dataset to download')
    parser_download.add_argument('lot', type=str, choices=_BUILTIN_DATASETS_LOT, help='level of theory')
    parser_download.add_argument('--root', type=str, default=None,
                                 help='Optional root directory to save the dataset, default folder is set as datasets/{dataset}-{lot}')

    # dataset download parser
    parser_h5info = subparsers.add_parser('h5info', help='show h5file informations',
                                            description="Show h5file informations.")
    parser_h5info.add_argument('path', type=str, help='path to a h5 dataset file or a directory')

    # parse args
    args = main_parser.parse_args()
    return args


def main():
    args = build_parser()

    if args.workflow == "download":
        download_builtin_dataset(args.dataset, args.lot, args.root)

    if args.workflow == "h5info":
        h5info(args.path)


if __name__ == '__main__':
    main()
