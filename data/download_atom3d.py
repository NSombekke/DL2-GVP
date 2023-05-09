import atom3d.datasets as da
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset")
parser.add_argument("--split", default=None)
args = parser.parse_args()

da.download_dataset(f'{args.dataset}', f"atom3d-data/{args.dataset.upper()}", split=args.split) # Download LBA dataset.