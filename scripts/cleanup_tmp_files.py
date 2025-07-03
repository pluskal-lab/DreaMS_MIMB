#!/usr/bin/env python3
import os
import re
import argparse

# Default target directory and patterns
DEFAULT_DIR = "/Users/macbook/CODE/DreaMS_MIMB/data/rawfiles"
DEFAULT_PATTERNS = [
    r".*dedup.*\.hdf5$",
    r".*dreams.*\.npy$",
    r".*high_quality.*\.hdf5$",
    r".*positive.*\.hdf5$",
    r".*\.hdf5$",
]

def delete_matching_files(directory, patterns):
    compiled_patterns = [re.compile(p) for p in patterns]
    files_deleted = []

    for fname in os.listdir(directory):
        full_path = os.path.join(directory, fname)
        if os.path.isfile(full_path):
            for pattern in compiled_patterns:
                if pattern.match(fname):
                    os.remove(full_path)
                    files_deleted.append(fname)
                    break
    return files_deleted

def main():
    parser = argparse.ArgumentParser(description="Delete temporary ML preprocessing files.")
    parser.add_argument(
        "-d", "--directory",
        type=str,
        default=DEFAULT_DIR,
        help="Target directory (default is the rawfiles folder)"
    )
    parser.add_argument(
        "-p", "--patterns",
        nargs="*",
        help="Custom regex patterns to match files (default is dedup, dreams, high_quality, positive)"
    )

    args = parser.parse_args()
    patterns = args.patterns if args.patterns else DEFAULT_PATTERNS

    deleted = delete_matching_files(args.directory, patterns)
    if deleted:
        print("Deleted files:")
        for f in deleted:
            print(f"  - {f}")
    else:
        print("No matching files found.")

if __name__ == "__main__":
    main()