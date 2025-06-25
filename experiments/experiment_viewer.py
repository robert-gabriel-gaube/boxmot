#!/usr/bin/env python3
import os
import sys
import json
import csv
import argparse
import pandas as pd

# never truncate rows or columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# expand the console width so columns never wrap
pd.set_option('display.width', None)

# allow columns to be arbitrarily wide
pd.set_option('display.max_colwidth', None)

def main():
    parser = argparse.ArgumentParser(
        description="Display info for a given experiment folder."
    )
    parser.add_argument(
        "experiment",
        help="Name of the experiment directory (e.g. Experiment_2)."
    )
    args = parser.parse_args()

    exp_dir = os.path.abspath(args.experiment)
    if not os.path.isdir(exp_dir):
        print(f"Error: directory not found: {exp_dir}", file=sys.stderr)
        sys.exit(1)

    # 1️⃣ Pretty-print config.json
    config_path = os.path.join(exp_dir, "config.json")
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        print("\nConfiguration (config.json):")
        print(json.dumps(config, indent=4, sort_keys=True, ensure_ascii=False))
    else:
        print("\nNo config.json found.")

    # 2️⃣ Check for 'better' or 'worse' flag file
    status = None
    for flag in ("better", "worse"):
        if os.path.isfile(os.path.join(exp_dir, flag)):
            status = flag
            break
    print(f"\nStatus: {status or 'unknown (no better/worse file)'}")

    # 3️⃣ Read results.csv
    results_path = os.path.join(exp_dir, "results.csv")
    if os.path.isfile(results_path):
        df = pd.read_csv(results_path)
        print(df['IDSW'])
    else:
        print("\nNo results.csv found.")

if __name__ == "__main__":
    main()
