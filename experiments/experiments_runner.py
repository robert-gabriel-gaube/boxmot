import os
import subprocess
import pandas as pd

def cleanup(experiment):
    for item in os.listdir(f'experiments/{experiment}'):
        item_path = os.path.join(f'experiments/{experiment}', item)
        if item != "config.json":
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    import shutil
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f"Failed to delete {item_path}: {e}")

def result_interpretation(original, results_run, output_dir):
    df_new = pd.read_csv(results_run, sep='\s+')
    df_orig = pd.read_csv(original,    sep='\s+')

    # Stack, label, diff
    data = pd.concat([df_new, df_orig], axis=0, ignore_index=True)
    data.index = ["Clustered", "Original"]
    data.loc["Diff"] = data.loc["Clustered"] - data.loc["Original"]
    if data.loc["Diff"]["MOTA"] > 0:
        with open(f'{output_dir}/better', 'w') as f:
            pass
    else:
        with open(f'{output_dir}/worse', 'w') as f:
            pass

    # Ensure output directory exists
    out_dir = os.path.dirname(output_dir)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    with open(f"{output_dir}/mota.txt", 'w') as f:
        f.write(str(data.loc['Clustered']['MOTA']))

    # Write to CSV
    data.to_csv(f'{output_dir}/results.csv', index=True)
    print(f"Wrote comparison to {output_dir}")

def run_tracker():
    cmd = [
        "python3",
        "tracking/val.py",
        "--yolo-model", "yolox_x.pt",
        "--reid-model", "osnet_x0_25_market1501.pt",
        "--tracking-method", "imprassoc",
        "--verbose",
        "--source", "./assets/MOT20/train"
    ]

    # This will print stdout/stderr directly to your console
    subprocess.run(cmd, check=True)

dir = os.listdir('/home/ubuntu/boxmot/runs/mot')
dir.sort()

experiments = os.listdir('experiments')
experiments = [d for d in experiments if 'Experiment' in d]
experiments.sort()

for experiment in experiments:
    # cleanup(experiment)

    # Copy contents of experiment
    with open(f'experiments/{experiment}/config.json', 'r') as f:
        data = f.read()
        with open('experiments/thresh_config.json', 'w') as o:
            o.write(data)

    try:
        run_tracker()
        dirs = os.listdir('runs/mot/')
        dir = f'yolox_x_osnet_x0_25_market1501_imprassoc_{len(dirs)}' if len(dirs) > 1 else 'yolox_x_osnet_x0_25_market1501_imprassoc'
        result_interpretation('experiments/original_results.txt', f'runs/mot/{dir}/pedestrian_summary.txt', f'experiments/{experiment}')
    except subprocess.CalledProcessError as e:
        print(f"Tracker failed with exit code {e}")
        with open(f"experiments/{experiment}/error", 'w') as f:
            pass

    