import json
import os

def generate_config():
    min_cluster = 3
    max_cluster = 35

    dirs = os.listdir('.')
    dirs = [int(d.split('_')[1]) for d in dirs if 'Experiment' in d]
    if len(dirs) > 0:
        dirs.sort()
        number = dirs[-1]
    else:
        number = 0

    config = {}
    config['experiment_name'] = f"Experiment_{number + 1}"
    config['experiment_description'] = 'Same but with lambda variable from 0.3 to 0.1 and proximity variable 0.2 to 0.05 and second_thresh variable from 0.19 to 0.12'
    cluster_config = {'cluster_eps' : 0.06, 'cluster_min_samples' : 3}
    config['cluster_config'] = cluster_config
    
    config['hyperparams'] = {
        'match_thresh' : {
            'is_variable' : True,
            'static_value' : 0.65,
            'is_ramp_up' : False,
            'cluster_min' : min_cluster,
            'cluster_max' : max_cluster,
            'val_min' : 0.45,
            'val_max' : 0.7
        },
        'second_match_thresh' : {
            'is_variable' : False,
            'static_value' : 0.19,
            'is_ramp_up' : False,
            'cluster_min' : min_cluster,
            'cluster_max' : max_cluster,
            'val_min' : 0.12,
            'val_max' : 0.19
        },
        'lambda_' : {
            'is_variable' : False,
            'static_value' : 0.2,
            'is_ramp_up' : False,
            'cluster_min' : min_cluster,
            'cluster_max' : max_cluster,
            'val_min' : 0.1,
            'val_max' : 0.3
        },
        'proximity_thresh' : {
            'is_variable' : False,
            'static_value' : 0.1,
            'is_ramp_up' : False,
            'cluster_min' : min_cluster,
            'cluster_max' : max_cluster,
            'val_min' : 0.05,
            'val_max' : 0.2
        } 
    }

    return config

config = generate_config()

os.makedirs(config['experiment_name'], exist_ok=True)

with open(f"{config['experiment_name']}/config.json", 'w') as f:
    json.dump(config, f)
