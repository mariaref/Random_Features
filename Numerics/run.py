import os
import time
import subprocess
import itertools
import argparse
import torch

from utils import copy_py, who_am_i

def create_script(params):
    script = '''#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --partition=p40_4,p100_4
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task={cpu}
#SBATCH --output={name}.out
#SBATCH --job-name={name}

module purge

for ((i={seed[0]}; i<={seed[1]}; i++)); do
  python main.py --seed $i --optimizer {optimizer} --lr {lr} --mom {mom} --bs {bs} --crit {crit} --save_at {name}.$i {no_bias} --zero_init_method centered --alpha {alpha} --dataset {dataset} --data_size {data_size} --width {width} --depth {depth} --input_dim {input_dim} --num_classes {num_classes} &
done

wait

'''.format(**params)

    with open('{}.sbatch'.format(params['name']), 'w') as f:
        f.write(script)
    with open('{}.params'.format(params['name']), 'wb') as f:
        torch.save(params, f)


def send_script(file):
    process = subprocess.Popen(['sbatch', file], stdout=subprocess.PIPE)


if __name__ == '__main__':
    
    exp_dir = 'r.{}'.format(int(time.time()))
    os.mkdir(exp_dir)
    copy_py(exp_dir) 
    os.chdir(exp_dir)

    grid = {
        'width':[2,    4,    5,    8,   12,   17,   25,   37,   53,   76,  110,
        159,  229,  330,  475,  685,  987, 1421, 2048],#2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
        'depth':[5],
        'input_dim':[10],
        'num_classes':[2],
        'optimizer': ['adam'],
        'lr': [0.1],
        'mom': [0.],
        'bs': [5000],
        'alpha': [1,10,100],
        'dataset': ['MNIST'],
        'data_size': [5000],
        'seed': [(41, 50),(51, 60)], # inclusive
        'cpu': [10], # must be "upper - lower + 1" 
        'crit': ['linear_hinge'],
    }

    def dict_product(d):
        keys = d.keys()
        for element in itertools.product(*d.values()):
            yield dict(zip(keys, element))

    for params in dict_product(grid):
        params['name'] = '{alpha}_{width}_{seed[0]:02d}-{seed[1]:02d}'.format(**params)
        # params['lr'] = min(1.e-4, 0.1 * params['width']**(-1.5))
        create_script(params)
        file_name = '{}.sbatch'.format(params['name'])
        send_script(file_name)


