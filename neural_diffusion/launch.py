import sys, os
import torch as th

def common(exe_code='train'):
    commands = []
    all_args = sys.argv[1:]
    num_gpus = th.cuda.device_count()
    if num_gpus > 1:
        commands = ['mpiexec', f'-n {num_gpus}', '--allow-run-as-root']
    commands += ['python', f'{exe_code}.py']
    commands += all_args
    os.system(' '.join(commands))

def main_train():
    common('neural_diffusion/train')

def main_sample():
    common('neural_diffusion/sample')

def main_eval():
    common('neural_diffusion/eval')