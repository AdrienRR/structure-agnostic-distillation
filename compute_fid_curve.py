#!/usr/bin/env python3
"""
Compute gFID for every checkpoint produced by a LightningDiT training run.

Usage (from the repository root):
    python compute_fid_curve.py --config configs/dit/2d_pool_align.yaml

Steps per checkpoint:
  1. accelerate launch inference.py  (GPU, generates 50K PNGs)
  2. tools/save_npz.py               (CPU, packs PNGs → npz)
  3. ADM evaluator (guided-diffusion/evaluations/evaluator.py, CPU TensorFlow)

Results are appended to output/<exp_name>/fid_curve.json (unguided) or
output/<exp_name>/fid_cfg<scale>.json (--cfg_scale > 1) after each checkpoint, so the
script is safe to resume.
"""

import os, sys, json, glob, subprocess, yaml, argparse

# ADM evaluator (OpenAI guided-diffusion) and its reference batch; see README.
ADM_EVALUATOR = os.environ.get('ADM_EVALUATOR', 'evaluations/evaluator.py')
ADM_PYTHON    = os.environ.get('ADM_PYTHON', sys.executable)   # interpreter with tensorflow
FID_REF_NPZ   = os.environ.get('FID_REF_NPZ', os.path.join(os.path.dirname(ADM_EVALUATOR),
                                                           'VIRTUAL_imagenet256_labeled.npz'))
PYTHON        = sys.executable
ACCELERATE    = 'accelerate'
SAMPLES_ROOT  = os.environ.get('SAMPLES_ROOT', 'samples')


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def sample_folder_name(cfg, ckpt_path, cfg_scale_override=None):
    """Reproduces the folder name logic from inference.py:do_sample."""
    model_type = cfg['model']['model_type'].replace('/', '-').lower()
    ckpt_stem  = os.path.splitext(os.path.basename(ckpt_path))[0]
    method     = cfg['sample']['sampling_method']
    steps      = cfg['sample']['num_sampling_steps']
    name = f"{model_type}-ckpt-{ckpt_stem}-{method}-{steps}"
    cfg_scale = cfg_scale_override if cfg_scale_override is not None else cfg['sample']['cfg_scale']
    cfg_interval_start = cfg['sample'].get('cfg_interval_start', 0)
    timestep_shift     = cfg['sample'].get('timestep_shift', 0)
    if cfg_scale > 1.0:
        name += f"-interval{cfg_interval_start:.2f}"
        name += f"-cfg{cfg_scale:.2f}"
        name += f"-shift{timestep_shift:.2f}"
    return name


def run(cmd, env=None, cwd=None):
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, env=env, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Command exited {result.returncode}: {' '.join(str(c) for c in cmd)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--num_gpus', type=int, default=8)
    parser.add_argument('--cfg_scale', type=float, default=1.0, help='CFG scale (1.0 = no guidance)')
    parser.add_argument('--per_proc_batch_size', type=int, default=None,
                        help='Per-GPU sampling batch (default: config); guidance doubles it, so halve for cfg > 1')
    parser.add_argument('--stride', type=int, default=1,
                        help='Evaluate every Nth checkpoint (newest-first, always includes newest)')
    parser.add_argument('--last_n', type=int, default=1,
                        help='Evaluate only the newest N checkpoints (default 1 = final/80k only). '
                             'Set high (e.g. 100) for a full curve.')
    parser.add_argument('--only_step', type=str, default=None,
                        help='Evaluate ONLY the checkpoint whose filename contains this step '
                             '(e.g. 0070000). Overrides --last_n/--stride.')
    args = parser.parse_args()

    cfg       = load_config(args.config)
    exp_dir   = os.path.join(cfg['train']['output_dir'], cfg['train']['exp_name'])
    ckpt_dir  = os.path.join(exp_dir, 'checkpoints')
    json_name = 'fid_curve.json' if args.cfg_scale <= 1.0 else f'fid_cfg{args.cfg_scale:g}.json'
    json_path = os.path.join(exp_dir, json_name)

    if os.path.exists(json_path):
        with open(json_path) as f:
            fid_results = json.load(f)
    else:
        fid_results = {}

    # Latest checkpoints first
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, '*.pt')), reverse=True)
    if not ckpts:
        sys.exit(f"No checkpoints found in {ckpt_dir}")
    if args.only_step:
        ckpts = [c for c in ckpts if args.only_step in os.path.basename(c)]
        if not ckpts:
            sys.exit(f"No checkpoint matching step {args.only_step} in {ckpt_dir}")
    else:
        ckpts = ckpts[:args.last_n]   # default 1: evaluate only the newest (final / 80k) checkpoint
        if args.stride > 1:
            ckpts = ckpts[::args.stride]   # subsample, newest always kept (index 0)
            print(f"Subsampled to every {args.stride}th checkpoint: {len(ckpts)} to evaluate")

    print(f"Found {len(ckpts)} checkpoints (newest first). Already evaluated: {len(fid_results)}")
    print(f"cfg_scale={args.cfg_scale}")

    env = os.environ.copy()

    for ckpt_path in ckpts:
        ckpt_stem = os.path.splitext(os.path.basename(ckpt_path))[0]
        if ckpt_stem in fid_results:
            print(f"[skip] step={ckpt_stem}  FID={fid_results[ckpt_stem]:.3f}")
            continue

        print(f"\n{'='*60}\nEvaluating checkpoint: {ckpt_stem}\n{'='*60}")

        # 1. Generate samples (GPU)
        run([
            ACCELERATE, 'launch',
            '--num_processes', str(args.num_gpus),
            'inference.py',
            '--config', args.config,
            '--ckpt_path', ckpt_path,
            '--cfg_scale', str(args.cfg_scale),
        ] + (['--per_proc_batch_size', str(args.per_proc_batch_size)] if args.per_proc_batch_size else []), env=env)

        # 2. Pack PNGs → npz (CPU)
        sample_dir = os.path.join(SAMPLES_ROOT, cfg['train']['exp_name'], sample_folder_name(cfg, ckpt_path, args.cfg_scale))
        npz_path   = sample_dir + '.npz'
        if not os.path.exists(npz_path):
            fid_num = cfg['sample']['fid_num']
            run([PYTHON, 'tools/save_npz.py', '--sample_dir', sample_dir, '--num', str(fid_num)], env=env)

        # 3. Compute FID (CPU-only TF)
        fid_env = env.copy()
        fid_env['CUDA_VISIBLE_DEVICES'] = ''
        # The evaluator caches its Inception graph in its cwd, so run it from its own
        # directory with absolute paths.
        result = subprocess.run(
            [ADM_PYTHON, os.path.abspath(ADM_EVALUATOR), os.path.abspath(FID_REF_NPZ), os.path.abspath(npz_path)],
            env=fid_env, cwd=os.path.dirname(os.path.abspath(ADM_EVALUATOR)),
            capture_output=True, text=True,
        )
        print(result.stdout[-3000:])
        if result.returncode != 0:
            print("STDERR:", result.stderr[-2000:])
            raise RuntimeError(f"fid_evaluator failed for {ckpt_stem}")

        fid_value = None
        for line in (result.stdout + result.stderr).splitlines():
            if line.strip().startswith('FID:'):
                try:
                    fid_value = float(line.split(':', 1)[1].strip())
                except ValueError:
                    pass

        if fid_value is None:
            raise RuntimeError(f"Could not parse FID from evaluator output for {ckpt_stem}")

        fid_results[ckpt_stem] = fid_value
        with open(json_path, 'w') as f:
            json.dump(fid_results, f, indent=2, sort_keys=True)
        print(f"[result] step={ckpt_stem}  FID={fid_value:.3f}  → {json_path}")

    print(f"\nDone. Final results:\n{json.dumps(fid_results, indent=2, sort_keys=True)}")


if __name__ == '__main__':
    main()
