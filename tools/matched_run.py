"""Bounded, sequential matched-framework experiment; archive every stage."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time
import zipfile

if __name__ == '__main__' and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.process_guard import stop


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--inputs', required=True)
    p.add_argument('--vllm-python', required=True)
    p.add_argument('--mini-backend', choices=['optimized', 'vllm'], default='optimized')
    p.add_argument('--models', nargs='+', default=['1b', '3b'], help='Unique frozen input labels')
    p.add_argument('--max-seconds', type=int, default=1800)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--reuse-environments', action='store_true', help='Reuse already installed Mini/vLLM environments')
    p.add_argument('--skip-tests', action='store_true', help='Only when this exact core already passed the full GPU suite')
    args = p.parse_args()
    if args.max_seconds <= 0:
        p.error('--max-seconds must be positive')
    if len(set(args.models)) != len(args.models) or any(
            re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', label) is None for label in args.models):
        p.error('--models requires unique alphanumeric labels (hyphen/underscore allowed)')
    for label in args.models:
        if not (Path(args.inputs)/('inputs-'+label+'.json')).is_file():
            p.error('Missing frozen input for model label: '+label)
    if args.mini_backend == 'vllm' and not args.reuse_environments:
        p.error('--mini-backend vllm requires --reuse-environments')
    if args.reuse_environments and not Path(args.vllm_python).is_file():
        p.error('Reused vLLM interpreter does not exist')
    mini_python = args.vllm_python if args.mini_backend == 'vllm' else sys.executable
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + args.max_seconds
    stages = []
    os.environ.update(DS_BUILD_OPS='0', VLLM_NO_USAGE_STATS='1', HF_HUB_DISABLE_TELEMETRY='1',
                      TOKENIZERS_PARALLELISM='false', CUDA_VISIBLE_DEVICES='0',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8')

    def archive():
        (out/'progress.json').write_text(json.dumps(stages, indent=2))
        entries = [p for folder in ('litemoe','tools','tests','configs','docs','LICENSES',str(out))
                   for p in Path(folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
        entries += list(Path(args.inputs).glob('*.json')) + [Path('pyproject.toml'),Path('THIRD_PARTY_NOTICES.md')]
        target = out.parent/(out.name+'-latest.zip')
        temporary = target.with_suffix('.tmp')
        with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as z:
            for p in entries: z.write(p,p.as_posix())
            z.writestr('MANIFEST.json',json.dumps({p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                                                for p in entries},indent=2))
        temporary.replace(target)

    def run(name, command, limit):
        row = dict(stage=name, command=command, started=time.time(), status='running')
        stages.append(row)
        (out/'progress.json').write_text(json.dumps(stages,indent=2))
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                row['status'] = 'deadline-exceeded'
                raise TimeoutError('Total experiment deadline exceeded before '+name)
            with (out/(name+'.log')).open('w') as log:
                try:
                    process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                except BaseException as error:
                    row.update(status='failed-to-start', error_type=type(error).__name__)
                    raise
                try:
                    code = process.wait(timeout=min(limit, max(0, deadline-time.monotonic())))
                    row.update(returncode=code,status='passed' if code==0 else 'failed')
                except BaseException as error:
                    row.update(status='interrupted-or-timeout', error_type=type(error).__name__)
                    try:
                        stop(process)
                    finally:
                        row['returncode'] = process.poll()
                    raise
        finally:
            row['finished']=time.time()
            archive()
            print(name,row['status'],flush=True)
        if row['status']!='passed': raise RuntimeError(name+' failed; inspect saved log')

    vp = args.vllm_python
    if not args.reuse_environments:
        run('install-mini',[sys.executable,'-m','pip','install','--no-build-isolation',
            'transformers==4.57.1','huggingface-hub==0.36.0','safetensors==0.6.2','pytest==8.4.2','deepspeed==0.19.8'],180)
        run('create-vllm-env',[sys.executable,'-m','venv','--without-pip',str(Path(vp).parents[1])],60)
        run('install-vllm',[sys.executable,'-m','pip','--python',vp,'install','--only-binary=:all:',
                           'vllm==0.32.0'],540)
    if not args.skip_tests:
        run('tests',[mini_python,'-m','pytest','-q'],120)
    extra = ['--smoke'] if args.smoke else []
    for model in args.models:
        inputs = str(Path(args.inputs)/('inputs-'+model+'.json'))
        for framework, python in (('mini',mini_python),('vllm',vp)):
            run(framework+'-'+model,[python,'-u','-m','tools.benchmark_vllm','run',
                '--framework',framework,'--mini-backend',args.mini_backend,'--inputs',inputs,'--output',str(out/(framework+'-'+model)),*extra],480)
        run('compare-'+model,[mini_python,'-m','tools.benchmark_vllm','compare','--inputs',inputs,
            '--mini',str(out/('mini-'+model)),'--vllm',str(out/('vllm-'+model)),
            '--output',str(out/('comparison-'+model+'.json'))],30)
    (out/'complete.json').write_text(json.dumps(dict(completed=True,smoke=args.smoke,
        framework_pairing=('sequential same allocation, shared pinned environment'
                           if args.mini_backend == 'vllm'
                           else 'sequential same allocation, separate dependency stacks'))))
    archive()


if __name__=='__main__':
    main()
