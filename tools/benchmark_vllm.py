"""External offline baseline using exactly the saved mini token workloads."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import hashlib
import math
import statistics


def legacy_main():
    p = argparse.ArgumentParser(description='Legacy baseline. Matched commands: prepare, run, compare; see docs/matched-vllm-protocol.md')
    p.add_argument('--config', required=True)
    p.add_argument('--inputs', required=True, help='Completed benchmark_layouts output directory')
    p.add_argument('--output', required=True)
    p.add_argument('--eager', action='store_true')
    args = p.parse_args()
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
    os.environ.setdefault('VLLM_NO_USAGE_STATS', '1')
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
    cfg = json.loads(Path(args.config).read_text())
    name = Path(args.config).stem.removeprefix('granite-')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    import torch
    from vllm import LLM, SamplingParams
    options = dict(model=cfg['model'], revision=cfg['revision'], dtype='bfloat16',
                   max_model_len=2048, max_num_seqs=32, gpu_memory_utilization=.35,
                   enable_prefix_caching=False, enable_chunked_prefill=True,
                   tensor_parallel_size=1, enforce_eager=args.eager, seed=0, moe_backend='triton')
    if not args.eager:
        options['compilation_config'] = dict(cudagraph_capture_sizes=[1, 8, 32])
    metadata = dict(options=options, versions={p: importlib.metadata.version(p) for p in
        ('vllm', 'torch', 'transformers', 'huggingface-hub', 'flashinfer-python')},
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), warmup=1, repeats=5,
        scope='wall time of offline LLM.generate including scheduler and detokenization; initialization excluded',
        numerical_contract='vLLM native BF16 kernels; output agreement recorded, not assumed to pass mini HF tolerance',
        reference_scope='mini engine E2E excludes final output-dictionary collection; compare with this scope caveat')
    (out / 'environment.json').write_text(json.dumps(metadata, indent=2))
    llm = LLM(**options)
    sampling = SamplingParams(temperature=0, max_tokens=128, ignore_eos=True, seed=0)
    for length in (128, 1024):
        for batch in (1, 8, 32):
            inp = Path(args.inputs) / f'input-{name}-b{batch}-p{length}.json'
            if not inp.exists():
                continue
            requests = json.loads(inp.read_text())
            prompts = [dict(prompt_token_ids=r['prompt']) for r in requests]
            ref = Path(args.inputs) / f'{name}-b{batch}-p{length}-ep1-async.jsonl'
            expected = json.loads(ref.read_text().splitlines()[0])['outputs'] if ref.exists() else None
            first = None
            for repeat in range(-1, 5):
                start = time.perf_counter()
                result = llm.generate(prompts, sampling, use_tqdm=False)
                elapsed = time.perf_counter() - start
                outputs = {r['request_id']: list(o.outputs[0].token_ids) for r, o in zip(requests, result)}
                if len(result) != len(requests) or any(len(v) != 128 for v in outputs.values()):
                    raise ValueError('Incomplete fixed-budget vLLM output')
                if first is None:
                    first = outputs
                row = dict(batch=batch, prompt_length=length, output_length=128, repeat=repeat,
                           elapsed_s=elapsed, e2e_tps=sum(map(len, outputs.values())) / elapsed,
                           repeat_equal=outputs == first, mini_equal=outputs == expected if expected else None,
                           outputs=outputs, texts={r['request_id']: o.outputs[0].text for r, o in zip(requests, result)})
                with (out / 'results.jsonl').open('a') as f:
                    f.write(json.dumps(row) + '\n')
                print('VLLM_BENCH', batch, length, repeat, round(row['e2e_tps'], 3),
                      'repeat_equal', row['repeat_equal'], 'mini_equal', row['mini_equal'], flush=True)
                if not row['repeat_equal']:
                    raise ValueError('vLLM repeated generation changed outputs')
    print('VLLM_COMPLETE', flush=True)


PROTOCOL = 'matched-offline-v1'
MATRIX = {'b1-p128': [128], 'b8-p128': [128] * 8, 'b32-p128': [128] * 32,
          'b1-p2048': [2048], 'b8-p1024': [1024] * 8,
          'mixed-b8': [32, 32, 64, 128, 256, 512, 1024, 2048]}
SCOPE = 'pretokenized submission through output collection and one final decode per request'


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def read_inputs(path):
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if data['protocol'] != PROTOCOL or set(data['cases']) != set(MATRIX):
        raise ValueError('Unknown or incomplete frozen matrix')
    for name, lengths in MATRIX.items():
        rows = data['cases'][name]
        if [len(r['prompt']) for r in rows] != lengths or any(r['budget'] != 128 for r in rows):
            raise ValueError('Frozen matrix shape/budget mismatch')
        ids = [r['request_id'] for r in rows]
        if len(set(ids)) != len(ids) or any(not isinstance(x, str) for x in ids):
            raise ValueError('Request IDs must be unique strings')
        if any(type(t) is not int or t < 0 for r in rows for t in r['prompt']):
            raise ValueError('Invalid prompt token')
    return data, hashlib.sha256(raw).hexdigest()


def validate_outputs(rows, outputs):
    if set(outputs) != {r['request_id'] for r in rows}:
        raise ValueError('Missing or extra request outputs')
    for r in rows:
        tokens = outputs[r['request_id']]
        if not isinstance(tokens, list) or len(tokens) != r['budget']:
            raise ValueError('Incomplete fixed-budget output')
        if any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError('Invalid output token')
    return sum(map(len, outputs.values()))


def timed_generation(generate, tokenizer, synchronize, clock=time.perf_counter):
    synchronize()
    start = clock()
    outputs, phases = generate()
    texts = {key: tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False) for key, tokens in outputs.items()}
    synchronize()
    return outputs, texts, phases, clock() - start


def prepare_matched(args):
    from transformers import AutoTokenizer
    from tools.experiment import workload
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(cfg['model'], revision=cfg['revision'])
    cases = {}
    for name, lengths in MATRIX.items():
        requests = workload(tokenizer, len(lengths), max(lengths), 128)
        cases[name] = [dict(request_id=r.request_id, prompt=r.prompt[:length], budget=128)
                       for r, length in zip(requests, lengths)]
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save(output, dict(protocol=PROTOCOL, model=cfg['model'], revision=cfg['revision'], cases=cases,
                     workload_kind='repeated authored text throughput stress; not an accuracy benchmark'))
    _, sha = read_inputs(output)
    print('INPUTS_FROZEN', sha, flush=True)


def run_matched(args):
    data, sha = read_inputs(args.inputs)
    profiles = ['b8-p128', 'mixed-b8'] if args.smoke else list(MATRIX)
    repeats = args.repeats if args.repeats is not None else (3 if args.smoke else 5)
    warmups = args.warmups
    if warmups < 1:
        raise ValueError('At least one warmup is required')
    if repeats < (3 if args.smoke else 5):
        raise ValueError('Use at least 3 smoke / 5 formal repetitions')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    args._created_output = True
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
    os.environ.setdefault('VLLM_NO_USAGE_STATS', '1')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
    import gc
    import subprocess
    import torch
    from transformers import AutoTokenizer
    from litemoe.cli import build, environment
    from litemoe.engine import Request
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Expose exactly one native BF16 GPU')
    torch.set_num_threads(1)
    torch.manual_seed(0)
    cfg = dict(model=data['model'], revision=data['revision'])
    metadata = dict(protocol=PROTOCOL, framework=args.framework, input_sha256=sha, **cfg,
                    scope=SCOPE, profiles=profiles, smoke=args.smoke, repeats=repeats, warmup=warmups,
                    sampling=dict(temperature=0, ignore_eos=True, max_tokens=128),
                    detokenization=dict(skip_special_tokens=False, clean_up_tokenization_spaces=False, calls_per_request=1),
                    device_identity=subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,driver_version',
                                                              '--format=csv,noheader'], text=True).strip(),
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), environment=environment())
    metadata['vllm_version'] = importlib.metadata.version('vllm') if args.framework == 'vllm' else None
    save(out / 'environment.json', metadata)
    tokenizer = AutoTokenizer.from_pretrained(cfg['model'], revision=cfg['revision'])
    started = time.perf_counter()
    if args.framework == 'vllm':
        from vllm import LLM, SamplingParams
        options = dict(**cfg, tokenizer_revision=cfg['revision'], dtype='bfloat16', max_model_len=4096,
                       max_num_seqs=32, gpu_memory_utilization=.75, enable_prefix_caching=False,
                       enable_chunked_prefill=True, tensor_parallel_size=1, seed=0, disable_log_stats=True)
        metadata['options'] = options
        llm = LLM(**options)
        sampling = SamplingParams(temperature=0, max_tokens=128, ignore_eos=True, seed=0, detokenize=False)
    else:
        torch.use_deterministic_algorithms(True)
        torch.utils.deterministic.fill_uninitialized_memory = False
        name = max(profiles, key=lambda n: len(MATRIX[n]) * max(MATRIX[n]))
        base, _, _ = build(**cfg, requests=[Request(**r) for r in data['cases'][name]],
                           backend=args.mini_backend, mode='graph-async', capacity=4096, ignore_eos=True)
        model, engine_type = base.model, type(base)
        del base
        gc.collect()
        torch.cuda.empty_cache()
        metadata['options'] = dict(backend=args.mini_backend, mode='graph-async', capacity=4096,
                                   host_process=None, deterministic=True, fill_uninitialized_memory=False)
    torch.cuda.synchronize()
    metadata['initialization_s'] = time.perf_counter() - started
    save(out / 'environment.json', metadata)
    for name in profiles:
        rows = data['cases'][name]
        if args.framework == 'mini':
            run = engine_type(model, [Request(**r) for r in rows], capacity=4096, ignore_eos=True,
                              mode='graph-async', host_process=None)
            generate = run.generate
        else:
            prompts = [dict(prompt_token_ids=r['prompt']) for r in rows]
            def generate():
                result = llm.generate(prompts, sampling, use_tqdm=False)
                if len(result) != len(rows) or any(len(o.outputs) != 1 for o in result):
                    raise ValueError('Incomplete vLLM request collection')
                return {r['request_id']: list(o.outputs[0].token_ids) for r, o in zip(rows, result)}, {}
        first = None
        try:
            for repeat in range(-warmups, repeats):
                outputs, texts, phases, elapsed = timed_generation(generate, tokenizer, torch.cuda.synchronize)
                count = validate_outputs(rows, outputs)
                if not math.isfinite(elapsed) or elapsed <= 0:
                    raise ValueError('Invalid elapsed time')
                if first is None:
                    first = outputs
                record = dict(profile=name, repeat=repeat, e2e_s=elapsed, e2e_tps=count / elapsed,
                              output_tokens=count, outputs=outputs, texts=texts, repeat_equal=first == outputs,
                              framework_phase_metrics=phases)
                with (out / 'results.jsonl').open('a', encoding='utf-8') as file:
                    file.write(json.dumps(record, ensure_ascii=False) + '\n')
                print('MATCHED', args.framework, name, repeat, round(count / elapsed, 3), flush=True)
        finally:
            if args.framework == 'mini':
                run.synchronize()
                if run.graph is not None:
                    run.graph.reset()
                    run.graph = None
                del generate, run
                gc.collect()
                torch.cuda.empty_cache()
    save(out / 'complete.json', dict(completed=True, profiles=profiles, repeats=repeats))


def compare_matched(args):
    data, sha = read_inputs(args.inputs)
    evidence = {}
    for framework in ('mini', 'vllm'):
        folder = Path(getattr(args, framework))
        env = json.loads((folder / 'environment.json').read_text(encoding='utf-8'))
        complete = json.loads((folder / 'complete.json').read_text(encoding='utf-8'))
        if not complete['completed'] or env['framework'] != framework or env['input_sha256'] != sha:
            raise ValueError('Incomplete or mismatched evidence')
        if complete['profiles'] != env['profiles'] or complete['repeats'] != env['repeats']:
            raise ValueError('Completion metadata disagrees')
        records = [json.loads(line) for line in (folder / 'results.jsonl').read_text(encoding='utf-8').splitlines()]
        if type(env.get('warmup')) is not int or env['warmup'] < 1:
            raise ValueError('Missing or invalid warmup count')
        expected = {(n, r) for n in env['profiles'] for r in range(-env['warmup'], env['repeats'])}
        if len(records) != len(expected) or {(r['profile'], r['repeat']) for r in records} != expected:
            raise ValueError('Missing or duplicated timing rows')
        for row in records:
            count = validate_outputs(data['cases'][row['profile']], row['outputs'])
            if count != row['output_tokens'] or not math.isfinite(row['e2e_s']) or row['e2e_s'] <= 0:
                raise ValueError('Invalid timing evidence')
        evidence[framework] = env, records
    a, b = (evidence[f][0] for f in ('mini', 'vllm'))
    for field in ('protocol', 'model', 'revision', 'scope', 'profiles', 'smoke', 'repeats', 'warmup', 'sampling',
                  'detokenization', 'device_identity', 'cuda_visible_devices'):
        if a[field] != b[field]:
            raise ValueError(f'Incompatible comparison field: {field}')
    if a['protocol'] != PROTOCOL or a['scope'] != SCOPE or not a['device_identity']:
        raise ValueError('Unrecognized counting/device protocol')
    profiles = ['b8-p128', 'mixed-b8'] if a['smoke'] else list(MATRIX)
    if a['profiles'] != profiles or a['repeats'] < (3 if a['smoke'] else 5):
        raise ValueError('Incomplete declared matrix/repetitions')
    if a['sampling'] != dict(temperature=0, ignore_eos=True, max_tokens=128) or a['detokenization'] != dict(
            skip_special_tokens=False, clean_up_tokenization_spaces=False, calls_per_request=1):
        raise ValueError('Unexpected sampling/output policy')
    summary = []
    for name in a['profiles']:
        selected = {f: [r for r in evidence[f][1] if r['profile'] == name and r['repeat'] >= 0]
                    for f in ('mini', 'vllm')}
        speeds = {f: statistics.median(r['output_tokens'] / r['e2e_s'] for r in rows)
                  for f, rows in selected.items()}
        mini, vllm = selected['mini'][0]['outputs'], selected['vllm'][0]['outputs']
        summary.append(dict(profile=name, **{f + '_tps': v for f, v in speeds.items()},
                            mini_over_vllm=speeds['mini'] / speeds['vllm'],
                            first_repeat_exact_requests=sum(mini[k] == vllm[k] for k in mini), requests=len(mini),
                            repeat_stable={f: all(r['outputs'] == rows[0]['outputs'] for r in rows)
                                           for f, rows in selected.items()}))
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    save(args.output, dict(protocol=PROTOCOL, input_sha256=sha, smoke=a['smoke'], rows=summary,
                           geometric_mean_ratio=math.exp(statistics.mean(math.log(r['mini_over_vllm']) for r in summary)),
                           environments={f: evidence[f][0] for f in evidence},
                           quality_status='output agreement only; independent semantic quality assessment required'))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ('prepare', 'run', 'compare'):
        return legacy_main()
    p = argparse.ArgumentParser(description='Matched Mini/vLLM offline benchmark; legacy flags remain supported')
    sub = p.add_subparsers(dest='action', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--config', required=True)
    prepare.add_argument('--output', required=True)
    run = sub.add_parser('run')
    run.add_argument('--framework', choices=['mini', 'vllm'], required=True)
    run.add_argument('--mini-backend', choices=['optimized','vllm'], default='optimized')
    run.add_argument('--inputs', required=True)
    run.add_argument('--output', required=True)
    run.add_argument('--smoke', action='store_true')
    run.add_argument('--repeats', type=int)
    run.add_argument('--warmups', type=int, default=3)
    compare = sub.add_parser('compare')
    for flag in ('inputs', 'mini', 'vllm', 'output'):
        compare.add_argument('--' + flag, required=True)
    args = p.parse_args()
    try:
        dict(prepare=prepare_matched, run=run_matched, compare=compare_matched)[args.action](args)
    except Exception as error:
        if args.action == 'run' and getattr(args, '_created_output', False):
            save(Path(args.output) / 'error.json', dict(error=repr(error)))
        raise


if __name__ == '__main__':
    main()
