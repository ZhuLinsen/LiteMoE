"""Validate the real CLI's subprocess plan without importing CUDA libraries."""
import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import zipfile

import pytest


@pytest.fixture
def runner(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'tools' / 'matched_run.py'
    spec = importlib.util.spec_from_file_location('matched_run_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)
    for filename in ('pyproject.toml', 'THIRD_PARTY_NOTICES.md'):
        Path(filename).write_text('fixture', encoding='utf-8')
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    for label in ('1b', '3b', '1b-base'):
        (inputs / ('inputs-' + label + '.json')).write_text('{}', encoding='utf-8')
    vp = tmp_path / 'vllm-env' / 'bin' / 'python'
    vp.parent.mkdir(parents=True)
    vp.touch()
    commands = []
    real_popen = module.subprocess.Popen

    class Process:
        def __init__(self, command, **kwargs):
            commands.append(command)

        def wait(self, timeout):
            return 0

    monkeypatch.setattr(module.subprocess, 'Popen', Process)

    def invoke(*extra):
        monkeypatch.setattr(sys, 'argv', [str(path), '--output', 'result', '--inputs', str(inputs),
                                        '--vllm-python', str(vp), *extra])
        return module.main()

    invoke.module = module
    invoke.real_popen = real_popen
    return invoke, commands, vp, inputs


def argument(command, name):
    return command[command.index(name) + 1]


def test_vllm_backend_uses_reused_interpreter_and_selected_labels(runner):
    invoke, commands, vp, _ = runner
    invoke('--mini-backend', 'vllm', '--reuse-environments', '--models', '1b-base', '3b')
    assert len(commands) == 7
    assert all(command[0] == str(vp) for command in commands)
    assert commands[0][1:] == ['-m', 'pytest', '-q']
    runs = [c for c in commands if 'run' in c]
    assert [argument(c, '--framework') for c in runs] == ['mini', 'vllm', 'mini', 'vllm']
    assert [Path(argument(c, '--inputs')).name for c in runs] == [
        'inputs-1b-base.json', 'inputs-1b-base.json', 'inputs-3b.json', 'inputs-3b.json']
    assert all(argument(c, '--mini-backend') == 'vllm' for c in runs)
    assert not any('pip' in c or 'venv' in c for c in commands)
    completed = json.loads(Path('result/complete.json').read_text())
    assert completed['completed']
    assert completed['framework_pairing'] == 'sequential same allocation, shared pinned environment'


def test_legacy_defaults_keep_install_path_backend_and_models(runner):
    invoke, commands, vp, _ = runner
    invoke()
    assert [c[0] for c in commands[:4]] == [sys.executable] * 4
    assert 'transformers==4.57.1' in commands[0]
    assert commands[1][1:4] == ['-m', 'venv', '--without-pip']
    assert 'vllm==0.32.0' in commands[2]
    runs = [c for c in commands if 'run' in c]
    assert [Path(argument(c, '--inputs')).name for c in runs] == [
        'inputs-1b.json', 'inputs-1b.json', 'inputs-3b.json', 'inputs-3b.json']
    assert [c[0] for c in runs] == [sys.executable, str(vp)] * 2
    assert all(argument(c, '--mini-backend') == 'optimized' for c in runs)


def test_skip_tests_and_smoke_preserve_existing_cli_meaning(runner):
    invoke, commands, _, _ = runner
    invoke('--reuse-environments', '--skip-tests', '--smoke', '--models', '1b')
    assert len(commands) == 3
    assert not any('pytest' in c for c in commands)
    assert all('--smoke' in c for c in commands if 'run' in c)
    assert json.loads(Path('result/complete.json').read_text())['smoke']


@pytest.mark.parametrize('labels', [
    ['1b', '1b'], ['../3b'], ['a/b'], [r'a\b'], ['..'], ['a:b'], [''], ['模型'],
])
def test_invalid_labels_fail_before_output_or_process_creation(runner, labels):
    invoke, commands, _, _ = runner
    with pytest.raises(SystemExit) as failure:
        invoke('--models', *labels)
    assert failure.value.code == 2
    assert commands == []
    assert not Path('result').exists()


def test_missing_input_fails_before_output_or_process_creation(runner):
    invoke, commands, _, _ = runner
    with pytest.raises(SystemExit):
        invoke('--models', 'missing')
    assert commands == []
    assert not Path('result').exists()


def test_vllm_backend_requires_reuse_before_output_or_installation(runner):
    invoke, commands, _, _ = runner
    with pytest.raises(SystemExit):
        invoke('--mini-backend', 'vllm')
    assert commands == []
    assert not Path('result').exists()


def test_reused_environment_must_exist_before_output_or_process_creation(runner):
    invoke, commands, vp, _ = runner
    vp.unlink()
    with pytest.raises(SystemExit):
        invoke('--reuse-environments', '--mini-backend', 'vllm')
    assert commands == []
    assert not Path('result').exists()


@pytest.mark.parametrize('seconds', ['0', '-1'])
def test_nonpositive_deadline_rejected_before_output_or_process(runner, seconds):
    invoke, commands, _, _ = runner
    with pytest.raises(SystemExit) as failure:
        invoke('--max-seconds', seconds)
    assert failure.value.code == 2
    assert commands == []
    assert not Path('result').exists()


@pytest.mark.parametrize('after_first_stage', [False, True])
def test_expired_deadline_archives_without_launching_next_stage(runner, monkeypatch, after_first_stage):
    invoke, commands, _, _ = runner
    readings = iter([0, 0, 0, 2] if after_first_stage else [0, 2])
    monkeypatch.setattr(invoke.module, 'time', SimpleNamespace(
        monotonic=lambda: next(readings), time=time.time))
    with pytest.raises(TimeoutError, match='before'):
        invoke('--reuse-environments', '--max-seconds', '1', '--models', '1b')
    assert len(commands) == int(after_first_stage)
    rows = json.loads(Path('result/progress.json').read_text())
    assert rows[-1]['status'] == 'deadline-exceeded'
    assert 'returncode' not in rows[-1]
    if after_first_stage:
        assert rows[0]['status'] == 'passed'
    with zipfile.ZipFile('result-latest.zip') as archive:
        assert archive.testzip() is None
        assert json.loads(archive.read('result/progress.json')) == rows
    assert not Path('result/complete.json').exists()


def test_real_cpu_timeout_stops_child_and_archives_evidence(runner, monkeypatch):
    invoke, commands, _, _ = runner
    children = []

    def start_cpu_child(command, **options):
        assert command[1:] == ['-m', 'pytest', '-q']
        child = invoke.real_popen([sys.executable, '-u', '-c',
            'import time; print("CPU child started", flush=True); time.sleep(30)'], **options)
        children.append(child)
        return child

    monkeypatch.setattr(invoke.module.subprocess, 'Popen', start_cpu_child)
    started = time.monotonic()
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            invoke('--reuse-environments', '--max-seconds', '2', '--models', '1b')
        assert len(children) == 1
        assert children[0].poll() is not None
        assert time.monotonic() - started < 10
        rows = json.loads(Path('result/progress.json').read_text())
        assert len(rows) == 1
        assert rows[0]['status'] == 'interrupted-or-timeout'
        assert rows[0]['error_type'] == 'TimeoutExpired'
        assert rows[0]['returncode'] == children[0].returncode
        with zipfile.ZipFile('result-latest.zip') as archive:
            assert archive.testzip() is None
            assert json.loads(archive.read('result/progress.json')) == rows
            log = archive.read('result/tests.log')
            assert b'CPU child started' in log
            manifest = json.loads(archive.read('MANIFEST.json'))
            assert manifest['result/tests.log'] == hashlib.sha256(log).hexdigest()
        assert not Path('result/complete.json').exists()
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)
