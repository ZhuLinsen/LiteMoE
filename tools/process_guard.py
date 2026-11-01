"""Host-independent Colab job supervisor. Run detached, after mounting Drive.

No credentials are accepted. The release endpoint comes from Colab's environment.
This is experiment infrastructure, not production inference code.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request
import zipfile


def atomic_write(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    if hashlib.sha256(temporary.read_bytes()).digest() != hashlib.sha256(data).digest():
        raise OSError('Archive readback failed')
    os.replace(temporary, path)


def checkpoint(source, destination):
    """Copy completed, valid stage archives; never accept a half-written ZIP."""
    records = []
    for path in sorted(Path(source).glob('*.zip')):
        try:
            with zipfile.ZipFile(path) as archive:
                if archive.testzip() is not None:
                    continue
            raw = path.read_bytes()
            # Revalidate the exact bytes copied, since producer may still write.
            import io
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                if archive.testzip() is not None:
                    continue
        except (OSError, zipfile.BadZipFile):
            continue
        target = Path(destination) / path.name
        digest = hashlib.sha256(raw).hexdigest()
        if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            atomic_write(target, raw)
        records.append(dict(file=path.name, bytes=len(raw), sha256=digest))
    return records


def unassign():
    """Same backend request as google.colab.runtime.unassign, without browser JS."""
    address = os.environ.get('TBE_RUNTIME_ADDR')
    if not address:
        raise RuntimeError('Missing Colab runtime management address')
    request = urllib.request.Request('http://' + address + '/unassign', method='POST')
    # Do not forward this internal request through an HTTP proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=15) as response:
        if response.status != 200:
            raise RuntimeError('Runtime release rejected')


def stop(proc):
    if proc.poll() is None:
        if os.name == 'posix':
            # The existing runner launches stages in separate process sessions.
            # Kill their descendants too; killing only the runner group leaks them.
            parents = {}
            for path in Path('/proc').glob('[0-9]*/stat'):
                try:
                    fields = path.read_text().rsplit(')', 1)[1].split()
                    parents[int(path.parent.name)] = int(fields[1])
                except (OSError, ValueError, IndexError):
                    continue
            victims = {proc.pid}
            while True:
                expanded = victims | {pid for pid, parent in parents.items() if parent in victims}
                if expanded == victims:
                    break
                victims = expanded
            for pid in victims - {proc.pid}:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
        proc.wait(timeout=15)


def supervise(command, source, destination, seconds, interval=30, release=unassign):
    destination = Path(destination)
    # Destination must already be a user-selected mounted persistent directory.
    if not destination.is_dir() or seconds <= 0 or interval <= 0:
        raise ValueError('Existing persistent destination and positive deadlines required')
    status = dict(state='starting', archives=[], errors=[], release='not_requested')
    proc = None
    started = time.monotonic()
    def persist():
        atomic_write(destination / 'guard-status.json', json.dumps(status, indent=2).encode())
    try:
        persist()  # Fail before starting an expensive job if Drive is unwritable.
        with (destination / 'supervisor.log').open('wb') as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            status.update(state='running', pid=proc.pid)
            while proc.poll() is None:
                remaining = seconds - (time.monotonic() - started)
                if remaining <= 0:
                    status['state'] = 'timed_out'
                    stop(proc)
                    break
                status['archives'] = checkpoint(source, destination)
                persist()
                try:
                    proc.wait(timeout=min(interval, remaining))
                except subprocess.TimeoutExpired:
                    pass
            status['returncode'] = proc.returncode
            if status['state'] != 'timed_out':
                status['state'] = 'finished' if proc.returncode == 0 else 'failed'
    except BaseException as error:
        status['state'] = 'supervisor_failed'
        status['errors'].append(repr(error))
    finally:
        try:
            if proc is not None:
                stop(proc)
            status['archives'] = checkpoint(source, destination)
        except BaseException as error:
            status['errors'].append('final_archive: ' + repr(error))
        # Save the release intent BEFORE the VM can disappear. A successful POST
        # is only an accepted request, not proof the allocation has terminated.
        status['release'] = 'requesting'
        try:
            persist()
        except BaseException as error:
            status['errors'].append('persist: ' + repr(error))
        try:
            release()
            status['release'] = 'request_accepted'
        except BaseException as error:
            status['release'] = 'failed'
            status['errors'].append('release: ' + repr(error))
        try:
            persist()
        except BaseException:
            print(json.dumps(status), flush=True)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Dedicated stage-archive directory')
    parser.add_argument('--destination', required=True, help='Existing mounted Drive job directory')
    parser.add_argument('--seconds', type=float, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command or args.command[0] != '--' or len(args.command) < 2:
        parser.error('Specify job command after --')
    if not os.environ.get('TBE_RUNTIME_ADDR'):
        parser.error('Colab runtime required; refusing to launch an unreleasable GPU job')
    result = supervise(args.command[1:], args.source, args.destination, args.seconds)
    raise SystemExit(0 if result['state'] == 'finished' and result['release'] == 'request_accepted'
                     and not result['errors'] else 1)
