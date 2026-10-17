#!/usr/bin/env bash
# Serial HTTP embeddings A/A/B on one idle card, with one server alive at a time.
# Usage: cache-ab.sh MAIN_BIN HEAD_BIN CHAT_GGUF EMBED_GGUF MMPROJ_GGUF OUT
set -euo pipefail
[ "$#" = 6 ] || { echo 'need MAIN_BIN HEAD_BIN CHAT_GGUF EMBED_GGUF MMPROJ_GGUF OUT' >&2; exit 64; }
source "${BASH_SOURCE[0]%/*}/../lease.sh"
export BLOOMERY_LEASE_CARD=docs/cards/embedding-cache.card
export ROUNDS=5
TIMING_GPU=0
WITNESS=(head loadavg cpu-freq meminfo gpus gpu-apps busiest)
lease_take
witness pre
lease_bounded 5400 python3 - "$@" <<'PY'
import base64
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import statistics
import struct
import subprocess
import sys
import time
import zlib
sys.path.insert(0, 'tools/ref/emb2')
from server_vectors import sha256 as digest

main, head, chat, embed, projector, output = map(Path, sys.argv[1:])
output.mkdir(parents=True, exist_ok=False)
rounds, count = int(os.environ["ROUNDS"]), 32
method = {
    'rounds': rounds, 'requests_per_workload': count,
    'arms': ['main-a', 'main-b', 'head'], 'cache_bytes': 67108864,
    'workloads': ['same-text', 'same-image', 'image-new-text', 'all-new'],
    'latency': 'HTTP connection, request, full response read and JSON parse; serial concurrency 1',
    'throughput': 'requests divided by block wall time, including vector validation and log writes',
    'warmup': 'one untimed request per workload; warm rows exclude it',
    'noise': f'maximum relative A/A difference across {rounds} paired rounds per workload and metric',
    'significant': 'p50 decrease and throughput increase above twice A/A noise in any hit workload',
    'miss_guard': 'all-new p50/p95 increase and throughput decrease no greater than twice A/A noise',
    'equality': 'all full-width unnormalized vectors must be bit-identical across every arm',
}
(output / 'method.json').write_text(json.dumps(method, indent=2) + '\n')
frozen = output / time.strftime('frozen-%Y%m%dT%H%M%SZ', time.gmtime())
frozen.mkdir()
for source, name in [(main, 'main'), (head, 'head'), (Path('tools/ref/emb2/cache-ab.sh'), 'cache-ab.sh'),
                     (Path('docs/cards/embedding-cache.card'), 'embedding-cache.card'),
                     (output / 'method.json', 'method.json')]:
    shutil.copy2(source, frozen / name)

manifest = {p.name: digest(p) for p in frozen.iterdir()}
(frozen / 'sha256.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('FROZEN', frozen, json.dumps(manifest), flush=True)
print('COMMITS', os.environ.get('NR_BASE'), os.environ.get('NR_HEAD'), flush=True)
print('MODELS', json.dumps({str(p): digest(p) for p in (chat, embed, projector)}), flush=True)
hardware = subprocess.check_output(['nvidia-smi'], text=True)
gb10 = 'GB10' in hardware
print('HARDWARE', hardware, flush=True)
print('CPU', Path('/proc/cpuinfo').read_text().split('model name', 1)[-1].splitlines()[0], flush=True)

def png(seed):
    width = height = 256
    rgb = bytes((seed % 251, (seed * 17) % 251, (seed * 43) % 251))
    raw = (b'\x00' + rgb * width) * height
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    data = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
    data += chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b'')
    return 'data:image/png;base64,' + base64.b64encode(data).decode()

text = 'Represent this document for retrieval: ' + 'The archive contains maps and photographs of coastal towns. ' * 16

def payload(workload, i):
    if workload == 'same-text':
        value = text
    else:
        seed = i + 11 if workload == 'all-new' else 1
        prompt = 'Describe this image.' if workload == 'same-image' else f'Describe image number {i}.'
        value = {'content': [{'type': 'text', 'text': prompt}, {'type': 'image_url', 'image_url': png(seed)}]}
    return json.dumps({'input': value, 'embd_normalize': -1}, separators=(',', ':')).encode()

cases = {name: [payload(name, i) for i in range(count)] for name in method['workloads']}
reference = {}
log = (output / 'requests.jsonl').open('w')
summary = []
active = None
cancelled = False
governors = {}
started_utc = time.time()

def governor(path, value):
    subprocess.run(['sudo', '-n', 'tee', str(path)], input=value + '\n', text=True, capture_output=True, check=True)

def stop():
    global active
    if active is None:
        return
    process, active = active, None
    if process.poll() is not None:
        raise RuntimeError(f'server exited unexpectedly: {process.returncode}')
    if not cancelled:
        process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
        raise RuntimeError('server needed SIGKILL')
    if process.returncode not in [0, -signal.SIGTERM]:
        raise RuntimeError(f'server stop failed: {process.returncode}')

def interrupted(signum, frame):
    global cancelled
    cancelled = True
    raise RuntimeError(f'interrupted by signal {signum}')

signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)

def witness(tag, pid=None):
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,name,used_memory', '--format=csv,noheader'], text=True).strip()
    if any(int(line.split(',')[0]) != pid for line in apps.splitlines() if line.strip()):
        raise RuntimeError('another GPU process: ' + apps)
    state = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,memory.free,pstate', '--format=csv'], text=True)
    print('WITNESS', tag, json.dumps({'load': os.getloadavg(), 'gpu': state, 'apps': apps,
          'memory': Path('/proc/meminfo').read_text()}), flush=True)

def idle():
    deadline, quiet = time.monotonic() + 600, None
    while time.monotonic() < deadline:
        if os.getloadavg()[0] < 1:
            quiet = quiet or time.monotonic()
            if time.monotonic() - quiet >= 30:
                return
        else:
            quiet = None
        time.sleep(2)
    raise RuntimeError('CPU load did not stay below 1 for 30 seconds')

def request(port, body):
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=120)
    started = time.perf_counter_ns()
    try:
        connection.request('POST', '/v1/embeddings', body, {'Content-Type': 'application/json', 'Connection': 'close'})
        response = connection.getresponse()
        data = json.loads(response.read())
        elapsed = (time.perf_counter_ns() - started) / 1e6
        if response.status != 200:
            raise RuntimeError(f'HTTP {response.status}: {data}')
        values = data['data'][0]['embedding']
        if not values or not all(math.isfinite(v) for v in values):
            raise RuntimeError('empty or nonfinite vector')
        bits = struct.pack('<' + 'f' * len(values), *values)
        key = hashlib.sha256(body).hexdigest()
        if key in reference and reference[key] != bits:
            raise RuntimeError('vector bits differ for request ' + key)
        reference[key] = bits
        return elapsed, key, hashlib.sha256(bits).hexdigest()
    finally:
        connection.close()

try:
    observed = {path.resolve(): path.read_text().strip() for path in
                Path('/sys/devices/system/cpu').glob('cpu[0-9]*/cpufreq/scaling_governor')}
    if not gb10:
        governors = observed
        for path in governors:
            governor(path, 'performance')
    print('GOVERNOR', 'unchanged' if gb10 else 'performance; original',
          json.dumps({str(p): v for p, v in observed.items()}), flush=True)
    for cycle in range(rounds):
        arms = ['main-a', 'main-b', 'head']
        arms = arms[cycle % 3:] + arms[:cycle % 3]
        for arm in arms:
            idle()
            witness(f'{cycle}/{arm}/pre')
            binary = frozen / ('head' if arm == 'head' else 'main')
            assert digest(binary) == manifest[binary.name], 'frozen binary changed'
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            command = [str(binary.resolve()), '-m', str(chat), '--place', 'a', '--ctx', '4096', '--parallel', '1',
                       '--host', '127.0.0.1', '--port', str(port), '--embd-model', str(embed), '--embd-mmproj', str(projector),
                       '--embd-ctx', '8192', '--embd-image-min-tokens', '1', '--embd-image-max-tokens', '70']
            if arm == 'head':
                command += ['--embd-cache-bytes', str(method['cache_bytes'])]
            print('COMMAND', cycle, arm, json.dumps(command), flush=True)
            server_log = (output / f'server-{cycle}-{arm}.log').open('w')
            active = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 300
            while True:
                if active.poll() is not None:
                    raise RuntimeError(f'server exited {active.returncode}; see {server_log.name}')
                try:
                    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=2)
                    conn.request('GET', '/health')
                    response = conn.getresponse()
                    response.read()
                    conn.close()
                    if response.status == 200:
                        break
                except OSError:
                    pass
                if time.monotonic() >= deadline:
                    raise RuntimeError('server startup timeout')
                time.sleep(0.5)
            for workload, bodies in cases.items():
                # Warm the text path without pre-seeding the all-new keys.
                warm = payload(workload, count + 1) if workload in ['all-new', 'image-new-text'] else bodies[0]
                request(port, warm)
                latencies = []
                block_started = time.perf_counter()
                for i, body in enumerate(bodies):
                    elapsed, key, vector = request(port, body)
                    latencies.append(elapsed)
                    row = dict(cycle=cycle, arm=arm, workload=workload, request=i, ms=elapsed, input=key, vector=vector)
                    log.write(json.dumps(row) + '\n')
                    log.flush()
                row = dict(cycle=cycle, arm=arm, workload=workload, n=count,
                           p50_ms=statistics.median(latencies), p95_ms=sorted(latencies)[math.ceil(.95 * count) - 1],
                           requests_s=count / (time.perf_counter() - block_started))
                summary.append(row)
                print('ROW', json.dumps(row), flush=True)
            witness(f'{cycle}/{arm}/post', active.pid)
            stop()
            server_log.close()
            witness(f'{cycle}/{arm}/stopped')
finally:
    cleanup_errors = []
    for path, original in governors.items():
        try:
            governor(path, original)
        except Exception as error:
            cleanup_errors.append(f'restoring {path}: {error}')
    try:
        stop()
    except Exception as error:
        cleanup_errors.append(f'stopping server: {error}')
    log.close()
    if cleanup_errors:
        raise RuntimeError('; '.join(cleanup_errors))
    print('GOVERNOR unchanged' if gb10 else 'GOVERNOR restored', flush=True)

journal = subprocess.check_output(([] if gb10 else ['sudo', '-n']) +
                                  ['journalctl', '-k', '--since', '@' + str(int(started_utc)), '--no-pager'], text=True)
if any('Xid' in line and 'NVRM' in line for line in journal.splitlines()):
    raise RuntimeError('NVIDIA Xid during the sitting')
(output / 'kernel.log').write_text(journal)

results = []
for workload in cases:
    row = dict(workload=workload, requests_per_arm=count * rounds)
    for metric in ['p50_ms', 'p95_ms', 'requests_s']:
        values = lambda arm: [s[metric] for s in summary if s['arm'] == arm and s['workload'] == workload]
        a, aa, b = values('main-a'), values('main-b'), values('head')
        noise = max(abs(x - y) / ((x + y) / 2) for x, y in zip(a, aa))
        baseline, candidate = statistics.median(a + aa), statistics.median(b)
        row[metric] = dict(main=baseline, head=candidate, delta_pct=100 * (candidate / baseline - 1),
                           aa_noise_pct=100 * noise, main_range=[min(a + aa), max(a + aa)], head_range=[min(b), max(b)])
    results.append(row)
miss = next(r for r in results if r['workload'] == 'all-new')
miss_ok = all(miss[m]['delta_pct'] <= 2 * miss[m]['aa_noise_pct'] for m in ['p50_ms', 'p95_ms'])
miss_ok &= miss['requests_s']['delta_pct'] >= -2 * miss['requests_s']['aa_noise_pct']
gain = any(-r['p50_ms']['delta_pct'] > 2 * r['p50_ms']['aa_noise_pct'] and
           r['requests_s']['delta_pct'] > 2 * r['requests_s']['aa_noise_pct'] for r in results if r != miss)
report = dict(method=method, results=results, miss_ok=miss_ok, meaningful=bool(miss_ok and gain), errors=0)
(output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
print('RESULT', json.dumps(report), flush=True)
table = ['| Workload | Main p50 ms | Cache p50 ms | Main p95 ms | Cache p95 ms | Main req/s | Cache req/s | p50 change | A/A p50 noise |',
         '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
for row in results:
    p50, p95, rate = row['p50_ms'], row['p95_ms'], row['requests_s']
    table.append(f"| {row['workload']} | {p50['main']:.3f} | {p50['head']:.3f} | {p95['main']:.3f} | {p95['head']:.3f} | {rate['main']:.2f} | {rate['head']:.2f} | {p50['delta_pct']:+.2f}% | {p50['aa_noise_pct']:.2f}% |")
table.append('')
table.append(f"All-new A/A noise: p50 {miss['p50_ms']['aa_noise_pct']:.2f}%, p95 {miss['p95_ms']['aa_noise_pct']:.2f}%, throughput {miss['requests_s']['aa_noise_pct']:.2f}%.")
table.append(f"All-new changes: p50 {miss['p50_ms']['delta_pct']:+.2f}%, p95 {miss['p95_ms']['delta_pct']:+.2f}%, throughput {miss['requests_s']['delta_pct']:+.2f}%.")
table.append(f"Completed {rounds} rotated A/A/B rounds, {count} measured requests per workload, arm and round, zero request or vector-equality errors.")
table.append(f"Decision: meaningful={report['meaningful']}, miss_ok={report['miss_ok']}.")
(output / 'summary.md').write_text('\n'.join(table) + '\n')
print('\n'.join(table), flush=True)
PY
witness post
