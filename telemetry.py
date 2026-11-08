"""Best-effort GPU telemetry; never changes the model or random state."""
from pathlib import Path
import csv
import datetime
import os
import subprocess
import threading
import time

class GPUMonitor:
    def __init__(self, output, interval=1.):
        self.output = Path(output)
        self.interval = interval
        self.stop = threading.Event()
        self.thread = None

    def _sample(self):
        path = self.output / 'gpu.csv'
        needs_header = not path.exists() or path.stat().st_size == 0
        with path.open('a', newline='', buffering=1) as stream:
            writer = csv.writer(stream)
            if needs_header:
                writer.writerow(['time_utc', 'invocation_seconds', 'physical_gpu_index', 'gpu_uuid',
                                 'utilization_percent', 'memory_used_mib', 'memory_total_mib',
                                 'power_w', 'temperature_c'])
            while not self.stop.is_set():
                tick = time.monotonic()
                try:
                    # Record device IDs; do not label global GPU0 as the CUDA-visible device.
                    command = ['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu',
                               '--format=csv,noheader,nounits']
                    response = subprocess.run(command, capture_output=True, text=True, timeout=3, check=True)
                    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    for row in csv.reader(response.stdout.splitlines()):
                        writer.writerow([now, time.monotonic() - self.started] + [x.strip() for x in row])
                except Exception as error:
                    with (self.output / 'telemetry_errors.txt').open('a') as errors:
                        errors.write(type(error).__name__ + ': ' + str(error) + '\n')
                self.stop.wait(max(.05, self.interval - (time.monotonic() - tick)))

    def __enter__(self):
        self.output.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, kind, value, traceback):
        self.stop.set()
        self.thread.join(timeout=5)
        with (self.output / 'run_times.txt').open('a') as stream:
            stream.write(f'utc={datetime.datetime.now(datetime.timezone.utc).isoformat()} '
                         f'wall_seconds={time.monotonic()-self.started:.6f} '
                         f'completed={kind is None}; includes training and evaluation; '
                         f'CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES", "unset")}\n')
