import psutil
import GPUtil
import time
import csv
from datetime import datetime

class ResourceLogger:
    def __init__(self, log_file='resource_usage.csv'):
        self.log_file = log_file
        self._init_csv()

    def _init_csv(self):
        with open(self.log_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'render_step',
                'render_fps',
                'render_time_s',
                'cpu_percent',
                'memory_percent',
                'memory_used_MB',
                'memory_total_MB',
                'gpu_id',
                'gpu_name',
                'gpu_load_percent',
                'gpu_mem_used_MB',
                'gpu_mem_free_MB',
                'gpu_temp_C'
            ])

    def log(self, step=None, fps=None, render_time=None):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory()
        gpus = GPUtil.getGPUs()

        if not gpus:
            rows = [[timestamp, step, fps, render_time, cpu, mem.percent,
                     mem.used // (1024**2), mem.total // (1024**2),
                     '', '', '', '', '', '']]
        else:
            rows = []
            for gpu in gpus:
                rows.append([
                    timestamp,
                    step,
                    fps,
                    render_time,
                    cpu,
                    mem.percent,
                    mem.used // (1024**2),
                    mem.total // (1024**2),
                    gpu.id,
                    gpu.name,
                    round(gpu.load * 100, 1),
                    gpu.memoryUsed,
                    gpu.memoryFree,
                    gpu.temperature
                ])

        with open(self.log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)