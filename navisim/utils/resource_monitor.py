import ray
import time
import psutil
import numpy as np
import threading
from typing import List, Dict, Any
import json
import csv
from datetime import datetime
import os

# Initialize Ray
ray.init()

# Resource monitoring utilities
class ResourceMonitor:
    def __init__(self):
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        self.timestamps = []
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self, interval=0.5):
        """Start monitoring system resources"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring and return collected data"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        return {
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'gpu_usage': self.gpu_usage,
            'timestamps': self.timestamps
        }
    
    def _monitor_loop(self, interval):
        """Internal monitoring loop"""
        while self.monitoring:
            # CPU and Memory usage
            cpu_percent = psutil.cpu_percent(interval=None)
            memory_percent = psutil.virtual_memory().percent
            
            # GPU usage (requires nvidia-ml-py3: pip install nvidia-ml-py3)
            gpu_percent = 0
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_percent = gpu_util.gpu
            except:
                # If no GPU or pynvml not available, set to 0
                gpu_percent = 0
            
            self.cpu_usage.append(cpu_percent)
            self.memory_usage.append(memory_percent)
            self.gpu_usage.append(gpu_percent)
            self.timestamps.append(datetime.now())
            
            time.sleep(interval)

# CPU-intensive task
@ray.remote
def cpu_intensive_task(task_id: int, size: int = 1000000) -> Dict[str, Any]:
    """CPU-intensive matrix multiplication task"""
    start_time = time.time()
    
    # Generate random matrices
    matrix_a = np.random.random((size // 1000, size // 1000))
    matrix_b = np.random.random((size // 1000, size // 1000))
    
    # Perform matrix multiplication
    result = np.dot(matrix_a, matrix_b)
    
    end_time = time.time()
    
    return {
        'task_id': task_id,
        'task_type': 'CPU',
        'execution_time': end_time - start_time,
        'result_shape': result.shape,
        'worker_pid': ray.worker.global_worker.worker_id
    }

# GPU-intensive task (simulated - replace with actual GPU operations)
@ray.remote(num_gpus=1)
def gpu_intensive_task(task_id: int, size: int = 1000000) -> Dict[str, Any]:
    """GPU-intensive task (simulated with numpy operations)"""
    start_time = time.time()
    
    # Simulate GPU-intensive computation
    # In real scenarios, use CuPy, PyTorch, or TensorFlow for actual GPU operations
    try:
        import cupy as cp
        # Actual GPU operations with CuPy
        matrix_a = cp.random.random((size // 1000, size // 1000))
        matrix_b = cp.random.random((size // 1000, size // 1000))
        result = cp.dot(matrix_a, matrix_b)
        cp.cuda.Stream.null.synchronize()  # Wait for GPU operations to complete
        result_shape = result.shape
    except ImportError:
        # Fallback to NumPy if CuPy not available
        print(f"GPU Task {task_id}: CuPy not available, using NumPy as fallback")
        matrix_a = np.random.random((size // 1000, size // 1000))
        matrix_b = np.random.random((size // 1000, size // 1000))
        result = np.dot(matrix_a, matrix_b)
        result_shape = result.shape
    
    end_time = time.time()
    
    return {
        'task_id': task_id,
        'task_type': 'GPU',
        'execution_time': end_time - start_time,
        'result_shape': result_shape,
        'worker_pid': ray.worker.global_worker.worker_id
    }

# Mixed CPU/GPU task
@ray.remote(num_cpus=2, num_gpus=0.5)
def mixed_compute_task(task_id: int) -> Dict[str, Any]:
    """Task that uses both CPU and GPU resources"""
    start_time = time.time()
    
    # CPU phase
    cpu_data = np.random.random((500, 500))
    cpu_result = np.linalg.svd(cpu_data)
    
    # GPU phase (simulated)
    try:
        import cupy as cp
        gpu_data = cp.asarray(cpu_data)
        gpu_result = cp.fft.fft2(gpu_data)
        cp.cuda.Stream.null.synchronize()
    except ImportError:
        # Fallback to NumPy
        gpu_result = np.fft.fft2(cpu_data)
    
    end_time = time.time()
    
    return {
        'task_id': task_id,
        'task_type': 'Mixed',
        'execution_time': end_time - start_time,
        'worker_pid': ray.worker.global_worker.worker_id
    }

def run_parallel_workload():
    """Run a mixed workload to demonstrate parallel GPU/CPU usage"""
    print("=== Ray Parallel GPU/CPU Compute Demo ===\n")
    
    # Initialize resource monitor
    monitor = ResourceMonitor()
    monitor.start_monitoring(interval=0.1)
    
    print("Ray cluster resources:")
    print(ray.cluster_resources())
    print()
    
    # Create a mixed workload
    tasks = []
    
    # Submit CPU-intensive tasks
    print("Submitting CPU-intensive tasks...")
    for i in range(4):
        task = cpu_intensive_task.remote(i, size=2000000)
        tasks.append(task)
    
    # Submit GPU-intensive tasks
    print("Submitting GPU-intensive tasks...")
    for i in range(2):
        task = gpu_intensive_task.remote(i + 100, size=1500000)
        tasks.append(task)
    
    # Submit mixed tasks
    print("Submitting mixed CPU/GPU tasks...")
    for i in range(2):
        task = mixed_compute_task.remote(i + 200)
        tasks.append(task)
    
    print(f"\nTotal tasks submitted: {len(tasks)}")
    print("Waiting for tasks to complete...")
    
    # Wait for all tasks to complete
    start_time = time.time()
    results = ray.get(tasks)
    end_time = time.time()
    
    # Stop monitoring
    resource_data = monitor.stop_monitoring()
    
    print(f"\nAll tasks completed in {end_time - start_time:.2f} seconds")
    
    # Analyze results
    print("\n=== Task Results ===")
    for result in results:
        print(f"Task {result['task_id']} ({result['task_type']}): "
              f"{result['execution_time']:.2f}s - Worker: {result['worker_pid']}")
    
    # Group results by type
    cpu_tasks = [r for r in results if r['task_type'] == 'CPU']
    gpu_tasks = [r for r in results if r['task_type'] == 'GPU']
    mixed_tasks = [r for r in results if r['task_type'] == 'Mixed']
    
    print(f"\nCPU tasks: {len(cpu_tasks)}, "
          f"Avg time: {np.mean([t['execution_time'] for t in cpu_tasks]):.2f}s")
    print(f"GPU tasks: {len(gpu_tasks)}, "
          f"Avg time: {np.mean([t['execution_time'] for t in gpu_tasks]):.2f}s")
    print(f"Mixed tasks: {len(mixed_tasks)}, "
          f"Avg time: {np.mean([t['execution_time'] for t in mixed_tasks]):.2f}s")
    
    # Display resource usage summary
    display_resource_summary(resource_data)
    
    # Export data for external analysis
    export_resource_data(resource_data, results)
    
    return results, resource_data

def display_resource_summary(resource_data):
    """Display comprehensive resource usage summary in text format"""
    if not resource_data['timestamps']:
        print("No resource data collected")
        return
    
    print("\n=== Resource Usage Summary ===")
    
    # Calculate statistics
    cpu_usage = resource_data['cpu_usage']
    memory_usage = resource_data['memory_usage']
    gpu_usage = resource_data['gpu_usage']
    
    def stats(data):
        if not data:
            return {'min': 0, 'max': 0, 'avg': 0, 'std': 0}
        return {
            'min': min(data),
            'max': max(data),
            'avg': sum(data) / len(data),
            'std': np.std(data)
        }
    
    cpu_stats = stats(cpu_usage)
    memory_stats = stats(memory_usage)
    gpu_stats = stats(gpu_usage)
    
    print(f"CPU Usage    - Min: {cpu_stats['min']:5.1f}%  Max: {cpu_stats['max']:5.1f}%  "
          f"Avg: {cpu_stats['avg']:5.1f}%  Std: {cpu_stats['std']:5.1f}%")
    print(f"Memory Usage - Min: {memory_stats['min']:5.1f}%  Max: {memory_stats['max']:5.1f}%  "
          f"Avg: {memory_stats['avg']:5.1f}%  Std: {memory_stats['std']:5.1f}%")
    print(f"GPU Usage    - Min: {gpu_stats['min']:5.1f}%  Max: {gpu_stats['max']:5.1f}%  "
          f"Avg: {gpu_stats['avg']:5.1f}%  Std: {gpu_stats['std']:5.1f}%")
    
    # Show timeline in text format
    print("\n=== Resource Usage Timeline (Text Chart) ===")
    create_text_chart(cpu_usage, "CPU Usage", "%")
    create_text_chart(memory_usage, "Memory Usage", "%")
    if any(gpu_usage):
        create_text_chart(gpu_usage, "GPU Usage", "%")
    
    # Show concurrent task analysis
    analyze_concurrent_execution(resource_data)

def create_text_chart(data, title, unit, width=60, height=10):
    """Create a simple ASCII chart"""
    if not data:
        return
    
    print(f"\n{title} Over Time:")
    
    # Normalize data to chart height
    max_val = max(data) if data else 1
    min_val = min(data) if data else 0
    
    # Create buckets for timeline
    bucket_size = max(1, len(data) // width)
    buckets = []
    for i in range(0, len(data), bucket_size):
        bucket = data[i:i + bucket_size]
        buckets.append(sum(bucket) / len(bucket) if bucket else 0)
    
    # Create chart
    for row in range(height, 0, -1):
        line = ""
        threshold = min_val + (max_val - min_val) * row / height
        for bucket_val in buckets:
            if bucket_val >= threshold:
                line += "█"
            else:
                line += " "
        print(f"{threshold:5.1f}|{line}")
    
    # X-axis
    print(f"     +{'-' * len(buckets)}")
    print(f"      0{' ' * (len(buckets) - 10)}time{' ' * 5}")
    print(f"Max: {max_val:.1f}{unit}, Min: {min_val:.1f}{unit}, Samples: {len(data)}")

def analyze_concurrent_execution(resource_data):
    """Analyze periods of high resource usage to identify concurrent execution"""
    print("\n=== Concurrent Execution Analysis ===")
    
    cpu_usage = resource_data['cpu_usage']
    gpu_usage = resource_data['gpu_usage']
    
    if not cpu_usage:
        return
    
    # Find periods of high CPU usage (indicating multiple tasks)
    cpu_threshold = 50  # Adjust based on your system
    high_cpu_periods = []
    
    for i, usage in enumerate(cpu_usage):
        if usage > cpu_threshold:
            high_cpu_periods.append(i)
    
    if high_cpu_periods:
        print(f"High CPU usage detected in {len(high_cpu_periods)} samples "
              f"({len(high_cpu_periods)/len(cpu_usage)*100:.1f}% of time)")
        
        # Find consecutive periods
        periods = []
        if high_cpu_periods:
            start = high_cpu_periods[0]
            end = high_cpu_periods[0]
            
            for i in range(1, len(high_cpu_periods)):
                if high_cpu_periods[i] == end + 1:
                    end = high_cpu_periods[i]
                else:
                    periods.append((start, end))
                    start = high_cpu_periods[i]
                    end = high_cpu_periods[i]
            periods.append((start, end))
        
        print(f"Found {len(periods)} distinct high-activity periods")
        for i, (start, end) in enumerate(periods[:5]):  # Show first 5
            duration = (end - start + 1) * 0.1  # assuming 0.1s intervals
            avg_cpu = sum(cpu_usage[start:end+1]) / (end - start + 1)
            print(f"  Period {i+1}: {duration:.1f}s duration, {avg_cpu:.1f}% avg CPU")
    
    # GPU analysis
    if any(gpu_usage):
        gpu_active = sum(1 for usage in gpu_usage if usage > 5)
        print(f"GPU was active for {gpu_active} samples "
              f"({gpu_active/len(gpu_usage)*100:.1f}% of time)")

def export_resource_data(resource_data, task_results):
    """Export resource data and task results for external analysis"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export resource data to CSV
    csv_filename = f"ray_resources_{timestamp}.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['timestamp', 'cpu_usage', 'memory_usage', 'gpu_usage'])
        
        for i, ts in enumerate(resource_data['timestamps']):
            writer.writerow([
                ts.isoformat(),
                resource_data['cpu_usage'][i] if i < len(resource_data['cpu_usage']) else 0,
                resource_data['memory_usage'][i] if i < len(resource_data['memory_usage']) else 0,
                resource_data['gpu_usage'][i] if i < len(resource_data['gpu_usage']) else 0
            ])
    
    # Export task results to JSON
    json_filename = f"ray_tasks_{timestamp}.json"
    with open(json_filename, 'w') as jsonfile:
        json.dump({
            'task_results': task_results,
            'summary': {
                'total_tasks': len(task_results),
                'cpu_tasks': len([t for t in task_results if t['task_type'] == 'CPU']),
                'gpu_tasks': len([t for t in task_results if t['task_type'] == 'GPU']),
                'mixed_tasks': len([t for t in task_results if t['task_type'] == 'Mixed']),
                'total_execution_time': sum(t['execution_time'] for t in task_results),
                'avg_execution_time': sum(t['execution_time'] for t in task_results) / len(task_results)
            }
        }, jsonfile, indent=2, default=str)
    
    print(f"\n=== Data Export ===")
    print(f"Resource data exported to: {csv_filename}")
    print(f"Task results exported to: {json_filename}")
    print(f"You can analyze these files with:")
    print(f"  - Python: pandas.read_csv('{csv_filename}')")
    print(f"  - R: read.csv('{csv_filename}')")
    print(f"  - Excel: Open {csv_filename} directly")

def plot_resource_usage(resource_data):
    """Legacy function - now redirects to text-based display"""
    print("\n=== Resource Visualization ===")
    print("GUI plotting not available on remote server.")
    print("Using text-based resource summary instead...")
    display_resource_summary(resource_data)

def demonstrate_concurrent_cpu_gpu():
    """Demonstrate concurrent CPU and GPU processing with detailed monitoring"""
    print("\n" + "="*70)
    print("CONCURRENT CPU/GPU PROCESSING DEMO")
    print("="*70)
    
    # Resource monitor for this specific demo
    monitor = ResourceMonitor()
    monitor.start_monitoring(interval=0.2)  # Higher frequency for better granularity
    
    print("This demo will run CPU and GPU tasks simultaneously to show:")
    print("1. Tasks start at the same time")
    print("2. CPU and GPU resources are used in parallel")
    print("3. Tasks complete independently")
    print()
    
    # Define concurrent tasks with clear timing
    @ray.remote(num_cpus=2)
    def cpu_heavy_task(task_id, duration=5):
        """CPU-intensive task with controlled duration"""
        start_time = time.time()
        print(f"🖥️  CPU Task {task_id} STARTED at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
        
        # CPU-intensive work
        result = 0
        target_time = start_time + duration
        
        while time.time() < target_time:
            # Keep CPU busy with meaningful work
            for i in range(100000):
                result += i ** 0.5
            
            # Brief check to avoid infinite loop
            if time.time() - start_time > duration + 1:
                break
        
        end_time = time.time()
        print(f"🖥️  CPU Task {task_id} COMPLETED at {datetime.now().strftime('%H:%M:%S.%f')[:-3]} "
              f"(Duration: {end_time - start_time:.2f}s)")
        
        return {
            'task_id': task_id,
            'task_type': 'CPU_HEAVY',
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time,
            'result_sample': result % 1000000
        }
    
    @ray.remote(num_gpus=1)
    def gpu_heavy_task(task_id, duration=5):
        """GPU-intensive task with controlled duration"""
        start_time = time.time()
        print(f"🎮 GPU Task {task_id} STARTED at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
        
        try:
            import cupy as cp
            # Actual GPU operations
            target_time = start_time + duration
            
            while time.time() < target_time:
                # GPU matrix operations
                size = 1000
                a = cp.random.random((size, size), dtype=cp.float32)
                b = cp.random.random((size, size), dtype=cp.float32)
                c = cp.dot(a, b)
                cp.cuda.Stream.null.synchronize()  # Ensure GPU work completes
                
                # Brief check to avoid infinite loop
                if time.time() - start_time > duration + 1:
                    break
            
            result_sample = float(cp.sum(c)) % 1000000
            
        except ImportError:
            print(f"🎮 GPU Task {task_id}: CuPy not available, simulating GPU work with NumPy")
            # Simulate GPU work with CPU operations
            target_time = start_time + duration
            
            while time.time() < target_time:
                size = 500  # Smaller for CPU simulation
                a = np.random.random((size, size)).astype(np.float32)
                b = np.random.random((size, size)).astype(np.float32)
                c = np.dot(a, b)
                
                # Brief check to avoid infinite loop
                if time.time() - start_time > duration + 1:
                    break
            
            result_sample = float(np.sum(c)) % 1000000
        
        end_time = time.time()
        print(f"🎮 GPU Task {task_id} COMPLETED at {datetime.now().strftime('%H:%M:%S.%f')[:-3]} "
              f"(Duration: {end_time - start_time:.2f}s)")
        
        return {
            'task_id': task_id,
            'task_type': 'GPU_HEAVY',
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time,
            'result_sample': result_sample
        }
    
    @ray.remote(num_cpus=1, num_gpus=0.5)
    def mixed_task(task_id, duration=4):
        """Task that alternates between CPU and GPU work"""
        start_time = time.time()
        print(f"🔄 Mixed Task {task_id} STARTED at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
        
        phases = []
        target_time = start_time + duration
        phase = 0
        
        while time.time() < target_time:
            phase_start = time.time()
            
            if phase % 2 == 0:
                # CPU phase
                cpu_result = sum(i ** 0.5 for i in range(500000))
                phases.append(f"CPU_{phase}")
            else:
                # GPU phase
                try:
                    import cupy as cp
                    a = cp.random.random((500, 500))
                    b = cp.dot(a, a.T)
                    cp.cuda.Stream.null.synchronize()
                except ImportError:
                    # CPU fallback
                    a = np.random.random((300, 300))
                    b = np.dot(a, a.T)
                phases.append(f"GPU_{phase}")
            
            phase += 1
            
            # Brief check to avoid infinite loop
            if time.time() - start_time > duration + 1:
                break
        
        end_time = time.time()
        print(f"🔄 Mixed Task {task_id} COMPLETED at {datetime.now().strftime('%H:%M:%S.%f')[:-3]} "
              f"(Duration: {end_time - start_time:.2f}s, Phases: {len(phases)})")
        
        return {
            'task_id': task_id,
            'task_type': 'MIXED',
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time,
            'phases_completed': len(phases)
        }
    
    # Show initial resources
    print("Ray Cluster Resources:")
    resources = ray.cluster_resources()
    for resource, amount in resources.items():
        print(f"  {resource}: {amount}")
    print()
    
    # Submit all tasks simultaneously
    print("Submitting tasks for concurrent execution...")
    demo_start = time.time()
    
    # Submit tasks with slight delays to show scheduling
    tasks = []
    
    # CPU tasks
    print("Submitting 2 CPU-heavy tasks...")
    for i in range(2):
        task = cpu_heavy_task.remote(i, duration=6)
        tasks.append(task)
        time.sleep(0.1)  # Brief delay to see scheduling
    
    # GPU tasks
    print("Submitting 2 GPU-heavy tasks...")
    for i in range(2):
        task = gpu_heavy_task.remote(i, duration=5)
        tasks.append(task)
        time.sleep(0.1)
    
    # Mixed tasks
    print("Submitting 2 mixed CPU/GPU tasks...")
    for i in range(2):
        task = mixed_task.remote(i, duration=4)
        tasks.append(task)
        time.sleep(0.1)
    
    print(f"\n🚀 All {len(tasks)} tasks submitted! Monitoring execution...")
    print()
    
    # Monitor task completion in real-time
    completed_tasks = []
    start_monitoring = time.time()
    
    while len(completed_tasks) < len(tasks):
        # Check for completed tasks
        ready, not_ready = ray.wait(tasks, timeout=0.5)
        
        for task in ready:
            if task not in completed_tasks:
                result = ray.get(task)
                completed_tasks.append(task)
                elapsed = time.time() - demo_start
                print(f"✅ {result['task_type']} Task {result['task_id']} finished "
                      f"(Total elapsed: {elapsed:.1f}s)")
        
        # Show current resource usage
        if len(completed_tasks) < len(tasks):
            available = ray.available_resources()
            cpu_used = resources.get('CPU', 0) - available.get('CPU', 0)
            gpu_used = resources.get('GPU', 0) - available.get('GPU', 0)
            print(f"📊 Currently using: {cpu_used:.1f}/{resources.get('CPU', 0)} CPUs, "
                  f"{gpu_used:.1f}/{resources.get('GPU', 0)} GPUs "
                  f"({len(tasks) - len(completed_tasks)} tasks remaining)")
        
        tasks = not_ready
        time.sleep(1)
    
    # Stop monitoring and analyze
    resource_data = monitor.stop_monitoring()
    total_time = time.time() - demo_start
    
    print(f"\n🎉 All tasks completed in {total_time:.2f} seconds!")
    
    # Analyze concurrent execution
    analyze_concurrent_execution_detailed(completed_tasks, resource_data, demo_start)
    
    return completed_tasks, resource_data

def analyze_concurrent_execution_detailed(task_results, resource_data, demo_start):
    """Detailed analysis of concurrent execution patterns"""
    print("\n" + "="*70)
    print("CONCURRENT EXECUTION ANALYSIS")
    print("="*70)
    
    # Get actual task results
    results = [ray.get(task) for task in task_results]
    
    # Timeline analysis
    print("\n📅 Task Timeline:")
    print("Task Type | Task ID | Start Time | End Time   | Duration")
    print("-" * 55)
    
    for result in sorted(results, key=lambda x: x['start_time']):
        start_offset = result['start_time'] - demo_start
        end_offset = result['end_time'] - demo_start
        print(f"{result['task_type']:9} | {result['task_id']:7} | "
              f"{start_offset:8.1f}s | {end_offset:8.1f}s | {result['duration']:6.1f}s")
    
    # Concurrency analysis
    print("\n🔄 Concurrency Analysis:")
    
    # Find overlapping time periods
    time_points = []
    for result in results:
        time_points.append((result['start_time'], 'start', result))
        time_points.append((result['end_time'], 'end', result))
    
    time_points.sort(key=lambda x: x[0])
    
    active_tasks = []
    max_concurrent = 0
    concurrent_periods = []
    
    for timestamp, event_type, task_result in time_points:
        if event_type == 'start':
            active_tasks.append(task_result)
        else:
            active_tasks.remove(task_result)
        
        max_concurrent = max(max_concurrent, len(active_tasks))
        
        if len(active_tasks) > 1:
            task_types = [t['task_type'] for t in active_tasks]
            offset = timestamp - demo_start
            concurrent_periods.append((offset, len(active_tasks), task_types))
    
    print(f"Maximum concurrent tasks: {max_concurrent}")
    
    # Check for true CPU/GPU concurrency
    cpu_gpu_concurrent = False
    for offset, count, types in concurrent_periods:
        has_cpu = any('CPU' in t for t in types)
        has_gpu = any('GPU' in t for t in types)
        if has_cpu and has_gpu:
            cpu_gpu_concurrent = True
            print(f"✅ CPU/GPU concurrency detected at {offset:.1f}s: {types}")
            break
    
    if not cpu_gpu_concurrent:
        print("❌ No simultaneous CPU/GPU execution detected")
    
    # Resource efficiency
    cpu_tasks = [r for r in results if 'CPU' in r['task_type']]
    gpu_tasks = [r for r in results if 'GPU' in r['task_type']]
    
    if cpu_tasks and gpu_tasks:
        avg_cpu_time = sum(t['duration'] for t in cpu_tasks) / len(cpu_tasks)
        avg_gpu_time = sum(t['duration'] for t in gpu_tasks) / len(gpu_tasks)
        
        print(f"\n📊 Performance Summary:")
        print(f"CPU tasks: {len(cpu_tasks)} tasks, avg {avg_cpu_time:.2f}s")
        print(f"GPU tasks: {len(gpu_tasks)} tasks, avg {avg_gpu_time:.2f}s")
        
        # Calculate theoretical vs actual time
        total_cpu_time = sum(t['duration'] for t in cpu_tasks)
        total_gpu_time = sum(t['duration'] for t in gpu_tasks)
        sequential_time = total_cpu_time + total_gpu_time
        actual_time = max(r['end_time'] for r in results) - min(r['start_time'] for r in results)
        
        efficiency = sequential_time / actual_time if actual_time > 0 else 1
        print(f"Parallelization efficiency: {efficiency:.2f}x speedup")
        print(f"Sequential time would be: {sequential_time:.1f}s")
        print(f"Actual parallel time: {actual_time:.1f}s")
    
    # Resource usage summary
    if resource_data['cpu_usage']:
        avg_cpu = sum(resource_data['cpu_usage']) / len(resource_data['cpu_usage'])
        max_cpu = max(resource_data['cpu_usage'])
        print(f"\n🖥️  CPU Usage: avg {avg_cpu:.1f}%, peak {max_cpu:.1f}%")
    
    if resource_data['gpu_usage'] and any(resource_data['gpu_usage']):
        avg_gpu = sum(resource_data['gpu_usage']) / len(resource_data['gpu_usage'])
        max_gpu = max(resource_data['gpu_usage'])
        print(f"🎮 GPU Usage: avg {avg_gpu:.1f}%, peak {max_gpu:.1f}%")
    
    print("\n✅ Concurrent CPU/GPU execution demo completed!")

def real_time_monitor():
    """Real-time resource monitoring for remote servers"""
    print("\n=== Real-time Resource Monitor ===")
    print("Press Ctrl+C to stop monitoring")
    print("Monitoring system resources every 2 seconds...")
    print()
    print("Time     | CPU%  | Memory% | GPU%  | Ray Tasks | Ray Actors")
    print("-" * 65)
    
    try:
        while True:
            # Get system resources
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            # Get GPU usage
            gpu_percent = 0
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_percent = gpu_util.gpu
            except:
                gpu_percent = 0
            
            # Get Ray cluster info
            try:
                cluster_resources = ray.cluster_resources()
                available_resources = ray.available_resources()
                
                # Count active tasks (approximation)
                cpu_used = cluster_resources.get('CPU', 0) - available_resources.get('CPU', 0)
                gpu_used = cluster_resources.get('GPU', 0) - available_resources.get('GPU', 0)
                
                ray_info = f"{cpu_used:.1f}CPU {gpu_used:.1f}GPU"
            except:
                ray_info = "N/A"
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"{timestamp} | {cpu_percent:4.1f}% | {memory_percent:6.1f}% | {gpu_percent:4.1f}% | {ray_info:>12}")
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")

def check_ray_dashboard():
    """Display Ray dashboard information for remote access"""
    print("\n=== Ray Dashboard Info ===")
    print("For remote server access, you have several options:")
    print()
    print("1. SSH Tunnel (Recommended):")
    print("   ssh -L 8265:localhost:8265 user@your-remote-server")
    print("   Then access: http://localhost:8265")
    print()
    print("2. Ray Dashboard CLI:")
    print("   ray status  # Show cluster status")
    print("   ray list tasks  # List active tasks")
    print("   ray list actors  # List active actors")
    print()
    print("3. Command-line monitoring:")
    print("   Run real_time_monitor() for live resource tracking")
    print()
    print("4. Export data for analysis:")
    print("   Resource data is automatically exported to CSV/JSON files")

def quick_concurrent_test():
    """Quick test to verify CPU/GPU can run concurrently"""
    print("\n=== Quick Concurrent Execution Test ===")
    print("This test runs 1 CPU task and 1 GPU task simultaneously")
    print("to verify concurrent execution capability.\n")
    
    @ray.remote(num_cpus=1)
    def quick_cpu_task():
        start = time.time()
        print(f"⚡ CPU task starting at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
        
        # CPU work for 3 seconds
        result = 0
        end_time = start + 3
        while time.time() < end_time:
            result += sum(range(10000))
        
        finish = time.time()
        print(f"⚡ CPU task finished at {datetime.now().strftime('%H:%M:%S.%f')[:-3]} ({finish-start:.1f}s)")
        return {'type': 'CPU', 'duration': finish - start}
    
    @ray.remote(num_gpus=1)
    def quick_gpu_task():
        start = time.time()
        print(f"🚀 GPU task starting at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
        
        try:
            import cupy as cp
            # GPU work for 3 seconds
            end_time = start + 3
            while time.time() < end_time:
                a = cp.random.random((500, 500))
                b = cp.dot(a, a)
                cp.cuda.Stream.null.synchronize()
        except ImportError:
            print("🚀 GPU task using NumPy fallback")
            end_time = start + 3
            while time.time() < end_time:
                a = np.random.random((300, 300))
                b = np.dot(a, a)
        
        finish = time.time()
        print(f"🚀 GPU task finished at {datetime.now().strftime('%H:%M:%S.%f')[:-3]} ({finish-start:.1f}s)")
        return {'type': 'GPU', 'duration': finish - start}
    
    # Submit both tasks at the same time
    print("Submitting both tasks...")
    cpu_task = quick_cpu_task.remote()
    gpu_task = quick_gpu_task.remote()
    
    # Wait for both to complete
    start_wait = time.time()
    results = ray.get([cpu_task, gpu_task])
    total_time = time.time() - start_wait
    
    print(f"\n✅ Both tasks completed in {total_time:.1f}s")
    print(f"If running sequentially, would take ~{sum(r['duration'] for r in results):.1f}s")
    
    if total_time < sum(r['duration'] for r in results) * 0.8:
        print("🎉 SUCCESS: Tasks ran concurrently!")
    else:
        print("⚠️  Tasks may have run sequentially")
    
    return results
    """Demonstrate how tasks are scheduled and show resource allocation"""
    print("\n=== Task Scheduling Demo ===")
    
    # Show current cluster resources
    print("Current Ray cluster resources:")
    resources = ray.cluster_resources()
    for resource, amount in resources.items():
        print(f"  {resource}: {amount}")
    
    print("\nAvailable resources:")
    available = ray.available_resources()
    for resource, amount in available.items():
        print(f"  {resource}: {amount}")
    
    # Submit tasks with different resource requirements
    print("\nSubmitting tasks with different resource requirements...")
    
    @ray.remote(num_cpus=1)
    def light_task(task_id):
        time.sleep(2)
        return f"Light task {task_id} completed"
    
    @ray.remote(num_cpus=2)
    def heavy_task(task_id):
        time.sleep(3)
        return f"Heavy task {task_id} completed"
    
    # Submit tasks
    tasks = []
    for i in range(3):
        tasks.append(light_task.remote(i))
    for i in range(2):
        tasks.append(heavy_task.remote(i))
    
    print(f"Submitted {len(tasks)} tasks")
    
    # Monitor resource usage while tasks run
    start_time = time.time()
    completed = []
    
    while len(completed) < len(tasks):
        # Check for completed tasks
        ready, not_ready = ray.wait(tasks, timeout=1)
        for task in ready:
            if task not in completed:
                result = ray.get(task)
                completed.append(task)
                print(f"✓ {result} (Total completed: {len(completed)}/{len(tasks)})")
        
        # Show current resource usage
        available = ray.available_resources()
        cpu_used = resources.get('CPU', 0) - available.get('CPU', 0)
        print(f"  Currently using {cpu_used:.1f}/{resources.get('CPU', 0)} CPUs")
        
        tasks = not_ready
    
    total_time = time.time() - start_time
    print(f"\nAll tasks completed in {total_time:.2f} seconds")

if __name__ == "__main__":
    try:
        # Run the main demo
        results, resource_data = run_parallel_workload()
        
        # Quick test option
        print("Run a quick 6-second test to verify concurrent execution?")
        response = input("Run quick test? (y/n): ").lower()
        if response.startswith('y'):
            quick_concurrent_test()
        
        # NEW: Detailed concurrent CPU/GPU demo
        print("\nRun the detailed concurrent CPU/GPU demo?")
        print("This will show CPU and GPU tasks running simultaneously with")
        print("real-time monitoring and detailed analysis.")
        response = input("Run concurrent demo? (y/n): ").lower()
        if response.startswith('y'):
            demonstrate_concurrent_cpu_gpu()
        
        # Additional demonstrations
        # demonstrate_task_scheduling()
        # check_ray_dashboard()
        
        # Optional: Start real-time monitoring
        # print("\n" + "="*60)
        # response = input("Would you like to start real-time monitoring? (y/n): ").lower()
        # if response.startswith('y'):
        #     real_time_monitor()
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up Ray
        ray.shutdown()
        print("\nRay cluster shutdown complete")