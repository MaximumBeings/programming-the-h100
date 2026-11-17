#!/usr/bin/env python3
"""Check the environment for this book without requiring PyTorch or a compiler run."""
import argparse
import ctypes
import datetime
import json
import platform
import re
import shutil
import subprocess
import sys


def command(arguments):
    try:
        process = subprocess.run(arguments, text=True, capture_output=True, timeout=30)
        return process.returncode, process.stdout.strip(), process.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return None, '', str(error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--minimum-gpus', type=int, default=1)
    parser.add_argument('--json', action='store_true', help='Print a machine-readable report.')
    args = parser.parse_args()
    if args.minimum_gpus < 1:
        parser.error('--minimum-gpus must be positive')
    errors, notes = [], []
    devices = []
    try:
        driver = ctypes.CDLL('libcuda.so.1')
        def checked(name, *arguments):
            code = getattr(driver, name)(*arguments)
            if code != 0:
                raise RuntimeError(f'{name} failed with CUDA driver error {code}')
        checked('cuInit', 0)
        count = ctypes.c_int()
        checked('cuDeviceGetCount', ctypes.byref(count))
        for ordinal in range(count.value):
            device, major, minor = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
            memory, name = ctypes.c_size_t(), ctypes.create_string_buffer(256)
            checked('cuDeviceGet', ctypes.byref(device), ordinal)
            checked('cuDeviceGetName', name, len(name), device)
            checked('cuDeviceComputeCapability', ctypes.byref(major), ctypes.byref(minor), device)
            checked('cuDeviceTotalMem_v2', ctypes.byref(memory), device)
            devices.append(dict(name=name.value.decode(), compute_capability=f'{major.value}.{minor.value}',
                                memory_mib=memory.value / (1024**2)))
    except (OSError, AttributeError, RuntimeError) as error:
        errors.append('CUDA driver discovery failed: ' + str(error))
    status, output, diagnostic = command(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'])
    driver_version = output.splitlines()[0] if status == 0 and output else None
    if len(devices) < args.minimum_gpus:
        errors.append(f'Requires {args.minimum_gpus} visible GPU(s); found {len(devices)}.')
    if devices and any(device['compute_capability'] != '9.0' for device in devices):
        errors.append('The WGMMA examples require Hopper compute capability 9.0 and sm_90a.')
    status, output, diagnostic = command(['nvcc', '--version'])
    match = re.search(r'release (\d+)\.(\d+), V([\d.]+)', output)
    toolkit = match[3] if status == 0 and match else None
    if toolkit is None:
        errors.append('nvcc was not found or its version could not be read.')
    elif (int(match[1]), int(match[2])) < (12, 0):
        errors.append('Hopper examples require a suitable CUDA 12+ toolkit.')
    elif toolkit != '12.8.93':
        notes.append('The captured book results used nvcc 12.8.93; this version needs its own validation.')
    available = {name: bool(shutil.which(name)) for name in ['ncu', 'compute-sanitizer', 'cuobjdump', 'nvdisasm']}
    if len(devices) == 1:
        notes.append('Single-GPU chapters are available; Chapter 11 workloads require 2 or 8 GPUs.')
    notes.append('Device discovery uses the CUDA Driver API and respects CUDA_VISIBLE_DEVICES.')
    notes.append('Driver compatibility, compiler version and the compiled architecture target are separate checks.')
    notes.append('This checks prerequisites, not kernel correctness, GPU interconnects or profiler permissions.')
    report = dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  os=platform.system(), python=platform.python_version(),
                  gpu_count=len(devices), devices=devices, driver=driver_version, nvcc=toolkit,
                  tools=available, status='PASS' if not errors else 'BLOCKED', errors=errors, notes=notes)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"H100 book preflight: {report['status']}")
        print(f"Visible GPUs: {len(devices)}; nvcc: {toolkit or 'unavailable'}")
        for device in devices:
            print(f"  {device['name']}; CC {device['compute_capability']}; {device['memory_mib']:g} MiB")
        for error in errors:
            print('BLOCKED:', error)
        for note in notes:
            print('NOTE:', note)
        print('Tools:', ', '.join(f'{name}={"found" if present else "missing"}' for name, present in available.items()))
    return 0 if not errors else 2


if __name__ == '__main__':
    sys.exit(main())
