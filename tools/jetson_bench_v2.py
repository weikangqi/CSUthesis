#!/usr/bin/env python3
"""
Jetson AGX Xavier TensorRT Benchmark v2
- FP16, INT8 (with random calibrator), Mixed precision
- Latency, memory, power via tegrastats
"""
import tensorrt as trt
import torch
import numpy as np
import time
import os
import threading
import json
import subprocess
import re

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
ONNX_PATH = os.path.expanduser("~/thesis_deploy/model.onnx")
ENGINE_DIR = os.path.expanduser("~/thesis_deploy")
WARMUP = 50
RUNS = 300

# Model I/O (from inspection)
# img_l: [1,3,480,640]  img_r: [1,3,480,640]
INPUT_SHAPES = {
    "img_l": (1, 3, 480, 640),
    "img_r": (1, 3, 480, 640),
}


# ── INT8 Calibrator ──────────────────────────────────────────────────────────

class RandomCalibrator(trt.IInt8EntropyCalibrator2):
    """Minimal entropy calibrator using random data (for engine build only)."""

    def __init__(self, n_batches=50):
        super().__init__()
        self.n_batches = n_batches
        self.batch_idx = 0
        self.cache_path = os.path.join(ENGINE_DIR, "calib_cache.bin")
        # Pre-allocate CUDA tensors for inputs
        self._buffers = {}
        for name, shape in INPUT_SHAPES.items():
            t = torch.zeros(shape, dtype=torch.float32, device="cuda")
            self._buffers[name] = t
        self._buf_list = list(self._buffers.values())
        self._ptrs = [t.data_ptr() for t in self._buf_list]

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.batch_idx >= self.n_batches:
            return None
        for t in self._buf_list:
            t.normal_()
        self.batch_idx += 1
        return self._ptrs

    def read_calibration_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_path, "wb") as f:
            f.write(cache)


# ── Power via tegrastats ─────────────────────────────────────────────────────

_HWMON4 = "/sys/devices/platform/c240000.i2c/i2c-1/1-0040/hwmon/hwmon4/"
_HWMON5 = "/sys/devices/platform/c240000.i2c/i2c-1/1-0041/hwmon/hwmon5/"


def _read_power_mw_once():
    """Sum all INA3221 channels for total board power (mW)."""
    total = 0.0
    found = False
    for base in [_HWMON4, _HWMON5]:
        for i in range(1, 4):
            try:
                v = float(open(f"{base}in{i}_input").read()) / 1000  # mV -> V
                c = float(open(f"{base}curr{i}_input").read())        # mA
                total += v * c  # mW
                found = True
            except Exception:
                pass
    return total if found else None


def read_power_samples(stop_event, samples):
    while not stop_event.is_set():
        p = _read_power_mw_once()
        if p is not None:
            samples.append(p)
        time.sleep(0.1)


# ── Memory ──────────────────────────────────────────────────────────────────

def get_memory_mb():
    try:
        for line in open(f"/proc/{os.getpid()}/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0


# ── Engine build ─────────────────────────────────────────────────────────────

def build_engine(onnx_path, precision):
    builder = trt.Builder(TRT_LOGGER)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  Parse error: {parser.get_error(i)}")
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

    calib = None
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)
        calib = RandomCalibrator(n_batches=50)
        config.int8_calibrator = calib
    elif precision == "mixed":
        # Mixed: FP16 globally, selective INT8 via calibrator
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.INT8)
        calib = RandomCalibrator(n_batches=50)
        config.int8_calibrator = calib

    print(f"  Building {precision} engine...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("  Build FAILED")
        return None

    out = os.path.join(ENGINE_DIR, f"model_{precision}.engine")
    with open(out, "wb") as f:
        f.write(serialized)
    print(f"  Saved: {out}")
    return out


# ── Benchmark ────────────────────────────────────────────────────────────────

def benchmark_engine(engine_path, label):
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # Allocate tensors
    bufs = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = [abs(s) for s in engine.get_tensor_shape(name)]
        dt = engine.get_tensor_dtype(name)
        torch_dt = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF:  torch.float16,
            trt.DataType.INT8:  torch.int8,
            trt.DataType.INT32: torch.int32,
        }.get(dt, torch.float32)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            t = torch.randn(shape, dtype=torch_dt, device="cuda")
        else:
            t = torch.zeros(shape, dtype=torch_dt, device="cuda")
        bufs[name] = t
        context.set_tensor_address(name, t.data_ptr())

    stream = torch.cuda.current_stream().cuda_stream
    torch.cuda.synchronize()

    # Warm-up
    for _ in range(WARMUP):
        context.execute_async_v3(stream)
        torch.cuda.synchronize()

    # Latency
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        context.execute_async_v3(stream)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    lat = float(np.mean(times))
    lat_std = float(np.std(times))

    # Memory
    mem = get_memory_mb()

    # Power (~3 s)
    pwr_samples = []
    stop = threading.Event()
    pt = threading.Thread(target=read_power_samples, args=(stop, pwr_samples))
    pt.start()
    t_end = time.time() + 3.0
    while time.time() < t_end:
        context.execute_async_v3(stream)
        torch.cuda.synchronize()
    stop.set()
    pt.join()

    avg_pwr_w = (float(np.mean(pwr_samples)) / 1000) if pwr_samples else None
    energy = (avg_pwr_w * lat / 1000) if avg_pwr_w else None

    return {
        "precision": label,
        "latency_ms": round(lat, 1),
        "latency_std_ms": round(lat_std, 2),
        "memory_mb": round(mem, 1),
        "avg_power_w": round(avg_pwr_w, 2) if avg_pwr_w else None,
        "energy_j_per_frame": round(energy, 4) if energy else None,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(f"TensorRT: {trt.__version__}  |  PyTorch: {torch.__version__}")
    print(f"Device:   {torch.cuda.get_device_name(0)}")
    print(f"ONNX:     {ONNX_PATH}")
    print("=" * 60)

    # Quick power sanity check
    p = _read_power_mw_once()
    print(f"Power sensor test: {p} mW  ({'OK' if p else 'not available - will skip power'})")

    configs = [
        ("fp16",  "FP16 engine"),
        ("int8",  "INT8 engine"),
        ("mixed", "Mixed precision engine"),
    ]

    results = []
    for key, label in configs:
        print(f"\n--- {label} ---")
        eng = os.path.join(ENGINE_DIR, f"model_{key}.engine")
        if not os.path.exists(eng):
            eng = build_engine(ONNX_PATH, key)
        if eng is None:
            continue
        print(f"  Benchmarking ({RUNS} iters)...")
        try:
            r = benchmark_engine(eng, label)
            results.append(r)
            print(f"  {json.dumps(r, indent=4)}")
        except Exception as e:
            print(f"  Benchmark failed: {e}")

    out = os.path.join(ENGINE_DIR, "bench_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")

    print("\n=== SUMMARY ===")
    hdr = f"{'Precision':<26} {'Lat(ms)':>8} {'Mem(MB)':>8} {'Pwr(W)':>7} {'Energy(J)':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r['precision']:<26} {r['latency_ms']:>8.1f} {r['memory_mb']:>8.0f} "
            f"{str(r['avg_power_w']):>7} {str(r['energy_j_per_frame']):>10}"
        )


if __name__ == "__main__":
    main()
