#!/usr/bin/env python3
"""
Jetson AGX Xavier TensorRT Benchmark (PyTorch-based, no pycuda needed)
Tests FP16, INT8, and Mixed precision engines
Measures: latency, memory, power
"""
import tensorrt as trt
import torch
import numpy as np
import time
import os
import threading
import json
import subprocess

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

ONNX_PATH = os.path.expanduser("~/thesis_deploy/model.onnx")
ENGINE_DIR = os.path.expanduser("~/thesis_deploy")
WARMUP = 30
RUNS = 200


def read_power_samples(stop_event, samples):
    """Thread: collect power samples from sysfs (Jetson INA3221)"""
    # Try standard Xavier power paths
    candidate_paths = [
        "/sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input",
        "/sys/bus/i2c/drivers/ina3221x/0-0040/iio:device0/in_power0_input",
        "/sys/bus/i2c/drivers/ina3221x/7-0040/iio:device0/in_power0_input",
    ]
    active_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            active_path = p
            break

    while not stop_event.is_set():
        if active_path:
            try:
                with open(active_path) as f:
                    val = float(f.read().strip())
                    if val > 100000:  # uW -> mW
                        val /= 1000
                    samples.append(val)
            except Exception:
                pass
        else:
            # Try tegrastats as fallback
            try:
                out = subprocess.check_output(
                    ["tegrastats", "--interval", "200"],
                    stderr=subprocess.DEVNULL, timeout=0.5
                ).decode()
                import re
                m = re.search(r'VDD_IN (\d+)mW', out)
                if m:
                    samples.append(float(m.group(1)))
            except Exception:
                pass
        time.sleep(0.1)


def get_memory_mb():
    """Get process RSS in MB"""
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0


def inspect_onnx(onnx_path):
    try:
        import onnx
        model = onnx.load(onnx_path)
        inputs = {}
        for inp in model.graph.input:
            dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            print(f"  Input  {inp.name}: {dims}")
            inputs[inp.name] = dims
        for out in model.graph.output:
            dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
            print(f"  Output {out.name}: {dims}")
        return inputs
    except Exception as e:
        print(f"  Could not inspect ONNX: {e}")
        return {}


def build_engine(onnx_path, precision="fp16"):
    """Build a TensorRT engine from ONNX"""
    builder = trt.Builder(TRT_LOGGER)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        ok = parser.parse(f.read())
    if not ok:
        for i in range(parser.num_errors):
            print(f"  Parse error: {parser.get_error(i)}")
        return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "mixed":
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.INT8)

    print(f"  Building engine (this may take a few minutes)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("  Build failed!")
        return None

    engine_path = os.path.join(ENGINE_DIR, f"model_{precision}.engine")
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"  Engine saved: {engine_path}")
    return engine_path


def benchmark_engine(engine_path, precision_name):
    """Benchmark TensorRT engine using PyTorch CUDA tensors"""
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # Allocate PyTorch CUDA tensors as TRT buffers
    tensors = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = [abs(s) for s in engine.get_tensor_shape(name)]
        trt_dtype = engine.get_tensor_dtype(name)

        if trt_dtype == trt.DataType.FLOAT:
            dtype = torch.float32
        elif trt_dtype == trt.DataType.HALF:
            dtype = torch.float16
        elif trt_dtype == trt.DataType.INT8:
            dtype = torch.int8
        elif trt_dtype == trt.DataType.INT32:
            dtype = torch.int32
        else:
            dtype = torch.float32

        t = torch.zeros(shape, dtype=dtype, device="cuda")
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            t = torch.randn(shape, dtype=dtype, device="cuda")
        tensors[name] = t
        context.set_tensor_address(name, t.data_ptr())

    # Synchronize stream
    torch.cuda.synchronize()

    # Warm-up
    for _ in range(WARMUP):
        context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()

    # Latency
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    latency_ms = float(np.mean(times))
    latency_std = float(np.std(times))

    # Memory (RSS after engine loaded + warmup)
    mem_mb = get_memory_mb()

    # Power: sample for ~3 s while running inference
    power_samples = []
    stop_event = threading.Event()
    pwr_thread = threading.Thread(
        target=read_power_samples, args=(stop_event, power_samples)
    )
    pwr_thread.start()

    t_end = time.time() + 3.0
    while time.time() < t_end:
        context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()

    stop_event.set()
    pwr_thread.join()

    avg_power_w = (float(np.mean(power_samples)) / 1000) if power_samples else None
    energy_j = (avg_power_w * latency_ms / 1000) if avg_power_w else None

    return {
        "precision": precision_name,
        "latency_ms": round(latency_ms, 1),
        "latency_std_ms": round(latency_std, 2),
        "memory_mb": round(mem_mb, 1),
        "avg_power_w": round(avg_power_w, 2) if avg_power_w else None,
        "energy_j_per_frame": round(energy_j, 4) if energy_j else None,
    }


def main():
    print("=" * 60)
    print(f"TensorRT: {trt.__version__}")
    print(f"PyTorch:  {torch.__version__}")
    print(f"CUDA:     {torch.cuda.get_device_name(0)}")
    print(f"ONNX:     {ONNX_PATH}")
    print("=" * 60)
    print("Model I/O:")
    inspect_onnx(ONNX_PATH)

    configs = [
        ("fp16", "FP16 engine"),
        ("int8", "INT8 engine"),
        ("mixed", "Mixed precision engine"),
    ]

    results = []
    for prec_key, prec_name in configs:
        print(f"\n--- {prec_name} ---")
        engine_path = os.path.join(ENGINE_DIR, f"model_{prec_key}.engine")
        if not os.path.exists(engine_path):
            engine_path = build_engine(ONNX_PATH, prec_key)
        if engine_path is None:
            print(f"  Skipping {prec_name}")
            continue
        print(f"  Running benchmark ({RUNS} iters, {WARMUP} warmup)...")
        try:
            r = benchmark_engine(engine_path, prec_name)
            results.append(r)
            print(f"  {json.dumps(r, indent=4)}")
        except Exception as e:
            print(f"  Benchmark failed: {e}")

    out_path = os.path.join(ENGINE_DIR, "bench_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    print("\n=== SUMMARY ===")
    hdr = f"{'Precision':<26} {'Latency(ms)':>12} {'Memory(MB)':>11} {'Power(W)':>9} {'Energy(J/f)':>12}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r['precision']:<26} {r['latency_ms']:>12.1f} {r['memory_mb']:>11.0f} "
            f"{str(r['avg_power_w']):>9} {str(r['energy_j_per_frame']):>12}"
        )


if __name__ == "__main__":
    main()
