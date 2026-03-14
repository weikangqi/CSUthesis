#!/usr/bin/env python3
"""
Jetson AGX Xavier TensorRT Benchmark
Tests FP16, INT8, and Mixed precision engines
Measures: latency, memory, power
"""
import tensorrt as trt
import numpy as np
import time
import os
import threading
import json

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

ONNX_PATH = os.path.expanduser("~/thesis_deploy/model.onnx")
ENGINE_DIR = os.path.expanduser("~/thesis_deploy")
WARMUP = 30
RUNS = 200


def read_power_samples(stop_event, samples):
    """Thread: collect power samples from sysfs"""
    # Jetson Xavier power sensor paths (try common locations)
    paths = [
        "/sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input",
        "/sys/bus/i2c/drivers/ina3221x/0-0040/iio:device0/in_power0_input",
        "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1/in1_input",
    ]
    active_path = None
    for p in paths:
        if os.path.exists(p):
            active_path = p
            break

    while not stop_event.is_set():
        if active_path:
            try:
                with open(active_path) as f:
                    val = float(f.read().strip())
                    # Convert to mW if in uW
                    if val > 100000:
                        val /= 1000
                    samples.append(val)
            except Exception:
                pass
        time.sleep(0.05)


def get_process_memory_mb():
    """Get current process RSS memory in MB"""
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0


def build_engine(onnx_path, precision="fp16"):
    """Build and serialize a TensorRT engine"""
    builder = trt.Builder(TRT_LOGGER)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        ok = parser.parse(f.read())
    if not ok:
        for i in range(parser.num_errors):
            print(f"  ONNX parse error: {parser.get_error(i)}")
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

    print(f"  Building {precision} engine (may take a few minutes)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print(f"  Failed to build {precision} engine")
        return None

    engine_path = os.path.join(ENGINE_DIR, f"model_{precision}.engine")
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"  Saved: {engine_path}")
    return engine_path


def benchmark_engine(engine_path, precision_name):
    """Load and benchmark a TensorRT engine"""
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401 - initializes CUDA

    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # Allocate I/O buffers
    device_buffers = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = [abs(s) for s in engine.get_tensor_shape(name)]
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        buf = cuda.mem_alloc(size)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            # Fill input with random data
            host = np.random.randn(*shape).astype(dtype)
            cuda.memcpy_htod(buf, host)
        context.set_tensor_address(name, int(buf))
        device_buffers.append(buf)

    stream = cuda.Stream()

    # Warm-up
    for _ in range(WARMUP):
        context.execute_async_v3(stream.handle)
        stream.synchronize()

    # --- Latency measurement ---
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        context.execute_async_v3(stream.handle)
        stream.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    latency_ms = float(np.mean(times))
    latency_std = float(np.std(times))

    # --- Memory measurement ---
    mem_mb = get_process_memory_mb()

    # --- Power measurement (sample for ~3 s during inference) ---
    power_samples = []
    stop_event = threading.Event()
    pwr_thread = threading.Thread(
        target=read_power_samples, args=(stop_event, power_samples)
    )
    pwr_thread.start()
    t_end = time.time() + 3.0
    while time.time() < t_end:
        context.execute_async_v3(stream.handle)
        stream.synchronize()
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


def inspect_onnx(onnx_path):
    try:
        import onnx
        model = onnx.load(onnx_path)
        print("ONNX inputs:")
        for inp in model.graph.input:
            dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            print(f"  {inp.name}: {dims}")
        print("ONNX outputs:")
        for out in model.graph.output:
            dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
            print(f"  {out.name}: {dims}")
    except Exception as e:
        print(f"  Could not inspect ONNX: {e}")


def main():
    print("=" * 60)
    print(f"TensorRT version: {trt.__version__}")
    print(f"ONNX path: {ONNX_PATH}")
    print("=" * 60)
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
        print(f"  Benchmarking...")
        r = benchmark_engine(engine_path, prec_name)
        results.append(r)
        print(f"  {json.dumps(r, indent=4)}")

    out_path = os.path.join(ENGINE_DIR, "bench_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    print("\n=== SUMMARY ===")
    print(f"{'Precision':<25} {'Latency(ms)':>12} {'Memory(MB)':>11} {'Power(W)':>9} {'Energy(J/f)':>12}")
    print("-" * 72)
    for r in results:
        print(
            f"{r['precision']:<25} {r['latency_ms']:>12.1f} {r['memory_mb']:>11.0f} "
            f"{str(r['avg_power_w']):>9} {str(r['energy_j_per_frame']):>12}"
        )


if __name__ == "__main__":
    main()
