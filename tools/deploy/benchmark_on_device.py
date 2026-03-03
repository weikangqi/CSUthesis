"""
Xiaomi 14 (骁龙 8 Gen3) 真机延迟测试
=====================================

流程：
  1. 用 mnnconvert 将 model.onnx 转换为 FP16 和 INT8 MNN 模型
  2. 从 MNN 2.8.0 Release 下载 Android ARM64 benchmark 二进制（仅首次）
  3. 通过 ADB 将模型和工具推到 /data/local/tmp/mnn_bench/
  4. 在设备上运行 benchmark，warm=10，repeat=100，绑定大核
  5. 拉回日志，解析延迟数据，打印并保存 CSV

运行方式：
  # 基本（自动下载 MNNBench，需要联网）
  python tools/deploy/benchmark_on_device.py

  # 已有转换好的模型，跳过转换
  python tools/deploy/benchmark_on_device.py --skip-convert

  # 指定 ADB 设备序列号
  python tools/deploy/benchmark_on_device.py --serial 172.16.101.70:5555

  # 只转换模型，不推到设备
  python tools/deploy/benchmark_on_device.py --convert-only
"""

import argparse
import csv
import io
import os
import re
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

MNN_VERSION    = '2.8.0'
MNN_ANDROID_URL = (
    'https://github.com/alibaba/MNN/releases/download/'
    f'{MNN_VERSION}/mnn_2.8.0_android_armv7_armv8_cpu_opencl_vulkan.zip'
)

DEPLOY_DIR   = Path(__file__).parent.parent.parent / 'output' / 'deploy'
ONNX_PATH    = DEPLOY_DIR / 'model.onnx'
FP16_PATH    = DEPLOY_DIR / 'model_fp16.mnn'
INT8_PATH    = DEPLOY_DIR / 'model_int8.mnn'
BENCH_CACHE  = DEPLOY_DIR / 'mnn_android'         # 解压 zip 到此处

DEVICE_TMP   = '/data/local/tmp/mnn_bench'
WARMUP       = 10
REPEAT       = 100

# 骁龙 8 Gen3：cpu0-3 为小核(510/520), cpu4-6 为中大核(720), cpu7 为超大核(X4)
# 论文设定：4 大核（taskset 0xF0 = cpu4-7）
TASKSET_MASK = '0xF0'

# ---------------------------------------------------------------------------
# ADB 工具
# ---------------------------------------------------------------------------

def adb(serial: str, *args, check=True, capture=True):
    """运行 adb 命令，返回 stdout 字符串。"""
    cmd = ['adb', '-s', serial] + list(args)
    r = subprocess.run(cmd, capture_output=capture, text=True, encoding='utf-8',
                       errors='replace')
    if check and r.returncode != 0:
        raise RuntimeError(f'ADB 失败: {" ".join(cmd)}\n{r.stderr}')
    return r.stdout.strip() if capture else ''


def adb_push(serial: str, local: Path, remote: str):
    print(f'  推送 {local.name} → {remote}')
    adb(serial, 'push', str(local), remote, capture=False)


def adb_shell(serial: str, cmd: str, capture=True):
    return adb(serial, 'shell', cmd, check=False, capture=capture)


# ---------------------------------------------------------------------------
# Step 1：转换模型
# ---------------------------------------------------------------------------

def convert_models(skip=False):
    """mnnconvert: ONNX → FP16 MNN + INT8 MNN"""
    if skip:
        print('[转换] --skip-convert，跳过')
        return

    if not ONNX_PATH.exists():
        print(f'[错误] ONNX 模型不存在: {ONNX_PATH}')
        print('       请先运行 python tools/deploy/export_to_mnn.py')
        sys.exit(1)

    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

    # FP16
    if FP16_PATH.exists():
        print(f'[转换] FP16 已存在，跳过: {FP16_PATH.name}')
    else:
        print('[转换] ONNX → FP16 MNN ...')
        r = subprocess.run([
            'mnnconvert', '-f', 'ONNX',
            '--modelFile', str(ONNX_PATH),
            '--MNNModel',  str(FP16_PATH),
            '--fp16',
        ], capture_output=True, text=True, timeout=300)
        # mnnconvert 在 Windows 下转换成功后可能以 segfault(139) 退出，
        # 以输出文件存在且大小 > 0 作为成功判断，忽略退出码。
        if not FP16_PATH.exists() or FP16_PATH.stat().st_size == 0:
            print(f'  [错误] FP16 转换失败（输出文件未生成）\n{r.stderr[:300]}')
            sys.exit(1)
        print(f'  FP16: {FP16_PATH.stat().st_size/1024/1024:.1f} MB')

    # INT8（权重量化，per-channel）
    if INT8_PATH.exists():
        print(f'[转换] INT8 已存在，跳过: {INT8_PATH.name}')
    else:
        print('[转换] ONNX → INT8 MNN (权重 per-channel) ...')
        r = subprocess.run([
            'mnnconvert', '-f', 'ONNX',
            '--modelFile',        str(ONNX_PATH),
            '--MNNModel',         str(INT8_PATH),
            '--weightQuantBits',  '8',
            '--weightQuantBlock', '0',
        ], capture_output=True, text=True, timeout=300)
        if not INT8_PATH.exists() or INT8_PATH.stat().st_size == 0:
            print(f'  [错误] INT8 转换失败（输出文件未生成）\n{r.stderr[:300]}')
            sys.exit(1)
        print(f'  INT8: {INT8_PATH.stat().st_size/1024/1024:.1f} MB')


# ---------------------------------------------------------------------------
# Step 2：下载 MNN Android benchmark 工具
# ---------------------------------------------------------------------------

def find_benchmark_bin() -> Path:
    """
    在解压目录中查找 arm64-v8a 的 benchmark 可执行文件。
    MNN Android zip 结构通常为 armv8/benchmark 或 arm64-v8a/benchmark。
    """
    candidates = list(BENCH_CACHE.glob('**/benchmark')) + \
                 list(BENCH_CACHE.glob('**/*Bench*'))
    # 优先 armv8/arm64
    arm64_bins = [p for p in candidates
                  if any(k in str(p).lower() for k in ('armv8', 'arm64', 'v8'))]
    if arm64_bins:
        return arm64_bins[0]
    if candidates:
        return candidates[0]
    return None


def ensure_benchmark_bin() -> Path:
    """返回本地 arm64 benchmark 二进制路径，如不存在则下载并解压。"""
    existing = find_benchmark_bin()
    if existing:
        print(f'[工具] 使用已缓存的 benchmark: {existing}')
        return existing

    zip_path = DEPLOY_DIR / f'mnn_{MNN_VERSION}_android.zip'

    if not zip_path.exists():
        print(f'[工具] 下载 MNN Android 包 ({MNN_VERSION}) ...')
        print(f'       URL: {MNN_ANDROID_URL}')
        try:
            urllib.request.urlretrieve(MNN_ANDROID_URL, str(zip_path),
                                       reporthook=_dl_progress)
            print()
        except Exception as e:
            print(f'\n  [错误] 下载失败: {e}')
            print('  请手动下载 MNN Android release 并解压到', BENCH_CACHE)
            sys.exit(1)
    else:
        print(f'[工具] 使用缓存 zip: {zip_path.name}')

    # 解压
    print(f'[工具] 解压 → {BENCH_CACHE}')
    BENCH_CACHE.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(zip_path)) as z:
        z.extractall(str(BENCH_CACHE))

    # 列出解压内容以便调试
    bins = list(BENCH_CACHE.glob('**/*'))
    print(f'  解压文件数: {len(bins)}')
    for p in bins[:15]:
        print(f'    {p.relative_to(BENCH_CACHE)}')
    if len(bins) > 15:
        print(f'    ... (共 {len(bins)} 个)')

    existing = find_benchmark_bin()
    if not existing:
        print('[错误] 未找到 benchmark 二进制，请检查解压目录:')
        print(f'       {BENCH_CACHE}')
        sys.exit(1)

    return existing


def _dl_progress(count, block_size, total_size):
    pct = count * block_size / total_size * 100
    done = int(pct / 2)
    sys.stdout.write(f'\r  [{"#"*done}{" "*(50-done)}] {min(pct,100):.1f}%')
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Step 3：推送到设备 & 运行 benchmark
# ---------------------------------------------------------------------------

def setup_device(serial: str, bench_bin: Path):
    """创建设备目录，推送模型和 benchmark 工具。"""
    adb_shell(serial, f'mkdir -p {DEVICE_TMP}')

    adb_push(serial, FP16_PATH, f'{DEVICE_TMP}/model_fp16.mnn')
    adb_push(serial, INT8_PATH, f'{DEVICE_TMP}/model_int8.mnn')
    adb_push(serial, bench_bin, f'{DEVICE_TMP}/benchmark')
    adb_shell(serial, f'chmod +x {DEVICE_TMP}/benchmark')


def run_benchmark(serial: str, model_remote: str, label: str,
                  num_thread: int = 4) -> dict:
    """
    调用设备上的 MNN benchmark 工具，解析并返回延迟统计。

    MNN benchmark 命令格式：
      ./benchmark <model_dir> <warmup> <repeat> <num_thread> [backend]
    或：
      ./benchmark <model_file> [options...]
    两种格式均常见，脚本会尝试两种。
    """
    bench_cmd_v1 = (
        f'cd {DEVICE_TMP} && '
        f'taskset {TASKSET_MASK} '
        f'./benchmark {model_remote} {WARMUP} {REPEAT} {num_thread} 0'
    )
    bench_cmd_v2 = (
        f'cd {DEVICE_TMP} && '
        f'taskset {TASKSET_MASK} '
        f'./benchmark --model {model_remote} '
        f'--warmup {WARMUP} --runLoops {REPEAT} --numThread {num_thread}'
    )

    print(f'\n[测试] {label} ({REPEAT} 次, {num_thread} 线程) ...')

    for cmd in (bench_cmd_v1, bench_cmd_v2):
        out = adb_shell(serial, cmd)
        result = _parse_benchmark_output(out)
        if result:
            return result
        # 尝试下一种格式

    # 如果两种格式都没有解析到，返回原始输出供人工查看
    print('  [warn] 未能自动解析输出，原始结果：')
    print(out[:800])
    return {'avg_ms': None, 'min_ms': None, 'max_ms': None, 'raw': out}


def _parse_benchmark_output(text: str) -> dict:
    """
    解析 MNN benchmark 输出中的延迟数值。
    常见格式：
      Avg= 116.0 ms   Min= 114.2 ms   Max= 118.5 ms
    或：
      avg time = 116.0 ms
    """
    patterns = [
        # 格式1: "Avg= X ms"
        (r'[Aa]vg[=\s:]+(\d+\.?\d*)\s*ms', 'avg_ms'),
        (r'[Mm]in[=\s:]+(\d+\.?\d*)\s*ms', 'min_ms'),
        (r'[Mm]ax[=\s:]+(\d+\.?\d*)\s*ms', 'max_ms'),
        # 格式2: "avg time = X"
        (r'avg\s*time\s*[=:]\s*(\d+\.?\d*)', 'avg_ms'),
    ]
    result = {}
    for pat, key in patterns:
        m = re.search(pat, text)
        if m:
            result[key] = float(m.group(1))

    if 'avg_ms' in result:
        return result
    # 尝试找任意数字行（最后备选）
    nums = re.findall(r'\b(\d{2,4}\.\d+)\s*ms', text)
    if nums:
        vals = [float(n) for n in nums]
        return {'avg_ms': sum(vals)/len(vals), 'min_ms': min(vals),
                'max_ms': max(vals)}
    return {}


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

def print_results(results: dict):
    print('\n' + '=' * 55)
    print('真机延迟测试结果  (Xiaomi 14 / 骁龙 8 Gen3 / CPU)')
    print('=' * 55)
    hdr = f'  {"模型":<20} {"均值(ms)":>9} {"最小(ms)":>9} {"最大(ms)":>9}'
    print(hdr)
    print('  ' + '-' * 50)
    for label, r in results.items():
        if r.get('avg_ms') is not None:
            avg = f'{r["avg_ms"]:.1f}'
            mn  = f'{r.get("min_ms", 0):.1f}' if r.get('min_ms') else '--'
            mx  = f'{r.get("max_ms", 0):.1f}' if r.get('max_ms') else '--'
            print(f'  {label:<20} {avg:>9} {mn:>9} {mx:>9}')
        else:
            print(f'  {label:<20}   解析失败，见上方原始输出')

    # 加速比
    fp16 = results.get('FP16', {}).get('avg_ms')
    int8 = results.get('INT8(权重量化)', {}).get('avg_ms')
    if fp16 and int8:
        print(f'\n  FP16 / INT8 加速比: {fp16/int8:.2f}×')
        print(f'  延迟减少: {(1-int8/fp16)*100:.1f}%')

    print('=' * 55)


def save_csv(results: dict, out_path: Path):
    rows = []
    for label, r in results.items():
        rows.append({
            '模型': label,
            '均值延迟(ms)': f'{r["avg_ms"]:.1f}' if r.get('avg_ms') else '',
            '最小延迟(ms)': f'{r.get("min_ms",0):.1f}' if r.get('min_ms') else '',
            '最大延迟(ms)': f'{r.get("max_ms",0):.1f}' if r.get('max_ms') else '',
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['模型','均值延迟(ms)','最小延迟(ms)','最大延迟(ms)'])
        w.writeheader()
        w.writerows(rows)
    print(f'\n结果已保存: {out_path}')


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Xiaomi 14 MNN 真机延迟测试')
    parser.add_argument('--serial',       default='172.16.101.70:5555',
                        help='ADB 设备序列号（默认 172.16.101.70:5555）')
    parser.add_argument('--skip-convert', action='store_true',
                        help='跳过模型转换（已有 .mnn 文件时使用）')
    parser.add_argument('--convert-only', action='store_true',
                        help='只转换模型，不推到设备')
    parser.add_argument('--num-thread',   type=int, default=4,
                        help='MNN 推理线程数（默认 4）')
    parser.add_argument('--output-csv',   default='output/deploy/device_latency.csv',
                        help='CSV 输出路径')
    args = parser.parse_args()

    serial = args.serial

    print('=' * 55)
    print('MNN 真机延迟测试流程')
    print('=' * 55)
    print(f'设备: {serial}')
    print(f'线程: {args.num_thread}  | CPU 绑定: taskset {TASKSET_MASK} (大核)')

    # Step 1: 转换
    print('\n[Step 1] 模型转换')
    convert_models(skip=args.skip_convert)

    if args.convert_only:
        print('\n[完成] --convert-only，已生成 MNN 模型文件：')
        for p in [FP16_PATH, INT8_PATH]:
            if p.exists():
                print(f'  {p}  ({p.stat().st_size/1024/1024:.1f} MB)')
        return

    # Step 2: 获取 benchmark 工具
    print('\n[Step 2] 获取 MNN benchmark 工具')
    bench_bin = ensure_benchmark_bin()
    print(f'  使用: {bench_bin}')

    # Step 3: 推送
    print('\n[Step 3] 推送到设备')
    setup_device(serial, bench_bin)

    # Step 4: 测试
    print('\n[Step 4] 设备推理测试')
    results = {}

    results['FP16'] = run_benchmark(
        serial, f'{DEVICE_TMP}/model_fp16.mnn', 'FP16 基线',
        num_thread=args.num_thread,
    )
    results['INT8(权重量化)'] = run_benchmark(
        serial, f'{DEVICE_TMP}/model_int8.mnn', 'INT8 权重量化',
        num_thread=args.num_thread,
    )

    # Step 5: 输出
    print_results(results)
    save_csv(results, Path(args.output_csv))

    print('\n[提示] 将上方均值延迟填入论文表 4.x (tab:quant_comparison)')
    print('       FP16 对应第三章基线 116.0ms（原估算值），如有差异请同步更新正文')


if __name__ == '__main__':
    main()
