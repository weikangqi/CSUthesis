/**
 * mnn_bench.cpp
 * 极简 MNN 端侧延迟测试工具
 * 编译为 ARM64 Android，在 Xiaomi 14 上通过 adb shell 运行
 *
 * 用法：./mnn_bench <model.mnn> [warmup=10] [repeat=100] [threads=4]
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#include "MNN/Interpreter.hpp"
#include "MNN/MNNDefine.h"
#include "MNN/Tensor.hpp"

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        Clock::now().time_since_epoch()).count();
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "用法: %s <model.mnn> [warmup] [repeat] [threads]\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    int warmup  = argc > 2 ? atoi(argv[2]) : 10;
    int repeat  = argc > 3 ? atoi(argv[3]) : 100;
    int threads = argc > 4 ? atoi(argv[4]) : 4;

    fprintf(stderr, "模型: %s\n预热: %d  计时: %d  线程: %d\n",
            model_path, warmup, repeat, threads);

    // 创建解释器
    auto* interp = MNN::Interpreter::createFromFile(model_path);
    if (!interp) {
        fprintf(stderr, "[错误] 无法加载模型: %s\n", model_path);
        return 1;
    }

    // 会话配置
    MNN::ScheduleConfig cfg;
    cfg.numThread = threads;
    cfg.type      = MNN_FORWARD_CPU;
    MNN::BackendConfig backendCfg;
    // Precision_Low = FP16 NEON 执行（ARMv8.2）
    // Precision_Normal = FP32 执行（即使权重是 FP16 存储的）
    backendCfg.precision = MNN::BackendConfig::Precision_Low;
    backendCfg.power     = MNN::BackendConfig::Power_High;
    cfg.backendConfig    = &backendCfg;

    auto* session = interp->createSession(cfg);
    if (!session) {
        fprintf(stderr, "[错误] 创建 Session 失败\n");
        return 1;
    }

    // 获取输入 tensor 并填充随机数据
    auto inputs = interp->getSessionInputAll(session);
    for (auto& kv : inputs) {
        MNN::Tensor* t = kv.second;
        MNN::Tensor host(t, MNN::Tensor::CAFFE);  // host-side copy
        int count = 1;
        for (int i = 0; i < host.dimensions(); ++i) count *= host.length(i);
        auto* data = host.host<float>();
        for (int i = 0; i < count; ++i) data[i] = 0.0f;
        t->copyFromHostTensor(&host);
    }

    fprintf(stderr, "开始推理...\n");

    // 预热
    for (int i = 0; i < warmup; ++i)
        interp->runSession(session);

    // 正式计时
    std::vector<double> times;
    times.reserve(repeat);
    for (int i = 0; i < repeat; ++i) {
        double t0 = now_ms();
        interp->runSession(session);
        times.push_back(now_ms() - t0);
    }

    // 统计
    double avg = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    double mn  = *std::min_element(times.begin(), times.end());
    double mx  = *std::max_element(times.begin(), times.end());

    // 标准差
    double var = 0;
    for (double t : times) var += (t - avg) * (t - avg);
    double std_dev = std::sqrt(var / times.size());

    // 输出（固定格式，供 Python 脚本解析）
    printf("=== MNNBench Result ===\n");
    printf("model:   %s\n", model_path);
    printf("threads: %d\n", threads);
    printf("warmup:  %d\n", warmup);
    printf("repeat:  %d\n", repeat);
    printf("Avg= %.2f ms\n", avg);
    printf("Min= %.2f ms\n", mn);
    printf("Max= %.2f ms\n", mx);
    printf("Std= %.2f ms\n", std_dev);

    fprintf(stderr, "完成。Avg=%.1f ms  Min=%.1f ms  Max=%.1f ms\n", avg, mn, mx);

    interp->releaseSession(session);
    MNN::Interpreter::destroy(interp);
    return 0;
}
