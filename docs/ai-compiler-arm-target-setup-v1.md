# AI Compiler 与 ARM Target Setup v1

## IR 与 Backend 边界

EdgeForge 有两个不同层次的 IR。`OperatorSpec` 是算子语义协议，描述 name、shape、dtype、layout 和属性，用于兼容性、correctness 和 benchmark identity；它不是 MLIR/LLVM，也不包含某个 ISA 的指令。BrainUICL 模型前端另用 `torch.export` 生成 `ExportedProgram`，保存为 `brainuicl.pt2`，并将节点摘要保存为 `ir.json`。后者是模型 Graph IR，不是所有后端都能直接加载的通用二进制。

`compiler.py` 是 EdgeForge pipeline orchestrator：它验证 `OperatorSpec`，根据 Kernel snapshot 分发到 reference、Triton 或 IREE adapter，并保存 compile/correctness/benchmark evidence。真实代码生成由外部 toolchain/backend 完成；Python reference 的 compile stage 只是生成 manifest，不是机器码编译。IREE 当前 adapter 是 prebuilt runtime-only contract，RKNN 当前只有 registry/preflight contract。

后端输入不相同：Triton 通常接 Python/Triton kernel 并 JIT 到 CUDA 程序，IREE 需要 MLIR/StableHLO/输入 module 再生成 VMFB，RKNN Toolkit2 将 ONNX/TFLite/PyTorch 等模型转换、量化为 `.rknn`，板端 Runtime 再加载它。ONNX Runtime 直接消费 ONNX graph。因而不能把 `brainuicl.pt2` 无条件广播给所有后端；需要针对每个后端的 lowering adapter、算子覆盖、layout/dtype 约束和 numerical correctness gate。当前 BrainUICL compile command 会重新加载模型并调用 `torch.compile`，后续再把 `pt2` 接入统一 lowering。

## ARM 配置器

只读 inventory：

```sh
PYTHONPATH=src python scripts/configure-arm-target.py \
  --output-root .edgeforge/arm-target-setup/v0.16.0-inventory
```

用户目录级配置和源码同步：

```sh
PYTHONPATH=src python scripts/configure-arm-target.py \
  --apply --sync-source \
  --output-root .edgeforge/arm-target-setup/v0.16.0-apply-v3
```

该命令不会安装系统包、写入 token 或启动 Worker。远端生成：

- `~/.local/src/edgeforge`：排除数据、checkpoint、日志、Artifact、`.git` 的 EdgeForge 源码；
- `~/.local/share/edgeforge/work`：Worker work root；
- `~/.local/state/edgeforge/logs`：版本化日志根目录；
- `~/.config/edgeforge/environment.json`：target/backend 状态；
- `~/.config/edgeforge/edgeforge.env.example`：凭证模板；
- `~/.local/share/edgeforge/run-worker.sh`：只广告 `python-reference` 的启动模板。

Orange Pi 的 RKNN 共享库或 demo 文件只能作为 runtime evidence。只有 `/dev/rknpu*`、RKNN Runtime、模型转换、算子覆盖和量化前后 correctness 全部通过后，才允许把 `rknn` 加入 `EDGEFORGE_BACKENDS`。

配置后的真实 CPU reference smoke 使用 `matmul` shape `[16,16,16]`、`fp32`、3 次重复：Orange Pi correctness 为 true、median 约 `1.014 ms`；P550 correctness 为 true、median 约 `3.002 ms`。两者 checksum 相同，说明算子语义和结果链路一致；该 Python reference 时间不是 CPU/GPU 性能排名，也不是 RKNN NPU 结果。
