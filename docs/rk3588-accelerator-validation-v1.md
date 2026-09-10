# RK3588 GPU/NPU 环境验证（EdgeForge 0.16.1）

本记录对应 `/home/undefined/Downloads/OrangePi_5_Ultra_RK3588_user manual_v1.0.pdf` 第 3.37 节（Methods of using NPU）。验证对象是 Orange Pi 5 Ultra 的 RK3588，不使用摄像头、不上传数据，也不替换系统库。手册示例假定板端 Debian 11；当前实际板端 `/etc/os-release` 为 Ubuntu 22.04.5，故 Runtime/包版本不能直接按手册截图推断，必须以本板 API smoke 为准。

## 手册要求与 EdgeForge 的边界

| 阶段 | 官方手册路径 | EdgeForge 处理 |
| --- | --- | --- |
| 模型转换/量化 | Ubuntu 18.04/20.04/22.04 PC 上安装 RKNN-Toolkit2；手册示例使用 v1.5.2 | 作为 host-side compiler adapter 的后续接入点，当前 smoke 不执行转换 |
| 板端 Runtime | RK3588 Debian 上的 `librknnrt.so`、`rknn_server` | 通过 RKNN C API 直接加载离线 `.rknn` 并运行；不要求板端 Python `rknn` 包 |
| PC 通过 ADB 推理 | `rknn_server` 作为 USB/ADB 代理 | 当前不依赖该路径；直接板端 C API 更适合 EdgeForge 的 Worker/Runtime 验证 |
| 应用部署 | RKNPU2 C 示例交叉编译后拷贝到板端 | 后续将由受控 RKNN backend manifest 生成，不把 demo 摄像头作为系统依赖 |

## 为什么此前会误报“不支持”

旧探测只查找 `/dev/rknpu*`。本板的内核驱动实际映射为 `/dev/dri/card1` 和 `/dev/dri/renderD129`，两者的 `/sys/class/drm/*/device/driver` 都指向 `RKNPU`；平台设备还包括 `fdab0000.npu` 与 `rknpu_dev.10.auto`。因此字符设备缺失只能说明“不是旧式节点布局”，不能说明 NPU 不存在。

旧探测还把 `vulkaninfo` 命令是否存在当成 Vulkan 驱动结论，并把 `rknn` Python 包当成板端 Runtime 要求。现在 Target Probe 分开保存 DRM RKNPU、平台设备、OpenCL 用户态、Vulkan loader/ICD、RKNN Runtime 文件和 Python 模块证据；只有实际 API smoke 成功才形成 correctness 证据。

## Orange Pi 实测结果（2026-08-24）

运行命令：

```sh
PYTHONPATH=src python3 scripts/rk3588-accelerator-smoke.py \
  --name orangepi --ssh-host orangepi --version 0.16.1 --repeat 3 \
  --output .edgeforge/rk3588-accelerator-smoke-v0.16.1.json \
  --log-dir logs
```

| 组件 | 结果 | 证据 |
| --- | --- | --- |
| Mali GPU / OpenCL | PASS | `Mali-G610 r0p0`，ARM OpenCL 3.0，4 compute units；离线 FP32 vector-add 256 元素，最大绝对误差 `0.0` |
| Vulkan loader | 可加载 | `libvulkan.so.1`，loader API `1.3.204` |
| Vulkan device | BLOCKED | `vkCreateInstance` 返回 `VK_ERROR_INCOMPATIBLE_DRIVER (-9)`；`/etc/vulkan/icd.d` 与 `/usr/share/vulkan/icd.d` 没有 Mali ICD manifest。此项是用户态注册缺失，不是 RK3588 硬件不支持 Vulkan 的证明 |
| RKNPU DRM/平台 | PASS（设备证据） | DRM `card1/renderD129` 驱动为 `RKNPU`；两个 NPU platform device 存在 |
| RKNN Runtime init/query/run | PASS | 系统 `/usr/lib/librknnrt.so`（API 1.4.0）和用户目录 RKNPU2 v1.5.2 fallback 均可加载 RK3588 `mobilenet_v1.rknn`；`rknn_init/query/inputs_set/run/outputs_get` 全部成功 |
| NPU driver | PASS（兼容性证据） | 系统 Runtime API `1.4.0` / fallback API `1.5.2` 均报告 driver `0.9.6`；3 次零输入输出 SHA-256 一致 |
| NPU latency | diagnostic only | 3 次板端 run 中位数约 `1.94 ms`；输出只验证 Runtime 执行和零输入确定性，不是有标签模型 correctness 或 EEG 模型性能结论 |

随附的 `/usr/share/rknn_demo/mobilenet_ssd.rknn` 在系统 `librknnrt.so` 1.4.0 下返回 `Invalid RKNN format`。这说明该 demo 模型与系统 Runtime/Toolkit 版本或目标平台不匹配，不说明 NPU 失效；同一系统 Runtime 对 RK3588 `mobilenet_v1.rknn` 已成功运行。验证用的 v1.5.2 Runtime 和 RK3588 MobileNet 模型放在 `~/.cache/edgeforge/rk3588-smoke/rknpu2-v1.5.2/`，没有覆盖 `/usr/lib`；升级系统 Runtime 前必须先做版本、驱动和回滚审计。

## 如何复现与读取结果

 `target-probe` 负责只读能力清单：

```sh
PYTHONPATH=src python3 scripts/probe-target.py \
  --name orangepi --ssh-host orangepi --version 0.16.1 \
  --output .edgeforge/v0.16.1-target-probe-orangepi.json
```

在板端源码已同步时，也可以直接使用 `PYTHONPATH=src python3 -m edgeforge accelerator-smoke --output <path>`；脚本形式额外提供 SSH 传输和远程日志封装。

`rk3588-accelerator-smoke.py` 的 GPU 测试是无输入文件的 OpenCL kernel；NPU 测试只读取显式 `.rknn` 文件并填充零输入。远程模式同步的仅是用户目录缓存中的 probe 源码，日志仍由本机按版本保存。返回 `partial` 表示 OpenCL/NPU 已通过但 Vulkan ICD 尚未注册；返回 `pass` 才表示三者都通过。任何状态都带有 `scientific_conclusion_allowed=false`。

部署前的 capability preflight 现在对 `rknn` 要求板端 Runtime 文件，并接受 `rk3588_npu_drm`、`rk3588_npu_device` 或 `rk3588_npu_platform` 任一 NPU 设备证据；不再要求板端安装 RKNN-Toolkit2 Python 包。真正的模型发布门禁仍需同一模型的 RKNN conversion digest、算子覆盖、数值 correctness 和 benchmark 证据。

从 0.16.1 起，`rknn`/`opencl`/`vulkan` preflight 还要求显式的已记录 API smoke JSON；文件与设备节点只能说明环境候选存在，不能单独解锁 backend：

```sh
PYTHONPATH=src python3 scripts/model-deploy-preflight.py \
  --manifest /tmp/rknn-manifest.json \
  --probe .edgeforge/v0.16.1-target-probe-orangepi.json \
  --runtime-validation .edgeforge/rk3588-accelerator-smoke-v0.16.1.json \
  --output /tmp/rknn-preflight.json --version 0.16.1
```

本次实际结果为 `PASS`；省略 `--runtime-validation` 会明确 `BLOCKED`。该 gate 仍不等价于 EEG 模型的有标签数值 correctness。

## 后续工作顺序

1. 在主机环境接入 RKNN-Toolkit2 v1.5.2 的 ONNX/`torch.export` adapter，把转换日志、target=`rk3588`、量化校准集 digest 和 `.rknn` digest 写入 Model Pipeline artifact；不在板端安装 Toolkit2。
2. 为 EEG 网络先选一个不含动态 shape/自定义算子的静态子图，完成 FP32 reference → RKNN INT8 子图的逐层误差对照，再扩大算子覆盖；模型转换失败必须保留失败日志。
3. 若确实需要 Vulkan，向 Orange Pi 镜像提供方确认对应 Mali-G610 Vulkan userspace 包/ICD manifest，先在用户目录或可回滚系统快照中验证；不要用 `libjpeg.so.8` 冒充 `libjpeg.so.62`，也不要把 Mesa software ICD 当作 Mali 结果。
4. 将本 smoke 注册为 EdgeForge 的 accelerator correctness stage；只有该 stage、模型 correctness 和目标架构/驱动证据同时通过，才允许 Worker 显式广告 `rknn` backend。当前 Worker 仍默认只广告 `python-reference`。

## 0.16.2 后续验证

0.16.1 的“Vulkan ICD 尚未配置”结论针对当时的系统默认搜索路径仍然成立；0.16.2 在不改系统的前提下发现并验证了用户目录 Rockchip `g610-g24p0` ICD candidate。详见 [RK3588 Vulkan 用户态验证](rk3588-vulkan-userspace-validation-v1.md)；两个版本的日志和 digest 独立保存。
