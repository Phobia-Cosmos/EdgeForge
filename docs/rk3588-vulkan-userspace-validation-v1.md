# RK3588 Vulkan 用户态验证（EdgeForge 0.16.2）

本记录补充 0.16.1 的 RK3588 GPU/NPU 验证。0.16.1 观察到系统 Ubuntu 22.04.5 的 `libvulkan.so.1` loader 可以加载，但 `/etc/vulkan/icd.d` 和 `/usr/share/vulkan/icd.d` 没有 Mali manifest；本版本继续保持系统目录不变，在 Orange Pi 用户目录隔离验证 Rockchip `libmali` 的不同 GPU build。

## 结果

| 候选 | 来源/用途 | Vulkan ICD entry points | EdgeForge API smoke | 结论 |
| --- | --- | --- | --- | --- |
| `libmali-valhall-g610-g13p0-x11-gbm.so` | 板上已安装系统包 `libmali-valhall-g610-g13p0-x11-gbm`；control candidate | 没有 `vk_icdGetInstanceProcAddr` / `vkGetInstanceProcAddr` | `vkCreateInstance=-9` | 不能作为 Vulkan ICD；不代表 RK3588 硬件不支持 Vulkan |
| `libmali-valhall-g610-g13p0-gbm.so` | Rockchip libmali 同源用户目录 candidate | 没有 loader ICD entry points | `vkCreateInstance=-9` | 同上，不能只凭文件名判断 |
| `libmali-valhall-g610-g24p0-gbm.so` | Rockchip libmali 同源用户目录 candidate | `vk_icdGetInstanceProcAddr`、`vk_icdGetPhysicalDeviceProcAddr`、`vk_icdNegotiateLoaderICDInterfaceVersion` 存在 | loader API `1.3.204`、instance `0`、physical device `1` | Mali-G610 Vulkan userspace API smoke PASS |

`g24p0` smoke 只证明当前 kernel/firmware、loader、ICD 和用户态库可以完成 instance/physical-device enumeration；它不是模型转换、Vulkan compute kernel correctness、EEG 算子覆盖或性能结论。OpenCL 的 FP32 vector-add 是独立的实际 kernel correctness smoke，RKNN 的 MobileNet 仍是零输入 deterministic runtime smoke。

## 隔离方式

候选库与 manifest 放在板端 `~/.cache/edgeforge/rk3588-vulkan-g24p0/`，EdgeForge 通过 `VK_ICD_FILENAMES` 在当前 Python 进程中临时选择 manifest，退出时恢复环境变量。没有执行 `sudo`、`apt install`、系统库替换、`ldconfig` 或 `/etc/vulkan` 写入。manifest 和库的 digest 绑定在 smoke JSON 中，避免后续把不同 build 混在一起。

## 复现

```sh
PYTHONPATH=src python3 scripts/probe-target.py \
  --name orangepi --ssh-host orangepi --version 0.16.2 \
  --vulkan-icd /home/orangepi/.cache/edgeforge/rk3588-vulkan-g24p0/icd/mali-g610-g24p0.json \
  --output .edgeforge/v0.16.2-target-probe-orangepi-g24p0.json --log-dir logs

PYTHONPATH=src python3 scripts/rk3588-accelerator-smoke.py \
  --name orangepi-all-g24p0 --ssh-host orangepi --version 0.16.2 \
  --vulkan-icd /home/orangepi/.cache/edgeforge/rk3588-vulkan-g24p0/icd/mali-g610-g24p0.json \
  --repeat 3 --output .edgeforge/rk3588-accelerator-smoke-v0.16.2-g24p0.json --log-dir logs
```

完整 smoke 返回 `status=pass`，其中 `gpu.opencl.status=pass`、`gpu.vulkan.status=pass`、`npu.status=pass`；系统默认不指定 `--vulkan-icd` 时仍返回 `gpu.status=partial`，这两个结果必须分别保留，不能覆盖旧记录。

## 对 EdgeForge 部署的影响

Backend registry 现在有显式的 `vulkan`/`opencl` target contract，但 `target_probe` 的 `vulkan_user_icd_manifest=true` 只表示用户目录 candidate 存在；`scripts/model-deploy-preflight.py` 仍要求同一目标名称的 Vulkan API smoke JSON，随后还需要真实模型 artifact、算子覆盖和 numerical correctness。当前 Worker 不会因为这次环境 smoke 自动广告 Vulkan 或 RKNN backend，避免把通用 runtime 能力误认为 EEG 模型已经可部署。

本版本的 `config/rk3588-vulkan-smoke-v1.json` + `v0.16.2-target-probe-orangepi-g24p0.json` + `rk3588-accelerator-smoke-v0.16.2-orangepi-g24p0.json` 已实际得到 preflight `PASS`；这是 capability/API gate，不是模型执行结果。
