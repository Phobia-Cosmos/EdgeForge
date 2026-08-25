"""Read-only RK3588 GPU/NPU runtime smoke probes.

The probe deliberately exercises user-space APIs instead of inferring a backend
from a board name or a single device node.  It is safe to run on a headless
board: the GPU test uses an offline OpenCL vector-add kernel and the NPU test
uses an offline RKNN model with deterministic zero input.  No camera, network
stream, model conversion, or system-package mutation is performed here.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _status(status: str, **fields: Any) -> dict[str, Any]:
    value: dict[str, Any] = {"status": status}
    value.update(fields)
    return value


def _decode_c_string(value: bytes | bytearray | None) -> str | None:
    if value is None:
        return None
    return bytes(value).split(b"\0", 1)[0].decode("utf-8", "replace")


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def _command_output(argv: list[str], timeout: float = 3.0) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "status": "timeout" if isinstance(error, subprocess.TimeoutExpired) else "unavailable",
            "exit_code": None,
            "stdout": "",
            "stderr": str(error),
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }
    return {
        "status": "succeeded" if completed.returncode == 0 else "failed",
        "exit_code": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


@contextmanager
def _temporary_environment(**updates: str | None):
    """Temporarily set environment variables for a vendor-loader probe.

    Vulkan chooses ICDs from the process environment when the loader is first
    used.  Keeping the override scoped to the probe prevents a board worker or
    a caller embedding EdgeForge from accidentally inheriting a diagnostic ICD.
    ``None`` removes a variable for the duration of the context.
    """

    previous: dict[str, str | None] = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _vulkan_manifest_evidence(manifest: str | Path | None) -> dict[str, Any]:
    """Read-only evidence for an optional Vulkan ICD manifest.

    A manifest and a loadable shared object are only *candidate* evidence.  A
    real Vulkan capability is recorded by :func:`probe_vulkan` after the loader
    creates an instance and enumerates a physical device.
    """

    if not manifest:
        return {"path": None, "status": "not-requested"}
    path = Path(manifest).expanduser()
    result: dict[str, Any] = {
        "path": str(path),
        "status": "missing",
        "sha256": None,
        "library_path": None,
        "library_exists": False,
        "library_sha256": None,
    }
    try:
        raw = path.read_bytes()
        result["sha256"] = hashlib.sha256(raw).hexdigest()
        payload = json.loads(raw.decode("utf-8"))
        icd = payload.get("ICD") if isinstance(payload, dict) else None
        library = icd.get("library_path") if isinstance(icd, dict) else None
        if isinstance(library, str) and library:
            library_path = Path(library).expanduser()
            if not library_path.is_absolute():
                library_path = path.parent / library_path
            result["library_path"] = str(library_path)
            result["library_exists"] = library_path.is_file()
            if library_path.is_file():
                result["library_sha256"] = hashlib.sha256(library_path.read_bytes()).hexdigest()
        result["status"] = "candidate"
    except (OSError, UnicodeError, json.JSONDecodeError, AttributeError, TypeError) as error:
        result["status"] = "invalid"
        result["error"] = str(error)
    return result


def collect_board_evidence() -> dict[str, Any]:
    """Collect kernel/device evidence without treating it as correctness."""

    drm: list[dict[str, str | None]] = []
    drm_root = Path("/sys/class/drm")
    if drm_root.exists():
        for node in sorted(drm_root.glob("card*")) + sorted(drm_root.glob("renderD*")):
            driver_link = node / "device" / "driver"
            driver_path: str | None = None
            try:
                driver_path = str(driver_link.resolve())
            except OSError:
                pass
            drm.append({"node": str(Path("/dev/dri") / node.name), "driver": driver_path})

    npu_platform = sorted(str(path) for path in Path("/sys/bus/platform/devices").glob("*npu*"))
    npu_char = sorted(str(path) for path in Path("/dev").glob("rknpu*"))
    firmware = sorted(str(path) for path in Path("/lib/firmware").glob("*mali*"))
    library_candidates = [
        "/lib/aarch64-linux-gnu/libmali.so.1",
        "/usr/lib/aarch64-linux-gnu/libmali.so.1",
        "/lib/aarch64-linux-gnu/libOpenCL.so.1",
        "/usr/lib/aarch64-linux-gnu/libOpenCL.so.1",
        "/lib/aarch64-linux-gnu/libvulkan.so.1",
        "/usr/lib/aarch64-linux-gnu/libvulkan.so.1",
        "/usr/lib/librknnrt.so",
        "/usr/lib/librknn_api.so",
    ]
    libraries = sorted(path for path in library_candidates if Path(path).exists())
    vulkan_icd_dirs = [Path("/etc/vulkan/icd.d"), Path("/usr/share/vulkan/icd.d")]
    vulkan_icds = sorted(str(path) for directory in vulkan_icd_dirs if directory.exists() for path in directory.glob("*.json"))
    opencl_icds = sorted(
        str(file_path)
        for directory in (Path("/etc/OpenCL/vendors"),)
        if directory.exists()
        for file_path in directory.glob("*.icd")
    )
    rknn_server = shutil.which("rknn_server")
    vulkaninfo_candidates = [
        shutil.which("vulkaninfo"),
        str(Path.home() / ".cache/edgeforge/rk3588-smoke/vulkan-tools/root/usr/bin/vulkaninfo"),
    ]
    vulkaninfo_path = next((path for path in vulkaninfo_candidates if path and Path(path).is_file()), None)
    vulkaninfo_probe = (
        _command_output([vulkaninfo_path, "--summary"], timeout=8.0)
        if vulkaninfo_path
        else {"status": "unavailable", "reason": "vulkaninfo command not found"}
    )
    server_process = _command_output(["pgrep", "-x", "rknn_server"], timeout=2.0) if rknn_server else {"status": "unavailable"}
    return {
        "architecture": platform.machine(),
        "kernel": platform.release(),
        "drm_nodes": drm,
        "rknpu_char_devices": npu_char,
        "rknpu_platform_devices": npu_platform,
        "rknpu_drm_nodes": [item["node"] for item in drm if (item.get("driver") or "").rstrip("/").split("/")[-1].upper() == "RKNPU"],
        "mali_firmware": firmware,
        "runtime_libraries": libraries,
        "vulkan_icd_manifests": vulkan_icds,
        "opencl_icd_manifests": opencl_icds,
        "rknn_server": {
            "path": rknn_server,
            "process": server_process,
        },
        "vulkaninfo": {"path": vulkaninfo_path, "probe": vulkaninfo_probe},
    }


# ---------------------------------------------------------------------------
# OpenCL smoke


_CL_SUCCESS = 0
_CL_DEVICE_TYPE_GPU = 1 << 2
_CL_DEVICE_NAME = 0x102B
_CL_DEVICE_VENDOR = 0x102C
_CL_DEVICE_VERSION = 0x102F
_CL_DRIVER_VERSION = 0x102D
_CL_DEVICE_MAX_COMPUTE_UNITS = 0x1002
_CL_MEM_READ_WRITE = 1 << 0
_CL_TRUE = 1


def _cl_info_string(lib: Any, fn: Any, obj: Any, key: int) -> str | None:
    size = ctypes.c_size_t(0)
    result = fn(obj, key, 0, None, ctypes.byref(size))
    if result != _CL_SUCCESS or size.value == 0:
        return None
    data = ctypes.create_string_buffer(size.value)
    result = fn(obj, key, size.value, data, None)
    return _decode_c_string(data.raw) if result == _CL_SUCCESS else None


def probe_opencl() -> dict[str, Any]:
    """Run a small Mali OpenCL vector-add kernel and verify its output."""

    candidates = ["libOpenCL.so.1", "libOpenCL.so", "libMaliOpenCL.so.1"]
    lib = None
    loaded = None
    errors: list[str] = []
    for candidate in candidates:
        try:
            lib = ctypes.CDLL(candidate)
            loaded = candidate
            break
        except OSError as error:
            errors.append(f"{candidate}: {error}")
    if lib is None:
        return _status("blocked", reason="OpenCL library unavailable", load_errors=errors)

    try:
        cl_platform_id = ctypes.c_void_p
        cl_device_id = ctypes.c_void_p
        cl_context = ctypes.c_void_p
        cl_command_queue = ctypes.c_void_p
        cl_mem = ctypes.c_void_p
        cl_program = ctypes.c_void_p
        cl_kernel = ctypes.c_void_p
        cl_uint = ctypes.c_uint32
        cl_int = ctypes.c_int32
        cl_size = ctypes.c_size_t

        get_platform_ids = lib.clGetPlatformIDs
        get_platform_ids.argtypes = [cl_uint, ctypes.POINTER(cl_platform_id), ctypes.POINTER(cl_uint)]
        get_platform_ids.restype = cl_int
        platform_count = cl_uint(0)
        result = get_platform_ids(0, None, ctypes.byref(platform_count))
        if result != _CL_SUCCESS or platform_count.value == 0:
            return _status("blocked", reason="OpenCL platform enumeration failed", error_code=int(result), library=loaded)
        platforms = (cl_platform_id * min(platform_count.value, 16))()
        result = get_platform_ids(len(platforms), platforms, None)
        if result != _CL_SUCCESS:
            return _status("blocked", reason="OpenCL platform enumeration failed", error_code=int(result), library=loaded)

        get_platform_info = lib.clGetPlatformInfo
        get_platform_info.argtypes = [cl_platform_id, cl_uint, cl_size, ctypes.c_void_p, ctypes.POINTER(cl_size)]
        get_platform_info.restype = cl_int
        get_device_ids = lib.clGetDeviceIDs
        get_device_ids.argtypes = [cl_platform_id, ctypes.c_ulong, cl_uint, ctypes.POINTER(cl_device_id), ctypes.POINTER(cl_uint)]
        get_device_ids.restype = cl_int
        get_device_info = lib.clGetDeviceInfo
        get_device_info.argtypes = [cl_device_id, cl_uint, cl_size, ctypes.c_void_p, ctypes.POINTER(cl_size)]
        get_device_info.restype = cl_int

        selected_platform = None
        selected_device = None
        platform_name = None
        device_count = 0
        for item in platforms[: platform_count.value]:
            count = cl_uint(0)
            query = get_device_ids(item, _CL_DEVICE_TYPE_GPU, 0, None, ctypes.byref(count))
            if query == _CL_SUCCESS and count.value:
                devices = (cl_device_id * min(count.value, 16))()
                query = get_device_ids(item, _CL_DEVICE_TYPE_GPU, len(devices), devices, None)
                if query == _CL_SUCCESS:
                    selected_platform = item
                    selected_device = devices[0]
                    device_count = int(count.value)
                    platform_name = _cl_info_string(lib, get_platform_info, item, 0x0902)
                    break
        if selected_device is None:
            return _status("blocked", reason="no OpenCL GPU device", platform_count=int(platform_count.value), library=loaded)

        device_name = _cl_info_string(lib, get_device_info, selected_device, _CL_DEVICE_NAME)
        device_vendor = _cl_info_string(lib, get_device_info, selected_device, _CL_DEVICE_VENDOR)
        device_version = _cl_info_string(lib, get_device_info, selected_device, _CL_DEVICE_VERSION)
        driver_version = _cl_info_string(lib, get_device_info, selected_device, _CL_DRIVER_VERSION)
        units = cl_uint(0)
        get_device_info(selected_device, _CL_DEVICE_MAX_COMPUTE_UNITS, ctypes.sizeof(units), ctypes.byref(units), None)

        create_context = lib.clCreateContext
        create_context.argtypes = [ctypes.c_void_p, cl_uint, ctypes.POINTER(cl_device_id), ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(cl_int)]
        create_context.restype = cl_context
        device_array = (cl_device_id * 1)(selected_device)
        error_code = cl_int(0)
        context = create_context(None, 1, device_array, None, None, ctypes.byref(error_code))
        if not context or error_code.value != _CL_SUCCESS:
            return _status("blocked", reason="OpenCL context creation failed", error_code=int(error_code.value), library=loaded, device=device_name)

        def release_context() -> None:
            try:
                lib.clReleaseContext(context)
            except Exception:
                pass

        try:
            create_queue = lib.clCreateCommandQueue
            create_queue.argtypes = [cl_context, cl_device_id, ctypes.c_ulong, ctypes.POINTER(cl_int)]
            create_queue.restype = cl_command_queue
            queue = create_queue(context, selected_device, 0, ctypes.byref(error_code))
            if not queue or error_code.value != _CL_SUCCESS:
                return _status("blocked", reason="OpenCL command queue creation failed", error_code=int(error_code.value), library=loaded, device=device_name)

            # Explicitly declare release ABIs.  ctypes otherwise defaults to
            # 32-bit integer arguments, truncating a 64-bit Mali handle during
            # cleanup and causing a board-side SIGSEGV.
            for release_name, release_type in (
                ("clReleaseKernel", cl_kernel),
                ("clReleaseProgram", cl_program),
                ("clReleaseMemObject", cl_mem),
                ("clReleaseCommandQueue", cl_command_queue),
                ("clReleaseContext", cl_context),
            ):
                release_function = getattr(lib, release_name)
                release_function.argtypes = [release_type]
                release_function.restype = cl_int

            n = 256
            values_a = (ctypes.c_float * n)(*[float(i) for i in range(n)])
            values_b = (ctypes.c_float * n)(*[float(2 * i + 1) for i in range(n)])
            values_out = (ctypes.c_float * n)()
            create_buffer = lib.clCreateBuffer
            create_buffer.argtypes = [cl_context, ctypes.c_ulong, cl_size, ctypes.c_void_p, ctypes.POINTER(cl_int)]
            create_buffer.restype = cl_mem
            buffer_a = create_buffer(context, _CL_MEM_READ_WRITE, ctypes.sizeof(values_a), None, ctypes.byref(error_code))
            buffer_b = create_buffer(context, _CL_MEM_READ_WRITE, ctypes.sizeof(values_b), None, ctypes.byref(error_code))
            buffer_out = create_buffer(context, _CL_MEM_READ_WRITE, ctypes.sizeof(values_out), None, ctypes.byref(error_code))
            if not buffer_a or not buffer_b or not buffer_out or error_code.value != _CL_SUCCESS:
                return _status("blocked", reason="OpenCL buffer allocation failed", error_code=int(error_code.value), library=loaded, device=device_name)

            write_buffer = lib.clEnqueueWriteBuffer
            write_buffer.argtypes = [cl_command_queue, cl_mem, cl_uint, cl_size, cl_size, ctypes.c_void_p, cl_uint, ctypes.c_void_p, ctypes.c_void_p]
            write_buffer.restype = cl_int
            for target, source in ((buffer_a, values_a), (buffer_b, values_b)):
                error_code.value = write_buffer(queue, target, _CL_TRUE, 0, ctypes.sizeof(source), ctypes.byref(source), 0, None, None)
                if error_code.value != _CL_SUCCESS:
                    return _status("blocked", reason="OpenCL input upload failed", error_code=int(error_code.value), library=loaded, device=device_name)

            source = b"__kernel void add(__global const float* a, __global const float* b, __global float* out) { size_t i = get_global_id(0); out[i] = a[i] + b[i]; }"
            create_program = lib.clCreateProgramWithSource
            create_program.argtypes = [cl_context, cl_uint, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(cl_size), ctypes.POINTER(cl_int)]
            create_program.restype = cl_program
            sources = (ctypes.c_char_p * 1)(source)
            lengths = (cl_size * 1)(len(source))
            program = create_program(context, 1, sources, lengths, ctypes.byref(error_code))
            if not program or error_code.value != _CL_SUCCESS:
                return _status("blocked", reason="OpenCL program creation failed", error_code=int(error_code.value), library=loaded, device=device_name)
            build_program = lib.clBuildProgram
            build_program.argtypes = [cl_program, cl_uint, ctypes.POINTER(cl_device_id), ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p]
            build_program.restype = cl_int
            error_code.value = build_program(program, 1, device_array, None, None, None)
            if error_code.value != _CL_SUCCESS:
                build_log = ""
                try:
                    get_build_info = lib.clGetProgramBuildInfo
                    get_build_info.argtypes = [cl_program, cl_device_id, cl_uint, cl_size, ctypes.c_void_p, ctypes.POINTER(cl_size)]
                    get_build_info.restype = cl_int
                    size = cl_size(0)
                    get_build_info(program, selected_device, 0x1183, 0, None, ctypes.byref(size))
                    data = ctypes.create_string_buffer(size.value or 1)
                    get_build_info(program, selected_device, 0x1183, len(data), data, None)
                    build_log = _decode_c_string(data.raw) or ""
                except Exception as exc:
                    build_log = str(exc)
                return _status("blocked", reason="OpenCL program build failed", error_code=int(error_code.value), build_log=build_log, library=loaded, device=device_name)

            create_kernel = lib.clCreateKernel
            create_kernel.argtypes = [cl_program, ctypes.c_char_p, ctypes.POINTER(cl_int)]
            create_kernel.restype = cl_kernel
            kernel = create_kernel(program, b"add", ctypes.byref(error_code))
            if not kernel or error_code.value != _CL_SUCCESS:
                return _status("blocked", reason="OpenCL kernel creation failed", error_code=int(error_code.value), library=loaded, device=device_name)
            set_arg = lib.clSetKernelArg
            set_arg.argtypes = [cl_kernel, cl_uint, cl_size, ctypes.c_void_p]
            set_arg.restype = cl_int
            for index, buffer in enumerate((buffer_a, buffer_b, buffer_out)):
                # ctypes returns an integer for a non-null c_void_p array
                # element.  Wrap it before taking its size/address; otherwise
                # x86_64 and aarch64 report ``this type has no size``.
                memory_handle = ctypes.c_void_p(buffer)
                error_code.value = set_arg(kernel, index, ctypes.sizeof(memory_handle), ctypes.byref(memory_handle))
                if error_code.value != _CL_SUCCESS:
                    return _status("blocked", reason="OpenCL kernel argument setup failed", error_code=int(error_code.value), library=loaded, device=device_name)
            enqueue_kernel = lib.clEnqueueNDRangeKernel
            enqueue_kernel.argtypes = [cl_command_queue, cl_kernel, cl_uint, ctypes.POINTER(cl_size), ctypes.POINTER(cl_size), ctypes.POINTER(cl_size), cl_uint, ctypes.c_void_p, ctypes.c_void_p]
            enqueue_kernel.restype = cl_int
            global_size = (cl_size * 1)(n)
            error_code.value = enqueue_kernel(queue, kernel, 1, None, global_size, None, 0, None, None)
            if error_code.value != _CL_SUCCESS:
                return _status("blocked", reason="OpenCL kernel enqueue failed", error_code=int(error_code.value), library=loaded, device=device_name)
            finish = lib.clFinish
            finish.argtypes = [cl_command_queue]
            finish.restype = cl_int
            error_code.value = finish(queue)
            if error_code.value != _CL_SUCCESS:
                return _status("blocked", reason="OpenCL queue finish failed", error_code=int(error_code.value), library=loaded, device=device_name)
            read_buffer = lib.clEnqueueReadBuffer
            read_buffer.argtypes = [cl_command_queue, cl_mem, cl_uint, cl_size, cl_size, ctypes.c_void_p, cl_uint, ctypes.c_void_p, ctypes.c_void_p]
            read_buffer.restype = cl_int
            error_code.value = read_buffer(queue, buffer_out, _CL_TRUE, 0, ctypes.sizeof(values_out), ctypes.byref(values_out), 0, None, None)
            if error_code.value != _CL_SUCCESS:
                return _status("blocked", reason="OpenCL output download failed", error_code=int(error_code.value), library=loaded, device=device_name)
            expected = [a + b for a, b in zip(values_a, values_b)]
            max_error = max(abs(float(actual) - expected[index]) for index, actual in enumerate(values_out))
            output_bytes = bytes(values_out)
            passed = max_error <= 1e-5
            return _status(
                "pass" if passed else "fail",
                library=loaded,
                platform=platform_name,
                platform_count=int(platform_count.value),
                device_count=device_count,
                device=device_name,
                vendor=device_vendor,
                device_version=device_version,
                driver_version=driver_version,
                max_compute_units=int(units.value),
                kernel="vector_add_fp32",
                elements=n,
                max_abs_error=max_error,
                output_sha256=hashlib.sha256(output_bytes).hexdigest(),
            )
        finally:
            # These functions are intentionally looked up lazily so a partial
            # vendor ICD still yields a structured failure rather than raising.
            for function_name, object_name in (
                ("clReleaseKernel", "kernel"),
                ("clReleaseProgram", "program"),
                ("clReleaseMemObject", "buffer_out"),
                ("clReleaseMemObject", "buffer_b"),
                ("clReleaseMemObject", "buffer_a"),
                ("clReleaseCommandQueue", "queue"),
            ):
                value = locals().get(object_name)
                if value:
                    try:
                        getattr(lib, function_name)(value)
                    except Exception:
                        pass
            release_context()
    except Exception as error:  # ctypes ABI differences should be visible in evidence.
        return _status("fail", reason="OpenCL probe exception", error=repr(error), library=loaded)


# ---------------------------------------------------------------------------
# Vulkan loader/device smoke


def _vulkan_version(value: int) -> str:
    return f"{value >> 22}.{(value >> 12) & 0x3FF}.{value & 0xFFF}"


def probe_vulkan(
    *,
    icd_manifest: str | Path | None = None,
    loader_library: str | Path | None = None,
) -> dict[str, Any]:
    """Probe the Vulkan loader and attempt instance/physical-device creation.

    ``icd_manifest`` is an opt-in, user-directory override intended for board
    bring-up.  It is never installed or copied by this function.  This makes it
    possible to distinguish three cases that otherwise look identical from a
    package inventory: a working loader, a manifest whose shared object is not
    an ICD, and a real physical device.  The default path retains the normal
    system Vulkan search behavior.
    """

    manifest_evidence = _vulkan_manifest_evidence(icd_manifest)
    library_name = str(loader_library) if loader_library else "libvulkan.so.1"
    environment_override = str(Path(icd_manifest).expanduser()) if icd_manifest else None
    if icd_manifest and (
        manifest_evidence.get("status") != "candidate"
        or not manifest_evidence.get("library_exists")
    ):
        return _status(
            "blocked",
            reason="Vulkan ICD manifest is invalid or its library is missing",
            loader=library_name,
            icd_manifest=manifest_evidence,
            icd_manifest_override=environment_override,
        )
    try:
        # The loader reads VK_ICD_FILENAMES while its global dispatch table is
        # initialized.  Set it before loading/calling any Vulkan entry point.
        with _temporary_environment(VK_ICD_FILENAMES=environment_override):
            lib = ctypes.CDLL(library_name)
            return _probe_vulkan_loaded_library(lib, library_name, manifest_evidence, environment_override)
    except OSError as error:
        return _status(
            "blocked",
            reason="Vulkan loader unavailable",
            error=str(error),
            loader=library_name,
            icd_manifest=manifest_evidence,
        )
    except Exception as error:
        return _status(
            "fail",
            reason="Vulkan probe exception",
            error=repr(error),
            loader=library_name,
            icd_manifest=manifest_evidence,
        )


def _probe_vulkan_loaded_library(
    lib: Any,
    library_name: str,
    manifest_evidence: dict[str, Any],
    environment_override: str | None,
) -> dict[str, Any]:
    """Run the ABI calls after the loader has been loaded.

    Keeping this in a separate function makes the loader/ICD contract easy to
    unit-test without requiring Vulkan libraries on the development host.
    """

    try:
        version = ctypes.c_uint32(0)
        enumerate_version = getattr(lib, "vkEnumerateInstanceVersion", None)
        version_result = -1
        if enumerate_version is not None:
            enumerate_version.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
            enumerate_version.restype = ctypes.c_int32
            version_result = int(enumerate_version(ctypes.byref(version)))

        class ExtensionProperties(ctypes.Structure):
            _fields_ = [("extensionName", ctypes.c_char * 256), ("specVersion", ctypes.c_uint32)]

        enumerate_extensions = lib.vkEnumerateInstanceExtensionProperties
        enumerate_extensions.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ExtensionProperties)]
        enumerate_extensions.restype = ctypes.c_int32
        extension_count = ctypes.c_uint32(0)
        extension_result = int(enumerate_extensions(None, ctypes.byref(extension_count), None))
        extensions: list[str] = []
        if extension_result == 0 and extension_count.value:
            extension_values = (ExtensionProperties * min(extension_count.value, 128))()
            extension_result = int(enumerate_extensions(None, ctypes.byref(extension_count), extension_values))
            if extension_result == 0:
                extensions = [_decode_c_string(item.extensionName) or "" for item in extension_values[: extension_count.value]]

        class ApplicationInfo(ctypes.Structure):
            _fields_ = [
                ("sType", ctypes.c_uint32),
                ("pNext", ctypes.c_void_p),
                ("pApplicationName", ctypes.c_char_p),
                ("applicationVersion", ctypes.c_uint32),
                ("pEngineName", ctypes.c_char_p),
                ("engineVersion", ctypes.c_uint32),
                ("apiVersion", ctypes.c_uint32),
            ]

        class InstanceCreateInfo(ctypes.Structure):
            _fields_ = [
                ("sType", ctypes.c_uint32),
                ("pNext", ctypes.c_void_p),
                ("flags", ctypes.c_uint32),
                ("pApplicationInfo", ctypes.POINTER(ApplicationInfo)),
                ("enabledLayerCount", ctypes.c_uint32),
                ("ppEnabledLayerNames", ctypes.c_void_p),
                ("enabledExtensionCount", ctypes.c_uint32),
                ("ppEnabledExtensionNames", ctypes.c_void_p),
            ]

        requested_version = min(version.value or ((1 << 22) | (2 << 12)), (1 << 22) | (2 << 12))
        app = ApplicationInfo(0, None, b"edgeforge-rk3588-smoke", 1, b"edgeforge", 1, requested_version)
        create_info = InstanceCreateInfo(1, None, 0, ctypes.pointer(app), 0, None, 0, None)
        instance = ctypes.c_void_p()
        create_instance = lib.vkCreateInstance
        create_instance.argtypes = [ctypes.POINTER(InstanceCreateInfo), ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        create_instance.restype = ctypes.c_int32
        create_result = int(create_instance(ctypes.byref(create_info), None, ctypes.byref(instance)))
        result: dict[str, Any] = {
            "loader": library_name,
            "loader_library": library_name,
            "loader_api_version": _vulkan_version(version.value) if version_result == 0 else None,
            "loader_version_result": version_result,
            "global_extension_count": int(extension_count.value),
            "global_extensions": extensions,
            "instance_result": create_result,
            "icd_manifest_override": environment_override,
            "icd_manifest": manifest_evidence,
        }
        if create_result != 0 or not instance:
            if create_result == -9 and environment_override and manifest_evidence.get("library_exists"):
                reason = "Vulkan ICD rejected or lacks loader entry points"
            else:
                reason = "no Vulkan ICD/device registered" if create_result == -9 else "Vulkan instance creation failed"
            return _status("blocked", reason=reason, **result)

        try:
            enumerate_devices = lib.vkEnumeratePhysicalDevices
            enumerate_devices.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_void_p)]
            enumerate_devices.restype = ctypes.c_int32
            count = ctypes.c_uint32(0)
            device_result = int(enumerate_devices(instance, ctypes.byref(count), None))
            result["physical_device_result"] = device_result
            result["physical_device_count"] = int(count.value)
            passed = device_result == 0 and count.value > 0
            return _status("pass" if passed else "blocked", reason=None if passed else "Vulkan instance has no physical device", **result)
        finally:
            destroy_instance = lib.vkDestroyInstance
            destroy_instance.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            destroy_instance.restype = None
            destroy_instance(instance, None)
    except Exception as error:
        return _status(
            "fail",
            reason="Vulkan probe exception",
            error=repr(error),
            loader=library_name,
            icd_manifest=manifest_evidence,
            icd_manifest_override=environment_override,
        )


# ---------------------------------------------------------------------------
# RKNN board-runtime smoke


RKNN_QUERY_IN_OUT_NUM = 0
RKNN_QUERY_INPUT_ATTR = 1
RKNN_QUERY_OUTPUT_ATTR = 2
RKNN_QUERY_PERF_RUN = 4
RKNN_QUERY_SDK_VERSION = 5
RKNN_TENSOR_TYPE_BYTES = {
    0: 4,  # FP32
    1: 2,  # FP16
    2: 1,  # INT8
    3: 1,  # UINT8
    4: 2,  # INT16
    5: 2,  # UINT16
    6: 4,  # INT32
    7: 4,  # UINT32
    8: 8,  # INT64
    9: 1,  # BOOL
}
RKNN_ERROR_NAMES = {
    0: "RKNN_SUCC",
    -1: "RKNN_ERR_FAIL",
    -2: "RKNN_ERR_TIMEOUT",
    -3: "RKNN_ERR_DEVICE_UNAVAILABLE",
    -4: "RKNN_ERR_MALLOC_FAIL",
    -5: "RKNN_ERR_PARAM_INVALID",
    -6: "RKNN_ERR_MODEL_INVALID",
    -7: "RKNN_ERR_CTX_INVALID",
    -8: "RKNN_ERR_INPUT_INVALID",
    -9: "RKNN_ERR_OUTPUT_INVALID",
    -10: "RKNN_ERR_DEVICE_UNMATCH",
    -11: "RKNN_ERR_INCOMPATILE_PRE_COMPILE_MODEL",
    -12: "RKNN_ERR_INCOMPATILE_OPTIMIZATION_LEVEL_VERSION",
    -13: "RKNN_ERR_TARGET_PLATFORM_UNMATCH",
}


class _RknnSDKVersion(ctypes.Structure):
    _fields_ = [("api_version", ctypes.c_char * 256), ("drv_version", ctypes.c_char * 256)]


class _RknnInputOutputNum(ctypes.Structure):
    _fields_ = [("n_input", ctypes.c_uint32), ("n_output", ctypes.c_uint32)]


class _RknnTensorAttr(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("n_dims", ctypes.c_uint32),
        ("dims", ctypes.c_uint32 * 16),
        ("name", ctypes.c_char * 256),
        ("n_elems", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("fmt", ctypes.c_int32),
        ("type", ctypes.c_int32),
        ("qnt_type", ctypes.c_int32),
        ("fl", ctypes.c_int8),
        ("zp", ctypes.c_int32),
        ("scale", ctypes.c_float),
        ("w_stride", ctypes.c_uint32),
        ("size_with_stride", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("h_stride", ctypes.c_uint32),
    ]


class _RknnInput(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("type", ctypes.c_int32),
        ("fmt", ctypes.c_int32),
    ]


class _RknnOutput(ctypes.Structure):
    _fields_ = [
        ("want_float", ctypes.c_uint8),
        ("is_prealloc", ctypes.c_uint8),
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
    ]


class _RknnPerfRun(ctypes.Structure):
    _fields_ = [("run_duration", ctypes.c_int64)]


def _default_model_path() -> Path | None:
    candidates = [
        os.environ.get("EDGEFORGE_RKNN_MODEL"),
        str(Path.home() / ".cache/edgeforge/rk3588-smoke/rknpu2-v1.5.2/mobilenet_v1.rknn"),
        "/usr/share/rknn_demo/mobilenet_ssd.rknn",
    ]
    return next((Path(item) for item in candidates if item and Path(item).is_file()), None)


def _default_runtime_path() -> Path | None:
    candidates = [
        os.environ.get("EDGEFORGE_RKNN_RUNTIME"),
        "/usr/lib/librknnrt.so",
        "/usr/lib/librknn_api.so",
        # A user-cache RKNPU2 runtime is a fallback for boards whose system
        # image does not ship a usable library; never prefer it over the
        # installed board runtime in a normal smoke.
        str(Path.home() / ".cache/edgeforge/rk3588-smoke/rknpu2-v1.5.2/librknnrt.so"),
    ]
    return next((Path(item) for item in candidates if item and Path(item).is_file()), None)


def _rknn_error(value: int) -> dict[str, Any]:
    return {"code": int(value), "name": RKNN_ERROR_NAMES.get(int(value), "UNKNOWN")}


def _tensor_summary(attr: _RknnTensorAttr) -> dict[str, Any]:
    return {
        "index": int(attr.index),
        "name": _decode_c_string(attr.name),
        "n_dims": int(attr.n_dims),
        "dims": [int(item) for item in attr.dims[: attr.n_dims]],
        "n_elems": int(attr.n_elems),
        "size": int(attr.size),
        "size_with_stride": int(attr.size_with_stride),
        "format": int(attr.fmt),
        "type": int(attr.type),
        "quantization_type": int(attr.qnt_type),
        "zero_point": int(attr.zp),
        "scale": float(attr.scale),
    }


def probe_rknn(model_path: str | Path | None = None, runtime_library: str | Path | None = None, repeat: int = 3) -> dict[str, Any]:
    """Load an offline RKNN model, execute deterministic zero-input inference."""

    model = Path(model_path) if model_path else _default_model_path()
    runtime = Path(runtime_library) if runtime_library else _default_runtime_path()
    if model is None:
        return _status("blocked", reason="no RKNN model found")
    if runtime is None:
        return _status("blocked", reason="no RKNN runtime library found", model=str(model))
    try:
        model_bytes = model.read_bytes()
        model_buffer = ctypes.create_string_buffer(model_bytes)
        # rknn_context is uint64_t on the aarch64 Linux ABI used by RK3588.
        context_type = ctypes.c_uint64
        lib = ctypes.CDLL(str(runtime))
        lib.rknn_init.argtypes = [ctypes.POINTER(context_type), ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
        lib.rknn_init.restype = ctypes.c_int32
        lib.rknn_destroy.argtypes = [context_type]
        lib.rknn_destroy.restype = ctypes.c_int32
        context = context_type(0)
        init_result = int(lib.rknn_init(ctypes.byref(context), ctypes.cast(model_buffer, ctypes.c_void_p), len(model_bytes), 0, None))
        common: dict[str, Any] = {
            "runtime_library": str(runtime),
            "runtime_sha256": hashlib.sha256(runtime.read_bytes()).hexdigest(),
            "model": str(model),
            "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
            "model_bytes": len(model_bytes),
            "init": _rknn_error(init_result),
            "model_correctness": {
                "status": "not-evaluated",
                "reason": "smoke verifies Runtime execution and zero-input determinism; no labeled reference comparison",
            },
        }
        if init_result != 0:
            return _status("blocked", reason="RKNN model init failed", **common)

        lib.rknn_query.argtypes = [context_type, ctypes.c_int32, ctypes.c_void_p, ctypes.c_uint32]
        lib.rknn_query.restype = ctypes.c_int32
        sdk = _RknnSDKVersion()
        sdk_result = int(lib.rknn_query(context, RKNN_QUERY_SDK_VERSION, ctypes.byref(sdk), ctypes.sizeof(sdk)))
        io = _RknnInputOutputNum()
        io_result = int(lib.rknn_query(context, RKNN_QUERY_IN_OUT_NUM, ctypes.byref(io), ctypes.sizeof(io)))
        common["sdk"] = {
            "result": _rknn_error(sdk_result),
            "api_version": _decode_c_string(sdk.api_version),
            "driver_version": _decode_c_string(sdk.drv_version),
        }
        common["io_query"] = _rknn_error(io_result)
        if io_result != 0 or io.n_input == 0 or io.n_output == 0 or io.n_input > 16 or io.n_output > 16:
            lib.rknn_destroy(context)
            return _status("blocked", reason="RKNN I/O metadata query failed", **common)

        inputs: list[_RknnTensorAttr] = []
        outputs: list[_RknnTensorAttr] = []
        for index in range(int(io.n_input)):
            attr = _RknnTensorAttr()
            attr.index = index
            result = int(lib.rknn_query(context, RKNN_QUERY_INPUT_ATTR, ctypes.byref(attr), ctypes.sizeof(attr)))
            if result != 0:
                lib.rknn_destroy(context)
                return _status("blocked", reason="RKNN input metadata query failed", index=index, error=_rknn_error(result), **common)
            inputs.append(attr)
        for index in range(int(io.n_output)):
            attr = _RknnTensorAttr()
            attr.index = index
            result = int(lib.rknn_query(context, RKNN_QUERY_OUTPUT_ATTR, ctypes.byref(attr), ctypes.sizeof(attr)))
            if result != 0:
                lib.rknn_destroy(context)
                return _status("blocked", reason="RKNN output metadata query failed", index=index, error=_rknn_error(result), **common)
            outputs.append(attr)
        common["inputs"] = [_tensor_summary(item) for item in inputs]
        common["outputs"] = [_tensor_summary(item) for item in outputs]

        lib.rknn_inputs_set.argtypes = [context_type, ctypes.c_uint32, ctypes.POINTER(_RknnInput)]
        lib.rknn_inputs_set.restype = ctypes.c_int32
        lib.rknn_run.argtypes = [context_type, ctypes.c_void_p]
        lib.rknn_run.restype = ctypes.c_int32
        lib.rknn_outputs_get.argtypes = [context_type, ctypes.c_uint32, ctypes.POINTER(_RknnOutput), ctypes.c_void_p]
        lib.rknn_outputs_get.restype = ctypes.c_int32
        lib.rknn_outputs_release.argtypes = [context_type, ctypes.c_uint32, ctypes.POINTER(_RknnOutput)]
        lib.rknn_outputs_release.restype = ctypes.c_int32
        input_buffers: list[Any] = []
        input_values: list[_RknnInput] = []
        for attr in inputs:
            byte_count = int(attr.size_with_stride or attr.size or (attr.n_elems * RKNN_TENSOR_TYPE_BYTES.get(int(attr.type), 1)))
            byte_count = max(byte_count, 1)
            buffer = (ctypes.c_ubyte * byte_count)()
            input_buffers.append(buffer)
            input_values.append(_RknnInput(int(attr.index), ctypes.cast(buffer, ctypes.c_void_p), byte_count, 1, int(attr.type), int(attr.fmt)))
        input_array = (_RknnInput * len(input_values))(*input_values)
        set_result = int(lib.rknn_inputs_set(context, len(input_values), input_array))
        common["inputs_set"] = _rknn_error(set_result)
        if set_result != 0:
            lib.rknn_destroy(context)
            return _status("blocked", reason="RKNN input setup failed", **common)

        repeat = max(1, min(int(repeat), 100))
        durations_ms: list[float] = []
        output_hashes: list[list[str]] = []
        perf_runs_us: list[int] = []
        for _ in range(repeat):
            started = time.perf_counter()
            run_result = int(lib.rknn_run(context, None))
            durations_ms.append((time.perf_counter() - started) * 1000.0)
            if run_result != 0:
                lib.rknn_destroy(context)
                return _status("blocked", reason="RKNN inference run failed", error=_rknn_error(run_result), **common)
            output_values = (_RknnOutput * len(outputs))()
            for index in range(len(outputs)):
                output_values[index].want_float = 1
                output_values[index].is_prealloc = 0
                output_values[index].index = index
            get_result = int(lib.rknn_outputs_get(context, len(outputs), output_values, None))
            if get_result != 0:
                lib.rknn_destroy(context)
                return _status("blocked", reason="RKNN output retrieval failed", error=_rknn_error(get_result), **common)
            hashes: list[str] = []
            for value in output_values:
                raw = ctypes.string_at(value.buf, int(value.size)) if value.buf and value.size else b""
                hashes.append(hashlib.sha256(raw).hexdigest())
            output_hashes.append(hashes)
            lib.rknn_outputs_release(context, len(outputs), output_values)
            perf = _RknnPerfRun()
            perf_result = int(lib.rknn_query(context, RKNN_QUERY_PERF_RUN, ctypes.byref(perf), ctypes.sizeof(perf)))
            if perf_result == 0:
                perf_runs_us.append(int(perf.run_duration))
        destroy_result = int(lib.rknn_destroy(context))
        deterministic = len(set(tuple(item) for item in output_hashes)) <= 1
        common.update(
            {
                "repeat": repeat,
                "run_duration_ms": durations_ms,
                "run_duration_median_ms": sorted(durations_ms)[len(durations_ms) // 2],
                "driver_perf_run_us": perf_runs_us,
                "output_sha256": output_hashes,
                "deterministic_zero_input": deterministic,
                "destroy": _rknn_error(destroy_result),
            }
        )
        return _status("pass" if deterministic and destroy_result == 0 else "fail", **common)
    except OSError as error:
        return _status("blocked", reason="RKNN runtime load failed", error=str(error), model=str(model), runtime_library=str(runtime))
    except Exception as error:
        return _status("fail", reason="RKNN probe exception", error=repr(error), model=str(model), runtime_library=str(runtime))


def run_accelerator_probe(
    *,
    name: str = "orangepi",
    model_path: str | Path | None = None,
    runtime_library: str | Path | None = None,
    repeat: int = 3,
    skip_opencl: bool = False,
    skip_vulkan: bool = False,
    skip_rknn: bool = False,
    vulkan_icd_manifest: str | Path | None = None,
    vulkan_loader_library: str | Path | None = None,
) -> dict[str, Any]:
    evidence = collect_board_evidence()
    gpu: dict[str, Any] = {
        "opencl": _status("skipped", reason="requested by caller") if skip_opencl else probe_opencl(),
        "vulkan": (
            _status("skipped", reason="requested by caller")
            if skip_vulkan
            else probe_vulkan(icd_manifest=vulkan_icd_manifest, loader_library=vulkan_loader_library)
        ),
    }
    gpu_passes = [item.get("status") == "pass" for item in gpu.values() if item.get("status") != "skipped"]
    gpu["status"] = "pass" if gpu_passes and all(gpu_passes) else ("partial" if any(gpu_passes) else "blocked")
    npu = _status("skipped", reason="requested by caller") if skip_rknn else probe_rknn(model_path=model_path, runtime_library=runtime_library, repeat=repeat)
    overall = "pass" if gpu["status"] == "pass" and npu.get("status") == "pass" else ("partial" if gpu["status"] in {"pass", "partial"} or npu.get("status") == "pass" else "blocked")
    return {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "probed_at_utc": _utc(),
        "probe_mode": "offline-api-smoke",
        "manual_reference": "OrangePi_5_Ultra_RK3588_user_manual_v1.0.pdf §3.37",
        "toolchain_contract": {
            "conversion_host": "Ubuntu PC + RKNN-Toolkit2",
            "board_runtime": "RKNPU2 librknnrt.so / rknn_server",
            "board_api": "RKNN C API",
            "camera_used": False,
        },
        "evidence": evidence,
        "gpu": gpu,
        "npu": npu,
        "status": overall,
        "scientific_conclusion_allowed": False,
    }


def result_digest(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
