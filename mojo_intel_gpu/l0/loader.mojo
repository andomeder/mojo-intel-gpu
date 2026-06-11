"""Level Zero dynamic library loader and function bindings.

Loads libze_loader.so at runtime via OwnedDLHandle and provides
type-safe Mojo wrappers for all Level Zero C functions.
"""
from std.ffi import OwnedDLHandle
from std.memory import alloc, memset
from .types import (
    ZeResult, ZeDeviceType, DeviceProperties, ComputeProperties,
    ZeGroupCount,
    ZeDriverHandle, ZeDeviceHandle, ZeContextHandle, ZeCommandListHandle,
    ZeModuleHandle, ZeKernelHandle,
    ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES, ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES,
    ZE_STRUCTURE_TYPE_CONTEXT_DESC, ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
    ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
    ZE_STRUCTURE_TYPE_MODULE_DESC, ZE_STRUCTURE_TYPE_KERNEL_DESC,
    ZE_STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC, ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES,
    ZE_INIT_DRIVER_TYPE_FLAG_GPU,
)


struct LevelZeroLibrary(Movable):
    """Level Zero runtime library loaded via dlopen.

    Provides access to all Level Zero C functions with zero-overhead FFI.
    The library is loaded once and function pointers are resolved on first use.
    """
    var _lib: OwnedDLHandle

    def __init__(out self) raises:
        """Load the Level Zero runtime library."""
        self._lib = OwnedDLHandle("libze_loader.so")

    # =========================================================================
    # Driver & Device enumeration
    # =========================================================================

    def init_drivers(self, count_ptr: Int, drivers_ptr: Int) -> ZeResult:
        """Initialize drivers and retrieve handles (replaces zeInit + zeDriverGet).

        Args:
            count_ptr: Int address of UInt32 count buffer.
            drivers_ptr: Int address of driver handles buffer, or Int(0) for count query.
        """
        # ze_init_driver_type_desc_t: stype(4)+pad(4)+pNext(8)+flags(4)+pad(4) = 24 bytes
        var desc = alloc[Int8](24)
        memset(desc, 0, 24)
        desc.bitcast[UInt32]()[0] = ZE_STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC
        desc.bitcast[UInt32]()[4] = ZE_INIT_DRIVER_TYPE_FLAG_GPU

        var result = self._lib.call["zeInitDrivers", Int32](
            count_ptr, drivers_ptr, Int(desc)
        )
        desc.free()
        return ZeResult(UInt32(result))

    def driver_get(self, count_ptr: Int, drivers_ptr: Int) -> ZeResult:
        """Get available drivers (deprecated, prefer init_drivers).

        Args:
            count_ptr: Int address of UInt32 count buffer.
            drivers_ptr: Int address of driver handles buffer, or Int(0) for count query.
        """
        var result = self._lib.call["zeDriverGet", Int32](
            count_ptr, drivers_ptr
        )
        return ZeResult(UInt32(result))
    def device_get(self, driver: Int, count_ptr: Int, devices_ptr: Int) -> ZeResult:
        """Get devices for a driver.

        Args:
            driver: Driver handle.
            count_ptr: Int address of UInt32 count buffer.
            devices_ptr: Int address of device handles buffer, or Int(0) for count query.
        """
        var result = self._lib.call["zeDeviceGet", Int32](
            driver, count_ptr, devices_ptr
        )
        return ZeResult(UInt32(result))

    def device_get_properties(self, device: Int, props_ptr: Int) -> ZeResult:
        """Get device properties (raw pointer to struct)."""
        var result = self._lib.call["zeDeviceGetProperties", Int32](
            device, props_ptr
        )
        return ZeResult(UInt32(result))

    def device_get_compute_properties(self, device: Int, props_ptr: Int) -> ZeResult:
        """Get device compute properties (raw pointer to struct)."""
        var result = self._lib.call["zeDeviceGetComputeProperties", Int32](
            device, props_ptr
        )
        return ZeResult(UInt32(result))

    def device_get_command_queue_group_properties(self, device: Int,
                                                   count_ptr: Int,
                                                   props_ptr: Int) -> ZeResult:
        """Get command queue group properties.

        Args:
            device: Device handle.
            count_ptr: Int address of UInt32 count buffer.
            props_ptr: Int address of queue group properties array, or Int(0) for count query.
        """
        var result = self._lib.call["zeDeviceGetCommandQueueGroupProperties", Int32](
            device, count_ptr, props_ptr
        )
        return ZeResult(UInt32(result))

    # =========================================================================
    # Context management
    # =========================================================================

    def context_create(self, driver: Int, ctx_out: Int) -> ZeResult:
        """Create a context.

        Args:
            driver: Driver handle.
            ctx_out: Int address of context handle output buffer.
        """
        # ze_context_desc_t: stype(4) + pad(4) + pNext(8) + flags(4) + pad(4) = 24 bytes
        var desc = alloc[Int8](24)
        memset(desc, 0, 24)
        # Set stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC (13)
        desc.bitcast[UInt32]()[0] = ZE_STRUCTURE_TYPE_CONTEXT_DESC

        var result = self._lib.call["zeContextCreate", Int32](
            driver, desc, ctx_out
        )
        desc.free()
        return ZeResult(UInt32(result))

    def context_destroy(self, ctx: Int) -> ZeResult:
        """Destroy a context."""
        var result = self._lib.call["zeContextDestroy", Int32](ctx)
        return ZeResult(UInt32(result))

    # =========================================================================
    # Command list (immediate — synchronous execution)
    # =========================================================================

    def command_list_create_immediate(self, ctx: Int, device: Int,
                                       cmdlist_out: Int,
                                       ordinal: UInt32 = 0) -> ZeResult:
        """Create an immediate command list (commands execute immediately).

        Args:
            ctx: Context handle.
            device: Device handle.
            cmdlist_out: Int address of command list handle output buffer.
            ordinal: Queue group ordinal (from queue group properties query).
        """
        # ze_command_queue_desc_t: stype(4)+pad(4)+pNext(8)+ordinal(4)+index(4)+
        #   flags(4)+mode(4)+priority(4)+pad(4) = 40 bytes
        var desc = alloc[Int8](40)
        memset(desc, 0, 40)
        desc.bitcast[UInt32]()[0] = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC
        # ordinal at u32[4] (offset 16), index at u32[5], mode at u32[7]
        desc.bitcast[UInt32]()[4] = ordinal
        desc.bitcast[UInt32]()[5] = 0  # index
        desc.bitcast[UInt32]()[7] = 2  # ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS

        var result = self._lib.call["zeCommandListCreateImmediate", Int32](
            ctx, device, desc, cmdlist_out
        )
        desc.free()
        return ZeResult(UInt32(result))

    def command_list_destroy(self, cmdlist: Int) -> ZeResult:
        """Destroy a command list."""
        var result = self._lib.call["zeCommandListDestroy", Int32](cmdlist)
        return ZeResult(UInt32(result))

    # =========================================================================
    # Memory allocation
    # =========================================================================

    def mem_alloc_device(self, ctx: Int, size: UInt64, alignment: UInt64,
                          device: Int, ptr_out: Int) -> ZeResult:
        """Allocate device memory.

        Args:
            ctx: Context handle.
            size: Number of bytes to allocate.
            alignment: Memory alignment.
            device: Device handle.
            ptr_out: Int address of pointer output buffer.
        """
        # ze_device_mem_alloc_desc_t: stype(4)+pad(4)+pNext(8)+flags(4)+ordinal(4) = 24 bytes
        var desc = alloc[Int8](24)
        memset(desc, 0, 24)
        desc.bitcast[UInt32]()[0] = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC

        var result = self._lib.call["zeMemAllocDevice", Int32](
            ctx, desc, size, alignment, device, ptr_out
        )
        desc.free()
        return ZeResult(UInt32(result))

    def mem_alloc_host(self, ctx: Int, size: UInt64, alignment: UInt64,
                        ptr_out: Int) -> ZeResult:
        """Allocate host-accessible memory.

        Args:
            ctx: Context handle.
            size: Number of bytes to allocate.
            alignment: Memory alignment.
            ptr_out: Int address of pointer output buffer.
        """
        # ze_host_mem_alloc_desc_t: stype(4)+pad(4)+pNext(8)+flags(4)+pad(4) = 24 bytes
        var desc = alloc[Int8](24)
        memset(desc, 0, 24)
        desc.bitcast[UInt32]()[0] = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC

        var result = self._lib.call["zeMemAllocHost", Int32](
            ctx, desc, size, alignment, ptr_out
        )
        desc.free()
        return ZeResult(UInt32(result))

    def mem_free(self, ctx: Int, ptr: Int) -> ZeResult:
        """Free device or host memory."""
        var result = self._lib.call["zeMemFree", Int32](ctx, ptr)
        return ZeResult(UInt32(result))

    # =========================================================================
    # Memory copy (immediate command list — synchronous)
    # =========================================================================

    def memcpy_htod(self, cmdlist: Int, dst_device: Int, src_host: Int,
                     size: UInt64) -> ZeResult:
        """Copy host to device (synchronous via immediate command list)."""
        var result = self._lib.call["zeCommandListAppendMemoryCopy", Int32](
            cmdlist, dst_device, src_host, size, Int(0), UInt32(0), Int(0)
        )
        if result != 0:
            return ZeResult(UInt32(result))
        # Sync
        result = self._lib.call["zeCommandListHostSynchronize", Int32](
            cmdlist, UInt64(0xFFFFFFFFFFFFFFFF)
        )
        return ZeResult(UInt32(result))

    def memcpy_dtoh(self, cmdlist: Int, dst_host: Int, src_device: Int,
                     size: UInt64) -> ZeResult:
        """Copy device to host (synchronous via immediate command list)."""
        var result = self._lib.call["zeCommandListAppendMemoryCopy", Int32](
            cmdlist, dst_host, src_device, size, Int(0), UInt32(0), Int(0)
        )
        if result != 0:
            return ZeResult(UInt32(result))
        # Sync
        result = self._lib.call["zeCommandListHostSynchronize", Int32](
            cmdlist, UInt64(0xFFFFFFFFFFFFFFFF)
        )
        return ZeResult(UInt32(result))

    # =========================================================================
    # Module & Kernel management
    # =========================================================================

    def module_create(self, ctx: Int, device: Int, spirv_data: Int,
                       spirv_size: UInt64, module_out: Int) -> ZeResult:
        """Create a module from SPIR-V binary.

        Args:
            ctx: Context handle.
            device: Device handle.
            spirv_data: Int address of SPIR-V binary data.
            spirv_size: Size of SPIR-V binary in bytes.
            module_out: Int address of module handle output buffer.
        """
        # ze_module_desc_t: stype(4)+pad(4)+pNext(8)+format(4)+pad(4)+inputSize(8)+
        #   pInputModule(8)+pBuildFlags(8)+pConstants(8) = 56 bytes
        var desc = alloc[Int8](56)
        memset(desc, 0, 56)
        desc.bitcast[UInt32]()[0] = ZE_STRUCTURE_TYPE_MODULE_DESC
        desc.bitcast[UInt32]()[4] = 0  # ZE_MODULE_FORMAT_IL_SPIRV = 0
        # inputSize at offset 24
        (desc + 24).bitcast[UInt64]()[0] = spirv_size
        # pInputModule at offset 32
        (desc + 32).bitcast[Int]()[0] = spirv_data
        # pBuildFlags at offset 40 — empty string
        (desc + 40).bitcast[Int]()[0] = Int(0)

        var build_log = alloc[Int8](8)
        memset(build_log, 0, 8)

        var result = self._lib.call["zeModuleCreate", Int32](
            ctx, device, desc, module_out, build_log
        )

        if result != 0:
            # Try to get build log
            var log_size = alloc[Int8](8)
            memset(log_size, 0, 8)
            _ = self._lib.call["zeModuleBuildLogGetString", Int32](
                build_log.bitcast[Int]()[0], log_size, Int(0)
            )
            var size_val = log_size.bitcast[Int]()[0]
            if size_val > 0:
                var log_buf = alloc[Int8](size_val + 1)
                _ = self._lib.call["zeModuleBuildLogGetString", Int32](
                    build_log.bitcast[Int]()[0], log_size, log_buf
                )
                print("Module build error:", String(log_buf, size_val))
                log_buf.free()
            log_size.free()

        _ = self._lib.call["zeModuleBuildLogDestroy", Int32](build_log.bitcast[Int]()[0])
        build_log.free()
        desc.free()
        return ZeResult(UInt32(result))

    def module_destroy(self, module: Int) -> ZeResult:
        """Destroy a module."""
        var result = self._lib.call["zeModuleDestroy", Int32](module)
        return ZeResult(UInt32(result))

    def kernel_create(self, module: Int, kernel_name: Int,
                       kernel_out: Int) -> ZeResult:
        """Create a kernel from a module.

        Args:
            module: Module handle.
            kernel_name: Int address of kernel name string.
            kernel_out: Int address of kernel handle output buffer.
        """
        # ze_kernel_desc_t: stype(4)+pad(4)+pNext(8)+flags(4)+pad(4)+pKernelName(8) = 32 bytes
        var desc = alloc[Int8](32)
        memset(desc, 0, 32)
        desc.bitcast[UInt32]()[0] = ZE_STRUCTURE_TYPE_KERNEL_DESC
        # pKernelName at offset 24
        (desc + 24).bitcast[Int]()[0] = kernel_name

        var result = self._lib.call["zeKernelCreate", Int32](
            module, desc, kernel_out
        )
        desc.free()
        return ZeResult(UInt32(result))

    def kernel_destroy(self, kernel: Int) -> ZeResult:
        """Destroy a kernel."""
        var result = self._lib.call["zeKernelDestroy", Int32](kernel)
        return ZeResult(UInt32(result))

    def kernel_set_group_size(self, kernel: Int, x: UInt32, y: UInt32,
                               z: UInt32) -> ZeResult:
        """Set kernel group size."""
        var result = self._lib.call["zeKernelSetGroupSize", Int32](
            kernel, x, y, z
        )
        return ZeResult(UInt32(result))

    def kernel_set_arg_value(self, kernel: Int, index: UInt32, size: UInt64,
                              value: UInt64) -> ZeResult:
        """Set kernel argument value.

        Args:
            kernel: Kernel handle.
            index: Argument index (0-based).
            size: Size of argument in bytes.
            value: Argument value (passed by value for scalars/pointers).
        """
        var value_buf = alloc[UInt64](1)
        value_buf[0] = value
        var result = self._lib.call["zeKernelSetArgumentValue", Int32](
            kernel, index, size, Int(value_buf)
        )
        value_buf.free()
        return ZeResult(UInt32(result))

    # =========================================================================
    # Kernel dispatch
    # =========================================================================

    def command_list_append_launch_kernel(self, cmdlist: Int, kernel: Int,
                                           group_count: ZeGroupCount,
                                           signal_event: Int,
                                           wait_events: Int,
                                           num_wait_events: UInt32) -> ZeResult:
        """Launch a kernel on the command list."""
        # ze_group_count_t is 12 bytes: x(4) + y(4) + z(4)
        var gc = alloc[Int8](12)
        gc.bitcast[UInt32]()[0] = group_count.x
        gc.bitcast[UInt32]()[1] = group_count.y
        gc.bitcast[UInt32]()[2] = group_count.z

        var result = self._lib.call["zeCommandListAppendLaunchKernel", Int32](
            cmdlist, kernel, gc, signal_event, num_wait_events, wait_events
        )
        gc.free()
        return ZeResult(UInt32(result))

    def synchronize(self, cmdlist: Int) -> ZeResult:
        """Synchronize the command list (wait for all commands to complete)."""
        var result = self._lib.call["zeCommandListHostSynchronize", Int32](
            cmdlist, UInt64(0xFFFFFFFFFFFFFFFF)
        )
        return ZeResult(UInt32(result))
