"""Level Zero dynamic library loader and function bindings.

Loads libze_loader.so at runtime via OwnedDLHandle and provides
type-safe Mojo wrappers for all Level Zero C functions.

All function symbols are resolved once at construction via
`OwnedDLHandle.get_symbol` and invoked through the C ABI, so calls are
zero-overhead (a direct CALL instruction) with no per-call dlsym.
"""
from std.ffi import OwnedDLHandle
from std.memory import Pointer, unsafe_memset
from std.memory.alloc import unsafe_alloc
from .types import (
    ZeResult, ZeGroupCount,
    ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES, ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES,
    ZE_STRUCTURE_TYPE_CONTEXT_DESC, ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
    ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
    ZE_STRUCTURE_TYPE_MODULE_DESC, ZE_STRUCTURE_TYPE_KERNEL_DESC,
    ZE_STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC, ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES,
    ZE_INIT_DRIVER_TYPE_FLAG_GPU,
)


@fieldwise_init
struct _CFn[Ret: AnyType]:
    """A resolved C function pointer, callable with the C ABI.

    Stores the dlsym address and invokes it through a C-ABI function
    pointer on each call. Mirrors the stdlib's internal _DLCallable.
    """

    var _addr: Int

    @always_inline
    def __call__[*T: AnyType](self, *args: *T) -> Self.Ret:
        var opaque = Pointer[NoneType, MutUntrackedOrigin](
            unsafe_from_address=self._addr
        )
        var typed_fn = Pointer(to=opaque).unsafe_bitcast[
            def(* a: * T) thin abi("C") -> Self.Ret
        ]()[]
        return typed_fn(*args)

struct LevelZeroLibrary(Movable):
    """Level Zero runtime library loaded via dlopen.

    Resolves all Level Zero C function symbols once at construction and
    exposes zero-overhead C-ABI wrappers for each. The library is
    reference-counted by the OS, so multiple LevelZeroLibrary instances
    share the same underlying handle.
    """

    var _lib: OwnedDLHandle
    var _init_drivers: _CFn[Int32]
    var _device_get: _CFn[Int32]
    var _device_get_properties: _CFn[Int32]
    var _device_get_compute_properties: _CFn[Int32]
    var _device_get_command_queue_group_properties: _CFn[Int32]
    var _context_create: _CFn[Int32]
    var _context_destroy: _CFn[Int32]
    var _command_list_create_immediate: _CFn[Int32]
    var _command_list_destroy: _CFn[Int32]
    var _mem_alloc_device: _CFn[Int32]
    var _mem_alloc_host: _CFn[Int32]
    var _mem_free: _CFn[Int32]
    var _memcpy: _CFn[Int32]
    var _memset: _CFn[Int32]
    var _host_synchronize: _CFn[Int32]
    var _module_create: _CFn[Int32]
    var _module_destroy: _CFn[Int32]
    var _module_build_log_get_string: _CFn[Int32]
    var _module_build_log_destroy: _CFn[Int32]
    var _kernel_create: _CFn[Int32]
    var _kernel_destroy: _CFn[Int32]
    var _kernel_set_group_size: _CFn[Int32]
    var _kernel_set_arg_value: _CFn[Int32]
    var _append_launch_kernel: _CFn[Int32]

    def __init__(out self) raises:
        """Load the Level Zero runtime library and resolve all symbols."""
        self._lib = OwnedDLHandle("libze_loader.so")
        self._init_drivers = Self._resolve(self._lib, "zeInitDrivers")
        self._device_get = Self._resolve(self._lib, "zeDeviceGet")
        self._device_get_properties = Self._resolve(self._lib, "zeDeviceGetProperties")
        self._device_get_compute_properties = Self._resolve(self._lib, "zeDeviceGetComputeProperties")
        self._device_get_command_queue_group_properties = Self._resolve(
            self._lib, "zeDeviceGetCommandQueueGroupProperties"
        )
        self._context_create = Self._resolve(self._lib, "zeContextCreate")
        self._context_destroy = Self._resolve(self._lib, "zeContextDestroy")
        self._command_list_create_immediate = Self._resolve(self._lib, "zeCommandListCreateImmediate")
        self._command_list_destroy = Self._resolve(self._lib, "zeCommandListDestroy")
        self._mem_alloc_device = Self._resolve(self._lib, "zeMemAllocDevice")
        self._mem_alloc_host = Self._resolve(self._lib, "zeMemAllocHost")
        self._mem_free = Self._resolve(self._lib, "zeMemFree")
        self._memcpy = Self._resolve(self._lib, "zeCommandListAppendMemoryCopy")
        self._memset = Self._resolve(self._lib, "zeCommandListAppendMemoryFill")
        self._host_synchronize = Self._resolve(self._lib, "zeCommandListHostSynchronize")
        self._module_create = Self._resolve(self._lib, "zeModuleCreate")
        self._module_destroy = Self._resolve(self._lib, "zeModuleDestroy")
        self._module_build_log_get_string = Self._resolve(self._lib, "zeModuleBuildLogGetString")
        self._module_build_log_destroy = Self._resolve(self._lib, "zeModuleBuildLogDestroy")
        self._kernel_create = Self._resolve(self._lib, "zeKernelCreate")
        self._kernel_destroy = Self._resolve(self._lib, "zeKernelDestroy")
        self._kernel_set_group_size = Self._resolve(self._lib, "zeKernelSetGroupSize")
        self._kernel_set_arg_value = Self._resolve(self._lib, "zeKernelSetArgumentValue")
        self._append_launch_kernel = Self._resolve(self._lib, "zeCommandListAppendLaunchKernel")

    @staticmethod
    def _resolve(lib: OwnedDLHandle, name: String) raises -> _CFn[Int32]:
        """Resolve a Level Zero symbol to a C-ABI callable."""
        var sym = lib.get_symbol[NoneType](name=name)
        if not sym:
            raise Error("Level Zero symbol not found: " + name)
        return _CFn[Int32](Int(sym.value()))

    @staticmethod
    def _check(result: Int32) -> ZeResult:
        """Wrap an Int32 Level Zero result into ZeResult."""
        return ZeResult(UInt32(result))

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
        var desc = unsafe_alloc[Int8](24)
        unsafe_memset(desc, 0, 24)
        desc.unsafe_bitcast[Scalar[DType.uint32]]().unsafe_store(0, ZE_STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC)
        desc.unsafe_bitcast[Scalar[DType.uint32]]().unsafe_store(4, ZE_INIT_DRIVER_TYPE_FLAG_GPU)

        var result = self._init_drivers(count_ptr, drivers_ptr, Int(desc))
        desc.unsafe_free()
        return Self._check(result)

    def device_get(self, driver: Int, count_ptr: Int, devices_ptr: Int) -> ZeResult:
        """Get devices for a driver.

        Args:
            driver: Driver handle.
            count_ptr: Int address of UInt32 count buffer.
            devices_ptr: Int address of device handles buffer, or Int(0) for count query.
        """
        return Self._check(self._device_get(driver, count_ptr, devices_ptr))

    def device_get_properties(self, device: Int, props_ptr: Int) -> ZeResult:
        """Get device properties (raw pointer to struct)."""
        return Self._check(self._device_get_properties(device, props_ptr))

    def device_get_compute_properties(self, device: Int, props_ptr: Int) -> ZeResult:
        """Get device compute properties (raw pointer to struct)."""
        return Self._check(self._device_get_compute_properties(device, props_ptr))

    def device_get_command_queue_group_properties(self, device: Int,
                                                   count_ptr: Int,
                                                   props_ptr: Int) -> ZeResult:
        """Get command queue group properties.

        Args:
            device: Device handle.
            count_ptr: Int address of UInt32 count buffer.
            props_ptr: Int address of queue group properties array, or Int(0) for count query.
        """
        return Self._check(
            self._device_get_command_queue_group_properties(device, count_ptr, props_ptr)
        )

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
        var desc = unsafe_alloc[Int8](24)
        unsafe_memset(desc, 0, 24)
        # Set stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC (13)
        desc.unsafe_bitcast[Scalar[DType.uint32]]().unsafe_store(0, ZE_STRUCTURE_TYPE_CONTEXT_DESC)

        var result = self._context_create(driver, desc, ctx_out)
        desc.unsafe_free()
        return Self._check(result)

    def context_destroy(self, ctx: Int) -> ZeResult:
        """Destroy a context."""
        return Self._check(self._context_destroy(ctx))

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
        var desc = unsafe_alloc[Int8](40)
        unsafe_memset(desc, 0, 40)
        desc.unsafe_bitcast[Scalar[DType.uint32]]().unsafe_store(0, ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC)
        # ordinal at u32[4] (offset 16), index at u32[5], mode at u32[7]
        desc.unsafe_bitcast[Scalar[DType.uint32]]().unsafe_store(4, ordinal)
        desc.unsafe_bitcast[Scalar[DType.uint32]]().unsafe_store(5, 0)  # index
        desc.unsafe_bitcast[Scalar[DType.uint32]]().unsafe_store(7, 2)  # ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS

        var result = self._command_list_create_immediate(ctx, device, desc, cmdlist_out)
        desc.unsafe_free()
        return Self._check(result)

    def command_list_destroy(self, cmdlist: Int) -> ZeResult:
        """Destroy a command list."""
        return Self._check(self._command_list_destroy(cmdlist))

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
        var desc = unsafe_alloc[Int8](24)
        unsafe_memset(desc, 0, 24)
        desc.unsafe_bitcast[Scalar[DType.uint32]]().unsafe_store(0, ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC)

        var result = self._mem_alloc_device(ctx, desc, size, alignment, device, ptr_out)
        desc.unsafe_free()
        return Self._check(result)

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
        var desc = unsafe_alloc[Int8](24)
        unsafe_memset(desc, 0, 24)
        desc.unsafe_bitcast[Scalar[DType.uint32]]().unsafe_store(0, ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC)

        var result = self._mem_alloc_host(ctx, desc, size, alignment, ptr_out)
        desc.unsafe_free()
        return Self._check(result)

    def mem_free(self, ctx: Int, ptr: Int) -> ZeResult:
        """Free device or host memory."""
        return Self._check(self._mem_free(ctx, ptr))

    # =========================================================================
    # Memory copy & fill (immediate command list — synchronous)
    # =========================================================================

    def memcpy_htod(self, cmdlist: Int, dst_device: Int, src_host: Int,
                     size: UInt64) -> ZeResult:
        """Copy host to device (synchronous via immediate command list)."""
        var result = self._memcpy(
            cmdlist, dst_device, src_host, size, Int(0), UInt32(0), Int(0)
        )
        if result != 0:
            return Self._check(result)
        return Self._check(self._host_synchronize(cmdlist, UInt64(0xFFFFFFFFFFFFFFFF)))

    def memcpy_dtoh(self, cmdlist: Int, dst_host: Int, src_device: Int,
                     size: UInt64) -> ZeResult:
        """Copy device to host (synchronous via immediate command list)."""
        var result = self._memcpy(
            cmdlist, dst_host, src_device, size, Int(0), UInt32(0), Int(0)
        )
        if result != 0:
            return Self._check(result)
        return Self._check(self._host_synchronize(cmdlist, UInt64(0xFFFFFFFFFFFFFFFF)))

    def memset_device(self, cmdlist: Int, dst_device: Int, value: UInt8,
                       size: UInt64) -> ZeResult:
        """Fill device memory with a byte value (synchronous)."""
        var pattern = unsafe_alloc[Int8](1)
        pattern.unsafe_store(0, Int8(value))
        var result = self._memset(
            cmdlist, dst_device, Int(pattern), UInt64(1), size,
            Int(0), UInt32(0), Int(0)
        )
        if result != 0:
            pattern.unsafe_free()
            return Self._check(result)
        result = self._host_synchronize(cmdlist, UInt64(0xFFFFFFFFFFFFFFFF))
        pattern.unsafe_free()
        return Self._check(result)

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
        var desc = unsafe_alloc[Int8](56)
        unsafe_memset(desc, 0, 56)
        desc.unsafe_bitcast[Scalar[DType.uint32]]().unsafe_store(0, ZE_STRUCTURE_TYPE_MODULE_DESC)
        desc.unsafe_bitcast[Scalar[DType.uint32]]().unsafe_store(4, 0)  # ZE_MODULE_FORMAT_IL_SPIRV = 0
        # inputSize at offset 24
        desc.unsafe_bitcast[Scalar[DType.uint64]]().unsafe_store(3, spirv_size)
        # pInputModule at offset 32
        desc.unsafe_bitcast[Scalar[DType.int]]().unsafe_store(4, spirv_data)
        # pBuildFlags at offset 40 — empty string
        desc.unsafe_bitcast[Scalar[DType.int]]().unsafe_store(5, Int(0))

        var build_log = unsafe_alloc[Int8](8)
        unsafe_memset(build_log, 0, 8)

        var result = self._module_create(ctx, device, desc, module_out, build_log)

        var build_log_handle = build_log.unsafe_bitcast[Scalar[DType.int]]().unsafe_load(0)
        if result != 0 and build_log_handle != 0:
            # Try to get build log (handle is only valid on build failure)
            var log_size = unsafe_alloc[Int8](8)
            unsafe_memset(log_size, 0, 8)
            _ = self._module_build_log_get_string(
                build_log_handle, log_size, Int(0)
            )
            var size_val = log_size.unsafe_bitcast[Scalar[DType.uint64]]().unsafe_load(0)
            if size_val > 0:
                var log_buf = unsafe_alloc[Int8](Int(size_val))
                _ = self._module_build_log_get_string(
                    build_log_handle, log_size, log_buf
                )
                print("Module build error:", String(log_buf, Int(size_val)))
                log_buf.unsafe_free()
            log_size.unsafe_free()

        if build_log_handle != Int(0):
            _ = self._module_build_log_destroy(build_log_handle)
        build_log.unsafe_free()
        desc.unsafe_free()
        return Self._check(result)

    def module_destroy(self, module: Int) -> ZeResult:
        """Destroy a module."""
        return Self._check(self._module_destroy(module))

    def kernel_create(self, module: Int, kernel_name: Int,
                       kernel_out: Int) -> ZeResult:
        """Create a kernel from a module.

        Args:
            module: Module handle.
            kernel_name: Int address of kernel name string.
            kernel_out: Int address of kernel handle output buffer.
        """
        # ze_kernel_desc_t: stype(4)+pad(4)+pNext(8)+flags(4)+pad(4)+pKernelName(8) = 32 bytes
        var desc = unsafe_alloc[Int8](32)
        unsafe_memset(desc, 0, 32)
        desc.unsafe_bitcast[Scalar[DType.uint32]]().unsafe_store(0, ZE_STRUCTURE_TYPE_KERNEL_DESC)
        # pKernelName at offset 24
        desc.unsafe_bitcast[Scalar[DType.int]]().unsafe_store(3, kernel_name)

        var result = self._kernel_create(module, desc, kernel_out)
        desc.unsafe_free()
        return Self._check(result)

    def kernel_destroy(self, kernel: Int) -> ZeResult:
        """Destroy a kernel."""
        return Self._check(self._kernel_destroy(kernel))

    def kernel_set_group_size(self, kernel: Int, x: UInt32, y: UInt32,
                               z: UInt32) -> ZeResult:
        """Set kernel group size."""
        return Self._check(self._kernel_set_group_size(kernel, x, y, z))

    def kernel_set_arg_value(self, kernel: Int, index: UInt32, size: UInt64,
                              value: UInt64) -> ZeResult:
        """Set kernel argument value.

        Args:
            kernel: Kernel handle.
            index: Argument index (0-based).
            size: Size of argument in bytes.
            value: Argument value (passed by value for scalars/pointers).
        """
        var value_buf = unsafe_alloc[UInt64](1)
        value_buf.unsafe_store(0, value)
        var result = self._kernel_set_arg_value(kernel, index, size, Int(value_buf))
        value_buf.unsafe_free()
        return Self._check(result)

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
        var gc = unsafe_alloc[Int8](12)
        gc.unsafe_bitcast[Scalar[DType.uint32]]().unsafe_store(0, group_count.x)
        gc.unsafe_bitcast[Scalar[DType.uint32]]().unsafe_store(1, group_count.y)
        gc.unsafe_bitcast[Scalar[DType.uint32]]().unsafe_store(2, group_count.z)

        var result = self._append_launch_kernel(
            cmdlist, kernel, gc, signal_event, num_wait_events, wait_events
        )
        gc.unsafe_free()
        return Self._check(result)

    def synchronize(self, cmdlist: Int) -> ZeResult:
        """Synchronize the command list (wait for all commands to complete)."""
        return Self._check(self._host_synchronize(cmdlist, UInt64(0xFFFFFFFFFFFFFFFF)))
