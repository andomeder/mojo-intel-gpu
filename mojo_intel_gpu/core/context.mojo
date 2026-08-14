"""GPU context management — initialization, device discovery, and lifecycle.

Provides IntelGPUContext as the primary entry point for Intel GPU operations.
Handles driver initialization, device selection, and resource cleanup.
"""
from std.memory import Pointer, unsafe_memset
from std.memory.alloc import unsafe_alloc
from ..l0.types import (
    ZeDeviceType, DeviceProperties, ComputeProperties,
    ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES, ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES,
    ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES,
    ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE,
)
from ..l0.loader import LevelZeroLibrary


@fieldwise_init
struct IntelGPUDevice(Copyable, Movable, Writable):
    """Represents a discovered Intel GPU device."""
    var handle: Int
    var properties: DeviceProperties
    var compute: ComputeProperties

    def write_to(self, mut writer: Some[Writer]):
        writer.write("IntelGPUDevice(", self.properties, ")")


struct IntelGPUContext(Movable):
    """High-level context for Intel GPU operations.

    Manages the Level Zero driver, device, context, and command list lifecycle.
    Similar to Mojo's DeviceContext for NVIDIA/AMD/Apple.

    Usage:
        ctx = IntelGPUContext()
        buf = ctx.allocate_device(1024)
        # ... use GPU ...
        ctx.close()
    """
    var _lib: LevelZeroLibrary
    var _driver: Int
    var _device: Int
    var _context: Int
    var _cmdlist: Int
    var _device_info: DeviceProperties
    var _compute_info: ComputeProperties

    def __init__(out self) raises:
        """Initialize Level Zero and discover an Intel GPU.

        Uses zeInitDrivers (replaces deprecated zeInit + zeDriverGet).
        Iterates all drivers to find one with a GPU device.
        Queries queue groups to find a compute-capable ordinal.

        Raises:
            Error: If no Intel GPU is found or initialization fails.
        """
        self._lib = LevelZeroLibrary()
        self._driver = Int(0)
        self._device = Int(0)
        self._context = Int(0)
        self._cmdlist = Int(0)
        self._device_info = DeviceProperties(0, 0, ZeDeviceType(0), 0, 0, "")
        self._compute_info = ComputeProperties(0, 0, 0, 0, 0, 0, List[UInt32]())

        # Get drivers via zeInitDrivers (replaces deprecated zeInit + zeDriverGet)
        var count_buf = unsafe_alloc[UInt32](1)
        count_buf.unsafe_offset(0)[] = 0
        var result = self._lib.init_drivers(Int(count_buf), Int(0))
        if result.is_error() or count_buf.unsafe_offset(0)[] == 0:
            count_buf.unsafe_free()
            raise Error("No Level Zero drivers found: " + String(result))
        var driver_count = count_buf.unsafe_offset(0)[]

        var drivers = unsafe_alloc[Int](Int(driver_count))
        result = self._lib.init_drivers(Int(count_buf), Int(drivers))
        count_buf.unsafe_free()
        if result.is_error():
            drivers.unsafe_free()
            raise Error("Failed to get drivers")

        # Iterate all drivers to find one with a GPU device
        self._driver = Int(0)
        self._device = Int(0)
        for d in range(Int(driver_count)):
            # Get devices for this driver
            var dev_count_buf = unsafe_alloc[UInt32](1)
            dev_count_buf.unsafe_offset(0)[] = 0
            result = self._lib.device_get(drivers.unsafe_offset(d)[], Int(dev_count_buf), Int(0))
            if result.is_error() or dev_count_buf.unsafe_offset(0)[] == 0:
                dev_count_buf.unsafe_free()
                continue
            var device_count = dev_count_buf.unsafe_offset(0)[]

            var devices = unsafe_alloc[Int](Int(device_count))
            result = self._lib.device_get(drivers.unsafe_offset(d)[], Int(dev_count_buf), Int(devices))
            dev_count_buf.unsafe_free()
            if result.is_error():
                devices.unsafe_free()
                continue

            # Find first GPU device
            for i in range(Int(device_count)):
                var props = unsafe_alloc[Int8](400)
                unsafe_memset(props, 0, 400)
                props.unsafe_bitcast[UInt32]().unsafe_offset(0)[] = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES
                var prop_result = self._lib.device_get_properties(devices.unsafe_offset(i)[], Int(props))
                if prop_result.is_success():
                    var dtype = props.unsafe_bitcast[UInt32]().unsafe_offset(4)[]  # type field (offset 16: stype+pad+pNext)
                    if dtype == ZeDeviceType.GPU:
                        self._device = devices.unsafe_offset(i)[]
                        self._driver = drivers.unsafe_offset(d)[]
                        self._device_info = self._parse_device_properties(props)
                        props.unsafe_free()
                        break
                props.unsafe_free()

            devices.unsafe_free()
            if self._device != Int(0):
                break

        drivers.unsafe_free()

        if self._device == Int(0):
            raise Error("No Intel GPU device found")

        # Get compute properties
        var cprops = unsafe_alloc[Int8](200)
        unsafe_memset(cprops, 0, 200)
        cprops.unsafe_bitcast[UInt32]().unsafe_offset(0)[] = ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES
        result = self._lib.device_get_compute_properties(self._device, Int(cprops))
        if result.is_success():
            self._compute_info = self._parse_compute_properties(cprops)
        cprops.unsafe_free()

        # Query command queue groups to find compute-capable ordinal
        var compute_ordinal: UInt32 = 0
        var qg_count_buf = unsafe_alloc[UInt32](1)
        qg_count_buf.unsafe_offset(0)[] = 0
        result = self._lib.device_get_command_queue_group_properties(
            self._device, Int(qg_count_buf), Int(0))
        if result.is_success() and qg_count_buf.unsafe_offset(0)[] > 0:
            var qg_count = qg_count_buf.unsafe_offset(0)[]
            # ze_command_queue_group_properties_t: stype(4)+pad(4)+pNext(8)+flags(4)+pad(4)+maxMemoryFillPatternSize(8)+numQueues(4)+pad(4) = 40 bytes
            var qg_props = unsafe_alloc[Int8](Int(qg_count) * 40)
            unsafe_memset(qg_props, 0, Int(qg_count) * 40)
            for q in range(Int(qg_count)):
                qg_props.unsafe_offset(q * 40).unsafe_bitcast[UInt32]().unsafe_offset(0)[] = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES
            result = self._lib.device_get_command_queue_group_properties(
                self._device, Int(qg_count_buf), Int(qg_props))
            if result.is_success():
                for q in range(Int(qg_count)):
                    var flags = qg_props.unsafe_offset(q * 40).unsafe_bitcast[UInt32]().unsafe_offset(4)[]  # flags at offset 16
                    if (flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) != 0:
                        compute_ordinal = UInt32(q)
                        break
            qg_props.unsafe_free()
        qg_count_buf.unsafe_free()

        # Create context
        var ctx_buf = unsafe_alloc[Int](1)
        ctx_buf.unsafe_offset(0)[] = 0
        result = self._lib.context_create(self._driver, Int(ctx_buf))
        if result.is_error():
            ctx_buf.unsafe_free()
            raise Error("Failed to create context")
        self._context = ctx_buf.unsafe_offset(0)[]
        ctx_buf.unsafe_free()

        # Create immediate command list with compute-capable queue group
        var cmd_buf = unsafe_alloc[Int](1)
        cmd_buf.unsafe_offset(0)[] = 0
        result = self._lib.command_list_create_immediate(
            self._context, self._device, Int(cmd_buf), compute_ordinal
        )
        if result.is_error():
            cmd_buf.unsafe_free()
            _ = self._lib.context_destroy(self._context)
            self._context = Int(0)
            raise Error("Failed to create command list")
        self._cmdlist = cmd_buf.unsafe_offset(0)[]
        cmd_buf.unsafe_free()

    def _parse_device_properties(self, raw: Pointer[Int8, MutUntrackedOrigin]) -> DeviceProperties:
        """Parse raw device properties buffer into DeviceProperties struct."""
        var u32 = raw.unsafe_bitcast[UInt32]()
        var u64 = raw.unsafe_bitcast[UInt64]()

        var vendor_id = u32.unsafe_offset(5)[]  # after stype(4)+pad(4)+pNext(8)
        var device_id = u32.unsafe_offset(6)[]
        var dtype = ZeDeviceType(u32.unsafe_offset(4)[])
        var clock = u32.unsafe_offset(9)[]
        var max_mem = u64.unsafe_offset(5)[]  # byte offset 40

        var name = ""
        var np = raw.unsafe_offset(112).unsafe_bitcast[Int8]()  # name at byte 112
        for i in range(256):
            var ch = np.unsafe_offset(i)[]
            if ch == 0:
                break
            name += chr(Int(ch))

        return DeviceProperties(vendor_id, device_id, dtype^, clock, max_mem, name)

    def _parse_compute_properties(self, raw: Pointer[Int8, MutUntrackedOrigin]) -> ComputeProperties:
        """Parse raw compute properties buffer."""
        var u32 = raw.unsafe_bitcast[UInt32]()

        var max_group_size = u32.unsafe_offset(4)[]  # after stype(4)+pad(4)+pNext(8)
        var max_group_x = u32.unsafe_offset(8)[]  # maxGroupCountX
        var max_group_y = u32.unsafe_offset(9)[]  # maxGroupCountY
        var max_group_z = u32.unsafe_offset(10)[]  # maxGroupCountZ
        var max_slm = u32.unsafe_offset(11)[]  # maxSharedLocalMemory
        var num_subgroups = u32.unsafe_offset(12)[]

        var subgroups = List[UInt32]()
        for i in range(min(Int(num_subgroups), 8)):
            subgroups.append(u32.unsafe_offset(13 + i)[])

        return ComputeProperties(max_group_size, max_group_x, max_group_y,
                                  max_group_z, max_slm, num_subgroups, subgroups^)

    # =========================================================================
    # Properties
    # =========================================================================

    def device(self) -> Int:
        """Get the device handle."""
        return self._device

    def context(self) -> Int:
        """Get the context handle."""
        return self._context

    def command_list(self) -> Int:
        """Get the command list handle."""
        return self._cmdlist

    def library(self) -> Pointer[LevelZeroLibrary, ImmUntrackedOrigin]:
        """Get a borrowed, non-owning reference to the shared Level Zero library."""
        return Pointer(to=self._lib).unsafe_origin_cast[ImmUntrackedOrigin]()

    def device_info(self) -> DeviceProperties:
        """Get device properties."""
        return self._device_info.copy()

    def compute_info(self) -> ComputeProperties:
        """Get compute properties."""
        return self._compute_info.copy()

    def is_initialized(self) -> Bool:
        """Check if the context is initialized."""
        return self._device != Int(0)

    # =========================================================================
    # Resource management
    # =========================================================================

    def close(mut self):
        """Clean up all resources. Call this when done with the GPU."""
        if self._cmdlist != Int(0):
            _ = self._lib.command_list_destroy(self._cmdlist)
            self._cmdlist = Int(0)
        if self._context != Int(0):
            _ = self._lib.context_destroy(self._context)
            self._context = Int(0)
        self._device = Int(0)
        self._driver = Int(0)

    def __deinit__(deinit self):
        """Automatic cleanup on destruction."""
        if self.is_initialized():
            self.close()

    # =========================================================================
    # Memory operations (delegated to LevelZeroLibrary)
    # =========================================================================

    def allocate_device(mut self, size: UInt64) raises -> Int:
        """Allocate device memory.

        Args:
            size: Number of bytes to allocate.

        Returns:
            Device pointer (Int).

        Raises:
            Error: If allocation fails.
        """
        var ptr_buf = unsafe_alloc[Int](1)
        ptr_buf.unsafe_offset(0)[] = 0
        var result = self._lib.mem_alloc_device(
            self._context, size, 64, self._device, Int(ptr_buf)
        )
        if result.is_error():
            ptr_buf.unsafe_free()
            raise Error("Device memory allocation failed: " + String(result))
        var ptr = ptr_buf.unsafe_offset(0)[]
        ptr_buf.unsafe_free()
        return ptr

    def allocate_host(mut self, size: UInt64) raises -> Int:
        """Allocate host-accessible memory.

        Args:
            size: Number of bytes to allocate.

        Returns:
            Host pointer (Int).

        Raises:
            Error: If allocation fails.
        """
        var ptr_buf = unsafe_alloc[Int](1)
        ptr_buf.unsafe_offset(0)[] = 0
        var result = self._lib.mem_alloc_host(
            self._context, size, 64, Int(ptr_buf)
        )
        if result.is_error():
            ptr_buf.unsafe_free()
            raise Error("Host memory allocation failed: " + String(result))
        var ptr = ptr_buf.unsafe_offset(0)[]
        ptr_buf.unsafe_free()
        return ptr

    def free_device(mut self, ptr: Int):
        """Free device memory."""
        if ptr != Int(0):
            _ = self._lib.mem_free(self._context, ptr)

    def free_host(mut self, ptr: Int):
        """Free host memory."""
        if ptr != Int(0):
            _ = self._lib.mem_free(self._context, ptr)

    def memcpy_htod(mut self, dst_device: Int, src_host: Int,
                    size: UInt64) raises:
        """Copy data from host to device.

        Args:
            dst_device: Device pointer.
            src_host: Host pointer.
            size: Number of bytes to copy.

        Raises:
            Error: If copy fails.
        """
        if dst_device == Int(0) or src_host == Int(0):
            raise Error("Host-to-device copy: null pointer")
        var result = self._lib.memcpy_htod(self._cmdlist, dst_device, src_host, size)
        if result.is_error():
            raise Error("Host-to-device copy failed: " + String(result))

    def memcpy_dtoh(mut self, dst_host: Int, src_device: Int,
                    size: UInt64) raises:
        """Copy data from device to host.

        Args:
            dst_host: Host pointer.
            src_device: Device pointer.
            size: Number of bytes to copy.

        Raises:
            Error: If copy fails.
        """
        if dst_host == Int(0) or src_device == Int(0):
            raise Error("Device-to-host copy: null pointer")
        var result = self._lib.memcpy_dtoh(self._cmdlist, dst_host, src_device, size)
        if result.is_error():
            raise Error("Device-to-host copy failed: " + String(result))

    def memset_device(mut self, dst_device: Int, value: UInt8,
                       size: UInt64) raises:
        """Fill device memory with a byte value.

        Args:
            dst_device: Device pointer.
            value: Byte value to fill.
            size: Number of bytes to fill.

        Raises:
            Error: If fill fails.
        """
        if dst_device == Int(0):
            raise Error("Device memset: null pointer")
        var result = self._lib.memset_device(self._cmdlist, dst_device, value, size)
        if result.is_error():
            raise Error("Device memset failed: " + String(result))

    def synchronize(mut self) raises:
        """Wait for all GPU operations to complete.

        Raises:
            Error: If synchronization fails.
        """
        var result = self._lib.synchronize(self._cmdlist)
        if result.is_error():
            raise Error("Synchronization failed: " + String(result))
