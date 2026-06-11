"""Level Zero type definitions — structs, enums, and constants.

These match the C API layouts from ze_api.h for correct FFI interop.
All struct sizes and field offsets are verified against the C headers.
"""


# =============================================================================
# Result codes
# =============================================================================

@fieldwise_init
struct ZeResult(Copyable, Movable, Writable, Equatable):
    """Level Zero result code wrapper."""
    var value: UInt32

    comptime SUCCESS: UInt32 = 0

    def is_success(self) -> Bool:
        return self.value == Self.SUCCESS

    def is_error(self) -> Bool:
        return self.value != Self.SUCCESS

    def write_to(self, mut writer: Some[Writer]):
        if self.is_success():
            writer.write("ZE_RESULT_SUCCESS")
        else:
            writer.write("ZE_RESULT_ERROR(0x", hex(Int(self.value)), ")")

    def __eq__(self, other: Self) -> Bool:
        return self.value == other.value


# =============================================================================
# Enumerations
# =============================================================================

@fieldwise_init
struct ZeDeviceType(Copyable, Movable, Writable):
    """Device type enumeration."""
    var value: UInt32

    comptime GPU: UInt32 = 1
    comptime CPU: UInt32 = 2
    comptime FPGA: UInt32 = 3
    comptime MCA: UInt32 = 4

    def is_gpu(self) -> Bool:
        return self.value == Self.GPU

    def write_to(self, mut writer: Some[Writer]):
        if self.value == Self.GPU:
            writer.write("GPU")
        elif self.value == Self.CPU:
            writer.write("CPU")
        elif self.value == Self.FPGA:
            writer.write("FPGA")
        elif self.value == Self.MCA:
            writer.write("MCA")
        else:
            writer.write("Unknown(", self.value, ")")


@fieldwise_init
struct ZeModuleFormat(Copyable, Movable, Writable):
    """Module format enumeration."""
    var value: UInt32

    comptime IL_SPIRV: UInt32 = 0
    comptime NATIVE: UInt32 = 1

    def write_to(self, mut writer: Some[Writer]):
        if self.value == Self.IL_SPIRV:
            writer.write("IL_SPIRV")
        elif self.value == Self.NATIVE:
            writer.write("NATIVE")
        else:
            writer.write("Unknown(", self.value, ")")


# =============================================================================
# Structure type IDs (for stype field)
# =============================================================================

comptime ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES: UInt32 = 3
comptime ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES: UInt32 = 4
comptime ZE_STRUCTURE_TYPE_CONTEXT_DESC: UInt32 = 13
comptime ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC: UInt32 = 14
comptime ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC: UInt32 = 21
comptime ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC: UInt32 = 22
comptime ZE_STRUCTURE_TYPE_MODULE_DESC: UInt32 = 27
comptime ZE_STRUCTURE_TYPE_KERNEL_DESC: UInt32 = 29
comptime ZE_STRUCTURE_TYPE_EVENT_POOL_DESC: UInt32 = 16
comptime ZE_STRUCTURE_TYPE_EVENT_DESC: UInt32 = 17
comptime ZE_STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC: UInt32 = 0x00020021
comptime ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES: UInt32 = 6

# Init driver type flags
comptime ZE_INIT_DRIVER_TYPE_FLAG_GPU: UInt32 = 1
comptime ZE_INIT_DRIVER_TYPE_FLAG_NPU: UInt32 = 2

# Command queue group property flags
comptime ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE: UInt32 = 1
comptime ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY: UInt32 = 2


# =============================================================================
# Opaque handle types (all are 8-byte pointers on x86_64)
# =============================================================================

comptime ZeDriverHandle = Int
comptime ZeDeviceHandle = Int
comptime ZeContextHandle = Int
comptime ZeCommandListHandle = Int
comptime ZeModuleHandle = Int
comptime ZeKernelHandle = Int
comptime ZeEventPoolHandle = Int
comptime ZeEventHandle = Int


# =============================================================================
# Device properties struct (matches ze_device_properties_t layout)
# =============================================================================

@fieldwise_init
struct DeviceProperties(Copyable, Movable, Writable):
    """Device properties from zeDeviceGetProperties.

    Layout matches ze_device_properties_t:
      stype(4) + pNext(8) + type(4) + vendorId(4) + deviceId(4) + flags(4) +
      subdeviceId(4) + coreClockRate(4) + maxMemAllocSize(8) + ... + name[256]
    """
    var vendor_id: UInt32
    var device_id: UInt32
    var device_type: ZeDeviceType
    var core_clock_mhz: UInt32
    var max_mem_alloc: UInt64
    var name: String

    def write_to(self, mut writer: Some[Writer]):
        writer.write("DeviceProperties(")
        writer.write("vendor=", hex(Int(self.vendor_id)))
        writer.write(", device=", hex(Int(self.device_id)))
        writer.write(", type=", self.device_type)
        writer.write(", clock=", self.core_clock_mhz, "MHz")
        writer.write(", max_mem=", Int(self.max_mem_alloc) // (1024 * 1024), "MB")
        writer.write(", name=", self.name)
        writer.write(")")


# =============================================================================
# Compute properties struct (matches ze_device_compute_properties_t)
# =============================================================================

@fieldwise_init
struct ComputeProperties(Copyable, Movable, Writable):
    """Compute properties from zeDeviceGetComputeProperties."""
    var max_group_size: UInt32
    var max_group_count_x: UInt32
    var max_group_count_y: UInt32
    var max_group_count_z: UInt32
    var max_shared_local_memory: UInt32
    var num_sub_group_sizes: UInt32
    var sub_group_sizes: List[UInt32]

    def write_to(self, mut writer: Some[Writer]):
        writer.write("ComputeProperties(")
        writer.write("max_group=", self.max_group_size)
        writer.write(", max_groups=(", self.max_group_count_x, ",",
                     self.max_group_count_y, ",", self.max_group_count_z, ")")
        writer.write(", slm=", self.max_shared_local_memory, "B")
        writer.write(", subgroups=", self.sub_group_sizes)
        writer.write(")")


# =============================================================================
# Group count for kernel launch
# =============================================================================

@fieldwise_init
struct ZeGroupCount(Copyable, Movable, Writable):
    """Kernel launch group count — plain 12-byte struct (3x uint32_t)."""
    var x: UInt32
    var y: UInt32
    var z: UInt32

    def write_to(self, mut writer: Some[Writer]):
        writer.write("ZeGroupCount(", self.x, ", ", self.y, ", ", self.z, ")")
