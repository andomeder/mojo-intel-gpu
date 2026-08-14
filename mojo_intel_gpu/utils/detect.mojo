"""Auto-detection utilities for Intel GPU hardware.

Provides functions to detect Intel GPUs and query their capabilities
without requiring manual Level Zero initialization.
"""
from std.memory import unsafe_memset
from std.memory.alloc import unsafe_alloc
from ..l0.types import (
    ZeDeviceType, DeviceProperties, ComputeProperties,
    ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES, ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES,
)
from ..l0.loader import LevelZeroLibrary


struct _GPUDeviceResult(Movable):
    """Internal result from GPU device search."""
    var driver_handle: Int
    var device_handle: Int
    var props_buffer: Pointer[Int8, MutUntrackedOrigin]

    def __init__(out self, driver: Int, device: Int, props: Pointer[Int8, MutUntrackedOrigin]):
        self.driver_handle = driver
        self.device_handle = device
        self.props_buffer = props

    def __deinit__(deinit self):
        self.props_buffer.unsafe_free()


struct IntelGPUDetector:
    """Detect and enumerate Intel GPU devices.

    Usage:
        detector = IntelGPUDetector()
        if detector.has_gpu():
            info = detector.get_gpu_info()
            print("Found:", info.name)
    """

    def __init__(out self):
        """Initialize the detector."""
        pass

    # =========================================================================
    # Private helpers — common driver/device enumeration
    # =========================================================================

    def _get_drivers(self, lib: LevelZeroLibrary) raises -> List[Int]:
        """Get all Level Zero driver handles."""
        var count_buf = unsafe_alloc[UInt32](1)
        count_buf.unsafe_offset(0)[] = 0
        var result = lib.init_drivers(Int(count_buf), Int(0))
        if result.is_error() or count_buf.unsafe_offset(0)[] == 0:
            count_buf.unsafe_free()
            raise Error("No Level Zero drivers found")
        var count = Int(count_buf.unsafe_offset(0)[])

        var drivers = unsafe_alloc[Int](count)
        result = lib.init_drivers(Int(count_buf), Int(drivers))
        count_buf.unsafe_free()
        if result.is_error():
            drivers.unsafe_free()
            raise Error("Failed to get drivers")

        var driver_list = List[Int]()
        for i in range(count):
            driver_list.append(drivers.unsafe_offset(i)[])
        drivers.unsafe_free()
        return driver_list^

    def _find_first_gpu_device(self, lib: LevelZeroLibrary,
                                drivers: List[Int]) raises -> _GPUDeviceResult:
        """Find first GPU device across all drivers.

        Returns _GPUDeviceResult with driver/device handles and props buffer.
        Raises if no GPU found.
        """
        for d in range(len(drivers)):
            var dev_count_buf = unsafe_alloc[UInt32](1)
            dev_count_buf.unsafe_offset(0)[] = 0
            var result = lib.device_get(drivers[d], Int(dev_count_buf), Int(0))
            if result.is_error() or dev_count_buf.unsafe_offset(0)[] == 0:
                dev_count_buf.unsafe_free()
                continue
            var device_count = Int(dev_count_buf.unsafe_offset(0)[])

            var devices = unsafe_alloc[Int](device_count)
            result = lib.device_get(drivers[d], Int(dev_count_buf), Int(devices))
            dev_count_buf.unsafe_free()
            if result.is_error():
                devices.unsafe_free()
                continue

            for i in range(device_count):
                var props = unsafe_alloc[Int8](400)
                unsafe_memset(props, 0, 400)
                props.unsafe_bitcast[UInt32]().unsafe_offset(0)[] = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES
                var prop_result = lib.device_get_properties(devices.unsafe_offset(i)[], Int(props))
                if prop_result.is_success():
                    var dtype = props.unsafe_bitcast[UInt32]().unsafe_offset(4)[]
                    if dtype == ZeDeviceType.GPU:
                        var driver = drivers[d]
                        var device = devices.unsafe_offset(i)[]
                        devices.unsafe_free()
                        return _GPUDeviceResult(driver, device, props)
                props.unsafe_free()

            devices.unsafe_free()

        raise Error("No Intel GPU found")

    def _count_gpu_devices(self, lib: LevelZeroLibrary,
                           drivers: List[Int]) raises -> Int:
        """Count GPU devices across all drivers."""
        var gpu_count = 0
        for d in range(len(drivers)):
            var dev_count_buf = unsafe_alloc[UInt32](1)
            dev_count_buf.unsafe_offset(0)[] = 0
            var result = lib.device_get(drivers[d], Int(dev_count_buf), Int(0))
            if result.is_error() or dev_count_buf.unsafe_offset(0)[] == 0:
                dev_count_buf.unsafe_free()
                continue
            var device_count = Int(dev_count_buf.unsafe_offset(0)[])

            var devices = unsafe_alloc[Int](device_count)
            result = lib.device_get(drivers[d], Int(dev_count_buf), Int(devices))
            dev_count_buf.unsafe_free()
            if result.is_error():
                devices.unsafe_free()
                continue

            for i in range(device_count):
                var props = unsafe_alloc[Int8](400)
                unsafe_memset(props, 0, 400)
                props.unsafe_bitcast[UInt32]().unsafe_offset(0)[] = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES
                var prop_result = lib.device_get_properties(devices.unsafe_offset(i)[], Int(props))
                if prop_result.is_success():
                    var dtype = props.unsafe_bitcast[UInt32]().unsafe_offset(4)[]
                    if dtype == ZeDeviceType.GPU:
                        gpu_count += 1
                props.unsafe_free()

            devices.unsafe_free()

        return gpu_count

    # =========================================================================
    # Public API
    # =========================================================================

    def has_gpu(self) -> Bool:
        """Check if any Intel GPU is available."""
        try:
            var lib = LevelZeroLibrary()
            var drivers = self._get_drivers(lib)
            var result = self._find_first_gpu_device(lib, drivers)
            return True
        except:
            return False

    def get_gpu_count(self) -> Int:
        """Get the number of Intel GPU devices available."""
        try:
            var lib = LevelZeroLibrary()
            var drivers = self._get_drivers(lib)
            return self._count_gpu_devices(lib, drivers)
        except:
            return 0

    def get_gpu_info(self) raises -> DeviceProperties:
        """Get information about the first Intel GPU found."""
        var lib = LevelZeroLibrary()
        var drivers = self._get_drivers(lib)
        var found = self._find_first_gpu_device(lib, drivers)
        var props = found.props_buffer

        var u32 = props.unsafe_bitcast[UInt32]()
        var u64 = props.unsafe_bitcast[UInt64]()

        var vendor_id = u32.unsafe_offset(5)[]
        var device_id = u32.unsafe_offset(6)[]
        var device_type = ZeDeviceType(u32.unsafe_offset(4)[])
        var clock = u32.unsafe_offset(9)[]
        var max_mem = u64.unsafe_offset(5)[]

        var name = ""
        var np = props.unsafe_offset(112)
        for j in range(256):
            var ch = np.unsafe_offset(j)[]
            if ch == 0:
                break
            name += chr(Int(ch))

        return DeviceProperties(vendor_id, device_id, device_type^,
                                clock, max_mem, name)

    def get_compute_info(self) raises -> ComputeProperties:
        """Get compute capabilities of the first Intel GPU found."""
        var lib = LevelZeroLibrary()
        var drivers = self._get_drivers(lib)
        var found = self._find_first_gpu_device(lib, drivers)
        var device = found.device_handle

        var cprops = unsafe_alloc[Int8](200)
        unsafe_memset(cprops, 0, 200)
        cprops.unsafe_bitcast[UInt32]().unsafe_offset(0)[] = ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES
        var result = lib.device_get_compute_properties(device, Int(cprops))
        if result.is_error():
            cprops.unsafe_free()
            raise Error("Failed to get compute properties")

        var u32 = cprops.unsafe_bitcast[UInt32]()
        var max_group_size = u32.unsafe_offset(4)[]
        var max_group_x = u32.unsafe_offset(8)[]
        var max_group_y = u32.unsafe_offset(9)[]
        var max_group_z = u32.unsafe_offset(10)[]
        var max_slm = u32.unsafe_offset(11)[]
        var num_subgroups = u32.unsafe_offset(12)[]

        var subgroups = List[UInt32]()
        for i in range(min(Int(num_subgroups), 8)):
            subgroups.append(u32.unsafe_offset(13 + i)[])

        cprops.unsafe_free()

        return ComputeProperties(max_group_size, max_group_x, max_group_y,
                                  max_group_z, max_slm, num_subgroups, subgroups^)


def detect_intel_gpu() -> Bool:
    """Quick check if an Intel GPU is available."""
    try:
        var detector = IntelGPUDetector()
        return detector.has_gpu()
    except:
        return False


def print_gpu_info() raises:
    """Print information about detected Intel GPU."""
    var detector = IntelGPUDetector()

    if not detector.has_gpu():
        print("No Intel GPU detected")
        return

    var info = detector.get_gpu_info()
    var compute = detector.get_compute_info()

    print("=== Intel GPU Detected ===")
    print("  Name:          ", info.name)
    print("  Vendor ID:     ", hex(Int(info.vendor_id)))
    print("  Device ID:     ", hex(Int(info.device_id)))
    print("  Type:          ", info.device_type)
    print("  Core Clock:    ", info.core_clock_mhz, " MHz")
    print("  Max Mem Alloc: ", Int(info.max_mem_alloc) // (1024 * 1024), " MB")
    print("  Max Group Size:", compute.max_group_size)
    print("  Max Group Cnt: (", compute.max_group_count_x, ",",
         compute.max_group_count_y, ",", compute.max_group_count_z, ")")
    print("  Max SLM:       ", compute.max_shared_local_memory, " bytes")
    print("  Sub-group Sizes:", compute.sub_group_sizes)
