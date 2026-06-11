"""Auto-detection utilities for Intel GPU hardware.

Provides functions to detect Intel GPUs and query their capabilities
without requiring manual Level Zero initialization.
"""
from std.memory import alloc, memset
from ..l0.types import (
    ZeDeviceType, DeviceProperties, ComputeProperties,
    ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES, ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES,
)
from ..l0.loader import LevelZeroLibrary


struct _GPUDeviceResult(Movable):
    """Internal result from GPU device search."""
    var driver_handle: Int
    var device_handle: Int
    var props_buffer: Int

    def __init__(out self, driver: Int, device: Int, props: Int):
        self.driver_handle = driver
        self.device_handle = device
        self.props_buffer = props

    def __del__(deinit self):
        if self.props_buffer != Int(0):
            UnsafePointer[Int8, MutExternalOrigin](unsafe_from_address=self.props_buffer).free()


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
        var count_buf = alloc[UInt32](1)
        count_buf[0] = 0
        var result = lib.init_drivers(Int(count_buf), Int(0))
        if result.is_error() or count_buf[0] == 0:
            count_buf.free()
            raise Error("No Level Zero drivers found")
        var count = Int(count_buf[0])

        var drivers = alloc[Int](count)
        result = lib.init_drivers(Int(count_buf), Int(drivers))
        count_buf.free()
        if result.is_error():
            drivers.free()
            raise Error("Failed to get drivers")

        var driver_list = List[Int]()
        for i in range(count):
            driver_list.append(drivers[i])
        drivers.free()
        return driver_list^

    def _find_first_gpu_device(self, lib: LevelZeroLibrary,
                                drivers: List[Int]) raises -> _GPUDeviceResult:
        """Find first GPU device across all drivers.

        Returns _GPUDeviceResult with driver/device handles and props buffer.
        Raises if no GPU found.
        """
        for d in range(len(drivers)):
            var dev_count_buf = alloc[UInt32](1)
            dev_count_buf[0] = 0
            var result = lib.device_get(drivers[d], Int(dev_count_buf), Int(0))
            if result.is_error() or dev_count_buf[0] == 0:
                dev_count_buf.free()
                continue
            var device_count = Int(dev_count_buf[0])

            var devices = alloc[Int](device_count)
            result = lib.device_get(drivers[d], Int(dev_count_buf), Int(devices))
            dev_count_buf.free()
            if result.is_error():
                devices.free()
                continue

            for i in range(device_count):
                var props = alloc[Int8](400)
                memset(props, 0, 400)
                props.bitcast[UInt32]()[0] = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES
                var prop_result = lib.device_get_properties(devices[i], Int(props))
                if prop_result.is_success():
                    var dtype = props.bitcast[UInt32]()[4]
                    if dtype == ZeDeviceType.GPU:
                        var driver = drivers[d]
                        var device = devices[i]
                        devices.free()
                        return _GPUDeviceResult(driver, device, Int(props))
                props.free()

            devices.free()

        raise Error("No Intel GPU found")

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

            var gpu_count = 0
            for d in range(len(drivers)):
                var dev_count_buf = alloc[UInt32](1)
                dev_count_buf[0] = 0
                var result = lib.device_get(drivers[d], Int(dev_count_buf), Int(0))
                if result.is_error() or dev_count_buf[0] == 0:
                    dev_count_buf.free()
                    continue
                var device_count = Int(dev_count_buf[0])

                var devices = alloc[Int](device_count)
                result = lib.device_get(drivers[d], Int(dev_count_buf), Int(devices))
                dev_count_buf.free()
                if result.is_error():
                    devices.free()
                    continue

                for i in range(device_count):
                    var props = alloc[Int8](400)
                    memset(props, 0, 400)
                    props.bitcast[UInt32]()[0] = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES
                    var prop_result = lib.device_get_properties(devices[i], Int(props))
                    if prop_result.is_success():
                        var dtype = props.bitcast[UInt32]()[4]
                        if dtype == ZeDeviceType.GPU:
                            gpu_count += 1
                    props.free()

                devices.free()

            return gpu_count
        except:
            return 0

    def get_gpu_info(self) raises -> DeviceProperties:
        """Get information about the first Intel GPU found."""
        var lib = LevelZeroLibrary()
        var drivers = self._get_drivers(lib)
        var found = self._find_first_gpu_device(lib, drivers)
        var props = found.props_buffer

        var u32 = UnsafePointer[UInt32, MutExternalOrigin](unsafe_from_address=props)
        var u64 = UnsafePointer[UInt64, MutExternalOrigin](unsafe_from_address=props)

        var vendor_id = u32[5]
        var device_id = u32[6]
        var device_type = ZeDeviceType(u32[4])
        var clock = u32[9]
        var max_mem = u64[5]

        var name = ""
        var np = UnsafePointer[Int8, MutExternalOrigin](unsafe_from_address=props + 112)
        for j in range(256):
            var ch = np[j]
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

        var cprops = alloc[Int8](200)
        memset(cprops, 0, 200)
        cprops.bitcast[UInt32]()[0] = ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES
        var result = lib.device_get_compute_properties(device, Int(cprops))
        if result.is_error():
            cprops.free()
            raise Error("Failed to get compute properties")

        var u32 = cprops.bitcast[UInt32]()
        var max_group_size = u32[4]
        var max_group_x = u32[8]
        var max_group_y = u32[9]
        var max_group_z = u32[10]
        var max_slm = u32[11]
        var num_subgroups = u32[12]

        var subgroups = List[UInt32]()
        for i in range(min(Int(num_subgroups), 8)):
            subgroups.append(u32[13 + i])

        cprops.free()

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
