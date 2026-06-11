"""Kernel management — loading, argument binding, and dispatch.

Provides Kernel as a high-level wrapper for SPIR-V kernel operations.
Handles module loading, argument setup, and launch configuration.
"""
from std.memory import alloc
from std.pathlib import Path
from ..l0.types import ZeResult, ZeGroupCount
from ..l0.loader import LevelZeroLibrary


struct Kernel(Movable):
    """High-level kernel wrapper for Intel GPU.

    Manages SPIR-V module loading, kernel creation, argument binding,
    and dispatch configuration.

    Usage:
        kernel = Kernel(ctx.context(), ctx.device(), ctx.command_list(), "vector_add.spv", "vector_add")
        kernel.set_arg_pointer(0, device_ptr_a)
        kernel.set_arg_pointer(1, device_ptr_b)
        kernel.set_arg_pointer(2, device_ptr_c)
        kernel.set_arg_value(3, 4, UInt64(n_value))
        kernel.launch(ZeGroupCount(4, 1, 1))
        ctx.synchronize()
    """
    var _lib: LevelZeroLibrary
    var _context: Int
    var _device: Int
    var _cmdlist: Int
    var _module: Int
    var _kernel: Int
    var _spv_data: List[UInt8]
    var _kernel_name: String

    def __init__(out self, context: Int, device: Int,
                 cmdlist: Int, spirv_path: String, kernel_name: String) raises:
        """Load a SPIR-V kernel from file.

        Args:
            context: Level Zero context handle.
            device: Level Zero device handle.
            cmdlist: Command list handle.
            spirv_path: Path to SPIR-V binary file.
            kernel_name: Name of the kernel function in the SPIR-V module.

        Raises:
            Error: If loading fails.
        """
        self._lib = LevelZeroLibrary()
        self._context = context
        self._device = device
        self._cmdlist = cmdlist
        self._module = Int(0)
        self._kernel = Int(0)
        self._spv_data = List[UInt8]()
        self._kernel_name = kernel_name

        # Read SPIR-V binary
        var path = Path(spirv_path)
        self._spv_data = path.read_bytes()
        if len(self._spv_data) == 0:
            raise Error("Failed to read SPIR-V file: " + spirv_path)

        var module_out = alloc[Int](1)
        module_out[0] = 0
        var result = self._lib.module_create(
            context, device,
            Int(self._spv_data.unsafe_ptr()),
            UInt64(len(self._spv_data)),
            Int(module_out)
        )
        if result.is_error():
            module_out.free()
            raise Error("Module creation failed: " + String(result))
        self._module = module_out[0]
        module_out.free()


        var kernel_out = alloc[Int](1)
        kernel_out[0] = 0
        result = self._lib.kernel_create(
            self._module,
            Int(kernel_name.unsafe_ptr()),
            Int(kernel_out)
        )
        if result.is_error():
            kernel_out.free()
            raise Error("Kernel creation failed: " + String(result))
        self._kernel = kernel_out[0]
        kernel_out.free()

    def __del__(deinit self):
        """Clean up kernel and module resources."""
        if self._kernel != Int(0):
            _ = self._lib.kernel_destroy(self._kernel)
        if self._module != Int(0):
            _ = self._lib.module_destroy(self._module)

    # =========================================================================
    # Properties
    # =========================================================================

    def module(self) -> Int:
        """Get the module handle."""
        return self._module

    def kernel(self) -> Int:
        """Get the kernel handle."""
        return self._kernel

    def is_valid(self) -> Bool:
        """Check if the kernel is valid (loaded)."""
        return self._kernel != Int(0)

    # =========================================================================
    # Argument binding
    # =========================================================================

    def set_arg_pointer(mut self, index: UInt32, device_ptr: Int) raises:
        """Set a kernel argument to a device pointer.

        Args:
            index: Argument index (0-based).
            device_ptr: Device pointer value.

        Raises:
            Error: If setting the argument fails.
        """
        var result = self._lib.kernel_set_arg_value(
            self._kernel, index, UInt64(8), UInt64(device_ptr)
        )
        if result.is_error():
            raise Error("Failed to set kernel arg " + String(index) +
                       ": " + String(result))

    def set_arg_value(mut self, index: UInt32, size: UInt64, value: UInt64) raises:
        """Set a kernel argument to a scalar value.

        Args:
            index: Argument index (0-based).
            size: Size of the argument in bytes.
            value: Argument value (as UInt64).

        Raises:
            Error: If setting the argument fails.
        """
        var result = self._lib.kernel_set_arg_value(
            self._kernel, index, size, value
        )
        if result.is_error():
            raise Error("Failed to set kernel arg " + String(index) +
                       ": " + String(result))

    def set_group_size(mut self, x: UInt32, y: UInt32, z: UInt32) raises:
        """Set the kernel group size.

        Args:
            x: Group size in X dimension.
            y: Group size in Y dimension.
            z: Group size in Z dimension.

        Raises:
            Error: If setting the group size fails.
        """
        var result = self._lib.kernel_set_group_size(self._kernel, x, y, z)
        if result.is_error():
            raise Error("Failed to set group size: " + String(result))

    # =========================================================================
    # Launch
    # =========================================================================

    def launch(mut self, group_count: ZeGroupCount) raises:
        """Launch the kernel.

        Args:
            group_count: Number of work groups (x, y, z).

        Raises:
            Error: If launch fails.
        """
        var result = self._lib.command_list_append_launch_kernel(
            self._cmdlist, self._kernel, group_count,
            Int(0),  # signal_event
            Int(0),  # wait_events
            UInt32(0)  # num_wait_events
        )
        if result.is_error():
            raise Error("Kernel launch failed: " + String(result))

