"""Memory management — typed device and host buffers.

Provides DeviceBuffer and HostBuffer as RAII wrappers for GPU memory.
Automatically frees memory when buffers go out of scope.
"""
from std.memory import alloc, memset
from ..l0.loader import LevelZeroLibrary
from std.sys import size_of


struct DeviceBuffer[T: AnyType](Movable):
    """RAII wrapper for device memory.

    Automatically frees device memory when the buffer goes out of scope.
    Creates its own LevelZeroLibrary internally (no shared ownership needed).

    Args:
        T: Element type (e.g., Float32, Int32).
    """
    var _ptr: Int
    var _count: Int
    var _lib: LevelZeroLibrary
    var _context: Int

    def __init__(out self, context: Int, device: Int, count: Int) raises:
        """Allocate device memory for `count` elements of type T.

        Creates its own LevelZeroLibrary handle internally.

        Args:
            context: Level Zero context handle.
            device: Level Zero device handle.
            count: Number of elements to allocate.

        Raises:
            Error: If allocation fails.
        """
        self._lib = LevelZeroLibrary()
        self._context = context
        self._count = count
        self._ptr = Int(0)

        var ptr_buf = alloc[Int](1)
        ptr_buf[0] = 0
        var size = UInt64(count * size_of[Self.T]())
        var result = self._lib.mem_alloc_device(
            context, size, 64, device,
            Int(ptr_buf)
        )
        self._ptr = ptr_buf[0]
        ptr_buf.free()
        if result.is_error():
            raise Error("Device buffer allocation failed: " + String(result))

    def __del__(deinit self):
        """Free device memory when buffer is destroyed."""
        if self._ptr != Int(0):
            _ = self._lib.mem_free(self._context, self._ptr)

    def ptr(self) -> Int:
        """Get the raw device pointer."""
        return self._ptr

    def count(self) -> Int:
        """Get the number of elements."""
        return self._count

    def size_bytes(self) -> UInt64:
        """Get the buffer size in bytes."""
        return UInt64(self._count * size_of[Self.T]())

    def is_valid(self) -> Bool:
        """Check if the buffer is valid (allocated)."""
        return self._ptr != Int(0)


struct HostBuffer[T: AnyType & ImplicitlyCopyable](Movable):
    """RAII wrapper for host-accessible memory.

    Automatically frees host memory when the buffer goes out of scope.
    Provides indexed access to elements via UnsafePointer[T].
    Creates its own LevelZeroLibrary internally (no shared ownership needed).

    Args:
        T: Element type (e.g., Float32, Int32).
    """
    var _ptr: Int
    var _count: Int
    var _lib: LevelZeroLibrary
    var _context: Int

    def __init__(out self, context: Int, count: Int) raises:
        """Allocate host memory for `count` elements of type T.

        Creates its own LevelZeroLibrary handle internally.

        Args:
            context: Level Zero context handle.
            count: Number of elements to allocate.

        Raises:
            Error: If allocation fails.
        """
        self._lib = LevelZeroLibrary()
        self._context = context
        self._count = count
        self._ptr = Int(0)

        var ptr_buf = alloc[Int](1)
        ptr_buf[0] = 0
        var size = UInt64(count * size_of[Self.T]())
        var result = self._lib.mem_alloc_host(
            context, size, 64,
            Int(ptr_buf)
        )
        self._ptr = ptr_buf[0]
        ptr_buf.free()
        if result.is_error():
            raise Error("Host buffer allocation failed: " + String(result))

    def __del__(deinit self):
        """Free host memory when buffer is destroyed."""
        if self._ptr != Int(0):
            _ = self._lib.mem_free(self._context, self._ptr)

    def ptr(self) -> Int:
        """Get the raw host pointer as Int for FFI."""
        return self._ptr

    def typed_ptr(self) -> UnsafePointer[Self.T, MutExternalOrigin]:
        """Get the host pointer as typed UnsafePointer[T] for element access."""
        return UnsafePointer[Self.T, MutExternalOrigin](unsafe_from_address=self._ptr)

    def count(self) -> Int:
        """Get the number of elements."""
        return self._count

    def size_bytes(self) -> UInt64:
        """Get the buffer size in bytes."""
        return UInt64(self._count * size_of[Self.T]())

    def is_valid(self) -> Bool:
        """Check if the buffer is valid (allocated)."""
        return self._ptr != Int(0)

    def __getitem__(self, idx: Int) -> Self.T:
        """Read an element at index."""
        return UnsafePointer[Self.T, MutExternalOrigin](unsafe_from_address=self._ptr)[idx]

    def __setitem__(mut self, idx: Int, value: Self.T):
        """Write an element at index."""
        UnsafePointer[Self.T, MutExternalOrigin](unsafe_from_address=self._ptr)[idx] = value
