"""Memory management — typed device and host buffers.

Provides DeviceBuffer and HostBuffer as RAII wrappers for GPU memory.
Automatically frees memory when buffers go out of scope.
"""

from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.sys import size_of
from ..l0.loader import LevelZeroLibrary


struct DeviceBuffer[T: AnyType](Movable):
    """RAII wrapper for device memory.

    Automatically frees device memory when the buffer goes out of scope.
    Borrows the Level Zero library and context handle from the owning
    `IntelGPUContext`, which must outlive this buffer.

    Args:
        T: Element type (e.g., Float32, Int32).
    """

    var _ptr: Int
    var _count: Int
    var _lib: Pointer[LevelZeroLibrary, ImmUntrackedOrigin]
    var _context: Int

    def __init__(out self, lib: Pointer[LevelZeroLibrary, ImmUntrackedOrigin],
                 context: Int, device: Int, count: Int) raises:
        """Allocate device memory for `count` elements of type T.

        Args:
            lib: Borrowed Level Zero library from the owning context.
            context: Level Zero context handle.
            device: Level Zero device handle.
            count: Number of elements to allocate.

        Raises:
            Error: If allocation fails.
        """
        if count <= 0:
            raise Error("Device buffer count must be positive, got " + String(count))
        self._lib = lib
        self._context = context
        self._count = count
        self._ptr = Int(0)

        var ptr_buf = unsafe_alloc[Int](1)
        ptr_buf.unsafe_offset(0)[] = 0
        var size = UInt64(count) * UInt64(size_of[Self.T]())
        var result = self._lib[].mem_alloc_device(
            context, size, 64, device,
            Int(ptr_buf)
        )
        self._ptr = ptr_buf.unsafe_offset(0)[]
        ptr_buf.unsafe_free()
        if result.is_error():
            raise Error("Device buffer allocation failed: " + String(result))

    def __deinit__(deinit self):
        """Free device memory when buffer is destroyed."""
        if self._ptr != Int(0):
            _ = self._lib[].mem_free(self._context, self._ptr)

    def ptr(self) -> Int:
        """Get the raw device pointer."""
        return self._ptr

    def count(self) -> Int:
        """Get the number of elements."""
        return self._count

    def size_bytes(self) -> UInt64:
        """Get the buffer size in bytes."""
        return UInt64(self._count) * UInt64(size_of[Self.T]())

    def is_valid(self) -> Bool:
        """Check if the buffer is valid (allocated)."""
        return self._ptr != Int(0)


struct HostBuffer[T: ImplicitlyCopyable & Deinitable](Movable):
    """RAII wrapper for host-accessible memory.

    Automatically frees host memory when the buffer goes out of scope.
    Provides indexed access to elements via Pointer[T].
    Borrows the Level Zero library and context handle from the owning
    `IntelGPUContext`, which must outlive this buffer.

    Args:
        T: Element type (e.g., Float32, Int32).
    """

    var _ptr: Int
    var _count: Int
    var _lib: Pointer[LevelZeroLibrary, ImmUntrackedOrigin]
    var _context: Int

    def __init__(out self, lib: Pointer[LevelZeroLibrary, ImmUntrackedOrigin],
                 context: Int, count: Int) raises:
        """Allocate host memory for `count` elements of type T.

        Args:
            lib: Borrowed Level Zero library from the owning context.
            context: Level Zero context handle.
            count: Number of elements to allocate.

        Raises:
            Error: If allocation fails.
        """
        if count <= 0:
            raise Error("Host buffer count must be positive, got " + String(count))
        self._lib = lib
        self._context = context
        self._count = count
        self._ptr = Int(0)

        var ptr_buf = unsafe_alloc[Int](1)
        ptr_buf.unsafe_offset(0)[] = 0
        var size = UInt64(count) * UInt64(size_of[Self.T]())
        var result = self._lib[].mem_alloc_host(
            context, size, 64,
            Int(ptr_buf)
        )
        self._ptr = ptr_buf.unsafe_offset(0)[]
        ptr_buf.unsafe_free()
        if result.is_error():
            raise Error("Host buffer allocation failed: " + String(result))

    def __deinit__(deinit self):
        """Free host memory when buffer is destroyed."""
        if self._ptr != Int(0):
            _ = self._lib[].mem_free(self._context, self._ptr)

    def ptr(self) -> Int:
        """Get the raw host pointer as Int for FFI."""
        return self._ptr

    def typed_ptr(self) -> Pointer[Self.T, MutUntrackedOrigin]:
        """Get the host pointer as typed Pointer[T] for element access.

        The returned pointer does not track the buffer's lifetime. It is valid
        only while `self` is alive and must not outlive the buffer.
        """
        return Pointer[Self.T, MutUntrackedOrigin](unsafe_from_address=self._ptr)

    def count(self) -> Int:
        """Get the number of elements."""
        return self._count

    def size_bytes(self) -> UInt64:
        """Get the buffer size in bytes."""
        return UInt64(self._count) * UInt64(size_of[Self.T]())

    def is_valid(self) -> Bool:
        """Check if the buffer is valid (allocated)."""
        return self._ptr != Int(0)

    def __getitem__(self, idx: Int) raises -> Self.T:
        """Read an element at index (bounds-checked)."""
        if idx < 0 or idx >= self._count:
            raise Error("Host buffer index out of range: " + String(idx))
        var p = Pointer[Self.T, MutUntrackedOrigin](unsafe_from_address=self._ptr)
        return p.unsafe_offset(idx)[]

    def __setitem__(mut self, idx: Int, value: Self.T) raises:
        """Write an element at index (bounds-checked)."""
        if idx < 0 or idx >= self._count:
            raise Error("Host buffer index out of range: " + String(idx))
        var p = Pointer[Self.T, MutUntrackedOrigin](unsafe_from_address=self._ptr)
        p.unsafe_offset(idx)[] = value
