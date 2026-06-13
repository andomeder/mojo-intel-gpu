// empty.cl — Empty kernel for launch latency measurement
//
// Returns immediately. Used to isolate the cost of kernel launch + sync
// from actual compute or memory work.

__kernel void empty() {
    // Intentional no-op. Measures pure L0 launch overhead.
}
