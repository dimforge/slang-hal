[shader("compute")]
[numthreads(64, 1, 1)]
func add_assign(
    uint3 invocation_id: SV_DispatchThreadID,
    RWStructuredBuffer<float> a,
    StructuredBuffer<float> b,
) {
    let thread_id = invocation_id.x;
    if (thread_id < a.getCount()) {
        a[thread_id] += b[thread_id];
    }
}