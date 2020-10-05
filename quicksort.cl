__kernel void quicksorting(const int CHUNK, const int RANK, const int SZ,
                      const __global int* arr,
                      __global int* result) {
    
    // Thread identifiers
    const int globalposition = get_global_id(0);

    if (globalposition == 0) {
        for (int i = 0; i < CHUNK; i++) {
            result[i] = arr[i];
        }
    }
}
