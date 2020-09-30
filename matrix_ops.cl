__kernel void multiply_matrices(const int NROW, const int RANK, const int SZ,
                      const __global int* A,
                      const __global int* B,
                      __global int* C) {
    
    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C
    const int globalCol = get_global_id(1); // Col ID of C

    for(int i = 0; i < NROW ; i++) {
        for(int j = 0; j < SZ; j++) {
            int mul = 0.0f;
            for (int z = 0; z < SZ; z++) {
                mul += A[i*SZ + z] * B[z*SZ + j];
            }
            C[i*SZ+j] = mul;
        }
    }
}