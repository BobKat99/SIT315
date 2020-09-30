__kernel void add_matrices(const int M, const int N, const int K,
                      const __global int* A,
                      const __global int* B,
                      __global int* C) {
    
    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)
 
    C[globalCol*M + globalRow] = A[globalCol*M + globalRow] + B[globalCol*M + globalRow];
    
}


__kernel void multiply_matrices(const int M, const int N, const int K,
                      const __global int* A,
                      const __global int* B,
                      __global int* C) {
    
    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)
    
    printf("globalRow: %d, globalCol: %d, M: %d, N: %d, K: %d\n", globalRow, globalCol, M, N, K);

    int acc = 0.0f;
    for (int k=0; k<K; k++) {
        acc += B[k*M + globalRow] * A[globalCol*K + k];
        // printf("acc: %d, B:%d, A:%d\n", B[k*M + globalRow], A[globalCol*K + k]);
    }
    printf("done with result of %d at %d\n", acc, globalCol*M + globalRow);
    C[globalCol*M + globalRow] = acc;
}




/*
    //printf("(%d,%d)\n ", globalRow, globalCol);
    // Compute a single element (loop over K)
    int acc = 0.0f;
    for (int k=0; k<K; k++) {
        acc += B[k*M + globalRow] * A[globalCol*K + k];
        printf("acc: %d, B:%d, A:%d)\n ", B[k*M + globalRow], A[globalCol*K + k];
    }
*/