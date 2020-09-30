__kernel void add_matrices(const int M, const int N, const int K,
                      const __global int* A,
                      const __global int* B,
                      __global int* C) {
    
    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)
 
    C[globalCol*M + globalRow] = A[globalCol*M + globalRow] + B[globalCol*M + globalRow];
    
}


__kernel void multiply_matrices(const int NROW, const int RANK, const int SZ,
                      const __global int* A,
                      const __global int* B,
                      __global int* C) {
    
    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)

    // printf("%d RANK: %d\n", A[12], RANK);

    
    if((globalRow >= (NROW*RANK)) && (globalRow < (NROW*(RANK + 1))) ){
        // printf("globalRow: %d, globalCol: %d, NROW: %d, RANK: %d, SZ: %d\n", globalRow, globalCol, NROW, RANK, SZ);
        int acc = 0.0f;
        for (int k=0; k<SZ; k++) {
            // printf("acc: %d, A at %d is %d, B at %d is %d\n", acc, globalCol*SZ + k, A[globalCol*SZ + k], k*SZ + globalRow, B[k*SZ + globalRow]);
            acc += A[globalCol*SZ + k]*B[k*SZ + globalRow];
        }
        // printf("done with result of %d \n", acc);
        C[globalCol*SZ + globalRow] = acc;
    }
}




/*
    //printf("(%d,%d)\n ", globalRow, globalCol);
    // Compute a single element (loop over K)
    int acc = 0.0f;
    for (int k=0; k<K; k++) {
        acc += B[k*M + globalRow] * A[globalCol*K + k];
     //   printf("(%d,%d), values = (%d, %d)\n ", k*M + globalRow, globalCol*K + k, A[k*M + globalRow] , B[globalCol*K + k]);
    }

        // for (long i = 0; i < NROW; i++)
    // {
    //     for (long j = 0; j < SZ; j++)
    //     {
    //         printf("%d ", A[i*SZ+j]);
    //     }
    //     printf("\n");
    // }
    // printf("globalRow: %d, globalCol: %d, NROW: %d, RANK: %d, SZ: %d\n", globalRow, globalCol, NROW, RANK, SZ);
    // printf("--------------------------------------\n");
*/