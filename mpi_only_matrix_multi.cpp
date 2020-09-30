#include <iostream>
#include<stdio.h>
#include<time.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <thread>
#include <mpi.h>

using namespace std;

// mpicxx ./mpi_only_matrix_multi.cpp -o mpionly.o
// mpirun -np 4 --hostfile ./cluster ./mpionly.o 1000

#define BILLION  1000000000L;
int NUM_THREADS = 4;

int SZ = 8;
int **A, **B, **C;

void init(int** &matrix, int rows, int cols, bool initialise);
void print( int** matrix, int rows, int cols);
void* add(void* block_id);
void* multiply(void* args);
void head(int num_processes);
void node(int process_rank, int num_processes);

int main(int argc, char** argv) {
    if(argc > 1) SZ = atoi(argv[1]);

    MPI_Init(NULL, NULL);

    int num_processes;
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    int process_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    if(process_rank == 0) //head
        head(num_processes);
    else
        node(process_rank, num_processes);
    
    MPI_Finalize();
}

void head(int num_processes)
{
    init(A, SZ, SZ, true), init(B, SZ, SZ, true), init(C, SZ, SZ, false);
    
    print(A, SZ, SZ);
    print(B, SZ, SZ);

    //my plan is to scatter A based on number of processes and broadcast B to all nodes
    int num_rows_per_process_from_A = SZ / num_processes;
    int num_elements_to_bcast = (SZ * SZ);
    int num_elements_to_scatter_or_gather = (SZ * SZ) / num_processes;


    MPI_Scatter(&A[0][0], num_elements_to_scatter_or_gather ,  MPI_INT , &A , 0, MPI_INT, 0 , MPI_COMM_WORLD);
    MPI_Bcast(&B[0][0], num_elements_to_bcast , MPI_INT , 0 , MPI_COMM_WORLD);

    
    //calculate the assigned rows of matrix C
    for(int i = 0; i < num_rows_per_process_from_A ; i++) {
        for(int j = 0; j < SZ; j++) {
            int mul = 0;
            for (int z = 0; z < SZ; z++) {
                mul += A[i][z] * B[z][j];
            }
            C[i][j] = mul;
        }
    }
    MPI_Gather(MPI_IN_PLACE, num_elements_to_scatter_or_gather , MPI_INT, &C[0][0] , num_elements_to_scatter_or_gather , MPI_INT, 0 , MPI_COMM_WORLD);
    //send the results back to the head node for merging and printing

    print(C, SZ, SZ);


}
void node(int process_rank, int num_processes)
{
    int num_rows_per_process_from_A = SZ / num_processes;
    int num_elements_to_bcast = (SZ * SZ);
    int num_elements_to_scatter_or_gather = (SZ * SZ) / num_processes;

    //receive my rows of matrix A, and all B
    init(A, num_rows_per_process_from_A , SZ, true), init(B, SZ, SZ, false), init(C, num_rows_per_process_from_A, SZ, false);

    MPI_Scatter(NULL, num_elements_to_scatter_or_gather , MPI_INT , &A[0][0], num_elements_to_scatter_or_gather, MPI_INT, 0 , MPI_COMM_WORLD);
    MPI_Bcast(&B[0][0], num_elements_to_bcast , MPI_INT , 0 , MPI_COMM_WORLD);
    
    
    //calculate the assigned rows of matrix C
    for(int i = 0; i < num_rows_per_process_from_A ; i++) {
        for(int j = 0; j < SZ; j++) {
            int mul = 0;
            for (int z = 0; z < SZ; z++) {
                mul += A[i][z] * B[z][j];
            }
            C[i][j] = mul;
        }
    }

    MPI_Gather(&C[0][0], num_elements_to_scatter_or_gather , MPI_INT, NULL, num_elements_to_scatter_or_gather , MPI_INT, 0 , MPI_COMM_WORLD);

}



void init(int** &A, int rows, int cols, bool initialise) {
    A = (int **) malloc(sizeof(int*) * rows * cols);  // number of rows * size of int* address in the memory
    int* tmp = (int *) malloc(sizeof(int) * cols * rows); 

    for(int i = 0 ; i < SZ ; i++) {
        A[i] = &tmp[i * cols];
    }
  

    if(!initialise) return;

    for(long i = 0 ; i < rows; i++) {
        for(long j = 0 ; j < cols; j++) {
            A[i][j] = rand() % 100; // any number less than 100
        }
    }
}

void print( int** A, int rows, int cols) {
  for(long i = 0 ; i < rows; i++) { //rows
        for(long j = 0 ; j < cols; j++) {  //cols
            printf("%d ",  A[i][j]); // print the cell value
        }
        printf("\n"); //at the end of the row, print a new line
    }
    printf("----------------------------\n");
}

void* add(void* block_id) {

    long block_sz = (SZ + 1) / NUM_THREADS; //how many rows in each thread to process
    long start =  (long) block_id * block_sz;  //where should I start? row 0 , row 500, ???
    long end   = start + block_sz; //where should I end
    if ( end > SZ ) end = SZ;
    printf("block id %ld, block size %ld, start %ld, end %ld\n" , (long)block_id, block_sz , start, end);
    
    for(long i = start ; i < end ; i++) {  //rows
        for(long j = 0 ; j < SZ; j++) { //cols
            C[i][j] = A[i][j] + B[i][j]; // c[0][0] = A[0][0] + B[0][0]
        }
    }
}

void* multiply(void* args) {

}