#include <iostream>
#include<stdio.h>
#include<time.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <thread>
#include <mpi.h>
#include <chrono>

using namespace std;
using nano_s = chrono::nanoseconds;

int SZ = 4;
int **A, **B, **C;

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
    
    // print(A, SZ, SZ);
    // print(B, SZ, SZ);
    auto t1 = chrono::steady_clock::now();

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
    //calculate time to milliseconds
    auto t2 = chrono::steady_clock::now();
    auto d_nano = chrono::duration_cast<nano_s>(t2-t1).count();
    float a = d_nano;
    float d_milli = a/1000000L;
    cout << "time elapse: " << d_milli << "ms" << endl;
    // print(C, SZ, SZ);


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

void print_arr(int arr[], int size) {
    for (int i = 0; i < size; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
}