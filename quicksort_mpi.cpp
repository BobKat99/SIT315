#include <iostream>
#include<stdio.h>
#include<time.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <thread>
#include <mpi.h>
#include <chrono>

// mpicxx quicksort_mpi.cpp -o quicksortmpi.o

using namespace std;
using nano_s = chrono::nanoseconds;

int SZ = 4;
int * arr;

void head(int num_processes);
void node(int process_rank, int num_processes);

void init(int * &arr, int size, bool initialise);
void print_arr(int * arr, int size);

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
    init(arr, SZ, true);
    // auto t1 = chrono::steady_clock::now();
    print_arr(arr, SZ);

    //my plan is to scatter A based on number of processes and broadcast B to all nodes
    int num_elements_to_scatter_or_gather = SZ / num_processes;


    MPI_Scatter(&arr[0], num_elements_to_scatter_or_gather ,  MPI_INT , &arr , 0, MPI_INT, 0 , MPI_COMM_WORLD);
    
    //sorting
    print_arr(arr, num_elements_to_scatter_or_gather);
   
    MPI_Gather(MPI_IN_PLACE, num_elements_to_scatter_or_gather , MPI_INT, &arr , num_elements_to_scatter_or_gather , MPI_INT, 0 , MPI_COMM_WORLD);
    
    // //calculate time to milliseconds
    // auto t2 = chrono::steady_clock::now();
    // auto d_nano = chrono::duration_cast<nano_s>(t2-t1).count();
    // float a = d_nano;
    // float d_milli = a/1000000L;
    // cout << "time elapse: " << d_milli << "ms" << endl;


}
void node(int process_rank, int num_processes)
{
    int num_elements_to_scatter_or_gather = SZ / num_processes;
    init(arr, num_elements_to_scatter_or_gather, false);

    //receive my rows of matrix A, and all B

    MPI_Scatter(NULL, num_elements_to_scatter_or_gather , MPI_INT , &arr[0], num_elements_to_scatter_or_gather, MPI_INT, 0 , MPI_COMM_WORLD);
    
    //sorting
    print_arr(arr, num_elements_to_scatter_or_gather);

    MPI_Gather(&arr[0], num_elements_to_scatter_or_gather , MPI_INT, NULL, num_elements_to_scatter_or_gather , MPI_INT, 0 , MPI_COMM_WORLD);

}

void init(int * &arr, int size, bool initialise) {
    arr = (int *) malloc(sizeof(int) * size);

    if(!initialise) return;

    for (int i = 0; i < SZ; i++) {
        arr[i] = rand() % 6;
    }
}

void print_arr(int * arr, int size) {
    for (int i = 0; i < size; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
}