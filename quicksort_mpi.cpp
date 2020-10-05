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
// mpirun -np 4 --hostfile ./cluster ./quicksortmpi.o 100

using namespace std;
using nano_s = chrono::nanoseconds;

int SZ = 4;
int * arr, *resultA, * dummyA;

void head(int num_processes);
void node(int process_rank, int num_processes);

void init(int * &arr, int size, bool initialise);
void quickSort(int * arr, int start, int end);
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
    init(resultA, SZ, false);

    int dumdum[SZ];
    auto t1 = chrono::steady_clock::now();
    // cout << "array in the beginning is: ";
    // print_arr(arr, SZ);

    int num_elements_to_scatter_or_gather = SZ / num_processes;
    init(dummyA, num_elements_to_scatter_or_gather, false);

    MPI_Scatter(&arr[0], num_elements_to_scatter_or_gather ,  MPI_INT , &arr , 0, MPI_INT, 0 , MPI_COMM_WORLD);
    
    //sorting
    quickSort(arr, 0, num_elements_to_scatter_or_gather - 1);

    for (int j = 0; j < num_elements_to_scatter_or_gather; j++) {
        dumdum[j] = arr[j];
    }

    int k = num_elements_to_scatter_or_gather; // the number element got sorted in the result
    for (int i = 1; i < num_processes; i++) {
        // MPI_Bcast(&dummyA[0], num_elements_to_scatter_or_gather, MPI_INT , i , MPI_COMM_WORLD);
        MPI_Recv(&dummyA[0], num_elements_to_scatter_or_gather, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int a = 0;
        int b = 0; 
        int c = 0;
        while(a < k && b < num_elements_to_scatter_or_gather) {
            if(dumdum[a] <= dummyA[b]) {
                resultA[c] = dumdum[a];
                a++;
            } else {
                resultA[c] = dummyA[b];
                b++;
            }
            c++;
        }

        while(a < k) {
            resultA[c] = dumdum[a];
            a++;
            c++;
        }

        while(b < num_elements_to_scatter_or_gather) {
            resultA[c] = dummyA[b];
            // printf("a: %d, b: %d, c: %d, result: %d \n", a, b, c, resultA[c]);
            b++;
            c++;
        }
        k += num_elements_to_scatter_or_gather;
        for (int m = 0; m < k; m++) {
            dumdum[m] = resultA[m];
        }
    }
    
    // MPI_Gather(MPI_IN_PLACE, num_elements_to_scatter_or_gather , MPI_INT, &arr[0] , num_elements_to_scatter_or_gather , MPI_INT, 0 , MPI_COMM_WORLD);
    // cout << "the final result after sorting is: ";
    // print_arr(resultA, SZ);

    // //calculate time to milliseconds
    auto t2 = chrono::steady_clock::now();
    auto d_nano = chrono::duration_cast<nano_s>(t2-t1).count();
    float a = d_nano;
    float d_milli = a/1000000L;
    cout << "time elapse: " << d_milli << "ms" << endl;


}
void node(int process_rank, int num_processes)
{
    int num_elements_to_scatter_or_gather = SZ / num_processes;
    init(arr, num_elements_to_scatter_or_gather, false);
    init(dummyA, num_elements_to_scatter_or_gather, false);

    MPI_Scatter(NULL, num_elements_to_scatter_or_gather , MPI_INT , &arr[0], num_elements_to_scatter_or_gather, MPI_INT, 0 , MPI_COMM_WORLD);

    //sorting
    quickSort(arr, 0, num_elements_to_scatter_or_gather - 1);
    for (int i = 0; i < num_elements_to_scatter_or_gather; i++) {
        dummyA[i] = arr[i];
    }

    // print_arr(arr, num_elements_to_scatter_or_gather);
    // MPI_Bcast(&dummyA[0], num_elements_to_scatter_or_gather, MPI_INT , process_rank , MPI_COMM_WORLD);

    MPI_Send(&dummyA[0], num_elements_to_scatter_or_gather, MPI_INT, 0, 0, MPI_COMM_WORLD);
    // MPI_Gather(&arr[0], num_elements_to_scatter_or_gather , MPI_INT, NULL, num_elements_to_scatter_or_gather , MPI_INT, 0 , MPI_COMM_WORLD);
}

void init(int * &arr, int size, bool initialise) {
    arr = (int *) malloc(sizeof(int) * size);

    if(!initialise) return;

    for (int i = 0; i < SZ; i++) {
        arr[i] = rand() % 100;
    }
}

void swap(int * arr, int i1, int i2) {
    int value = arr[i1];
    arr[i1] = arr[i2];
    arr[i2] = value;
}

void quickSort(int * arr, int start, int end) {
    if (start < end) {
        int pivot = end;
        for (int i = start; i < pivot; i++) {
            if (arr[i] >= arr[pivot]) {
                swap(arr, i, pivot);
                swap(arr, i, (pivot - 1));
                pivot--;
                i--;
            }
        }
        quickSort(arr, start, (pivot-1));
        quickSort(arr, (pivot + 1), end);
    }
}

void print_arr(int * arr, int size) {
    for (int i = 0; i < size; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
}