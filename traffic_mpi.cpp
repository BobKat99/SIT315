#include <iostream>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <thread>
#include <mpi.h>
#include <chrono>
#include <string>
#include <sstream>
#include <fstream>

using namespace std;
using nano_s = chrono::nanoseconds;

#define NUMBER_HOUR 4
#define NUMBER_SIGN 6
#define MINUTE_MEA 5
#define MINUTES_IN_HOUR 60

int measures = NUMBER_HOUR * NUMBER_SIGN * (MINUTES_IN_HOUR / MINUTE_MEA);

// mpicxx ./mpi_only_matrix_multi.cpp -o mpionly.o
// mpirun -np 4 --hostfile ./cluster ./mpionly.o 1000

int NumProd = 1;
int NumCon = 1;
int **A;

void init(int** &matrix, int rows, int cols, bool initialise);
void print( int** matrix, int rows, int cols);
void head(int num_processes);
void node(int process_rank, int num_processes);

int main(int argc, char** argv) {
    if(argc > 1) {
        NumProd = atoi(argv[1]);
        NumCon = atoi(argv[2]);
    }
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
    ifstream MyReadFile("data.csv");

    int num_data_local = measures / num_processes;

    string work_str;

    for (int i = 0; i < num_processes; i++){
        string str = "";
        for(int j = 0; j < num_data_local; j++) {
            string inside_str;
            getline(MyReadFile, inside_str);
            str += inside_str + "\n";
        }
        if(i == 0){
            work_str = str;
        } else {
            MPI_Send(str.c_str(), str.length(), MPI_CHAR, i, 0, MPI_COMM_WORLD);
        }
    }

    // ifstream MyReadFile2("data.csv");
    // MyReadFile2.read(arrStr, sizeChar);
    
    // int num_elements_to_scatter_or_gather = str.length() / num_processes;

    // MPI_Scatter(&arrStr[0], num_elements_to_scatter_or_gather ,  MPI_CHAR , &arrStr , 0, MPI_INT, 0 , MPI_COMM_WORLD);

    // for (int i =0; i < num_elements_to_scatter_or_gather; i++) {
    //     cout << arrStr[i];
    // }
    // MPI_Gather(MPI_IN_PLACE, num_elements_to_scatter_or_gather , MPI_INT, &C[0][0] , num_elements_to_scatter_or_gather , MPI_INT, 0 , MPI_COMM_WORLD);
}
void node(int process_rank, int num_processes)
{
    // note using pragma for for the sum of prod and consu, then print inside func to check

    int num_data_local = measures / num_processes;
    int sizeChar;
    int maxCharInOneLine = 50;
    sizeChar = maxCharInOneLine * num_data_local;

    char * strA;
    strA = new char [sizeChar];

    MPI_Recv(&strA[0], sizeChar, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    string work_str(strA);

    // cout << work_str;

    // MPI_Gather(&C[0][0], num_elements_to_scatter_or_gather , MPI_INT, NULL, num_elements_to_scatter_or_gather , MPI_INT, 0 , MPI_COMM_WORLD);
}



void init(int** &A, int rows, int cols, bool initialise) {
    A = (int **) malloc(sizeof(int*) * rows * cols);  // number of rows * size of int* address in the memory
    int* tmp = (int *) malloc(sizeof(int) * cols * rows); 

    for(int i = 0 ; i < 100 ; i++) {
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