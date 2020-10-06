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
#include <omp.h>

using namespace std;
using nano_s = chrono::nanoseconds;

// mpicxx ./testTraffic.cpp -o traffic.o -fopenmp
// mpirun -np 4 --hostfile ./cluster ./traffic.o 2 2

// the data in general
struct trafficData {
    int hours, minute, id, cars;
};

// the data after processing to create the sum
struct resultSum {
    int id, sum;
};

// the below is for creating a type of queue corresponding witht the customized dataType
// define default capacity of the queue
#define SIZE 10
// Class for queue
class queue
{
	trafficData *arr;   	// array to store queue elements
	int capacity;   // maximum capacity of the queue
	int front;  	// front points to front element in the queue (if any)
	int rear;   	// rear points to last element in the queue
	int count;  	// current size of the queue

public:
	queue(int size = SIZE);	 // constructor
	~queue();				   // destructor

	void dequeue();
	void enqueue(trafficData x);
	trafficData peek();
	int size();
	bool isEmpty();
	bool isFull();
};
// Constructor to initialize queue
queue::queue(int size)
{
	arr = new trafficData[size];
	capacity = size;
	front = 0;
	rear = -1;
	count = 0;
}
// Destructor to free memory allocated to the queue
queue::~queue()
{
	delete[] arr;
}
// Utility function to remove front element from the queue
void queue::dequeue()
{
	// check for queue underflow
	if (isEmpty())
	{
		cout << "UnderFlow\nProgram Terminated\n";
		exit(EXIT_FAILURE);
	}
	front = (front + 1) % capacity;
	count--;
}
// Utility function to add an item to the queue
void queue::enqueue(trafficData item)
{
	// check for queue overflow
	if (isFull())
	{
		cout << "OverFlow\nProgram Terminated\n";
		exit(EXIT_FAILURE);
	}
	rear = (rear + 1) % capacity;
	arr[rear] = item;
	count++;
}
// Utility function to return front element in the queue
trafficData queue::peek()
{
	if (isEmpty())
	{
		cout << "UnderFlow\nProgram Terminated\n";
		exit(EXIT_FAILURE);
	}
	return arr[front];
}
// Utility function to return the size of the queue
int queue::size()
{
	return count;
}
// Utility function to check if the queue is empty or not
bool queue::isEmpty()
{
	return (size() == 0);
}
// Utility function to check if the queue is full or not
bool queue::isFull()
{
	return (size() == capacity);
}

#define NUMBER_HOUR 24
#define NUMBER_SIGN 24
#define MINUTE_MEA 5
#define MINUTES_IN_HOUR 60

int measures = NUMBER_HOUR * NUMBER_SIGN * (MINUTES_IN_HOUR / MINUTE_MEA);

// mpicxx ./mpi_only_matrix_multi.cpp -o mpionly.o
// mpirun -np 4 --hostfile ./cluster ./mpionly.o 1000

int NumProd = 1;
int NumCon = 1;
int ** resultMatrix;

omp_lock_t lockProd;
omp_lock_t lockCon;

void init(int ** &matrix, int rows, int cols);
void print( int ** matrix, int rows, int cols);
void producer(string myText, queue &que);
bool qAccessCon(queue &que, int ** &matrix, int row_per);

void printMatrix (resultSum matrix[NUMBER_HOUR][NUMBER_SIGN]);
void sorting(resultSum matrix[NUMBER_HOUR][NUMBER_SIGN]);
void printResult (resultSum matrix[NUMBER_HOUR][NUMBER_SIGN], int hour, int numberMax);

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
    // extract data and spread
    auto t1 = chrono::steady_clock::now();
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
    
    MyReadFile.close();

    // begin produce and consume 
    queue que(8); // the queue is only 8 for litmit buffer
    int hours_inside = NUMBER_HOUR / num_processes;
    init(resultMatrix, NUMBER_HOUR, NUMBER_SIGN);
    int num_elements_to_scatter_or_gather = hours_inside*NUMBER_SIGN;

    stringstream str_strm(work_str);

    int total_thread = NumCon + NumProd;

    // set up multi threading
    omp_init_lock(&lockProd);
    omp_init_lock(&lockCon);
    omp_set_num_threads(total_thread);

    int count_prod = 0;
    int count_con = 0; 

    // multi threading with each thread is a consumer or producer
    #pragma omp parallel for 
    for(int i = 0; i < total_thread ; i++) {
        // cout << "[processing at thread " << i << "]" << endl;
        if (i < NumProd) {
            while (count_prod < num_data_local) {
                omp_set_lock(&lockProd);
                string record;
                getline(str_strm, record);
                producer(record, que);
                count_prod++;
                omp_unset_lock(&lockProd);
            }
        } else {
            while(count_con < num_data_local) {
                omp_set_lock(&lockCon);
                bool check = false;
                while(!check) {
                    check = qAccessCon(que, resultMatrix, hours_inside);
                };
                count_con++;
                omp_unset_lock(&lockCon);
            }
        }
    }

    omp_destroy_lock(&lockProd);
    omp_destroy_lock(&lockCon);

    // sequential code
    // for (int i = 0; i < num_data_local; i++) {
    //     string record;
    //     getline(str_strm, record);
    //     producer(record, que);
    // }

    // bool check = true;
    // while(check) {
    //     check = qAccessCon(que, resultMatrix, hours_inside);
    // };

    MPI_Gather(MPI_IN_PLACE, num_elements_to_scatter_or_gather , MPI_INT, &resultMatrix[0][0] , num_elements_to_scatter_or_gather , MPI_INT, 0 , MPI_COMM_WORLD);
    auto t2 = chrono::steady_clock::now();

    // calculate the time to check the traffic
    auto d_nano = chrono::duration_cast<nano_s>(t2-t1).count();
    float a = d_nano;
    float d_milli = a/1000000L;
    cout << "time elapse: " << d_milli << "ms" << endl;

    // sorting the result to find the less crowded signals
    resultSum resultFinal[NUMBER_HOUR][NUMBER_SIGN];

    for(int i = 0 ; i < NUMBER_HOUR; i++) {
        for(int j = 0 ; j < NUMBER_SIGN; j++) {
            resultSum conVal{j, resultMatrix[i][j]};
            resultFinal[i][j] = conVal;
        }
    }

    sorting(resultFinal);

    // printMatrix(resultFinal);

    // find the highest records
    int inputH, inputM;
    cout <<"Enter the hour to display: ";
    cin >> inputH;
    cout <<"Enter the number of highest records display: ";
    cin >> inputM;
    printResult(resultFinal, inputH, inputM);

    // print(resultMatrix, NUMBER_HOUR, NUMBER_SIGN);
}
void node(int process_rank, int num_processes)
{
    // cout << "hello at node " << process_rank << endl;
    int num_data_local = measures / num_processes;
    int sizeChar;
    int maxCharInOneLine = 50;
    sizeChar = maxCharInOneLine * num_data_local;

    char * strA;
    strA = new char [sizeChar];

    MPI_Recv(&strA[0], sizeChar, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    string work_str(strA);

    // begin produce and consume 
    queue que(8);
    int hours_inside = NUMBER_HOUR / num_processes;
    init(resultMatrix, hours_inside, NUMBER_SIGN);
    int num_elements_to_scatter_or_gather = hours_inside*NUMBER_SIGN;

    // cout << work_str << endl;

    stringstream str_strm(work_str);

    int total_thread = NumCon + NumProd;

    omp_init_lock(&lockProd);
    omp_init_lock(&lockCon);
    omp_set_num_threads(total_thread);

    int count_prod = 0;
    int count_con = 0; 

    #pragma omp parallel for 
    for(int i = 0; i < total_thread ; i++) {
        if (i < NumProd) {
            while (count_prod < num_data_local) {
                omp_set_lock(&lockProd);
                string record;
                getline(str_strm, record);
                producer(record, que);
                count_prod++;
                omp_unset_lock(&lockProd);
            }
        } else {
            while(count_con < num_data_local) {
                omp_set_lock(&lockCon);
                bool check = false;
                while(!check) {
                    check = qAccessCon(que, resultMatrix, hours_inside);
                };
                count_con++;
                omp_unset_lock(&lockCon);
            }
        }
    }

    omp_destroy_lock(&lockProd);
    omp_destroy_lock(&lockCon);

    // for (int i = 0; i < num_data_local; i++) {
    //     string record;
    //     getline(str_strm, record);
    //     producer(record, que);
    // }

    // bool check = true;
    // while(check) {
    //     check = qAccessCon(que, resultMatrix, hours_inside);
    // };

    MPI_Gather(&resultMatrix[0][0], num_elements_to_scatter_or_gather , MPI_INT, NULL, num_elements_to_scatter_or_gather , MPI_INT, 0 , MPI_COMM_WORLD);
}

bool qAccessProd(trafficData data, queue &que) {
    if (!que.isFull()) {
        que.enqueue(data);
        return true;
    } else {
        return false;
    }
}

bool qAccessCon(queue &que, int ** &matrix, int row_per) {
    if (!que.isEmpty()) {
        trafficData theData = que.peek();
        int row = theData.hours%row_per;
        matrix[row][theData.id] += theData.cars;
        que.dequeue();

        return true;
    } else {
        return false;
    }
}

void producer(string myText, queue &que) {
        //make a stringstream of the result which the rows inside the dataset
        //myText here is the line of data
        // cout << myText;
        stringstream str_strm(myText);
        trafficData data;
        string tmp;
        int count = 0;
        while (getline(str_strm, tmp, ',')) {
            switch (count)
            {
                case 0:
                    {
                        stringstream s(tmp);
                        string tmp2;
                        int count2 = 0;
                        while (getline(s, tmp2, ':')) {
                            switch (count2)
                            {
                            case 0:
                                data.hours = stoi(tmp2);
                                break;
                            case 1:
                                data.minute = stoi(tmp2);
                                break;
                            default:
                                cout << "error2" << endl;
                                break;
                            }
                            count2++;
                        }
                    }
                    break;
                case 1:
                    data.id = stoi(tmp);
                    break;
                case 2:
                    data.cars = stoi(tmp);
                    break;
                default:
                    cout << "error" << endl;
                    break;
            }   
            count++;
        }
        // printf(": hours %d, id %d, and cars %d\n", data.hours, data.id, data.cars);
        bool check = false;
        while(!check) {
            check = qAccessProd(data, que);
        };
}

void init(int ** &A, int rows, int cols) {
    A = (int **) malloc(sizeof(int*) * rows * cols);  // number of rows * size of int* address in the memory
    int* tmp = (int *) malloc(sizeof(int) * cols * rows); 

    for(int i = 0 ; i < cols ; i++) {
        A[i] = &tmp[i * cols];
    }

    for(long i = 0 ; i < rows; i++) {
        for(long j = 0 ; j < cols; j++) {
            A[i][j] = 0;
        }
    }
}

void print(int ** A, int rows, int cols) {
  for(long i = 0 ; i < rows; i++) { //rows
        for(long j = 0 ; j < cols; j++) {  //cols
            printf("%d ", A[i][j]); // print the cell value
        }
        printf("\n"); //at the end of the row, print a new line
    }
    printf("----------------------------\n");
}

// print the highest records
void printResult (resultSum matrix[NUMBER_HOUR][NUMBER_SIGN], int hour, int numberMax) {
    if (numberMax > NUMBER_SIGN || hour > NUMBER_HOUR || matrix[hour][0].id == -1) {
        cout << "error" << endl;
    } else {
        cout << "Hour of [" << hour << "]: "; 
        for (int i = (NUMBER_SIGN - 1); i > (NUMBER_SIGN - numberMax - 1); i--) {
            cout << "id-" << matrix[hour][i].id << " "; 
        }
        cout << endl;
    }
}

void swap(resultSum matrix[NUMBER_HOUR][NUMBER_SIGN], int i1, int i2, int j1, int j2) {
    resultSum value = matrix[i1][j1];
    matrix[i1][j1] = matrix[i2][j2];
    matrix[i2][j2] = value;
}

void sorting(resultSum matrix[NUMBER_HOUR][NUMBER_SIGN]) {
    for (int i = 0; i < NUMBER_HOUR; i++) {
        for (int j = 0; j < NUMBER_SIGN; j++) {
            if (matrix[i][j].id == -1) {
                break;
            }
            for (int g = 0; g < NUMBER_SIGN - j - 1; g++) {
                if (matrix[i][g].sum > matrix[i][g+1].sum) {
                    swap(matrix, i, i, g, g+1);
                }
            }
        }
    }
}

void printMatrix (resultSum matrix[NUMBER_HOUR][NUMBER_SIGN]) {
    for (int i = 0; i < NUMBER_HOUR; i++) {
        cout << "Hour [" << i << "]: ";
        for (int j = 0; j < NUMBER_SIGN; j++) {
            if (matrix[i][j].id == -1) {
                break;
            }
            cout << "ID" << matrix[i][j].id << ": " << matrix[i][j].sum << ", ";
        }
        cout << endl;
    }
}