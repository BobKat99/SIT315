#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <omp.h>
#include <chrono>

using namespace std;
using nano_s = chrono::nanoseconds;

// the data in general
struct trafficData {
    int hours, minute, id, cars;
};

// the data after processing to create the sum
struct resultSum {
    int id=-1, sum;
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

// a global elements
#define HOURS_OF_DAY 24
#define NUMBER_OF_SIGN 6

queue que(8);
resultSum matrixResult[HOURS_OF_DAY][NUMBER_OF_SIGN];
bool productionDone = false;
//start reading file
ifstream MyReadFile("data.csv");

bool qAccess(bool isProducer, trafficData data) {
    if(isProducer) {
        if (!que.isFull()) {
            que.enqueue(data);
            return true;
        } else {
            return false;
        }
    } else {
        if (!que.isEmpty()) {
            trafficData theData = que.peek();
            if (matrixResult[theData.hours][theData.id].sum == 0) {
                resultSum finalResult{theData.id, theData.cars};
                matrixResult[theData.hours][theData.id] = finalResult;
            } else {
                matrixResult[theData.hours][theData.id].sum += theData.cars;
            }
            que.dequeue();

            return true;
        } else {
            return false;
        }
    }
}

void producer() {
    string myText;
    // Use a while loop together with the getline() function to read the file line by line
    while (getline (MyReadFile, myText)) {
        //make a stringstream of the result which the rows inside the dataset
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
        bool check = false;
        while(!check) {
            check = qAccess(true, data);
        }
    }
    // Close the file
    MyReadFile.close();
}

void consumer() {
    while(!productionDone) {
        while(!que.isEmpty()) {
            qAccess(false, trafficData{0,0});
        }
    }
}

void swap(resultSum matrix[HOURS_OF_DAY][NUMBER_OF_SIGN], int i1, int i2, int j1, int j2) {
    resultSum value = matrix[i1][j1];
    matrix[i1][j1] = matrix[i2][j2];
    matrix[i2][j2] = value;
}

void sorting(resultSum matrix[HOURS_OF_DAY][NUMBER_OF_SIGN]) {
    for (int i = 0; i < HOURS_OF_DAY; i++) {
        for (int j = 0; j < NUMBER_OF_SIGN; j++) {
            if (matrix[i][j].id == -1) {
                break;
            }
            for (int g = 0; g < NUMBER_OF_SIGN - j - 1; g++) {
                if (matrix[i][g].sum > matrix[i][g+1].sum) {
                    swap(matrix, i, i, g, g+1);
                }
            }
        }
    }
}

void printMatrix (resultSum matrix[HOURS_OF_DAY][NUMBER_OF_SIGN]) {
    for (int i = 0; i < HOURS_OF_DAY; i++) {
        cout << "Hour [" << i << "]: ";
        for (int j = 0; j < NUMBER_OF_SIGN; j++) {
            if (matrix[i][j].id == -1) {
                break;
            }
            cout << "ID" << matrix[i][j].id << ": " << matrix[i][j].sum << ", ";
        }
        cout << endl;
    }
}

void printResult (resultSum matrix[HOURS_OF_DAY][NUMBER_OF_SIGN], int hour, int numberMax) {
    if (numberMax > NUMBER_OF_SIGN || hour > HOURS_OF_DAY || matrix[hour][0].id == -1) {
        cout << "error" << endl;
    } else {
        cout << "Hour of [" << hour << "]: "; 
        for (int i = 0; i < numberMax; i++) {
            cout << "id-" << matrix[hour][i].id << " "; 
        }
        cout << endl;
    }
}

int main() {
    int num_prod;
    int num_consu;

    cout <<"Enter the number of producer: ";
    cin >> num_prod;
    cout <<"Enter the number of consumer: ";
    cin >> num_consu;

    auto t1 = chrono::steady_clock::now();
    #pragma omp parallel
       {
           #pragma omp sections
           {
               #pragma omp section
               {
                    #pragma omp parallel for
                    for(int i = 0; i < num_prod; i++) {
                        producer();
                    }
                    productionDone = true;
               }
               #pragma omp section
               {
                   #pragma omp parallel for
                    for(int i = 0; i < num_consu; i++) {
                        consumer();
                    }
               }
           }
       }
    sorting(matrixResult);
    auto t2 = chrono::steady_clock::now();

    auto d_nano_normal = chrono::duration_cast<nano_s>(t2-t1).count();

    printMatrix(matrixResult);
    cout << "time elapse in nanosecond is: " << d_nano_normal << endl;

    int inputH, inputM;
    cout <<"Enter the hour to display: ";
    cin >> inputH;
    cout <<"Enter the max number to display: ";
    cin >> inputM;
    printResult(matrixResult, inputH, inputM);
}