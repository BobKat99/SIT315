#include <iostream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <omp.h>

using namespace std;
using nano_s = chrono::nanoseconds;

#define NUM_THREADS 8
static pthread_mutex_t mutex;

struct threadData
{
    int ** matrixA;
    int ** matrixB;
    int ** result;
    int n;
    int acc;
};

void printMatrix (int ** matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}

void dotProd (int ** matrixA, int ** matrixB, int ** result, int start, int end, int n) {
    if (end <= n) {
        for (int a = start; a < end; a++) {
            for (int b = 0; b < n; b++) {
                int mul = 0;
                for (int c = 0; c < n; c++) {
                    mul += matrixA[a][c] * matrixB[c][b];
                }
                result[a][b] = mul;
            }
        }
    }
}

void * calculateMatrix(void * data)
{
    pthread_mutex_lock(&mutex);

    struct threadData *the_data;
    the_data = (threadData*) data;

    int ** matrixA = the_data->matrixA;
    int ** matrixB = the_data->matrixB;
    int ** result = the_data->result;
    int i = the_data->acc;
    int n = the_data->n;

    int start = i*(n/NUM_THREADS);
    int end;
    if (i == NUM_THREADS - 1) {
        end = n;
    } else {
        end = (i+1)*(n/NUM_THREADS);
    }

    dotProd(matrixA, matrixB, result, start, end, n);

    pthread_mutex_unlock(&mutex);

    pthread_exit(NULL);
}

int main() {
    //create random numbers
    srand(time(NULL));
    int n = rand() % 20 + 1;
    n = 1000;
    //create maxtrix
    int ** matrixA;
    matrixA = new int *[n];
    for (int i = 0; i < n; i++) {
        matrixA[i] = new int[n];
    }
    int ** matrixB;
    matrixB = new int *[n];
    for (int i = 0; i < n; i++) {
        matrixB[i] = new int[n];
    }
    int ** result;
    result = new int *[n];
    for (int i = 0; i < n; i++) {
        result[i] = new int[n];
    }
    //fill matrix with random number
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrixA[i][j] = rand() % 100;
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrixB[i][j] = rand() % 100;
        }
    }

    // calculate using pthread
    // Thread set up
    pthread_t threadId[NUM_THREADS];
    threadData td[NUM_THREADS];
    // calculate
    auto t3 = chrono::steady_clock::now();
    for (int i = 0; i < NUM_THREADS; i++) {
        td[i] = {matrixA, matrixB, result, n, i};
        int rc = pthread_create(&threadId[i], NULL, calculateMatrix, (void *)&td[i]);
        if (rc) {
            cout << "Error:unable to create thread" <<endl;
            exit(-1);
        }
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        int err = pthread_join(threadId[i], NULL);
        if (err) {
            cout << "Error in thread join" << endl;
        }
    }
    auto t4 = chrono::steady_clock::now();

    // calculate using openMP
    omp_set_num_threads(NUM_THREADS);
    auto t5 = chrono::steady_clock::now();
    #pragma omp parallel for
    for (int a = 0; a < n; a++) {
        for (int b = 0; b < n; b++) {
            int mul = 0;
            for (int c = 0; c < n; c++) {
                mul += matrixA[a][c] * matrixB[c][b];
            }
            result[a][b] = mul;
        }
    }
    auto t6 = chrono::steady_clock::now();

    //calculate the result normally
    auto t1 = chrono::steady_clock::now();
    dotProd(matrixA, matrixB, result, 0, n, n);
    auto t2 = chrono::steady_clock::now();

    // calulate the time
    auto d_nano_normal = chrono::duration_cast<nano_s>(t2-t1).count();
    auto d_nano_pthread = chrono::duration_cast<nano_s>(t4-t3).count();
    auto d_nano_omp = chrono::duration_cast<nano_s>(t6-t5).count();

    float a = d_nano_normal;
    float b = d_nano_pthread;
    float c = d_nano_omp;
    float p_th = (b/a)*100;
    float mp = (c/a)*100;

    //print out the final result
    cout << "the time needed for normal calculation in nanosecond is: " << d_nano_normal << endl;
    cout << "the time needed for pthread calculation in nanosecond is: " << d_nano_pthread << " and is " << p_th << "% of normal" << endl;
    cout << "the time needed for openMP calculation in nanosecond is: " << d_nano_omp << " and is " << mp << "% of normal" << endl << endl;
    cout << "the result matrix of " << n << "x" << n << " and threads of " << NUM_THREADS << endl;
    // printMatrix(result, n);
    return 0;
}