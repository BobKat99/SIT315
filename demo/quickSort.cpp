#include <iostream>
#include <string>
#include <cstdlib>
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <chrono>

using namespace std;
using nano_s = chrono::nanoseconds;

#define NUM_THREADS 4

void print_arr(int arr[], int size) {
    for (int i = 0; i < size; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
}

void swap(int arr[], int i1, int i2) {
    int value = arr[i1];
    arr[i1] = arr[i2];
    arr[i2] = value;
}

void quickSort(int arr[], int start, int end) {
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

void quickSortPara(int arr[], int start, int end) {
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
       #pragma omp parallel
       {
           #pragma omp sections
           {
               #pragma omp section
               {
                    quickSort(arr, start, (pivot-1));
               }
               #pragma omp section
               {
                    quickSort(arr, (pivot + 1), end);
               }
           }
       }
    }
}

void quickSortPara2(int arr[], int start, int end) {
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
       #pragma omp parallel
       {
           #pragma omp sections
           {
               #pragma omp section
               {
                    quickSortPara2(arr, start, (pivot-1));
               }
               #pragma omp section
               {
                    quickSortPara2(arr, (pivot + 1), end);
               }
           }
       }
    }
}

int main() {
    //set up random numbers
    srand(time(NULL));
    int n = rand() % 20 + 1;
    n = 1000000;

    // create 2 idential arrays with random numbers
    int arr[n];
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 100;
    }
    int arr2[n];
    for (int i = 0; i < n; i++) {
        arr2[i] = arr[i];
    }

    // cout << "the array before is: ";
    // print_arr(arr, n);
    // cout << endl;
    // sort normally
    auto t1 = chrono::steady_clock::now();
    quickSort(arr, 0, n-1);
    auto t2 = chrono::steady_clock::now();

    // cout << "the array after normal is: ";
    // print_arr(arr, n);
    // cout << endl;

    // sort using concurrent programming with 2 threads
    auto t3 = chrono::steady_clock::now();
    quickSortPara(arr2, 0, n-1);
    auto t4 = chrono::steady_clock::now();

    // cout << "the array after openMP is: ";
    // print_arr(arr2, n);
    // cout << endl;

    // sort using concurrent programming with multi threads
    auto t5 = chrono::steady_clock::now();
    quickSortPara2(arr2, 0, n-1);
    auto t6 = chrono::steady_clock::now();

     // calulate the time
    auto d_nano_normal = chrono::duration_cast<nano_s>(t2-t1).count();
    auto d_nano_omp = chrono::duration_cast<nano_s>(t4-t3).count();
     auto d_nano_omp_multi = chrono::duration_cast<nano_s>(t6-t5).count();

    float a = d_nano_normal;
    float b = d_nano_omp_multi;
    float c = d_nano_omp;
    float mp = (c/a)*100;
    float mp2 = (b/a)*100;

    //print out the final results
    cout << "the time needed for normal calculation in nanosecond is: " << d_nano_normal << endl;
    cout << "the time needed for openMP 2 threads calculation in nanosecond is: " << d_nano_omp << " and is " << mp << "% of normal" << endl;
    cout << "the time needed for openMP multi calculation in nanosecond is: " << d_nano_omp_multi << " and is " << mp2 << "% of normal" << endl << endl;

    return 0;
}