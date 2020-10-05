
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <thread>
#include <mpi.h>
#include <chrono>
#include <CL/cl.h>

// mpicxx -opencl ./quicksort_opencl.cpp -o quicksortCL.o -std=c++11 -lOpenCL
// mpirun -np 4 --hostfile ./cluster ./mpiopencl.o 500

using namespace std;
using nano_s = chrono::nanoseconds;

int SZ = 4;
const int TS = 4;
int * arr, *resultA;

void init(int * &arr, int size, bool initialise);
void print_arr(int * arr, int size);

void head(int num_processes);
void node(int process_rank, int num_processes);

cl_device_id device_id;
cl_context context;
cl_program program;
cl_kernel kernel;
cl_command_queue queue;

cl_event event = NULL;
int err;

cl_mem bufArr, bufResult;

size_t local[1] = {(size_t)TS};
size_t global[1] = {(size_t)SZ}; // number of rows and cols or basically the number of threads with indices i and j where i is the row and j is the col of the matric C

cl_device_id create_device();
cl_program build_program(cl_context ctx, cl_device_id dev, const char *filename);

void setup_openCL_device_context_queue_kernel(char* filename, char* kernelname);
void setup_kernel_memory(int num_rows_per_process_from_A);
void copy_kernel_args(int num_rows_per_process_from_A, int rank);
void free_memory();

int main(int argc, char **argv)
{
    if (argc > 1)
        SZ = atoi(argv[1]);

    MPI_Init(NULL, NULL);

    int num_processes;
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    int process_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    if (process_rank == 0)
    {
        head(num_processes);
    }
    else
    {
        node(process_rank, num_processes);
    }

    MPI_Finalize();
}

void head(int num_processes)
{
    init(arr, SZ, true);
    init(resultA, SZ, false);

    print_arr(arr, SZ);
    // auto t1 = chrono::steady_clock::now();

    int num_elements_to_scatter_or_gather = SZ / num_processes;
    
    MPI_Scatter(&arr[0], num_elements_to_scatter_or_gather ,  MPI_INT , &arr , 0, MPI_INT, 0 , MPI_COMM_WORLD);
    //Start of OpenCL
    local[0] = 2;
    global[0] = SZ;

    setup_openCL_device_context_queue_kernel( (char*) "./quicksort.cl" , (char*) "quicksorting");
    setup_kernel_memory(num_elements_to_scatter_or_gather);
    copy_kernel_args(num_elements_to_scatter_or_gather, 0);
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, &event);
    clWaitForEvents(1, &event);
    clEnqueueReadBuffer(queue, bufResult, CL_TRUE, 0, num_elements_to_scatter_or_gather *sizeof(int), &resultA[0], 0, NULL, NULL);
    //end of OpenCL

    MPI_Gather(MPI_IN_PLACE, num_elements_to_scatter_or_gather, MPI_INT, &resultA[0], num_elements_to_scatter_or_gather, MPI_INT, 0, MPI_COMM_WORLD);
    print_arr(resultA, SZ);
    //calculate time to milliseconds
    // auto t2 = chrono::steady_clock::now();
    // auto d_nano = chrono::duration_cast<nano_s>(t2-t1).count();
    // float a = d_nano;
    // float d_milli = a/1000000L;

    // cout << "time elapse: " << d_milli << "ms" << endl;
    free_memory();
}

void node(int process_rank, int num_processes)
{
    int num_elements_to_scatter_or_gather = SZ / num_processes;
    init(arr, num_elements_to_scatter_or_gather, false);
    init(resultA, num_elements_to_scatter_or_gather, false);

    MPI_Scatter(NULL, num_elements_to_scatter_or_gather, MPI_INT, &arr[0], num_elements_to_scatter_or_gather, MPI_INT, 0, MPI_COMM_WORLD);

    //Start of OpenCL
    local[0] = 2;
    global[0] = SZ;

    setup_openCL_device_context_queue_kernel( (char*) "./quicksort.cl" , (char*) "quicksorting");
    setup_kernel_memory(num_elements_to_scatter_or_gather);
    copy_kernel_args(num_elements_to_scatter_or_gather, process_rank);
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, local, 0, NULL, &event);
    clWaitForEvents(1, &event);
    clEnqueueReadBuffer(queue, bufResult, CL_TRUE, 0, num_elements_to_scatter_or_gather *sizeof(int), &resultA[0], 0, NULL, NULL);
    //End of OpenCL
   
    MPI_Gather(&resultA[0], num_elements_to_scatter_or_gather, MPI_INT, NULL, num_elements_to_scatter_or_gather, MPI_INT, 0, MPI_COMM_WORLD);
    free_memory();
}

void free_memory()
{

    clReleaseKernel(kernel);
    clReleaseMemObject(bufArr);
    clReleaseMemObject(bufResult);

    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);
}

void copy_kernel_args(int elements, int rank)
{
    //NOTE that we modified the first parameter (Rows of A)
    clSetKernelArg(kernel, 0, sizeof(int), (void *)&elements);
    clSetKernelArg(kernel, 1, sizeof(int), (void *)&rank);
    clSetKernelArg(kernel, 2, sizeof(int), (void *)&SZ);

    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&bufArr);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&bufResult);
    if (err < 0)
    {
        perror("Couldn't create a kernel argument");
        printf("error = %d", err);
        exit(1);
    }
}

void setup_kernel_memory(int elements)
{
     //NOTE that we modified the bufA to only cover rows of A and C
    bufArr = clCreateBuffer(context, CL_MEM_READ_ONLY, elements * sizeof(int), NULL, NULL);
    bufResult = clCreateBuffer(context, CL_MEM_READ_WRITE, elements * sizeof(int), NULL, NULL);

    // Copy matrices to the GPU
    clEnqueueWriteBuffer(queue, bufArr, CL_TRUE, 0, elements * sizeof(int), &arr[0], 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufResult, CL_TRUE, 0, elements * sizeof(int), &resultA[0], 0, NULL, NULL);
}

void init(int *&arr, int size, bool initialise)
{
    arr = (int *) malloc(sizeof(int) * size);

    if(!initialise) return;

    for (int i = 0; i < SZ; i++) {
        arr[i] = rand() % 100;
    }
}

void print_arr(int * arr, int size) {
    for (int i = 0; i < size; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
}

void setup_openCL_device_context_queue_kernel(char* filename, char* kernelname) {
    device_id = create_device();
    cl_int err;
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
   if(err < 0) {
      perror("Couldn't create a context");
      exit(1);   
    }

    program = build_program(context, device_id, filename );
    queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
    if(err < 0) {
      perror("Couldn't create a command queue");
      exit(1);   
    };

    kernel = clCreateKernel(program, kernelname, &err);
    if(err < 0) {
      perror("Couldn't create a kernel");
      printf("error =%d", err);
      exit(1);
    };

}

cl_program build_program(cl_context ctx, cl_device_id dev, const char *filename)
{
    cl_program program;
    FILE *program_handle;
    char *program_buffer, *program_log;
    size_t program_size, log_size;

    /* Read program file and place content into buffer */
    program_handle = fopen(filename, "r");
    if (program_handle == NULL)
    {
        perror("Couldn't find the program file");
        exit(1);
    }
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char *)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);

    /* Create program from file 

    Creates a program from the source code in the add_numbers.cl file. 
    Specifically, the code reads the file's content into a char array 
    called program_buffer, and then calls clCreateProgramWithSource.
    */
    program = clCreateProgramWithSource(ctx, 1,
                                        (const char **)&program_buffer, &program_size, &err);
    if (err < 0)
    {
        perror("Couldn't create the program");
        exit(1);
    }
    free(program_buffer);

    /* Build program 

    The fourth parameter accepts options that configure the compilation. 
    These are similar to the flags used by gcc. For example, you can 
    define a macro with the option -DMACRO=VALUE and turn off optimization 
    with -cl-opt-disable.
    */
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err < 0)
    {

        /* Find size of log and print to std output */
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
                              0, NULL, &log_size);
        program_log = (char *)malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
                              log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }

    return program;
}

cl_device_id create_device()
{
    cl_platform_id platform;
    cl_device_id dev;
    int err;

    /* Identify a platform */
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err < 0)
    {
        perror("Couldn't identify a platform");
        exit(1);
    }

    // Access a device
    // GPU
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    if (err == CL_DEVICE_NOT_FOUND)
    {
        // CPU
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
    }
    if (err < 0)
    {
        perror("Couldn't access any devices");
        exit(1);
    }

    return dev;
}