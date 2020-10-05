// A utility function to swap two elements 
void swap(__global int* arr, int i1, int i2) {
    int value = arr[i1];
    arr[i1] = arr[i2];
    arr[i2] = value;
}

int partition(__global int* arr, int l, int h) 
{ 
	int x = arr[h]; 
	int i = (l - 1); 

	for (int j = l; j <= h - 1; j++) { 
		if (arr[j] <= x) { 
			i++; 
			swap(arr, i, j); 
		} 
	} 
	swap(arr, i + 1, h); 
	return (i + 1); 
} 

void quickSortIterative(__global int* arr, int l, int h) 
{
    int stack[131069];
	// Create an auxiliary stack
	int top = -1; 

	stack[++top] = l; 
	stack[++top] = h; 

	while (top >= 0) { 
		h = stack[top--]; 
		l = stack[top--]; 
		int p = partition(arr, l, h);  
		if (p - 1 > l) { 
			stack[++top] = l; 
			stack[++top] = p - 1; 
		} 
		if (p + 1 < h) { 
			stack[++top] = p + 1; 
			stack[++top] = h; 
		} 
	} 
}

__kernel void quicksorting(const int CHUNK, const int RANK, const int SZ,
                      __global int* arr) {
    
    // Thread identifiers
    const int globalposition = get_global_id(0);

    if (globalposition == 0) {
        quickSortIterative(arr, 0, CHUNK - 1);
    }
}
