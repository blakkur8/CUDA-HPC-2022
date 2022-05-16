#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <stdio.h>

#define iterations 256
#define results_size iterations * iterations * iterations
#define frequencies_group_by_size iterations * iterations * iterations * 2


struct Offset {
    int x;
    int y;
    int z;
};

__global__ void add_kernel(unsigned int *results,  int x_offset, int y_offset, int z_offset) {
    int i = blockIdx.x;
    int j = threadIdx.x;


    // Largest number that i and j can be is 255, the offset allows the number to be larger
    // do to limits in number of threads cuda allows
    unsigned int x = i + (x_offset * iterations);
    unsigned int y = j + (y_offset * iterations);

    unsigned int result;

    // thread_array is an independent part of the results array for each thread that each thread can use,
    // to prevent race conditions.
    
    // All CUDA cores uses the same 1D array, but each core needs to have it own index-range in the array,
    // the core is the only one that can modify that part of the array.
    // Each result in the array is followed by its frequncy
    // Each thread in the array can have max 256 results
    // Each block has 256 threads
    // So to get block number 2 you need start pointer +=  (2 * iterations * iterations * 1)
    // where 1 is the index of block 2.
    unsigned int *thread_array = &results[((2 * iterations * j) + (2 * iterations * iterations * i))];

    int next_free_index = 0;
    int z = 0;

    for (int k=0; k < iterations; k++) {
        z = k + (z_offset * iterations);
        
        // The function the is being computed
        result = (x & y) ^ ((!x) & z);


        for (int l=0; l <= next_free_index; l++) {
            if (l == next_free_index) {
                // First instance of the result 

                next_free_index ++;

                // Assign the result to free location
                thread_array[(l * 2)] = result;

                // Initialize the counter for that particular result
                thread_array[(l * 2 + 1)] = 1;
                break;
            } else if(thread_array[(l * 2)] == result){
                // The result exits

                // Iterate the counter for that particular result
                thread_array[(l * 2 + 1)] ++;
                break;
            }
        }
    }
}

__global__ void merge_threads_arrays(unsigned int *frequencies_group_by_threads, unsigned int* frequencies_group_by_block) {
    // This function merges the result for each thread in the same block

    // Input:
    //  [[BLOCK 1], [BLOCK 2], ...]
    //  BLOCK 1 -> [[Thread 1], [Thread 2], ...]
    //  Thread 1 -> [[result 1, count], [result 2, count]]
    // The array is 1-dimensional but it is better to think about it as a 3-dimensional array

    // Output:
    //  [[BLOCK 1], [BLOCK 2], ...]
    //  BLOCK 1 -> [[result 1, count], [result 2, count], ...]
    // The array is 1-dimensional but it is better to think about it as a 3-dimensional array

    int i = blockIdx.x;

    unsigned int result, result_frequency;

    // Keeps track of the current thread in block i
    unsigned int* current_thread = frequencies_group_by_threads + (i * iterations * iterations * 2);

    // Start index of the array that the threads in block i are merge into
    unsigned int* block_frequencies = frequencies_group_by_block + (i * iterations * iterations * 2);

    int next_free_index = 0;

    for (int thread_index=0; thread_index < iterations; thread_index++) {
        for (int result_index=0; result_index < iterations; result_index++) {

            result = current_thread[result_index * 2];
            result_frequency = current_thread[result_index * 2 + 1];


            if(result_frequency == 0) break;

            for (int ii=0; ii <= next_free_index; ii++) {

                if (ii == next_free_index) {
                    // First instance of the result

                    next_free_index ++;
                    block_frequencies[ii * 2] = result;
                    block_frequencies[ii * 2 + 1] = result_frequency;
                    break;
                } else if(block_frequencies[ii * 2] == result) {

                    // The result exits
                    
                    block_frequencies[ii * 2 + 1] += result_frequency;
                    break;
                }
            }
        }

        // Each thread has 256 results, each result has 2 values (result, count)
        // so each thread has total 256 * 2 entries.
        current_thread +=  iterations * 2;
    }
}


int start_brute_force (int *results, unsigned int *frequencies, int inner_iterations) {
    unsigned int *cuda_frequencies_count = nullptr;
    unsigned int *cuda_block_frequencies_count = nullptr;

    unsigned int *frequencies_count = (unsigned int *) malloc(frequencies_group_by_size * sizeof(int));
    unsigned int *frequencies_group_by_blocks = (unsigned int*) malloc(frequencies_group_by_size* sizeof(int));

    cudaMalloc((void **) &cuda_frequencies_count, frequencies_group_by_size * sizeof(int));
    cudaMalloc((void **) &cuda_block_frequencies_count, frequencies_group_by_size * sizeof(int));

    Offset offset;

    // These 3 for loops create an offset so the program is able to run numbers > 256
    for (int i=0; i < inner_iterations; i++) {
        for (int j=0; j < inner_iterations; j++) {
            for (int k=0; k < inner_iterations; k++) {
                offset = {i, j, k};

                // Clear memory before using it again.
                cudaMemset(cuda_frequencies_count, 0, frequencies_group_by_size * sizeof(int));
                cudaMemset(cuda_block_frequencies_count, 0, frequencies_group_by_size * sizeof(int));
                memset(frequencies_group_by_blocks, 0, frequencies_group_by_size * sizeof(int));

                add_kernel<<<iterations, iterations>>>(cuda_frequencies_count, offset.x, offset.y, offset.z);

                cudaDeviceSynchronize();

                cudaMemcpy(frequencies_count, cuda_frequencies_count, frequencies_group_by_size * sizeof(int), cudaMemcpyDeviceToHost);

                // There frequencies calulate are group by blockId and threadId, and need to be merge together
                // merge_threads_arrays merges the frequencies of all threads within a block to a single result
                merge_threads_arrays<<<256, 1>>>(cuda_frequencies_count, cuda_block_frequencies_count);

                cudaDeviceSynchronize();

                cudaMemcpy(frequencies_group_by_blocks, cuda_block_frequencies_count, frequencies_group_by_size * sizeof(int), cudaMemcpyDeviceToHost);
                unsigned int result, result_frequency;

                unsigned int* current_block;

                // Merge each block array to a single result array
                for (int block_index=0; block_index < iterations; block_index++) {

                    current_block = frequencies_group_by_blocks + (block_index * iterations * iterations * 2);
                    for (int i=0; i < iterations * iterations; i++){
                        result = current_block[i * 2];
                        result_frequency = current_block[i * 2 + 1];

                        // If frequency is 0 there are no more result left in the current block
                        if(result_frequency == 0) break;
                        
                        // Sum all results by index as the result and value as frequency
                        frequencies[result] += result_frequency;
                    }
                }

            }
        }
    }
    cudaFree(cuda_block_frequencies_count);
    cudaFree(cuda_frequencies_count);
    free(frequencies_count);

    return 0;
}


int main(int argc, char * argv[]) {
    int N = atoi(argv[1]);

    unsigned int start_number = 0;
    unsigned int end_number = N;
    unsigned int range_size = end_number - start_number;

    // For timing the program
    clock_t start, end;
    double run_time;
    
    
    // Each CUDA iterations can run maxmium of N=256 times. 
    // For number N > 256 it needs to run x iterations for each parameter (i, j, k).
    // where inner_iterations = x. 
    int inner_iterations = range_size / iterations;

    int *results = (int *) malloc(results_size * sizeof(int));
    unsigned int *frequencies = (unsigned int *) malloc(4294967296 * sizeof(int));

    // For timing
    start = clock();

    start_brute_force(results, frequencies, inner_iterations);
    
    // For timing
    end = clock();
    run_time = ((double) (end - start)) / CLOCKS_PER_SEC;

    unsigned long long int temp_count = 0;

    // Prints out the frequency of each results
    for (int i=0; i < range_size; i++) {

        printf("%d: %d \n", i, frequencies[i]);


        temp_count += frequencies[i];
    }

    printf("%llu \n", temp_count);
    printf("Run time: %f \n", run_time);

    free(results);
    free(frequencies);
    return 0;
}
