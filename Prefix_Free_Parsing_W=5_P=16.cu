
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <algorithm>
#include <chrono>
#include <assert.h>
#include <utility>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <stdint.h> 

#define MAX_THREADS_PER_BLOCK 512
#define MAX_SHARED_MEM_PER_BLOCK 2048
#define Elements_Per_Block 4
#define P 16
#define W 5
#define Size_Max_Sort 10000000
#define Max_Threads_Per_SM 512
#define Max_Sort_in_Shared_Mem 8192

using namespace std;



/*
const ui prime_numbers[] = {
    23,	29,
31,	37,	41,	43,	47	,53,	59	,61	,67	,71,
73,	79	,83	,89	,97	,101	,103,	107,	109,	113
};
*/
int n;
int n_enlarged;
int num_blocks_n;
int num_blocks_num_blocks_n;
int num_blocks_n_enlarged;
int num_blocks_num_blocks_n_enlarged;
int size_D;
int num_blocks_size_D;
int size_S;
int smallest_two_potenz_larger_than_size_S;
int num_blocks_smallest_two_potenz_larger_than_size_S;
int num_blocks_size_S;
int size_Suffixes_Without_Same_Preeceding_Char;
int num_blocks_size_Suffixes_Without_Same_Preeceding_Char;
int smallest_two_potenz_larger_than_size_Suffixes_Without_Same_Preeceding_Char;
int num_blocks_smallest_two_potenz_larger_than_size_Suffixes_Without_Same_Preeceding_Char;
int size_Ranks;
int size_CPU_Blocks_Array;
int size_Block_L_Values;
int size_Block_L_Values_times_8192;
int num_blocks_size_Block_L_Values_times_8192;

unsigned char* Dev_Input;
unsigned char* input;

unsigned char* Dev_Output;
unsigned char* output;

int* Res;

int* Auxiliar1;
int* Auxiliar2;
int* Auxiliar3;
int* Block_L_Values;
int* D;
int* D_Suffix_Array;
int* Suffixes_Without_Same_Preeceding_Char;
unsigned long long* Ranks;
unsigned char* Preeceding_Characters;
short* L_Values;
int* Dev_CPU_Blocks_Array;
int* CPU_Blocks_Array;
int* Block_Info_Radix_Sorting;
int* Block_Group_Info;
int* Sum_Group_Index;
int* Group_Offset;

unsigned int* S_L;
unsigned int* S_Interval_Length;
unsigned short* S_Length_Suffix;
unsigned int* S_Suffix_End;

unsigned int* S_Pos;
unsigned int* S_Rank;
unsigned long long* Cur_Prefix_S;

unsigned int* S_Pos_New;
unsigned int* S_Rank_New;
unsigned long long* Cur_Prefix_S_New;

unsigned int* S_L_Final;
unsigned int* S_Interval_Length_Final;
unsigned short* S_Length_Suffix_Final;


void Scan_input() {

    FILE* f = fopen("input.txt", "r");


    fscanf(f, "%d", &n);
    //scanf("%d", &n);
    input = new unsigned char[n + 1 + W];
    output = new unsigned char[n + 1];

    memset(output, 1, n);
    output[n] = '\0';
    int val;
    //fscanf(f, "%c", &val);
    //fscanf(f, "%c", &val);
    
    for (int i = 0; i < n; i++) {
        fscanf(f, "%d", &val);
        input[i] = val;
        if (!(input[i] >= 2 && input[i] != 255))
            printf("%d %d ", i, val);
        assert(input[i] >= 2 && input[i] != 255);
    }
    
    /*
    fscanf(f, "%c", &val);
    for (int i = 0; i < n; i++) {
        fscanf(f, "%c", &val);
        input[i]= val+128;
        if (!(input[i] >= 2 && input[i] != 255))
            printf("%d %c ", i, input[i]);
        assert(input[i] >= 2 && input[i] != 255);
    }
    */
    //scanf("%s", (input));
    for (int i = 0; i < W; i++)
        input[n + i] = 1;
    input[n + W] = '\0';
    n_enlarged = n + W;
}

int smallest_two_potenz_larger_than_k(int k) {
    int temp = 1;

    while (temp < k) {
        temp <<= 1;

    }

    return temp;
}



void Compute_Params() {
    num_blocks_n_enlarged = (n_enlarged - 1 + MAX_THREADS_PER_BLOCK * Elements_Per_Block) / (MAX_THREADS_PER_BLOCK * Elements_Per_Block);
    num_blocks_num_blocks_n_enlarged = (num_blocks_n_enlarged + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
    num_blocks_n = (n - 1 + MAX_THREADS_PER_BLOCK * Elements_Per_Block) / (MAX_THREADS_PER_BLOCK * Elements_Per_Block);
    num_blocks_num_blocks_n = (num_blocks_n + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;

}

void Malloc_And_Copy_On_GPU() {
    cudaMalloc((void**)&Dev_Input, (n_enlarged + 1) * sizeof(unsigned char));
    cudaMemcpy(Dev_Input, input, (n_enlarged + 1) * sizeof(unsigned char), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&Dev_Output, (n + 1) * sizeof(unsigned char));
    cudaMemcpy(Dev_Output, output, (n + 1) * sizeof(unsigned char), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&Res, sizeof(int));

    cudaMalloc((void**)&Auxiliar1, num_blocks_n_enlarged * sizeof(int));

    cudaMalloc((void**)&Auxiliar2, num_blocks_n_enlarged * sizeof(int));



}

__device__ int smallest_two_potenz_larger_than_k_device(int k) {
    int temp = 1;

    while (temp < k) {
        temp <<= 1;

    }

    return temp;
}


__device__ int maximum(int a, int b) {
    return ((a >= b) ? a : b);
}

int maximum1(int a, int b) {
    return ((a >= b) ? a : b);
}





__device__ int minimum(int a, int b) {
    return ((a <= b) ? a : b);
}

__device__ __forceinline__ unsigned long long pack8(const unsigned char* data) {
    unsigned long long result = 0ULL;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        result |= (unsigned long long)data[i] << ((7 - i) * 8);
    }
    return result;
}

__device__ __forceinline__ unsigned long long pack8_not_reversed(const unsigned char* data) {
    unsigned long long result = 0ULL;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        result |= (unsigned long long)data[i] << ((i) * 8);
    }
    return result;
}


__global__ void Init_Vector(int* Arr, int n, int val) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id < n) {
        Arr[thread_id] = val;
    }
}
__global__ void Init_Vector(unsigned int* Arr, unsigned int n, int val) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id < n) {
        Arr[thread_id] = val;
    }
}
__global__ void Init_Vector1(int* Arr, int n, int val, int offset) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id < n) {
        Arr[(thread_id + offset)] = val;
    }
}


__global__ void Init_Vector(int* Arr, int n, int val, int mult, int offset) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id < n) {
        Arr[(thread_id + offset) * mult] = val;
    }
}


__global__ void Parallel_Block_Sum(int* Z, int* Y, int n) {
    int i = threadIdx.x;
    int l = MAX_SHARED_MEM_PER_BLOCK * Elements_Per_Block * blockIdx.x;
    int r = l + MAX_SHARED_MEM_PER_BLOCK * Elements_Per_Block;
    __shared__ int A[MAX_SHARED_MEM_PER_BLOCK];
    for (int j = l + i; j < r; j += MAX_THREADS_PER_BLOCK) {

        A[j - l] = ((j < n) ? Z[j] : 0);
    }


    for (int stride = MAX_SHARED_MEM_PER_BLOCK >> 1; stride > 0; stride >>= 1) {
        __syncthreads();
        if (i < stride) {
            A[i] += A[i + stride];
        }
    }

    if (threadIdx.x == 0) {
        Y[blockIdx.x] = A[i];
    }
}

__global__ void Init_Res(int* Res, int* Auxiliar, int index) {
    Res[0] = Auxiliar[index];
}


void Parallel_Sum(int* A, int* B, int* Res, int size) {
    int num_blocks = (size - 1 + MAX_SHARED_MEM_PER_BLOCK * Elements_Per_Block) / (MAX_SHARED_MEM_PER_BLOCK * Elements_Per_Block);

    while (num_blocks > 1) {
        Parallel_Block_Sum << <num_blocks, MAX_THREADS_PER_BLOCK >> > (A, B, size);
        size = num_blocks;

        if (num_blocks > 1) {

            int* temp = A;
            A = B;
            B = temp;

            num_blocks = (num_blocks - 1 + MAX_SHARED_MEM_PER_BLOCK * Elements_Per_Block) / (MAX_SHARED_MEM_PER_BLOCK * Elements_Per_Block);
        }
    }
    // Schreibe Ergebnis in Res
    Init_Res << <1, 1 >> > (Res, B, 0);
}





__global__ void Prefix_Sum_Kernel(int* Array1, int* Array2, int n, int stride) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id < n) {
        int index = thread_id - stride;
        if (index >= 0)
            Array1[thread_id] += Array2[index];
        //else
            //Array1[thread_id] = Array2[thread_id];

    }
}

__global__ void Pointer_Jumping_Kernel(int* Array1, int* Array2, int n, int stride) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id < n) {
        int index = thread_id + stride;
        if (index < n) {
            if (Array2[thread_id] == -1)
                Array1[thread_id] = Array2[index];
            else
                Array1[thread_id] = Array2[thread_id];
        }
        //else 
           // Array1[thread_id] = Array2[thread_id];

    }
}

__global__ void Pointer_Jumping_Kernel_Block_L_Values(int* Array1, int* Array2, int n, int stride, int size_D) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    int block_id = (thread_id) / 8192;
    int rest = (thread_id)-block_id * 8192;
    if (block_id < n) {
        int index = (block_id + stride);
        if (index < n) {
            index = index * 8192 + rest;
            if (Array2[thread_id] == size_D)
                Array1[thread_id] = Array2[index];
            else
                Array1[thread_id] = Array2[thread_id];
        }
        //else 
           // Array1[thread_id] = Array2[thread_id];

    }
}

__global__ void Prefix_Sum_Radix_Sort_Kernel(int* Array1, int* Array2, int* Block_Group_Info, int n, int stride) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    int block_id = (thread_id) / 128;
    int rest = (thread_id)-block_id * 128;
    if (block_id < n) {
        int block_group_info = Block_Group_Info[block_id];
        int Block_Group_Info_index = (block_id - stride);
        if (Block_Group_Info_index >= 0 && block_group_info == Block_Group_Info[Block_Group_Info_index]) {
            int index1 = Block_Group_Info_index + rest * n;
            int index = block_id + rest * n;
            Array1[index] += Array2[index1];
        }


    }
}

__global__ void Copy_Array(int* Array1, int* Array2, int n) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id < n) {
        Array1[thread_id] = Array2[thread_id];
    }
}

void Prefix_Sum(int* A, int* B, int* Res, int size) {
    int num_blocks = (size - 1 + MAX_THREADS_PER_BLOCK) / (MAX_THREADS_PER_BLOCK);

    for (int stride = 1; stride < size; stride <<= 1) {
        Copy_Array << <num_blocks, MAX_THREADS_PER_BLOCK >> > (B, A, size);

        Prefix_Sum_Kernel << <num_blocks, MAX_THREADS_PER_BLOCK >> > (A, B, size, stride);
    }

    Init_Res << <1, 1 >> > (Res, A, size - 1);
}

void Prefix_Sum_Radix_Sort(int* A, int* B, int* Block_Group_Info, int* Res, int size) {
    int size_1 = size * 128;
    int num_blocks = (size_1 - 1 + MAX_THREADS_PER_BLOCK) / (MAX_THREADS_PER_BLOCK);
    for (int stride = 1; stride < size; stride <<= 1) {
        Copy_Array << <num_blocks, MAX_THREADS_PER_BLOCK >> > (B, A, size_1);

        Prefix_Sum_Radix_Sort_Kernel << <num_blocks, MAX_THREADS_PER_BLOCK >> > (A, B, Block_Group_Info, size, stride);
    }

    // Init_Res << <1, 1 >> > (Res, A, size - 1);
}



void Pointer_Jumping(int* A, int* B, int size) {
    int num_blocks = (size - 1 + MAX_THREADS_PER_BLOCK) / (MAX_THREADS_PER_BLOCK);

    for (int stride = 1; stride < size; stride <<= 1) {
        Copy_Array << <num_blocks, MAX_THREADS_PER_BLOCK >> > (B, A, size);

        Pointer_Jumping_Kernel << <num_blocks, MAX_THREADS_PER_BLOCK >> > (A, B, size, stride);
    }


}
void Pointer_Jumping_Block_L_Values(int* A, int* B, int size, int size_D) {
    int size_1 = size * 8192;

    int num_blocks = (size_1 - 1 + MAX_THREADS_PER_BLOCK) / (MAX_THREADS_PER_BLOCK);
    for (int stride = 1; stride < size; stride <<= 1) {
        Copy_Array << <num_blocks, MAX_THREADS_PER_BLOCK >> > (B, A, size_1);

        Pointer_Jumping_Kernel_Block_L_Values << <num_blocks, MAX_THREADS_PER_BLOCK >> > (A, B, size, stride, size_D);
    }


}




__global__ void Compute_D1(unsigned char* Dev_Input, int* D_Block_Sum, int n) {
    __shared__ unsigned char Sub_String[MAX_THREADS_PER_BLOCK * Elements_Per_Block + W];
    __shared__ short Sum[MAX_THREADS_PER_BLOCK];
    unsigned char first_word_in_D[W];
    for (int i = 0; i < W; i++)
        first_word_in_D[i] = Dev_Input[i];
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    int l = MAX_THREADS_PER_BLOCK * Elements_Per_Block * block_id + thread_id;
    int r = l + W - 1;
    int local_sum = 0;
    for (int j = 0; j < Elements_Per_Block; j++) {
        Sub_String[thread_id + W - 1 + MAX_THREADS_PER_BLOCK * j] = ((r + MAX_THREADS_PER_BLOCK * j < n) ? Dev_Input[r + MAX_THREADS_PER_BLOCK * j] : 0);

    }
    if (thread_id < W - 1 && l < n)
        Sub_String[thread_id] = Dev_Input[l];

    __syncthreads();
    for (int j = 0; j < Elements_Per_Block && (r + MAX_THREADS_PER_BLOCK * j < n); j++) {
        int potenz_2_mod_P = 1;
        int res = 0;
        int l1 = thread_id + MAX_THREADS_PER_BLOCK * j;
        int r1 = l1 + W - 1;
        bool equal = true;
        for (int i = r1; i >= l1; i--) {

            equal = ((equal == true) ? Sub_String[i] == first_word_in_D[i - l1] : false);
            res = (res + potenz_2_mod_P * int(Sub_String[i])) % P;
            potenz_2_mod_P = (potenz_2_mod_P * 2) % P;
        }

        if ((res == 0 && (l > 0 || (l == 0 && l1 > 0))) || (equal && (l > 0 || (l == 0 && l1 > 0)))) {
            local_sum++;

        }
    }

    Sum[thread_id] = local_sum;
    for (int stride = MAX_THREADS_PER_BLOCK >> 1; stride > 0; stride >>= 1) {
        __syncthreads();

        if (thread_id < stride) {
            Sum[thread_id] = Sum[thread_id] + Sum[thread_id + stride];

        }
    }

    if (thread_id == 0)
        D_Block_Sum[block_id] = Sum[0];




}


__global__ void Compute_D2(unsigned char* Dev_Input, int* D_Blocks_Prefix_Sum, int* D, int* D_Suffix_Array, int n) {
    __shared__ unsigned char Sub_String[MAX_THREADS_PER_BLOCK * Elements_Per_Block + W];
    __shared__ short Prefix_Sum[MAX_THREADS_PER_BLOCK * Elements_Per_Block];
    short Saved_Sums[Elements_Per_Block];
    unsigned char first_word_in_D[W];
    for (int i = 0; i < W; i++)
        first_word_in_D[i] = Dev_Input[i];
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    int l = MAX_THREADS_PER_BLOCK * Elements_Per_Block * block_id + thread_id;
    int r = l + W - 1;
    int offset = ((block_id - 1 >= 0) ? D_Blocks_Prefix_Sum[block_id - 1] : 0);
    for (int j = 0; j < Elements_Per_Block; j++) {
        Sub_String[thread_id + W - 1 + MAX_THREADS_PER_BLOCK * j] = ((r + MAX_THREADS_PER_BLOCK * j < n) ? Dev_Input[r + MAX_THREADS_PER_BLOCK * j] : 0);
        Prefix_Sum[thread_id + MAX_THREADS_PER_BLOCK * j] = 0;
    }

    if (thread_id < W - 1 && l < n)
        Sub_String[thread_id] = Dev_Input[l];

    __syncthreads();
    for (int j = 0; j < Elements_Per_Block && (r + MAX_THREADS_PER_BLOCK * j < n); j++) {
        int potenz_2_mod_P = 1;
        int res = 0;
        int l1 = thread_id + MAX_THREADS_PER_BLOCK * j;
        int r1 = l1 + W - 1;
        bool equal = true;
        for (int i = r1; i >= l1; i--) {
            equal = ((equal == true) ? Sub_String[i] == first_word_in_D[i - l1] : false);
            res = (res + potenz_2_mod_P * int(Sub_String[i])) % P;
            potenz_2_mod_P = (potenz_2_mod_P * 2) % P;
        }

        if ((res == 0 && (l > 0 || (l == 0 && l1 > 0))) || (equal && (l > 0 || (l == 0 && l1 > 0)))) {
            Prefix_Sum[l1] = 1;
        }
    }

    for (int stride = 1; stride < (MAX_THREADS_PER_BLOCK * Elements_Per_Block); stride <<= 1) {
        __syncthreads();
        for (int j = 0; j < Elements_Per_Block; j++) {
            int l1 = thread_id + MAX_THREADS_PER_BLOCK * j;

            if (l1 >= stride) {
                Saved_Sums[j] = Prefix_Sum[l1 - stride];

            }
        }
        __syncthreads();
        for (int j = 0; j < Elements_Per_Block; j++) {
            int l1 = thread_id + MAX_THREADS_PER_BLOCK * j;

            if (l1 >= stride) {
                Prefix_Sum[l1] = Prefix_Sum[l1] + Saved_Sums[j];
            }
        }
    }
    __syncthreads();
    for (int j = 0; j < Elements_Per_Block; j++) {
        int l1 = thread_id + MAX_THREADS_PER_BLOCK * j;
        if ((l1 > 0 && Prefix_Sum[l1] > Prefix_Sum[l1 - 1]) || (l1 == 0 && Prefix_Sum[0] > 0)) {
            int index = ((l1 > 0) ? Prefix_Sum[l1 - 1] : 0) + offset;
            int index_ans = r + MAX_THREADS_PER_BLOCK * j;

            D[index] = index_ans;
            D_Suffix_Array[index] = index_ans;
        }

    }


}

__global__ void Compute_D3(int* D, int* D_Suffix_Array, int n_enlarged, int size_D) {
    D[(size_D - 1)] = n_enlarged - 1;
    D_Suffix_Array[(size_D - 1)] = n_enlarged - 1;
}

__global__ void Compute_D4(int* D, int* D_Suffix_Array, int size_D) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id < size_D) {
        int length_d = ((thread_id > 0) ? D[thread_id] - D[(thread_id - 1)] + W : D[thread_id] + 1);
        if (length_d >= 8192 || length_d <= W)
            printf("%d ", length_d);
        assert(length_d < 8192);
        assert(length_d > W);


        D[thread_id + 1 * size_D] = length_d;
        D_Suffix_Array[thread_id + 1 * size_D] = length_d;
        D_Suffix_Array[thread_id + 2 * size_D] = thread_id;
    }
}

__global__ void D_Check(int* D, unsigned char* Dev_Input, int size_D) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id == 0) {
        for (int j = 0; j < size_D; j++) {
            if (D[j] == 89 || D[j] == 85) {
                printf("\n");
                for (int k = j; k < j + 6; k++) {
                    printf("%d %d ", D[k], k);
                    int lenght_d = D[k + size_D];
                    for (int i = D[k] - lenght_d + 1; i <= D[k]; i++) {
                        printf("%c ", Dev_Input[i]);
                    }
                }
            }

        }
    }
}




__global__ void Compute_D_Suffix_Array1(int* D_Suffix_Array, int* Pointer_list, unsigned char* Dev_Input, int size_D) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id < size_D) {
        /*
        if (thread_id == 0) {
            for (int j = 0; j < size_D; j++) {
                int val11 = D_Suffix_Array[j];
                int val12 = D_Suffix_Array[j + 1 * size_D];
                printf("\n %d %d ", j, val11);
                for (int index_val1 = val11 - val12 + 1; index_val1 <= val11; index_val1++) {
                    char char_val1 = Dev_Input[index_val1];
                    printf("%c ", char_val1);
                }
            }
        }
      */
        bool equal = true;
        if (thread_id < size_D - 1) {
            int val11 = D_Suffix_Array[thread_id];
            int val12 = D_Suffix_Array[thread_id + 1 * size_D];


            int val21 = D_Suffix_Array[thread_id + 1];
            int val22 = D_Suffix_Array[thread_id + 1 + 1 * size_D];





            for (int index_val1 = val11 - val12 + 1, index_val2 = val21 - val22 + 1; index_val1 <= val11 && index_val2 <= val21; index_val1++, index_val2++) {
                unsigned char char_val1 = Dev_Input[index_val1];
                unsigned char char_val2 = Dev_Input[index_val2];
                if (char_val1 > char_val2) {
                    equal = false;
                    break;
                }
                else if (char_val1 < char_val2) {
                    equal = false;
                    break;
                }
            }

            if (equal == true) {
                if (val12 > val22)
                    equal = false;
                else if (val12 < val22)
                    equal = false;
            }

            if (equal == false)
                Pointer_list[thread_id] = thread_id;

        }
        else {
            Pointer_list[thread_id] = thread_id;
        }



    }
}

__global__ void Compute_D_Suffix_Array2(int* D_Suffix_Array, int* D, int* Prefix_Sum, int2* Suffix_Array, int size_D) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id < size_D) {
        int rank = Prefix_Sum[thread_id];
        int index = D_Suffix_Array[thread_id + 2 * size_D];
        D[index + 2 * size_D] = rank;
        Suffix_Array[thread_id] = make_int2(index, 0);
        Prefix_Sum[thread_id]++;

    }



}




__global__ void Init_D_Suffix_Ranks(int2* New_D_Suffix_Array, int* D, unsigned long long* Ranks_Tuple, int size_D_Suffix_Array_Sorting, int size_D, int prefix_length) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id < size_D_Suffix_Array_Sorting) {

        int index = New_D_Suffix_Array[thread_id].x;
        unsigned int rank1 = D[index + 2 * size_D];
        unsigned int rank2 = ((index + prefix_length < size_D) ? D[(index + prefix_length) + 2 * size_D] : 0);
        // if ((index == 7 || index== 11 || index==13) && prefix_length>0) {
       //     printf("\n %d %d %d %d",index,rank1,rank2, index + prefix_length);
       // }
        unsigned long long rank = rank2;
        rank |= ((unsigned long long)rank1) << (32);
        Ranks_Tuple[thread_id] = rank;
    }
}

__global__ void Reduce_D_Suffix_Array1(int2* Suffix_Array, unsigned long long* Ranks_Tuple, int size_D_Suffix_Array_Sorting, int* Prefix_Sum1, int* Prefix_Sum2, int size_D) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id < size_D_Suffix_Array_Sorting) {

        unsigned long long rank = Ranks_Tuple[thread_id];
        bool cond1 = (thread_id == 0) || (thread_id > 0 && rank != Ranks_Tuple[thread_id - 1]);
        bool cond2 = (thread_id == size_D_Suffix_Array_Sorting - 1) || (thread_id < size_D_Suffix_Array_Sorting - 1 && rank != Ranks_Tuple[thread_id + 1]);
        if (cond1) {
            Prefix_Sum1[thread_id] = 1;
        }
        if (!(cond1 && cond2)) {
            Prefix_Sum2[thread_id] = 1;
        }

    }
}

__global__ void Reduce_D_Suffix_Array2(unsigned long long* Ranks_Tuple, int size_D_Suffix_Array_Sorting, int* Prefix_Sum1, int* Prefix_Sum3, int size_D) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id < size_D_Suffix_Array_Sorting) {
        unsigned long long rank = Ranks_Tuple[thread_id];
        bool cond1 = (thread_id == 0) || (thread_id > 0 && rank != Ranks_Tuple[thread_id - 1]);

        if (cond1) {
            int index = Prefix_Sum1[thread_id] - 1;
            Prefix_Sum3[index] = thread_id;

        }


    }
}

__global__ void Reduce_D_Suffix_Array3(int2* Suffix_Array, int2* Suffix_Array_New, int* Suffix_Array_Final, int* D, unsigned long long* Ranks_Tuple, int size_D_Suffix_Array_Sorting, int* Prefix_Sum1, int* Prefix_Sum2, int* Prefix_Sum3, int size_D) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id < size_D_Suffix_Array_Sorting) {

        int2 cur_val = Suffix_Array[thread_id];
        unsigned long long rank = Ranks_Tuple[thread_id];
        bool cond1 = (thread_id == 0) || (thread_id > 0 && rank != Ranks_Tuple[thread_id - 1]);
        bool cond2 = (thread_id == size_D_Suffix_Array_Sorting - 1) || (thread_id < size_D_Suffix_Array_Sorting - 1 && rank != Ranks_Tuple[thread_id + 1]);
        if (!(cond1 && cond2)) {

            int same_group_index = Prefix_Sum1[thread_id] - 1;
            int leftmost = Prefix_Sum3[same_group_index] + 1;
            int new_rank = leftmost + cur_val.y;
            D[(cur_val.x) + 2 * size_D] = new_rank;


            int index = Prefix_Sum2[thread_id] - 1;
            int diff = thread_id - index;
            cur_val.y += diff;
            Suffix_Array_New[index] = cur_val;

            // if (cur_val.x == 11)
             //   printf("\n ## %d %d %d %d", cur_val.x, same_group_index, leftmost, new_rank);


        }
        else {
            int rank_new = (thread_id + 1) + cur_val.y;
            Suffix_Array_Final[rank_new - 1] = cur_val.x;
            D[(cur_val.x) + 2 * size_D] = rank_new;

            //if (cur_val.x == 13)
              //  printf("\n %d %d %d", cur_val.x, rank_new, thread_id);

        }

    }
}




__global__ void Compute_D_Suffix_Array3(int* New_D_Suffix_Array, int* D, int size_D) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id < size_D) {

        int index = New_D_Suffix_Array[thread_id];

        if (index - 1 >= 0) {
            D[(index - 1) + 2 * size_D] = thread_id;
            //if (D[(index - 1)] == 84 || D[(index - 1)] == 88)
             //   printf("\n %d %d ", D[(index - 1)],D[index]);
        }

        if (thread_id == size_D - 1)
            D[(thread_id)+2 * size_D] = -1;


    }



}




__global__ void RadixSort2(int* Prefix_Sum1, int* Prefix_Sum2, int* Block_Group_Info, int* Sum_Group_Index, int n) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    int block_id = (thread_id) / 256;
    int rest = (thread_id)-block_id * 256;
    if (block_id < n) {
        int block_group_info = Block_Group_Info[block_id];
        int group_sum_index = Sum_Group_Index[block_id];
        int index1 = block_id - 1 + rest * n;
        int index2 = group_sum_index + rest * n;

        if ((block_id - 1 >= 0 && block_group_info != Block_Group_Info[block_id - 1]) || block_id == 0) {
            Prefix_Sum2[index2] = ((index1 >= 0) ? Prefix_Sum1[index1] : 0);
        }


    }
}

__global__ void RadixSort3(int* Prefix_Sum1, int* Prefix_Sum2, int* Sum_Group_Index, int n) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    int block_id = (thread_id) / 256;
    int rest = (thread_id)-block_id * 256;
    if (block_id < n) {

        int group_sum_index = Sum_Group_Index[block_id];
        int index1 = block_id + rest * n;
        int index2 = group_sum_index + rest * n;
        Prefix_Sum1[index1] -= Prefix_Sum2[index2];



    }
}

__global__ void RadixSort_D1(int* D_Suffix_Array, int* D_Suffix_Array_Rank, int* D_Suffix_Array_New, int* D_Suffix_Array_Rank_New, int* D_Suffix_Array_Final, unsigned char* Dev_Input, int* Block_Info_Radix_Sorting, int* Prefix_Sum, int* Block_Group_Info, int* Sum_Group_Index, int size_D_Sorting, int it, int size_Block_Info_Radix_Sorting, int size_D) {
    __shared__ unsigned char memory[3072];


    int l = Block_Info_Radix_Sorting[blockIdx.x];
    int r = Block_Info_Radix_Sorting[blockIdx.x + size_Block_Info_Radix_Sorting];

    int diff = r - l + 1;
    int two_potenz = smallest_two_potenz_larger_than_k_device(diff);
    bool cond = (blockIdx.x > 0 && Block_Group_Info[blockIdx.x] == Block_Group_Info[blockIdx.x - 1]);
    bool complete_sort = (Sum_Group_Index[blockIdx.x] == blockIdx.x) && (!cond);

    // if (blockIdx.x == 1)
      //  printf("sgrt");

    if (complete_sort && two_potenz <= 256) {
        uint8_t* pos = (uint8_t*)memory;
        unsigned long long* cur_prefix = (unsigned long long*)(memory + maximum(8, two_potenz));
        unsigned char prefix_thread[8];
        for (int i = threadIdx.x; i < two_potenz; i += 64) {
            pos[i] = i;
        }

        for (int i = threadIdx.x; i < two_potenz; i += 64) {
            if (l + i <= r) {
                int suffix_end = D_Suffix_Array[l + i];
                int index = suffix_end - D_Suffix_Array[l + i + size_D_Sorting] + 1 + it;
                for (int j = 0; j < 8; j++, index++)
                    prefix_thread[j] = ((index <= suffix_end) ? Dev_Input[index] : 0);

                cur_prefix[i] = pack8(prefix_thread);
            }
            else
                cur_prefix[i] = 0xFFFFFFFFFFFFFFFFULL;
        }

        for (int k = 2; k <= two_potenz; k <<= 1)
        {
            for (int j = k >> 1; j > 0; j = j >> 1)
            {
                __syncthreads();

                for (unsigned short i = threadIdx.x; i < two_potenz; i += 64) {
                    unsigned short ij;
                    ij = i ^ j;

                    if (ij > i)
                    {


                        unsigned long long val11 = cur_prefix[i];
                        uint8_t val12 = pos[i];


                        unsigned long long val21 = cur_prefix[ij];
                        uint8_t val22 = pos[ij];


                        bool cond = ((i & k) == 0);
                        bool val1_greater_then_val2;
                        bool val2_greater_then_val1;



                        val1_greater_then_val2 = ((val11) > (val21));
                        val2_greater_then_val1 = ((val11) < (val21));

                        if (complete_sort && val1_greater_then_val2 == false && val2_greater_then_val1 == false && l + val12 <= r && l + val22 <= r) {


                            int suffix_end1 = D_Suffix_Array[l + val12];
                            int suffix_length1 = D_Suffix_Array[l + val12 + size_D_Sorting];


                            int suffix_end2 = D_Suffix_Array[l + val22];
                            int suffix_length2 = D_Suffix_Array[l + val22 + size_D_Sorting];

                            for (int index_val1 = suffix_end1 - suffix_length1 + it + 9, index_val2 = suffix_end2 - suffix_length2 + it + 9; index_val1 <= suffix_end1 && index_val2 <= suffix_end2; index_val1++, index_val2++) {

                                unsigned char char_val1 = Dev_Input[index_val1];
                                unsigned char char_val2 = Dev_Input[index_val2];
                                if (char_val1 > char_val2) {
                                    val1_greater_then_val2 = true;
                                    break;
                                }
                                else if (char_val1 < char_val2) {
                                    val2_greater_then_val1 = true;
                                    break;
                                }
                            }

                            if (val1_greater_then_val2 == false && val2_greater_then_val1 == false) {
                                if (suffix_length1 > suffix_length2)
                                    val1_greater_then_val2 = true;
                                else if (suffix_length1 < suffix_length2)
                                    val2_greater_then_val1 = true;

                            }
                        }


                        if (((cond) && (val1_greater_then_val2))
                            || ((!cond) && (val2_greater_then_val1))) {

                            cur_prefix[i] = val21;
                            pos[i] = val22;


                            cur_prefix[ij] = val11;
                            pos[ij] = val12;

                        }


                    }

                }
            }
        }
        // if (blockIdx.x == 3)
          //   printf("sgrt");
        __syncthreads();
        for (int i = threadIdx.x; i < diff; i += 64) {

            int read_index = l + pos[i];

            int val1 = D_Suffix_Array[read_index];
            int val2 = D_Suffix_Array_Rank[read_index];
            int val3 = D_Suffix_Array[read_index + size_D_Sorting];
            int val4 = D_Suffix_Array[read_index + 2 * size_D_Sorting];

            val2 += i;
            D_Suffix_Array_Final[val2] = val1;
            D_Suffix_Array_Final[val2 + size_D] = val3;
            D_Suffix_Array_Final[val2 + 2 * size_D] = val4;


        }


    }
    else {

        short* pos = (short*)memory;
        unsigned char* cur_characters = (unsigned char*)(memory + 512 * 2);
        short* block_sums = (short*)(memory + 512 * 3);

        for (int i = threadIdx.x; i < two_potenz; i += 64) {
            pos[i] = i;
        }

        for (int i = threadIdx.x; i < 256; i += 64) {
            block_sums[i] = 0;
        }


        for (int i = threadIdx.x; i < two_potenz; i += 64) {
            if (l + i <= r) {

                int suffix_end = D_Suffix_Array[l + i];
                int index = suffix_end - D_Suffix_Array[l + i + size_D_Sorting] + 1 + it;

                cur_characters[i] = (((index <= suffix_end) ? Dev_Input[index] : 0));

            }
            else
                cur_characters[i] = 255;

        }
        for (int k = 2; k <= two_potenz; k <<= 1)
        {
            for (int j = k >> 1; j > 0; j = j >> 1)
            {
                __syncthreads();

                for (unsigned short i = threadIdx.x; i < two_potenz; i += 64) {
                    unsigned short ij;
                    ij = i ^ j;

                    if (ij > i)
                    {


                        unsigned char val11 = cur_characters[i];
                        short val12 = pos[i];


                        unsigned char val21 = cur_characters[ij];
                        short val22 = pos[ij];


                        bool cond = ((i & k) == 0);
                        bool val1_greater_then_val2 = ((val11) > (val21));
                        bool val2_greater_then_val1 = ((val11) < (val21));


                        if (((cond) && (val1_greater_then_val2))
                            || ((!cond) && (val2_greater_then_val1))) {

                            cur_characters[i] = val21;
                            pos[i] = val22;


                            cur_characters[ij] = val11;
                            pos[ij] = val12;

                        }


                    }

                }
            }
        }


        __syncthreads();
        for (int i = threadIdx.x; i < diff; i += 64) {

            if ((i == 0) || (i > 0 && cur_characters[i] != cur_characters[i - 1]))
                block_sums[cur_characters[i]] = i;
        }
        __syncthreads();
        for (int i = threadIdx.x; i < diff; i += 64) {
            if ((i == diff - 1) || (i < diff && cur_characters[i] != cur_characters[i + 1]))
                block_sums[cur_characters[i]] = i - block_sums[cur_characters[i]] + 1;
        }



        __syncthreads();

        for (int i = threadIdx.x; i < 256; i += 64) {
            if (block_sums[i] > 0)
                Prefix_Sum[blockIdx.x + i * size_Block_Info_Radix_Sorting] = block_sums[i];
            //if (i == 65) {
             //   printf("\n %d", block_sums[i]);
            //}
        }




        for (int i = threadIdx.x, cnt = 0; i < diff; i += 64, cnt++) {


            int read_index = l + pos[i];
            int write_index = l + i;

            D_Suffix_Array_New[write_index] = D_Suffix_Array[read_index];

            D_Suffix_Array_Rank_New[write_index] = D_Suffix_Array_Rank[read_index];

            D_Suffix_Array_New[write_index + size_D_Sorting] = D_Suffix_Array[read_index + size_D_Sorting];

            D_Suffix_Array_New[write_index + 2 * size_D_Sorting] = D_Suffix_Array[read_index + 2 * size_D_Sorting];



        }




    }




}




__global__ void RadixSort_D_RL_1(int* D_Suffix_Array, int* D_Suffix_Array_Rank, int* D_Suffix_Array_New, int* D_Suffix_Array_Rank_New, int* D_Suffix_Array_Final, unsigned char* Dev_Input, int* Block_Info_Radix_Sorting, int* Prefix_Sum, int* Block_Group_Info, int* Sum_Group_Index, int size_D_Sorting, int it, int size_Block_Info_Radix_Sorting, int size_D, int size_Dev_Input) {
    __shared__ char memory[3072];


    int l = Block_Info_Radix_Sorting[blockIdx.x];
    int r = Block_Info_Radix_Sorting[blockIdx.x + size_Block_Info_Radix_Sorting];

    int diff = r - l + 1;
    int two_potenz = smallest_two_potenz_larger_than_k_device(diff);
    bool cond = (blockIdx.x > 0 && Block_Group_Info[blockIdx.x] == Block_Group_Info[blockIdx.x - 1]);
    bool complete_sort = (Sum_Group_Index[blockIdx.x] == blockIdx.x) && (!cond);

    // if (blockIdx.x == 1)
      //  printf("sgrt");

    if (complete_sort && two_potenz <= 256) {
        uint8_t* pos = (uint8_t*)memory;
        unsigned long long* cur_prefix = (unsigned long long*)(memory + maximum(8, two_potenz));
        unsigned char prefix_thread[8];
        for (int i = threadIdx.x; i < two_potenz; i += 64) {
            pos[i] = i;
        }

        for (int i = threadIdx.x; i < two_potenz; i += 64) {
            if (l + i <= r) {
                int suffix_end = D_Suffix_Array[l + i];
                //if (suffix_end == 175827)
                //    printf("sdrgsdtg");
                int index = suffix_end - it;

                unsigned char cur_char = 0;
                for (int j = 0; j < 8; j++, index--) {
                    cur_char = ((index == -1) ? Dev_Input[size_Dev_Input - W - 1] : 0);
                    prefix_thread[j] = ((index >= 0) ? Dev_Input[index] : cur_char);
                }



                cur_prefix[i] = pack8(prefix_thread);
            }
            else
                cur_prefix[i] = 0xFFFFFFFFFFFFFFFFULL;
        }

        for (int k = 2; k <= two_potenz; k <<= 1)
        {
            for (int j = k >> 1; j > 0; j = j >> 1)
            {
                __syncthreads();

                for (unsigned short i = threadIdx.x; i < two_potenz; i += 64) {
                    unsigned short ij;
                    ij = i ^ j;

                    if (ij > i)
                    {


                        unsigned long long val11 = cur_prefix[i];
                        uint8_t val12 = pos[i];


                        unsigned long long val21 = cur_prefix[ij];
                        uint8_t val22 = pos[ij];


                        bool cond = ((i & k) == 0);
                        bool val1_greater_then_val2;
                        bool val2_greater_then_val1;



                        val1_greater_then_val2 = ((val11) > (val21));
                        val2_greater_then_val1 = ((val11) < (val21));

                        if (complete_sort && val1_greater_then_val2 == false && val2_greater_then_val1 == false && l + val12 <= r && l + val22 <= r) {


                            int suffix_end1 = D_Suffix_Array[l + val12];
                            int suffix_length1 = D_Suffix_Array[l + val12 + size_D_Sorting];


                            int suffix_end2 = D_Suffix_Array[l + val22];
                            int suffix_length2 = D_Suffix_Array[l + val22 + size_D_Sorting];

                            for (int index_val1 = suffix_end1 - (it + 8), index_val2 = suffix_end2 - (it + 8); index_val1 > suffix_end1 - suffix_length1 && index_val2 > suffix_end2 - suffix_length2; index_val1--, index_val2--) {

                                unsigned char char_val1 = Dev_Input[index_val1];
                                unsigned char char_val2 = Dev_Input[index_val2];
                                if (char_val1 > char_val2) {
                                    val1_greater_then_val2 = true;
                                    break;
                                }
                                else if (char_val1 < char_val2) {
                                    val2_greater_then_val1 = true;
                                    break;
                                }
                            }

                            if (val1_greater_then_val2 == false && val2_greater_then_val1 == false) {
                                if (suffix_length1 > suffix_length2)
                                    val1_greater_then_val2 = true;
                                else if (suffix_length1 < suffix_length2)
                                    val2_greater_then_val1 = true;
                                else {
                                    int index_val1 = (suffix_end1 - suffix_length1 >= 0) ? suffix_end1 - suffix_length1 : size_Dev_Input - W - 1;
                                    int index_val2 = (suffix_end2 - suffix_length2 >= 0) ? suffix_end2 - suffix_length2 : size_Dev_Input - W - 1;
                                    unsigned char char_val1 = Dev_Input[index_val1];
                                    unsigned char char_val2 = Dev_Input[index_val2];
                                    if (char_val1 > char_val2) {
                                        val1_greater_then_val2 = true;

                                    }
                                    else if (char_val1 < char_val2) {
                                        val2_greater_then_val1 = true;

                                    }
                                }

                            }
                        }


                        if (((cond) && (val1_greater_then_val2))
                            || ((!cond) && (val2_greater_then_val1))) {

                            cur_prefix[i] = val21;
                            pos[i] = val22;


                            cur_prefix[ij] = val11;
                            pos[ij] = val12;

                        }


                    }

                }
            }
        }
        // if (blockIdx.x == 3)
          //   printf("sgrt");
        __syncthreads();
        for (int i = threadIdx.x; i < diff; i += 64) {

            int read_index = l + pos[i];

            int val1 = D_Suffix_Array[read_index];
            int val2 = D_Suffix_Array_Rank[read_index];
            int val3 = D_Suffix_Array[read_index + size_D_Sorting];
            int val4 = D_Suffix_Array[read_index + 2 * size_D_Sorting];
            /*
            if (val1 == 175827) {
                for (int k = 0; k < diff; k++) {
                    read_index = l + pos[k];
                    int suffix_end = D_Suffix_Array[read_index];
                    int length_D = D_Suffix_Array[read_index + size_D_Sorting];
                    printf("\n");

                    for (int j = 0; j < length_D; j++) {
                        printf("%c ", Dev_Input[suffix_end-j]);
                    }
                    if(suffix_end - length_D>=0)
                       printf("%c ", Dev_Input[suffix_end - length_D]);
                    else
                        printf("%c ", Dev_Input[size_Dev_Input - W - 1]);

                    printf("%d %d", suffix_end, length_D);


               }
            }
            */

            val2 += i;
            D_Suffix_Array_Final[val2] = val1;
            D_Suffix_Array_Final[val2 + size_D] = val3;
            D_Suffix_Array_Final[val2 + 2 * size_D] = val4;


        }


    }
    else {

        short* pos = (short*)memory;
        unsigned char* cur_characters = (unsigned char*)(memory + 512 * 2);
        short* block_sums = (short*)(memory + 512 * 3);

        for (int i = threadIdx.x; i < two_potenz; i += 64) {
            pos[i] = i;
        }

        for (int i = threadIdx.x; i < 256; i += 64) {
            block_sums[i] = 0;
        }


        for (int i = threadIdx.x; i < two_potenz; i += 64) {
            if (l + i <= r) {

                int suffix_end = D_Suffix_Array[l + i];

                int index = suffix_end - it;

                index = (index >= 0) ? index : size_Dev_Input - W - 1;
                cur_characters[i] = Dev_Input[index];


            }
            else
                cur_characters[i] = 255;
        }
        for (int k = 2; k <= two_potenz; k <<= 1)
        {
            for (int j = k >> 1; j > 0; j = j >> 1)
            {
                __syncthreads();

                for (unsigned short i = threadIdx.x; i < two_potenz; i += 64) {
                    unsigned short ij;
                    ij = i ^ j;

                    if (ij > i)
                    {


                        unsigned char val11 = cur_characters[i];
                        short val12 = pos[i];


                        unsigned char val21 = cur_characters[ij];
                        short val22 = pos[ij];


                        bool cond = ((i & k) == 0);
                        bool val1_greater_then_val2 = ((val11) > (val21));
                        bool val2_greater_then_val1 = ((val11) < (val21));


                        if (((cond) && (val1_greater_then_val2))
                            || ((!cond) && (val2_greater_then_val1))) {

                            cur_characters[i] = val21;
                            pos[i] = val22;


                            cur_characters[ij] = val11;
                            pos[ij] = val12;

                        }


                    }

                }
            }
        }


        __syncthreads();
        for (int i = threadIdx.x; i < diff; i += 64) {

            if ((i == 0) || (i > 0 && cur_characters[i] != cur_characters[i - 1]))
                block_sums[cur_characters[i]] = i;
        }
        __syncthreads();
        for (int i = threadIdx.x; i < diff; i += 64) {
            if ((i == diff - 1) || (i < diff && cur_characters[i] != cur_characters[i + 1]))
                block_sums[cur_characters[i]] = i - block_sums[cur_characters[i]] + 1;
        }



        __syncthreads();

        for (int i = threadIdx.x; i < 256; i += 64) {
            if (block_sums[i] > 0)
                Prefix_Sum[blockIdx.x + i * size_Block_Info_Radix_Sorting] = block_sums[i];
            //if (i == 65) {
             //   printf("\n %d", block_sums[i]);
            //}
        }

        for (int i = threadIdx.x, cnt = 0; i < diff; i += 64, cnt++) {


            int read_index = l + pos[i];
            int write_index = l + i;

            D_Suffix_Array_New[write_index] = D_Suffix_Array[read_index];

            D_Suffix_Array_Rank_New[write_index] = D_Suffix_Array_Rank[read_index];

            D_Suffix_Array_New[write_index + size_D_Sorting] = D_Suffix_Array[read_index + size_D_Sorting];

            D_Suffix_Array_New[write_index + 2 * size_D_Sorting] = D_Suffix_Array[read_index + 2 * size_D_Sorting];



        }



    }




}






__global__ void RadixSort_D4(int* D_Suffix_Array, int* D_Suffix_Array_Rank, int* D_Suffix_Array_New, int* D_Suffix_Array_Rank_New, int* D_Suffix_Array_Final, unsigned char* Dev_Input, int* Block_Info_Radix_Sorting, int* Prefix_Sum, int* Block_Group_Info, int* Sum_Group_Index, int* Group_Offset, int size_D_Sorting, int it, int size_Block_Info_Radix_Sorting, int size_D) {

    __shared__ unsigned char cur_characters[512];

    __shared__ int in_block_sums[256];
    __shared__ int block_sums_total[256];
    __shared__ bool all_equal[64];
    int thread_id = threadIdx.x;
    int l = Block_Info_Radix_Sorting[blockIdx.x];
    int r = Block_Info_Radix_Sorting[blockIdx.x + size_Block_Info_Radix_Sorting];

    int diff = r - l + 1;
    int two_potenz = smallest_two_potenz_larger_than_k_device(diff);
    int group_sum_index = Sum_Group_Index[blockIdx.x];
    bool cond = (blockIdx.x > 0 && Block_Group_Info[blockIdx.x] == Block_Group_Info[blockIdx.x - 1]);
    bool complete_sort = (group_sum_index == blockIdx.x) && (!cond);
    int group_offset = Group_Offset[blockIdx.x];


    all_equal[thread_id] = true;

    __syncthreads();
    if (!(complete_sort && two_potenz <= 256)) {

        for (int i = threadIdx.x; i < two_potenz; i += 64) {
            if (l + i <= r) {

                int suffix_end = D_Suffix_Array[l + i];
                int index = suffix_end - D_Suffix_Array[l + i + size_D_Sorting] + 1 + it;

                cur_characters[i] = (((index <= suffix_end) ? Dev_Input[index] : 0));
                if (index <= suffix_end)
                    all_equal[thread_id] = false;
            }
            else
                cur_characters[i] = 255;
        }
        __syncthreads();

        for (int stride = 32; stride > 0; stride >>= 1) {
            __syncthreads();
            if (thread_id < stride && all_equal[thread_id + stride] == false) {
                all_equal[thread_id] = false;
            }
        }
        //if (all_equal[0] == false && complete_sort == false)
           // printf("sdvgbsd");


        for (int i = thread_id; i < 256; i += 64) {
            in_block_sums[i] = 0;
        }

        // for (int i = thread_id; i < 128; i += 64) {
         //    block_sums_left[i] = (cond ? Prefix_Sum[blockIdx.x - 1 + i * size_Block_Info_Radix_Sorting] : 0);

        // }


        for (int i = thread_id; i < 256; i += 64) {
            block_sums_total[i] = Prefix_Sum[group_sum_index + i * size_Block_Info_Radix_Sorting];

        }



        for (int stride = 1; stride < 256; stride <<= 1) {

            __syncthreads();

            int val1;
            int val2;
            int val3;
            int val4;

            if (thread_id - stride >= 0)
                val1 = block_sums_total[thread_id - stride];

            if (thread_id + 64 - stride >= 0)
                val2 = block_sums_total[thread_id + 64 - stride];

            if (thread_id + 128 - stride >= 0)
                val3 = block_sums_total[thread_id + 128 - stride];

            if (thread_id + 192 - stride >= 0)
                val4 = block_sums_total[thread_id + 192 - stride];

            __syncthreads();

            if (thread_id - stride >= 0)
                block_sums_total[thread_id] = block_sums_total[thread_id] + val1;

            if (thread_id + 64 - stride >= 0)
                block_sums_total[thread_id + 64] = block_sums_total[thread_id + 64] + val2;

            if (thread_id + 128 - stride >= 0)
                block_sums_total[thread_id + 128] = block_sums_total[thread_id + 128] + val3;

            if (thread_id + 192 - stride >= 0)
                block_sums_total[thread_id + 192] = block_sums_total[thread_id + 192] + val4;
        }




        __syncthreads();
        for (int i = thread_id; i < diff; i += 64) {

            if ((i == 0) || (i > 0 && cur_characters[i] != cur_characters[i - 1]))
                in_block_sums[cur_characters[i]] = i;
        }




        __syncthreads();
        if (all_equal[0] == false) {
            for (int i = thread_id; i < diff; i += 64) {

                int read_index = l + i;

                int val1 = D_Suffix_Array[read_index];
                int val2 = D_Suffix_Array_Rank[read_index];
                int val3 = D_Suffix_Array[read_index + size_D_Sorting];
                int val4 = D_Suffix_Array[read_index + 2 * size_D_Sorting];

                int cur_character = cur_characters[i];
                int in_block_offset = (i - in_block_sums[cur_characters[i]]);
                int in_group_cur_character_left_offset = ((cond) ? Prefix_Sum[blockIdx.x - 1 + cur_characters[i] * size_Block_Info_Radix_Sorting] : 0);
                int in_group_smaller_then_cur_character_offset = ((cur_character - 1 >= 0) ? block_sums_total[cur_character - 1] : 0);
                int write_index;

                val2 += in_group_smaller_then_cur_character_offset;
                write_index = in_block_offset + in_group_cur_character_left_offset + in_group_smaller_then_cur_character_offset + group_offset;

                //printf("\n %d %d %d %d", val5, cur_characters[i],i,blockIdx.x);

                D_Suffix_Array_New[write_index] = val1;
                D_Suffix_Array_Rank_New[write_index] = val2;
                D_Suffix_Array_New[write_index + size_D_Sorting] = val3;
                D_Suffix_Array_New[write_index + 2 * size_D_Sorting] = val4;

                // if (write_index == 154) {
                    //   printf("\n f %d %d %d %d %d", val5, in_block_offset, in_group_cur_character_left_offset, in_group_smaller_then_cur_character_offset, group_offset);
                    //}

                // if (blockIdx.x==1) {
                    //    printf("\n %d %d %d %d %d %d",val5, write_index, in_block_offset, in_group_cur_character_left_offset, in_group_smaller_then_cur_character_offset, group_offset);
                    //}

            }
        }
        else {
            for (int i = thread_id; i < diff; i += 64) {

                int read_index = l + i;

                int val1 = D_Suffix_Array[read_index];
                int val2 = D_Suffix_Array_Rank[read_index];
                int val3 = D_Suffix_Array[read_index + size_D_Sorting];
                int val4 = D_Suffix_Array[read_index + 2 * size_D_Sorting];



                int in_group_cur_character_left_offset = ((cond) ? Prefix_Sum[blockIdx.x - 1 + cur_characters[i] * size_Block_Info_Radix_Sorting] : 0);


                val2 += i + in_group_cur_character_left_offset;


                //printf("\n %d %d %d %d", val5, cur_characters[i],i,blockIdx.x);

                D_Suffix_Array_Rank_New[read_index] = 1e9;
                D_Suffix_Array_Final[val2] = val1;
                D_Suffix_Array_Final[val2 + size_D] = val3;
                D_Suffix_Array_Final[val2 + 2 * size_D] = val4;

                // if (write_index == 154) {
                    //   printf("\n f %d %d %d %d %d", val5, in_block_offset, in_group_cur_character_left_offset, in_group_smaller_then_cur_character_offset, group_offset);
                    //}

                // if (blockIdx.x==1) {
                    //    printf("\n %d %d %d %d %d %d",val5, write_index, in_block_offset, in_group_cur_character_left_offset, in_group_smaller_then_cur_character_offset, group_offset);
                    //}

            }
        }


    }
    else {
        for (int i = thread_id; i < diff; i += 64) {

            int write_index = l + i;
            D_Suffix_Array_Rank_New[write_index] = 1e9;

        }

    }

}

__global__ void RadixSort_D_RL_4(int* D_Suffix_Array, int* D_Suffix_Array_Rank, int* D_Suffix_Array_New, int* D_Suffix_Array_Rank_New, int* D_Suffix_Array_Final, unsigned char* Dev_Input, int* Block_Info_Radix_Sorting, int* Prefix_Sum, int* Block_Group_Info, int* Sum_Group_Index, int* Group_Offset, int size_D_Sorting, int it, int size_Block_Info_Radix_Sorting, int size_D, int size_Dev_Input) {

    __shared__ unsigned char cur_characters[512];

    __shared__ int in_block_sums[256];
    __shared__ int block_sums_total[256];
    __shared__ bool all_equal[64];
    int thread_id = threadIdx.x;
    int l = Block_Info_Radix_Sorting[blockIdx.x];
    int r = Block_Info_Radix_Sorting[blockIdx.x + size_Block_Info_Radix_Sorting];

    int diff = r - l + 1;
    int two_potenz = smallest_two_potenz_larger_than_k_device(diff);
    int group_sum_index = Sum_Group_Index[blockIdx.x];
    bool cond = (blockIdx.x > 0 && Block_Group_Info[blockIdx.x] == Block_Group_Info[blockIdx.x - 1]);
    bool complete_sort = (group_sum_index == blockIdx.x) && (!cond);
    int group_offset = Group_Offset[blockIdx.x];


    all_equal[thread_id] = true;

    __syncthreads();
    if (!(complete_sort && two_potenz <= 256)) {

        for (int i = threadIdx.x; i < two_potenz; i += 64) {
            if (l + i <= r) {

                int suffix_end = D_Suffix_Array[l + i];


                int index = suffix_end - it;
                int length_D = D_Suffix_Array[l + i + size_D_Sorting];
                if (length_D > it) {
                    all_equal[thread_id] = false;
                }

                index = (index >= 0) ? index : size_Dev_Input - W - 1;
                cur_characters[i] = Dev_Input[index];


            }
            else
                cur_characters[i] = 255;
        }
        __syncthreads();

        for (int stride = 32; stride > 0; stride >>= 1) {
            __syncthreads();
            if (thread_id < stride && all_equal[thread_id + stride] == false) {
                all_equal[thread_id] = false;
            }
        }
        //if (all_equal[0] == false && complete_sort == false)
           // printf("sdvgbsd");


        for (int i = thread_id; i < 256; i += 64) {
            in_block_sums[i] = 0;
        }

        // for (int i = thread_id; i < 128; i += 64) {
         //    block_sums_left[i] = (cond ? Prefix_Sum[blockIdx.x - 1 + i * size_Block_Info_Radix_Sorting] : 0);

        // }


        for (int i = thread_id; i < 256; i += 64) {
            block_sums_total[i] = Prefix_Sum[group_sum_index + i * size_Block_Info_Radix_Sorting];

        }



        for (int stride = 1; stride < 256; stride <<= 1) {

            __syncthreads();

            int val1;
            int val2;
            int val3;
            int val4;

            if (thread_id - stride >= 0)
                val1 = block_sums_total[thread_id - stride];

            if (thread_id + 64 - stride >= 0)
                val2 = block_sums_total[thread_id + 64 - stride];

            if (thread_id + 128 - stride >= 0)
                val3 = block_sums_total[thread_id + 128 - stride];

            if (thread_id + 192 - stride >= 0)
                val4 = block_sums_total[thread_id + 192 - stride];

            __syncthreads();

            if (thread_id - stride >= 0)
                block_sums_total[thread_id] = block_sums_total[thread_id] + val1;

            if (thread_id + 64 - stride >= 0)
                block_sums_total[thread_id + 64] = block_sums_total[thread_id + 64] + val2;

            if (thread_id + 128 - stride >= 0)
                block_sums_total[thread_id + 128] = block_sums_total[thread_id + 128] + val3;

            if (thread_id + 192 - stride >= 0)
                block_sums_total[thread_id + 192] = block_sums_total[thread_id + 192] + val4;
        }




        __syncthreads();
        for (int i = thread_id; i < diff; i += 64) {

            if ((i == 0) || (i > 0 && cur_characters[i] != cur_characters[i - 1]))
                in_block_sums[cur_characters[i]] = i;
        }




        __syncthreads();
        if (all_equal[0] == false) {
            for (int i = thread_id; i < diff; i += 64) {

                int read_index = l + i;

                int val1 = D_Suffix_Array[read_index];
                int val2 = D_Suffix_Array_Rank[read_index];
                int val3 = D_Suffix_Array[read_index + size_D_Sorting];
                int val4 = D_Suffix_Array[read_index + 2 * size_D_Sorting];

                int cur_character = cur_characters[i];
                int in_block_offset = (i - in_block_sums[cur_characters[i]]);
                int in_group_cur_character_left_offset = ((cond) ? Prefix_Sum[blockIdx.x - 1 + cur_characters[i] * size_Block_Info_Radix_Sorting] : 0);
                int in_group_smaller_then_cur_character_offset = ((cur_character - 1 >= 0) ? block_sums_total[cur_character - 1] : 0);
                int write_index;

                val2 += in_group_smaller_then_cur_character_offset;
                write_index = in_block_offset + in_group_cur_character_left_offset + in_group_smaller_then_cur_character_offset + group_offset;

                //printf("\n %d %d %d %d", val5, cur_characters[i],i,blockIdx.x);

                D_Suffix_Array_New[write_index] = val1;
                D_Suffix_Array_Rank_New[write_index] = val2;
                D_Suffix_Array_New[write_index + size_D_Sorting] = val3;
                D_Suffix_Array_New[write_index + 2 * size_D_Sorting] = val4;

                // if (write_index == 154) {
                    //   printf("\n f %d %d %d %d %d", val5, in_block_offset, in_group_cur_character_left_offset, in_group_smaller_then_cur_character_offset, group_offset);
                    //}

                // if (blockIdx.x==1) {
                    //    printf("\n %d %d %d %d %d %d",val5, write_index, in_block_offset, in_group_cur_character_left_offset, in_group_smaller_then_cur_character_offset, group_offset);
                    //}

            }
        }
        else {

            //printf("\n %d", diff);
            for (int i = thread_id; i < diff; i += 64) {

                int read_index = l + i;

                int val1 = D_Suffix_Array[read_index];
                int val2 = D_Suffix_Array_Rank[read_index];
                int val3 = D_Suffix_Array[read_index + size_D_Sorting];
                int val4 = D_Suffix_Array[read_index + 2 * size_D_Sorting];



                int cur_character = cur_characters[i];
                int in_group_cur_character_left_offset = ((cond) ? Prefix_Sum[blockIdx.x - 1 + cur_characters[i] * size_Block_Info_Radix_Sorting] : 0);
                int in_group_smaller_then_cur_character_offset = ((cur_character - 1 >= 0) ? block_sums_total[cur_character - 1] : 0);
                int in_block_offset = (i - in_block_sums[cur_characters[i]]);

                val2 += in_block_offset + in_group_cur_character_left_offset + in_group_smaller_then_cur_character_offset;


                //printf("\n %d %d %d %d", val5, cur_characters[i],i,blockIdx.x);

                D_Suffix_Array_Rank_New[read_index] = 1e9;
                D_Suffix_Array_Final[val2] = val1;
                D_Suffix_Array_Final[val2 + size_D] = val3;
                D_Suffix_Array_Final[val2 + 2 * size_D] = val4;

                // if (write_index == 154) {
                    //   printf("\n f %d %d %d %d %d", val5, in_block_offset, in_group_cur_character_left_offset, in_group_smaller_then_cur_character_offset, group_offset);
                    //}

                // if (blockIdx.x==1) {
                    //    printf("\n %d %d %d %d %d %d",val5, write_index, in_block_offset, in_group_cur_character_left_offset, in_group_smaller_then_cur_character_offset, group_offset);
                    //}

            }
        }


    }
    else {
        for (int i = thread_id; i < diff; i += 64) {

            int write_index = l + i;
            D_Suffix_Array_Rank_New[write_index] = 1e9;

        }

    }

}

__global__ void Reduce_D1(int* D_Suffix_Array_Rank, int* Prefix_Sum, int size_D_Sorting, int size_D) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id < size_D_Sorting) {

        unsigned int val2 = D_Suffix_Array_Rank[thread_id];//curr rank;

        if (val2 != 1e9) {
            Prefix_Sum[thread_id] = 1;
        }
    }
}

__global__ void Reduce_D2(int* D_Suffix_Array, int* D_Suffix_Array_Rank, int* D_Suffix_Array_New, int* D_Suffix_Array_Rank_New, int* Prefix_Sum, int* Prefix_Sum2, int size_D_Sorting, int size_D_new) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id < size_D_Sorting) {

        int val1 = D_Suffix_Array[thread_id];
        int val2 = D_Suffix_Array_Rank[thread_id];
        int val3 = D_Suffix_Array[thread_id + size_D_Sorting];
        int val4 = D_Suffix_Array[thread_id + 2 * size_D_Sorting];


        bool cond2 = (thread_id == size_D_Sorting - 1) || ((thread_id < size_D_Sorting - 1) && val2 != D_Suffix_Array_Rank[thread_id + 1]);

        if (val2 != 1e9) {
            int write_index = ((thread_id > 0) ? Prefix_Sum[thread_id - 1] : 0);
            D_Suffix_Array_New[write_index] = val1;
            D_Suffix_Array_Rank_New[write_index] = val2;
            D_Suffix_Array_New[write_index + size_D_new] = val3;
            D_Suffix_Array_New[write_index + 2 * size_D_new] = val4;
            if (cond2 && write_index + 1 < size_D_new) {
                Prefix_Sum2[write_index + 1] = 1;
                // printf("\n %d %d %d %d", val5, thread_id, size_S_new, S[thread_id + 1 + 4 * size_S_Sorting]);
            }
        }
    }
}



__global__ void Reduce_3(int* Prefix_Sum1, int* Prefix_Sum2, int size_S_Sorting) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id < size_S_Sorting) {
        int sum = Prefix_Sum1[thread_id];
        if ((thread_id < size_S_Sorting - 1 && sum != Prefix_Sum1[thread_id + 1]) || (thread_id == size_S_Sorting - 1))
            Prefix_Sum2[sum] = thread_id;


    }
}

__global__ void Reduce_4(int* Prefix_Sum1, int* Prefix_Sum2, int size_S_Sorting) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id < size_S_Sorting) {
        int sum = Prefix_Sum1[thread_id];
        Prefix_Sum1[thread_id] = Prefix_Sum2[sum];



    }
}


__global__ void Reduce_5(int* Prefix_Sum, int* Pointer_Jumping, int size_S_Sorting) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id < size_S_Sorting) {


        int group_info = Pointer_Jumping[thread_id];
        int diff = group_info - thread_id + 1;

        bool cond = (thread_id == 0) || ((thread_id > 0) && group_info != Pointer_Jumping[thread_id - 1]);
        if (diff % 512 == 0 || cond) {
            Prefix_Sum[thread_id] = 1;
            // printf("\n %d ", thread_id);
        }

    }
}

__global__ void Reduce_6(int* Prefix_Sum, int* Pointer_Jumping, int* Block_Info_Radix_Sorting, int* Block_Group_Info, int* Sum_Group_Index, int* Group_Offset_Temp, int size_S_Sorting, int size_Block_Info_Radix_Sorting) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id < size_S_Sorting) {


        int group_info = Pointer_Jumping[thread_id];
        int diff = group_info - thread_id + 1;
        int rest = diff % 512;
        bool cond1 = (thread_id == 0) || ((thread_id > 0) && group_info != Pointer_Jumping[thread_id - 1]);
        bool cond2 = rest == 0;
        if (cond2 || cond1) {

            int index = ((thread_id > 0) ? Prefix_Sum[thread_id - 1] : 0);
            Block_Group_Info[index] = group_info;
            Block_Info_Radix_Sorting[index] = thread_id;
            Block_Info_Radix_Sorting[index + size_Block_Info_Radix_Sorting] = ((cond2) ? thread_id + 512 - 1 : (thread_id + rest - 1));
            Sum_Group_Index[index] = Prefix_Sum[group_info] - 1;
            if (cond1)
                Group_Offset_Temp[group_info] = thread_id;
        }

    }
}

__global__ void Reduce_7(int* Prefix_Sum, int* Pointer_Jumping, int* Group_Offset_Temp, int* Group_Offset, int size_S_Sorting, int size_Block_Info_Radix_Sorting) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id < size_S_Sorting) {


        int group_info = Pointer_Jumping[thread_id];
        int diff = group_info - thread_id + 1;
        int rest = diff % 512;
        bool cond1 = (thread_id == 0) || ((thread_id > 0) && group_info != Pointer_Jumping[thread_id - 1]);
        bool cond2 = rest == 0;
        if (cond2 || cond1) {

            int index = ((thread_id > 0) ? Prefix_Sum[thread_id - 1] : 0);

            Group_Offset[index] = Group_Offset_Temp[group_info];

        }

    }
}

__global__ void Init_Block_Info_Radix_Sorting(int* Block_Info_Radix_Sorting, int* Block_Group_Info, int* Sum_Group_Index, int* Group_Offset, int size_Block_Info_Radix_Sorting, int size_S_Sorting) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id < size_Block_Info_Radix_Sorting) {
        Block_Group_Info[thread_id] = 0;
        Block_Info_Radix_Sorting[thread_id] = (thread_id) * 512;
        Block_Info_Radix_Sorting[thread_id + size_Block_Info_Radix_Sorting] = ((thread_id + 1) * 512 < size_S_Sorting) ? (thread_id + 1) * 512 - 1 : size_S_Sorting - 1;
        Sum_Group_Index[thread_id] = size_Block_Info_Radix_Sorting - 1;
        Group_Offset[thread_id] = 0;
    }
}

void Sort_D_Left_To_Rigth() {
    int size_D_Sorting = size_D;
    int size_Block_Info_Radix_Sorting = (size_D_Sorting + 512 - 1) / 512;
    int size_Auxiliar1 = 0;

    int* D_Suffix_Array_New;
    int* D_Suffix_Array_Rank;
    int* D_Suffix_Array_Rank_New;
    int* D_Suffix_Array_Final;
    cudaMalloc((void**)&D_Suffix_Array_New, (size_D * 3) * sizeof(int));
    cudaMalloc((void**)&D_Suffix_Array_Rank, (size_D) * sizeof(int));
    cudaMalloc((void**)&D_Suffix_Array_Rank_New, (size_D) * sizeof(int));
    cudaMalloc((void**)&D_Suffix_Array_Final, (size_D * 3) * sizeof(int));
    cudaMalloc((void**)&Block_Info_Radix_Sorting, (size_Block_Info_Radix_Sorting * 2) * sizeof(int));
    cudaMalloc((void**)&Block_Group_Info, size_Block_Info_Radix_Sorting * sizeof(int));
    cudaMalloc((void**)&Sum_Group_Index, size_Block_Info_Radix_Sorting * sizeof(int));
    cudaMalloc((void**)&Group_Offset, size_Block_Info_Radix_Sorting * sizeof(int));

    Init_Block_Info_Radix_Sorting << < (size_Block_Info_Radix_Sorting + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK >> > (Block_Info_Radix_Sorting, Block_Group_Info, Sum_Group_Index, Group_Offset, size_Block_Info_Radix_Sorting, size_D);

    Init_Vector << < (size_D_Sorting + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK >> > (D_Suffix_Array_Rank, size_D_Sorting, 0);

    int it = 0;
    while (size_D_Sorting > 0) {


        int size_Block_Info_Radix_Sorting_256 = size_Block_Info_Radix_Sorting * 256;
        //printf("\n %d %d %d %d", size_S_Sorting, size_Block_Info_Radix_Sorting, size_Block_Info_Radix_Sorting_256,it);
        if (size_Auxiliar1 < maximum1(size_Block_Info_Radix_Sorting_256, size_D_Sorting)) {

            if (size_Auxiliar1 > 0) {
                cudaFree(Auxiliar1);
                cudaFree(Auxiliar2);
                cudaFree(Auxiliar3);
            }
            size_Auxiliar1 = maximum1(size_Block_Info_Radix_Sorting_256, size_D_Sorting);
            cudaMalloc((void**)&Auxiliar1, maximum1(size_Block_Info_Radix_Sorting_256, size_D_Sorting) * sizeof(int));
            cudaMalloc((void**)&Auxiliar2, maximum1(size_Block_Info_Radix_Sorting_256, size_D_Sorting) * sizeof(int));
            cudaMalloc((void**)&Auxiliar3, maximum1(size_Block_Info_Radix_Sorting_256, size_D_Sorting) * sizeof(int));
        }
        int num_blocks_size_Block_Info_Radix_Sorting_256 = (size_Block_Info_Radix_Sorting_256 + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
        int num_blocks_size_D_Sorting = (size_D_Sorting + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;

        Init_Vector << <num_blocks_size_Block_Info_Radix_Sorting_256, MAX_THREADS_PER_BLOCK >> > (Auxiliar1, size_Block_Info_Radix_Sorting_256, 0);
        //Init_Vector << <num_blocks_size_Block_Info_Radix_Sorting_256, MAX_THREADS_PER_BLOCK >> > (Auxiliar2, size_Block_Info_Radix_Sorting_256, 0);

        RadixSort_D1 << <size_Block_Info_Radix_Sorting, 64 >> > (D_Suffix_Array, D_Suffix_Array_Rank, D_Suffix_Array_New, D_Suffix_Array_Rank_New, D_Suffix_Array_Final, Dev_Input, Block_Info_Radix_Sorting, Auxiliar1, Block_Group_Info, Sum_Group_Index, size_D_Sorting, it, size_Block_Info_Radix_Sorting, size_D);


        swap(D_Suffix_Array, D_Suffix_Array_New);
        swap(D_Suffix_Array_Rank, D_Suffix_Array_Rank_New);
        //Prefix_Sum_Radix_Sort(Auxiliar1, Auxiliar2, Block_Group_Info, Res, size_Block_Info_Radix_Sorting);

        thrust::inclusive_scan(
            thrust::device_pointer_cast(Auxiliar1),
            thrust::device_pointer_cast(Auxiliar1 + size_Block_Info_Radix_Sorting_256),
            thrust::device_pointer_cast(Auxiliar1)
        );

        RadixSort2 << < num_blocks_size_Block_Info_Radix_Sorting_256, MAX_THREADS_PER_BLOCK >> > (Auxiliar1, Auxiliar2, Block_Group_Info, Sum_Group_Index, size_Block_Info_Radix_Sorting);

        RadixSort3 << < num_blocks_size_Block_Info_Radix_Sorting_256, MAX_THREADS_PER_BLOCK >> > (Auxiliar1, Auxiliar2, Sum_Group_Index, size_Block_Info_Radix_Sorting);

        RadixSort_D4 << <size_Block_Info_Radix_Sorting, 64 >> > (D_Suffix_Array, D_Suffix_Array_Rank, D_Suffix_Array_New, D_Suffix_Array_Rank_New, D_Suffix_Array_Final, Dev_Input, Block_Info_Radix_Sorting, Auxiliar1, Block_Group_Info, Sum_Group_Index, Group_Offset, size_D_Sorting, it, size_Block_Info_Radix_Sorting, size_D);

        swap(D_Suffix_Array, D_Suffix_Array_New);
        swap(D_Suffix_Array_Rank, D_Suffix_Array_Rank_New);

        Init_Vector << <num_blocks_size_D_Sorting, MAX_THREADS_PER_BLOCK >> > (Auxiliar1, size_D_Sorting, 0);
        //Init_Vector << <num_blocks_size_S_Sorting, MAX_THREADS_PER_BLOCK >> > (Auxiliar2, size_S_Sorting, 0);

        Reduce_D1 << < num_blocks_size_D_Sorting, MAX_THREADS_PER_BLOCK >> > (D_Suffix_Array_Rank, Auxiliar1, size_D_Sorting, size_D);

        thrust::inclusive_scan(
            thrust::device_pointer_cast(Auxiliar1),
            thrust::device_pointer_cast(Auxiliar1 + size_D_Sorting),
            thrust::device_pointer_cast(Auxiliar1)
        );


        int size_D_New;

        cudaMemcpy(&size_D_New, Auxiliar1 + (size_D_Sorting - 1), sizeof(int), cudaMemcpyDeviceToHost);


        if (size_D_New == 0)
            break;

        Init_Vector << <(size_D_New + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK >> > (Auxiliar2, size_D_New, 0);
        //Init_Vector << <(size_S_New + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK >> > (Auxiliar3, size_S_New, 0);

        Reduce_D2 << < num_blocks_size_D_Sorting, MAX_THREADS_PER_BLOCK >> > (D_Suffix_Array, D_Suffix_Array_Rank, D_Suffix_Array_New, D_Suffix_Array_Rank_New, Auxiliar1, Auxiliar2, size_D_Sorting, size_D_New);

        size_D_Sorting = size_D_New;
        num_blocks_size_D_Sorting = (size_D_Sorting + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;


        swap(D_Suffix_Array, D_Suffix_Array_New);
        swap(D_Suffix_Array_Rank, D_Suffix_Array_Rank_New);


        thrust::inclusive_scan(
            thrust::device_pointer_cast(Auxiliar2),
            thrust::device_pointer_cast(Auxiliar2 + size_D_Sorting),
            thrust::device_pointer_cast(Auxiliar2)
        );

        Reduce_3 << < num_blocks_size_D_Sorting, MAX_THREADS_PER_BLOCK >> > (Auxiliar2, Auxiliar1, size_D_Sorting);

        Reduce_4 << < num_blocks_size_D_Sorting, MAX_THREADS_PER_BLOCK >> > (Auxiliar2, Auxiliar1, size_D_Sorting);

        Init_Vector << <num_blocks_size_D_Sorting, MAX_THREADS_PER_BLOCK >> > (Auxiliar1, size_D_Sorting, 0);
        //Init_Vector << <num_blocks_size_S_Sorting, MAX_THREADS_PER_BLOCK >> > (Auxiliar3, size_S_Sorting, 0);

        Reduce_5 << < num_blocks_size_D_Sorting, MAX_THREADS_PER_BLOCK >> > (Auxiliar1, Auxiliar2, size_D_Sorting);

        thrust::inclusive_scan(
            thrust::device_pointer_cast(Auxiliar1),
            thrust::device_pointer_cast(Auxiliar1 + size_D_Sorting),
            thrust::device_pointer_cast(Auxiliar1)
        );

        int size_Block_Info_Radix_Sorting_New;

        cudaMemcpy(&size_Block_Info_Radix_Sorting_New, Auxiliar1 + (size_D_Sorting - 1), sizeof(int), cudaMemcpyDeviceToHost);


        if (size_Block_Info_Radix_Sorting < size_Block_Info_Radix_Sorting_New) {
            cudaFree(Block_Info_Radix_Sorting);
            cudaFree(Block_Group_Info);
            cudaFree(Sum_Group_Index);
            cudaFree(Group_Offset);
            cudaMalloc((void**)&Block_Info_Radix_Sorting, (size_Block_Info_Radix_Sorting_New * 2) * sizeof(int));
            cudaMalloc((void**)&Block_Group_Info, size_Block_Info_Radix_Sorting_New * sizeof(int));
            cudaMalloc((void**)&Sum_Group_Index, size_Block_Info_Radix_Sorting_New * sizeof(int));
            cudaMalloc((void**)&Group_Offset, size_Block_Info_Radix_Sorting_New * sizeof(int));

        }


        size_Block_Info_Radix_Sorting = size_Block_Info_Radix_Sorting_New;
        Reduce_6 << < num_blocks_size_D_Sorting, MAX_THREADS_PER_BLOCK >> > (Auxiliar1, Auxiliar2, Block_Info_Radix_Sorting, Block_Group_Info, Sum_Group_Index, Auxiliar3, size_D_Sorting, size_Block_Info_Radix_Sorting);

        Reduce_7 << <num_blocks_size_D_Sorting, MAX_THREADS_PER_BLOCK >> > (Auxiliar1, Auxiliar2, Auxiliar3, Group_Offset, size_D_Sorting, size_Block_Info_Radix_Sorting);
        it++;

    }






    cudaFree(Auxiliar3);
    cudaFree(Block_Info_Radix_Sorting);
    cudaFree(Block_Group_Info);
    cudaFree(Sum_Group_Index);
    cudaFree(Group_Offset);

    cudaFree(D_Suffix_Array);
    cudaFree(D_Suffix_Array_Rank);


    cudaFree(D_Suffix_Array_New);
    cudaFree(D_Suffix_Array_Rank_New);




    cudaFree(Auxiliar1);
    cudaFree(Auxiliar2);


    D_Suffix_Array = D_Suffix_Array_Final;
}

void Sort_D_Rigth_To_Left() {
    //D_Check << <1, 1 >> > (D, Dev_Input, size_D);
    int size_D_Sorting = size_D;
    int size_Block_Info_Radix_Sorting = (size_D_Sorting + 512 - 1) / 512;
    int size_Auxiliar1 = 0;

    int* D_Suffix_Array_New;
    int* D_Suffix_Array_Rank;
    int* D_Suffix_Array_Rank_New;
    int* D_Suffix_Array_Final;
    cudaMalloc((void**)&D_Suffix_Array_New, (size_D * 3) * sizeof(int));
    cudaMalloc((void**)&D_Suffix_Array_Rank, (size_D) * sizeof(int));
    cudaMalloc((void**)&D_Suffix_Array_Rank_New, (size_D) * sizeof(int));
    cudaMalloc((void**)&D_Suffix_Array_Final, (size_D * 3) * sizeof(int));
    cudaMalloc((void**)&Block_Info_Radix_Sorting, (size_Block_Info_Radix_Sorting * 2) * sizeof(int));
    cudaMalloc((void**)&Block_Group_Info, size_Block_Info_Radix_Sorting * sizeof(int));
    cudaMalloc((void**)&Sum_Group_Index, size_Block_Info_Radix_Sorting * sizeof(int));
    cudaMalloc((void**)&Group_Offset, size_Block_Info_Radix_Sorting * sizeof(int));

    Init_Block_Info_Radix_Sorting << < (size_Block_Info_Radix_Sorting + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK >> > (Block_Info_Radix_Sorting, Block_Group_Info, Sum_Group_Index, Group_Offset, size_Block_Info_Radix_Sorting, size_D);

    Init_Vector << < (size_D_Sorting + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK >> > (D_Suffix_Array_Rank, size_D_Sorting, 0);

    int it = 0;
    while (size_D_Sorting > 0) {


        int size_Block_Info_Radix_Sorting_256 = size_Block_Info_Radix_Sorting * 256;
        //printf("\n %d %d %d %d", size_S_Sorting, size_Block_Info_Radix_Sorting, size_Block_Info_Radix_Sorting_256,it);
        if (size_Auxiliar1 < maximum1(size_Block_Info_Radix_Sorting_256, size_D_Sorting)) {

            if (size_Auxiliar1 > 0) {
                cudaFree(Auxiliar1);
                cudaFree(Auxiliar2);
                cudaFree(Auxiliar3);
            }
            size_Auxiliar1 = maximum1(size_Block_Info_Radix_Sorting_256, size_D_Sorting);
            cudaMalloc((void**)&Auxiliar1, maximum1(size_Block_Info_Radix_Sorting_256, size_D_Sorting) * sizeof(int));
            cudaMalloc((void**)&Auxiliar2, maximum1(size_Block_Info_Radix_Sorting_256, size_D_Sorting) * sizeof(int));
            cudaMalloc((void**)&Auxiliar3, maximum1(size_Block_Info_Radix_Sorting_256, size_D_Sorting) * sizeof(int));
        }
        int num_blocks_size_Block_Info_Radix_Sorting_256 = (size_Block_Info_Radix_Sorting_256 + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
        int num_blocks_size_D_Sorting = (size_D_Sorting + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;

        Init_Vector << <num_blocks_size_Block_Info_Radix_Sorting_256, MAX_THREADS_PER_BLOCK >> > (Auxiliar1, size_Block_Info_Radix_Sorting_256, 0);
        //Init_Vector << <num_blocks_size_Block_Info_Radix_Sorting_256, MAX_THREADS_PER_BLOCK >> > (Auxiliar2, size_Block_Info_Radix_Sorting_256, 0);

        RadixSort_D_RL_1 << <size_Block_Info_Radix_Sorting, 64 >> > (D, D_Suffix_Array_Rank, D_Suffix_Array_New, D_Suffix_Array_Rank_New, D_Suffix_Array_Final, Dev_Input, Block_Info_Radix_Sorting, Auxiliar1, Block_Group_Info, Sum_Group_Index, size_D_Sorting, it, size_Block_Info_Radix_Sorting, size_D, n_enlarged);

        swap(D, D_Suffix_Array_New);
        swap(D_Suffix_Array_Rank, D_Suffix_Array_Rank_New);
        //Prefix_Sum_Radix_Sort(Auxiliar1, Auxiliar2, Block_Group_Info, Res, size_Block_Info_Radix_Sorting);

        thrust::inclusive_scan(
            thrust::device_pointer_cast(Auxiliar1),
            thrust::device_pointer_cast(Auxiliar1 + size_Block_Info_Radix_Sorting_256),
            thrust::device_pointer_cast(Auxiliar1)
        );

        RadixSort2 << < num_blocks_size_Block_Info_Radix_Sorting_256, MAX_THREADS_PER_BLOCK >> > (Auxiliar1, Auxiliar2, Block_Group_Info, Sum_Group_Index, size_Block_Info_Radix_Sorting);

        RadixSort3 << < num_blocks_size_Block_Info_Radix_Sorting_256, MAX_THREADS_PER_BLOCK >> > (Auxiliar1, Auxiliar2, Sum_Group_Index, size_Block_Info_Radix_Sorting);

        RadixSort_D_RL_4 << <size_Block_Info_Radix_Sorting, 64 >> > (D, D_Suffix_Array_Rank, D_Suffix_Array_New, D_Suffix_Array_Rank_New, D_Suffix_Array_Final, Dev_Input, Block_Info_Radix_Sorting, Auxiliar1, Block_Group_Info, Sum_Group_Index, Group_Offset, size_D_Sorting, it, size_Block_Info_Radix_Sorting, size_D, n_enlarged);

        swap(D, D_Suffix_Array_New);
        swap(D_Suffix_Array_Rank, D_Suffix_Array_Rank_New);

        Init_Vector << <num_blocks_size_D_Sorting, MAX_THREADS_PER_BLOCK >> > (Auxiliar1, size_D_Sorting, 0);
        //Init_Vector << <num_blocks_size_S_Sorting, MAX_THREADS_PER_BLOCK >> > (Auxiliar2, size_S_Sorting, 0);

        Reduce_D1 << < num_blocks_size_D_Sorting, MAX_THREADS_PER_BLOCK >> > (D_Suffix_Array_Rank, Auxiliar1, size_D_Sorting, size_D);

        thrust::inclusive_scan(
            thrust::device_pointer_cast(Auxiliar1),
            thrust::device_pointer_cast(Auxiliar1 + size_D_Sorting),
            thrust::device_pointer_cast(Auxiliar1)
        );


        int size_D_New;

        cudaMemcpy(&size_D_New, Auxiliar1 + (size_D_Sorting - 1), sizeof(int), cudaMemcpyDeviceToHost);


        if (size_D_New == 0)
            break;

        Init_Vector << <(size_D_New + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK >> > (Auxiliar2, size_D_New, 0);
        //Init_Vector << <(size_S_New + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK >> > (Auxiliar3, size_S_New, 0);

        Reduce_D2 << < num_blocks_size_D_Sorting, MAX_THREADS_PER_BLOCK >> > (D, D_Suffix_Array_Rank, D_Suffix_Array_New, D_Suffix_Array_Rank_New, Auxiliar1, Auxiliar2, size_D_Sorting, size_D_New);

        size_D_Sorting = size_D_New;
        num_blocks_size_D_Sorting = (size_D_Sorting + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;


        swap(D, D_Suffix_Array_New);
        swap(D_Suffix_Array_Rank, D_Suffix_Array_Rank_New);


        thrust::inclusive_scan(
            thrust::device_pointer_cast(Auxiliar2),
            thrust::device_pointer_cast(Auxiliar2 + size_D_Sorting),
            thrust::device_pointer_cast(Auxiliar2)
        );

        Reduce_3 << < num_blocks_size_D_Sorting, MAX_THREADS_PER_BLOCK >> > (Auxiliar2, Auxiliar1, size_D_Sorting);

        Reduce_4 << < num_blocks_size_D_Sorting, MAX_THREADS_PER_BLOCK >> > (Auxiliar2, Auxiliar1, size_D_Sorting);

        Init_Vector << <num_blocks_size_D_Sorting, MAX_THREADS_PER_BLOCK >> > (Auxiliar1, size_D_Sorting, 0);
        //Init_Vector << <num_blocks_size_S_Sorting, MAX_THREADS_PER_BLOCK >> > (Auxiliar3, size_S_Sorting, 0);

        Reduce_5 << < num_blocks_size_D_Sorting, MAX_THREADS_PER_BLOCK >> > (Auxiliar1, Auxiliar2, size_D_Sorting);

        thrust::inclusive_scan(
            thrust::device_pointer_cast(Auxiliar1),
            thrust::device_pointer_cast(Auxiliar1 + size_D_Sorting),
            thrust::device_pointer_cast(Auxiliar1)
        );

        int size_Block_Info_Radix_Sorting_New;

        cudaMemcpy(&size_Block_Info_Radix_Sorting_New, Auxiliar1 + (size_D_Sorting - 1), sizeof(int), cudaMemcpyDeviceToHost);


        if (size_Block_Info_Radix_Sorting < size_Block_Info_Radix_Sorting_New) {
            cudaFree(Block_Info_Radix_Sorting);
            cudaFree(Block_Group_Info);
            cudaFree(Sum_Group_Index);
            cudaFree(Group_Offset);
            cudaMalloc((void**)&Block_Info_Radix_Sorting, (size_Block_Info_Radix_Sorting_New * 2) * sizeof(int));
            cudaMalloc((void**)&Block_Group_Info, size_Block_Info_Radix_Sorting_New * sizeof(int));
            cudaMalloc((void**)&Sum_Group_Index, size_Block_Info_Radix_Sorting_New * sizeof(int));
            cudaMalloc((void**)&Group_Offset, size_Block_Info_Radix_Sorting_New * sizeof(int));

        }


        size_Block_Info_Radix_Sorting = size_Block_Info_Radix_Sorting_New;
        Reduce_6 << < num_blocks_size_D_Sorting, MAX_THREADS_PER_BLOCK >> > (Auxiliar1, Auxiliar2, Block_Info_Radix_Sorting, Block_Group_Info, Sum_Group_Index, Auxiliar3, size_D_Sorting, size_Block_Info_Radix_Sorting);

        Reduce_7 << <num_blocks_size_D_Sorting, MAX_THREADS_PER_BLOCK >> > (Auxiliar1, Auxiliar2, Auxiliar3, Group_Offset, size_D_Sorting, size_Block_Info_Radix_Sorting);
        it++;

    }






    cudaFree(Auxiliar3);
    cudaFree(Block_Info_Radix_Sorting);
    cudaFree(Block_Group_Info);
    cudaFree(Sum_Group_Index);
    cudaFree(Group_Offset);

    cudaFree(D);
    cudaFree(D_Suffix_Array_Rank);


    cudaFree(D_Suffix_Array_New);
    cudaFree(D_Suffix_Array_Rank_New);




    cudaFree(Auxiliar1);
    cudaFree(Auxiliar2);


    D = D_Suffix_Array_Final;
}

void Sort_Suffix_Array() {
    int2* Suffix_Array;
    int2* Suffix_Array_New;
    unsigned long long* Ranks_Tuple;
    int* Suffix_Array_Final;
    int size_D_Suffix_Array_Sorting = size_D;
    int num_blocks_size_D_Suffix_Array_Sorting = (size_D_Suffix_Array_Sorting + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;

    cudaMalloc((void**)&Ranks_Tuple, (size_D) * sizeof(unsigned long long));
    cudaMalloc((void**)&Suffix_Array, (size_D) * sizeof(int2));
    cudaMalloc((void**)&Suffix_Array_New, (size_D) * sizeof(int2));
    cudaMalloc((void**)&Suffix_Array_Final, (size_D) * sizeof(int));
    cudaMalloc((void**)&Auxiliar3, (size_D) * sizeof(int));

    Compute_D_Suffix_Array2 << <num_blocks_size_D, MAX_THREADS_PER_BLOCK >> > (D_Suffix_Array, D, Auxiliar1, Suffix_Array, size_D);

    cudaFree(D_Suffix_Array);
    //D_Check<<<1,1>>>(D, Dev_Input, size_D);
    int prefix_length = 1;
    while (size_D_Suffix_Array_Sorting > 0) {
        Init_D_Suffix_Ranks << <num_blocks_size_D_Suffix_Array_Sorting, MAX_THREADS_PER_BLOCK >> > (Suffix_Array, D, Ranks_Tuple, size_D_Suffix_Array_Sorting, size_D, prefix_length);

        thrust::sort_by_key(
            thrust::device_pointer_cast(Ranks_Tuple),
            thrust::device_pointer_cast(Ranks_Tuple + size_D_Suffix_Array_Sorting),
            thrust::device_pointer_cast(Suffix_Array)
        );

        Init_Vector << <num_blocks_size_D_Suffix_Array_Sorting, MAX_THREADS_PER_BLOCK >> > (Auxiliar1, size_D_Suffix_Array_Sorting, 0);

        Init_Vector << <num_blocks_size_D_Suffix_Array_Sorting, MAX_THREADS_PER_BLOCK >> > (Auxiliar2, size_D_Suffix_Array_Sorting, 0);

        Reduce_D_Suffix_Array1 << <num_blocks_size_D_Suffix_Array_Sorting, MAX_THREADS_PER_BLOCK >> > (Suffix_Array, Ranks_Tuple, size_D_Suffix_Array_Sorting, Auxiliar1, Auxiliar2, size_D);

        thrust::inclusive_scan(
            thrust::device_pointer_cast(Auxiliar1),
            thrust::device_pointer_cast(Auxiliar1 + size_D_Suffix_Array_Sorting),
            thrust::device_pointer_cast(Auxiliar1)
        );

        thrust::inclusive_scan(
            thrust::device_pointer_cast(Auxiliar2),
            thrust::device_pointer_cast(Auxiliar2 + size_D_Suffix_Array_Sorting),
            thrust::device_pointer_cast(Auxiliar2)
        );


        Reduce_D_Suffix_Array2 << <num_blocks_size_D_Suffix_Array_Sorting, MAX_THREADS_PER_BLOCK >> > (Ranks_Tuple, size_D_Suffix_Array_Sorting, Auxiliar1, Auxiliar3, size_D);

        Reduce_D_Suffix_Array3 << <num_blocks_size_D_Suffix_Array_Sorting, MAX_THREADS_PER_BLOCK >> > (Suffix_Array, Suffix_Array_New, Suffix_Array_Final, D, Ranks_Tuple, size_D_Suffix_Array_Sorting, Auxiliar1, Auxiliar2, Auxiliar3, size_D);

        cudaMemcpy(&size_D_Suffix_Array_Sorting, Auxiliar2 + (size_D_Suffix_Array_Sorting - 1), sizeof(int), cudaMemcpyDeviceToHost);

        swap(Suffix_Array, Suffix_Array_New);

        num_blocks_size_D_Suffix_Array_Sorting = (size_D_Suffix_Array_Sorting + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;

        prefix_length <<= 1;
    }

    Compute_D_Suffix_Array3 << <num_blocks_size_D, MAX_THREADS_PER_BLOCK >> > (Suffix_Array_Final, D, size_D);


    cudaFree(Auxiliar1);
    cudaFree(Auxiliar2);
    cudaFree(Auxiliar3);
    cudaFree(Suffix_Array_Final);
    cudaFree(Ranks_Tuple);
    cudaFree(Suffix_Array_New);
    cudaFree(Suffix_Array);
}


void Compute_D() {
    int* D_Blocks_Prefix_Sum = Auxiliar1;
    Init_Vector << <num_blocks_num_blocks_n_enlarged, MAX_THREADS_PER_BLOCK >> > (D_Blocks_Prefix_Sum, num_blocks_n_enlarged, 0);
    Init_Vector << <num_blocks_num_blocks_n_enlarged, MAX_THREADS_PER_BLOCK >> > (Auxiliar2, num_blocks_n_enlarged, 0);

    Compute_D1 << <num_blocks_n, MAX_THREADS_PER_BLOCK >> > (Dev_Input, D_Blocks_Prefix_Sum, n);
    //Prefix_Sum(D_Blocks_Prefix_Sum, Auxiliar2, Res, num_blocks_n);

    thrust::inclusive_scan(
        thrust::device_pointer_cast(Auxiliar1),
        thrust::device_pointer_cast(Auxiliar1 + num_blocks_n),
        thrust::device_pointer_cast(Auxiliar1)
    );

    cudaMemcpy(&size_D, Auxiliar1 + (num_blocks_n - 1), sizeof(int), cudaMemcpyDeviceToHost);

    size_D++;



    cudaMalloc((void**)&D, (size_D * 3) * sizeof(int));
    cudaMalloc((void**)&D_Suffix_Array, (size_D * 3) * sizeof(int));


    num_blocks_size_D = (size_D + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;


    Compute_D2 << <num_blocks_n, MAX_THREADS_PER_BLOCK >> > (Dev_Input, D_Blocks_Prefix_Sum, D, D_Suffix_Array, n);

    size_Block_L_Values = (size_D + 8192 - 1) / 8192;
    size_Block_L_Values_times_8192 = size_Block_L_Values * 8192;
    num_blocks_size_Block_L_Values_times_8192 = (size_Block_L_Values_times_8192 + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;

    cudaFree(Auxiliar1);
    cudaFree(Auxiliar2);


    Compute_D3 << <1, 1 >> > (D, D_Suffix_Array, n_enlarged, size_D);

    Compute_D4 << <num_blocks_size_D, MAX_THREADS_PER_BLOCK >> > (D, D_Suffix_Array, size_D);


    Sort_D_Left_To_Rigth();

    cudaMalloc((void**)&Auxiliar1, (size_D) * sizeof(int));
    cudaMalloc((void**)&Auxiliar2, (size_D) * sizeof(int));

    //if (smallest_two_potenz_larger_than_size_D - size_D > 0)
      //  Init_Vector1 << < (smallest_two_potenz_larger_than_size_D - size_D + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK >> > (Auxiliar1, smallest_two_potenz_larger_than_size_D - size_D, 1e9, size_D);

    Init_Vector << <num_blocks_size_D, MAX_THREADS_PER_BLOCK >> > (Auxiliar1, size_D, -1);
    Init_Vector << <num_blocks_size_D, MAX_THREADS_PER_BLOCK >> > (Auxiliar2, size_D, -1);

    Compute_D_Suffix_Array1 << <num_blocks_size_D, MAX_THREADS_PER_BLOCK >> > (D_Suffix_Array, Auxiliar1, Dev_Input, size_D);

    Pointer_Jumping(Auxiliar1, Auxiliar2, size_D);




    //int size = smallest_two_potenz_larger_than_size_D;
    /*
    for (int k = 2; k <= size; k <<= 1)
    {
        for (int j = k >> 1; j > 0; j = j >> 1)
        {

            Bitonic_Sort_D_Suffix_Array << <num_blocks_smallest_two_potenz_larger_than_size_D, MAX_THREADS_PER_BLOCK >> > (New_D_Suffix_Array, D, j, k, size, size_D);

        }
    }
    */
    Sort_Suffix_Array();

    Sort_D_Rigth_To_Left();


    /*
    for (int k = 2; k <= size; k <<= 1)
    {
        for (int j = k >> 1; j > 0; j = j >> 1)
        {

            Bitonic_Sort_D_After_E << <num_blocks_smallest_two_potenz_larger_than_size_D, MAX_THREADS_PER_BLOCK >> > (D, Dev_Input, j, k, size, n_enlarged);

        }
    }
    */
    //D_Check << <1, 1 >> > (D, Dev_Input, size_D, smallest_two_potenz_larger_than_size_D);


}


__global__ void Compute_S1(int* D, unsigned char* Dev_Input, int* Prefix_Sum, short* L_Values, int size_D) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id < size_D) {



        int val11 = D[thread_id];
        int val12 = D[thread_id + 1 * size_D];
        /*
        if (thread_id == 0) {
            for (int j = 0; j < size_D; j++) {
                val11 = D[j*3];
                val12 = D[j*3 + 1];
                printf("\n");
                for (int index_val1 = val11 - val12 + 1; index_val1 <= val11; index_val1++) {
                    char char_val1 = Dev_Input[index_val1];
                    printf("%c", char_val1);
                }
            }
            val11 = D[idx];
            val12 = D[idx + 1];
        }

       */
        if (thread_id == 0) {
            Prefix_Sum[0] = val12 - W;
            L_Values[0] = 0;
        }
        else {



            int val21 = D[thread_id - 1];
            int val22 = D[thread_id - 1 + 1 * size_D];


            int cnt_prefix_length = 0;

            for (int index_val1 = val11, index_val2 = val21; index_val1 >= val11 - val12 + 1 && index_val2 >= val21 - val22 + 1; index_val1--, index_val2--, cnt_prefix_length++) {
                unsigned char char_val1 = Dev_Input[index_val1];
                unsigned char char_val2 = Dev_Input[index_val2];
                if (char_val1 > char_val2) {
                    break;
                }
                else if (char_val1 < char_val2) {
                    break;
                }
            }
            Prefix_Sum[thread_id] = val12 - maximum(W, cnt_prefix_length);

            L_Values[thread_id] = cnt_prefix_length;


        }

    }



}


__global__ void Compute_S2(unsigned int* S_L, unsigned short* S_Length_Suffix, int* D, int* Prefix_Sum, int size_D, int size_S) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id < size_D) {
        int num_suffixes_l = ((thread_id > 0) ? Prefix_Sum[thread_id] - Prefix_Sum[(thread_id)-1] : Prefix_Sum[0]);
        int offset_l = ((thread_id > 0) ? Prefix_Sum[thread_id - 1] : 0);


        int val12 = D[thread_id + 1 * size_D];
        //if (4 == D[thread_id])
         //   printf("srdgr");

        int l = val12 - num_suffixes_l;
        int r = val12;
        for (int i = l; i < r; i++) {
            int index = i - l + offset_l;
            S_Length_Suffix[index] = i + 1;
            S_L[index] = thread_id;

        }

    }
}





__global__ void Compute_S3(short* L_Values, int* Block_L_Values, int size_D) {


    __shared__ short L_values_local[8192];
    __shared__ short pos[8192];
    int pos_offset = blockIdx.x * 8192;

    for (unsigned short i = threadIdx.x; i < 8192; i += 1024) {
        pos[i] = i;
        L_values_local[i] = ((pos_offset + i < size_D) ? L_Values[pos_offset + i] : 32767);


    }


    for (int k = 2; k <= 8192; k <<= 1)
    {
        for (int j = k >> 1; j > 0; j = j >> 1)
        {
            __syncthreads();

            for (unsigned short i = threadIdx.x; i < 8192; i += 1024) {
                unsigned short ij;
                ij = i ^ j;

                if (ij > i)
                {
                    int idx = i;
                    int ijdx = ij;

                    short val11 = L_values_local[idx];
                    short val12 = pos[idx];


                    short val21 = L_values_local[ijdx];
                    short val22 = pos[ijdx];


                    bool cond = ((i & k) == 0);
                    bool val1_greater_then_val2 = ((val11) > (val21)) || ((val11 == val21) && (val12) > (val22));
                    bool val2_greater_then_val1 = ((val11) < (val21)) || ((val11 == val21) && (val12) < (val22));


                    if (((cond) && (val1_greater_then_val2))
                        || ((!cond) && (val2_greater_then_val1))) {

                        L_values_local[idx] = val21;
                        pos[idx] = val22;


                        L_values_local[ijdx] = val11;
                        pos[ijdx] = val12;

                    }


                }

            }
        }
    }
    __syncthreads();

    for (unsigned short i = threadIdx.x; i < 8192; i += 1024) {
        short val = L_values_local[i];
        short val1 = pos[i];
        if ((i == 0 || (i > 0 && val != L_values_local[i - 1])) && (pos_offset + val1 < size_D)) {

            Block_L_Values[blockIdx.x * 8192 + val] = val1 + pos_offset;

        }

    }

}




__global__ void Compute_S4(unsigned int* S_Interval_Length, int* D, int* Prefix_Sum, short* L_Values, int* Block_L_Values, int size_D, int size_Block_L_Values, int size_S) {


    __shared__ short L_values_local[8192];
    __shared__ int Block_L_Values_local[8192];
    int pos_offset = blockIdx.x * 8192;
    int Prefix_Sum_Thread_Values[8];
    short D_Word_lengths_Thread_Values[8];
    short L_Thread_Values[8];
    int Save_Min_Values[8];
    bool cond = blockIdx.x + 1 < size_Block_L_Values;

    for (short i = threadIdx.x; i < 8192; i += 1024) {
        Block_L_Values_local[i] = ((cond) ? Block_L_Values[(blockIdx.x + 1) * 8192 + i] : size_D);
    }


    for (short i = threadIdx.x, k = 0; i < 8192; i += 1024, k++) {

        Prefix_Sum_Thread_Values[k] = ((pos_offset + i - 1 < size_D && pos_offset + i - 1 >= 0) ? Prefix_Sum[pos_offset + i - 1] : 0);
        L_values_local[i] = ((pos_offset + i < size_D) ? L_Values[pos_offset + i] : 32767);
        L_Thread_Values[k] = ((pos_offset + i < size_D) ? maximum(L_Values[pos_offset + i], W) : 32767);
        D_Word_lengths_Thread_Values[k] = ((pos_offset + i < size_D) ? D[(pos_offset + i) + 1 * size_D] : -1);
    }
    __syncthreads();


    for (short stride = 1; stride < 8192; stride <<= 1) {

        __syncthreads();

        for (short i = threadIdx.x, k = 0; k < 8; i += 1024, k++) {
            if (i - stride >= 0)
                Save_Min_Values[k] = Block_L_Values_local[i - stride];


        }


        __syncthreads();

        for (short i = threadIdx.x, k = 0; k < 8; i += 1024, k++) {
            if (i - stride >= 0)
                Block_L_Values_local[i] = ((Save_Min_Values[k] <= Block_L_Values_local[i]) ? Save_Min_Values[k] : Block_L_Values_local[i]);

        }




    }



    __syncthreads();

    for (short i = threadIdx.x, k = 0; i < 8192; i += 1024, k++) {
        short l = L_Thread_Values[k];
        short r = D_Word_lengths_Thread_Values[k];
        if (r != -1) {
            short last = i + 1;
            for (short j = r; j > l; j--) {
                bool found = false;
                for (short h = last; h < 8192;) {
                    if (L_values_local[h] < j) {
                        last = h;
                        int index = (Prefix_Sum_Thread_Values[k] + (j - (l + 1)));
                        S_Interval_Length[index] = h - 1 - i;

                        found = true;
                        break;
                    }
                    else
                        h++;
                }
                if (!found) {
                    int index = (Prefix_Sum_Thread_Values[k] + (j - (l + 1)));

                    S_Interval_Length[index] = Block_L_Values_local[j - 1] - 1 - (i + pos_offset);
                    /*
                    if (S[index] == -25733)
                        printf("\n %d %d %d",j-1, blockIdx.x, pos_offset);
                        */
                }
            }
        }
    }
}
__global__ void Init_S_Suffix_End_and_S_Pos(unsigned int* S_Suffix_End, unsigned int* S_Pos, unsigned int* S_L, int* D, int size_S) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id < size_S) {
        S_Pos[thread_id] = thread_id;
        int D_index = S_L[thread_id];
        S_Suffix_End[thread_id] = D[D_index];
    }
}

__global__ void Load_8_Byte_Chunk_S(unsigned long long* Cur_Prefix_S, unsigned int* S_Suffix_End, unsigned int* S_Pos, unsigned short* S_Length_Suffix, unsigned char* Dev_Input, int size_S_Sorting, int it) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id < size_S_Sorting) {
        unsigned char prefix_thread[8];
        int S_index = S_Pos[thread_id];
        int suffix_end = S_Suffix_End[S_index];
        int index = suffix_end - S_Length_Suffix[S_index] + 1 + it;
        for (int j = 0; j < 8; j++, index++)
            prefix_thread[j] = ((index <= suffix_end) ? Dev_Input[index] : 0);

        Cur_Prefix_S[thread_id] = pack8_not_reversed(prefix_thread);
    }
}

__global__ void RadixSort_S1(unsigned int* S_Pos, unsigned int* S_Rank, unsigned long long* Cur_Prefix_S, unsigned int* S_Pos_New, unsigned int* S_Rank_New, unsigned long long* Cur_Prefix_S_New, unsigned int* S_L, unsigned int* S_Interval_Length, unsigned short* S_Length_Suffix, unsigned int* S_L_Final, unsigned int* S_Interval_Length_Final, unsigned short* S_Length_Suffix_Final, unsigned int* S_Suffix_End, unsigned char* Dev_Input, int* Block_Info_Radix_Sorting, int* Prefix_Sum, int* Block_Group_Info, int* Sum_Group_Index, int size_S_Sorting, int it, int size_Block_Info_Radix_Sorting, int size_S) {
    __shared__ unsigned char memory[3072];


    int l = Block_Info_Radix_Sorting[blockIdx.x];
    int r = Block_Info_Radix_Sorting[blockIdx.x + size_Block_Info_Radix_Sorting];

    int diff = r - l + 1;
    int two_potenz = smallest_two_potenz_larger_than_k_device(diff);
    bool cond = (blockIdx.x > 0 && Block_Group_Info[blockIdx.x] == Block_Group_Info[blockIdx.x - 1]);
    bool complete_sort = (Sum_Group_Index[blockIdx.x] == blockIdx.x) && (!cond);

    // if (blockIdx.x == 1)
      //  printf("sgrt");

    if (complete_sort && two_potenz <= 256) {
        uint8_t* pos = (uint8_t*)memory;
        unsigned long long* cur_prefix = (unsigned long long*)(memory + maximum(8, two_potenz));
        unsigned char prefix_thread[8];
        for (int i = threadIdx.x; i < two_potenz; i += 64) {
            pos[i] = i;
        }

        for (int i = threadIdx.x; i < two_potenz; i += 64) {
            if (l + i <= r) {
                int S_index = S_Pos[l + i];
                int suffix_end = S_Suffix_End[S_index];
                int index = suffix_end - S_Length_Suffix[S_index] + 1 + it;
                for (int j = 0; j < 8; j++, index++)
                    prefix_thread[j] = ((index <= suffix_end) ? Dev_Input[index] : 0);

                cur_prefix[i] = pack8(prefix_thread);
            }
            else
                cur_prefix[i] = 0xFFFFFFFFFFFFFFFFULL;
        }

        for (int k = 2; k <= two_potenz; k <<= 1)
        {
            for (int j = k >> 1; j > 0; j = j >> 1)
            {
                __syncthreads();

                for (unsigned short i = threadIdx.x; i < two_potenz; i += 64) {
                    unsigned short ij;
                    ij = i ^ j;

                    if (ij > i)
                    {


                        unsigned long long val11 = cur_prefix[i];
                        uint8_t val12 = pos[i];


                        unsigned long long val21 = cur_prefix[ij];
                        uint8_t val22 = pos[ij];


                        bool cond = ((i & k) == 0);
                        bool val1_greater_then_val2;
                        bool val2_greater_then_val1;



                        val1_greater_then_val2 = ((val11) > (val21));
                        val2_greater_then_val1 = ((val11) < (val21));

                        if (complete_sort && val1_greater_then_val2 == false && val2_greater_then_val1 == false && l + val12 <= r && l + val22 <= r) {

                            unsigned int S_index1 = S_Pos[l + val12];
                            int suffix_end1 = S_Suffix_End[S_index1];
                            int suffix_length1 = S_Length_Suffix[S_index1];

                            unsigned int S_index2 = S_Pos[l + val22];
                            int suffix_end2 = S_Suffix_End[S_index2];
                            int suffix_length2 = S_Length_Suffix[S_index2];

                            for (int index_val1 = suffix_end1 - suffix_length1 + it + 9, index_val2 = suffix_end2 - suffix_length2 + it + 9; index_val1 <= suffix_end1 && index_val2 <= suffix_end2; index_val1++, index_val2++) {

                                unsigned char char_val1 = Dev_Input[index_val1];
                                unsigned char char_val2 = Dev_Input[index_val2];
                                if (char_val1 > char_val2) {
                                    val1_greater_then_val2 = true;
                                    break;
                                }
                                else if (char_val1 < char_val2) {
                                    val2_greater_then_val1 = true;
                                    break;
                                }
                            }

                            if (val1_greater_then_val2 == false && val2_greater_then_val1 == false) {
                                if (suffix_length1 > suffix_length2)
                                    val1_greater_then_val2 = true;
                                else if (suffix_length1 < suffix_length2)
                                    val2_greater_then_val1 = true;

                            }
                        }


                        if (((cond) && (val1_greater_then_val2))
                            || ((!cond) && (val2_greater_then_val1))) {

                            cur_prefix[i] = val21;
                            pos[i] = val22;


                            cur_prefix[ij] = val11;
                            pos[ij] = val12;

                        }


                    }

                }
            }
        }
        // if (blockIdx.x == 3)
          //   printf("sgrt");
        __syncthreads();
        for (int i = threadIdx.x; i < diff; i += 64) {

            int read_index = l + pos[i];

            unsigned int S_index = S_Pos[read_index];
            unsigned int val2 = S_Rank[read_index];

            unsigned int val1 = S_L[S_index];
            unsigned int val3 = S_Interval_Length[S_index];
            unsigned short val4 = S_Length_Suffix[S_index];
            val2 += i;
            S_L_Final[val2] = val1;
            S_Interval_Length_Final[val2] = val3;
            S_Length_Suffix_Final[val2] = val4;


        }


    }
    else {

        short* pos = (short*)memory;
        unsigned char* cur_characters = (unsigned char*)(memory + 512 * 2);
        short* block_sums = (short*)(memory + 512 * 3);

        for (int i = threadIdx.x; i < two_potenz; i += 64) {
            pos[i] = i;
        }

        for (int i = threadIdx.x; i < 256; i += 64) {
            block_sums[i] = 0;
        }


        for (int i = threadIdx.x; i < two_potenz; i += 64) {
            if (l + i <= r) {


                cur_characters[i] = Cur_Prefix_S[l + i] >> ((it % 8) * 8);

            }
            else
                cur_characters[i] = 255;
        }
        for (int k = 2; k <= two_potenz; k <<= 1)
        {
            for (int j = k >> 1; j > 0; j = j >> 1)
            {
                __syncthreads();

                for (unsigned short i = threadIdx.x; i < two_potenz; i += 64) {
                    unsigned short ij;
                    ij = i ^ j;

                    if (ij > i)
                    {


                        unsigned char val11 = cur_characters[i];
                        short val12 = pos[i];


                        unsigned char val21 = cur_characters[ij];
                        short val22 = pos[ij];


                        bool cond = ((i & k) == 0);
                        bool val1_greater_then_val2 = ((val11) > (val21));
                        bool val2_greater_then_val1 = ((val11) < (val21));


                        if (((cond) && (val1_greater_then_val2))
                            || ((!cond) && (val2_greater_then_val1))) {

                            cur_characters[i] = val21;
                            pos[i] = val22;


                            cur_characters[ij] = val11;
                            pos[ij] = val12;

                        }


                    }

                }
            }
        }


        __syncthreads();
        for (int i = threadIdx.x; i < diff; i += 64) {

            if ((i == 0) || (i > 0 && cur_characters[i] != cur_characters[i - 1]))
                block_sums[cur_characters[i]] = i;
        }
        __syncthreads();
        for (int i = threadIdx.x; i < diff; i += 64) {
            if ((i == diff - 1) || (i < diff && cur_characters[i] != cur_characters[i + 1]))
                block_sums[cur_characters[i]] = i - block_sums[cur_characters[i]] + 1;
        }



        __syncthreads();

        for (int i = threadIdx.x; i < 256; i += 64) {
            if (block_sums[i] > 0)
                Prefix_Sum[blockIdx.x + i * size_Block_Info_Radix_Sorting] = block_sums[i];
            //if (i == 65) {
             //   printf("\n %d", block_sums[i]);
            //}
        }




        for (int i = threadIdx.x, cnt = 0; i < diff; i += 64, cnt++) {


            int read_index = l + pos[i];
            int write_index = l + i;

            S_Pos_New[write_index] = S_Pos[read_index];

            S_Rank_New[write_index] = S_Rank[read_index];

            Cur_Prefix_S_New[write_index] = Cur_Prefix_S[read_index];

        }


    }




}







__global__ void RadixSort_S4(unsigned int* S_Pos, unsigned int* S_Rank, unsigned long long* Cur_Prefix_S, unsigned int* S_Pos_New, unsigned int* S_Rank_New, unsigned long long* Cur_Prefix_S_New, int* Block_Info_Radix_Sorting, int* Prefix_Sum, int* Block_Group_Info, int* Sum_Group_Index, int* Group_Offset, int size_S_Sorting, int it, int size_Block_Info_Radix_Sorting) {

    __shared__ unsigned char cur_characters[512];
    //__shared__ int block_sums_left[128];
    __shared__ int in_block_sums[256];
    __shared__ int block_sums_total[256];

    int thread_id = threadIdx.x;
    int l = Block_Info_Radix_Sorting[blockIdx.x];
    int r = Block_Info_Radix_Sorting[blockIdx.x + size_Block_Info_Radix_Sorting];

    int diff = r - l + 1;
    int two_potenz = smallest_two_potenz_larger_than_k_device(diff);
    int group_sum_index = Sum_Group_Index[blockIdx.x];
    bool cond = (blockIdx.x > 0 && Block_Group_Info[blockIdx.x] == Block_Group_Info[blockIdx.x - 1]);
    bool complete_sort = (group_sum_index == blockIdx.x) && (!cond);
    int group_offset = Group_Offset[blockIdx.x];
    if (!(complete_sort && two_potenz <= 256)) {

        for (int i = threadIdx.x; i < two_potenz; i += 64) {
            if (l + i <= r) {
                cur_characters[i] = Cur_Prefix_S[l + i] >> ((it % 8) * 8);
            }
            else
                cur_characters[i] = 255;
        }

        for (int i = thread_id; i < 256; i += 64) {
            in_block_sums[i] = 0;
        }

        //for (int i = thread_id; i < 128; i += 64) {
          //   block_sums_left[i] = (cond ? Prefix_Sum[blockIdx.x - 1 + i * size_Block_Info_Radix_Sorting] : 0);

       //  }


        for (int i = thread_id; i < 256; i += 64) {
            block_sums_total[i] = Prefix_Sum[group_sum_index + i * size_Block_Info_Radix_Sorting];

        }



        for (int stride = 1; stride < 256; stride <<= 1) {

            __syncthreads();

            int val1;
            int val2;
            int val3;
            int val4;

            if (thread_id - stride >= 0)
                val1 = block_sums_total[thread_id - stride];

            if (thread_id + 64 - stride >= 0)
                val2 = block_sums_total[thread_id + 64 - stride];

            if (thread_id + 128 - stride >= 0)
                val3 = block_sums_total[thread_id + 128 - stride];

            if (thread_id + 192 - stride >= 0)
                val4 = block_sums_total[thread_id + 192 - stride];

            __syncthreads();

            if (thread_id - stride >= 0)
                block_sums_total[thread_id] = block_sums_total[thread_id] + val1;

            if (thread_id + 64 - stride >= 0)
                block_sums_total[thread_id + 64] = block_sums_total[thread_id + 64] + val2;

            if (thread_id + 128 - stride >= 0)
                block_sums_total[thread_id + 128] = block_sums_total[thread_id + 128] + val3;

            if (thread_id + 192 - stride >= 0)
                block_sums_total[thread_id + 192] = block_sums_total[thread_id + 192] + val4;
        }




        __syncthreads();
        for (int i = thread_id; i < diff; i += 64) {

            if ((i == 0) || (i > 0 && cur_characters[i] != cur_characters[i - 1]))
                in_block_sums[cur_characters[i]] = i;
        }




        __syncthreads();

        for (int i = thread_id; i < diff; i += 64) {

            int read_index = l + i;


            unsigned int val1 = S_Pos[read_index];
            unsigned int val2 = S_Rank[read_index];
            unsigned long long val3 = Cur_Prefix_S[read_index];


            int cur_character = cur_characters[i];
            int in_block_offset = (i - in_block_sums[cur_characters[i]]);
            int in_group_cur_character_left_offset = ((cond) ? Prefix_Sum[blockIdx.x - 1 + cur_characters[i] * size_Block_Info_Radix_Sorting] : 0);
            int in_group_smaller_then_cur_character_offset = ((cur_character - 1 >= 0) ? block_sums_total[cur_character - 1] : 0);
            int write_index;

            val2 += in_group_smaller_then_cur_character_offset;
            write_index = in_block_offset + in_group_cur_character_left_offset + in_group_smaller_then_cur_character_offset + group_offset;

            //printf("\n %d %d %d %d", val5, cur_characters[i],i,blockIdx.x);

            S_Pos_New[write_index] = val1;
            S_Rank_New[write_index] = val2;
            Cur_Prefix_S_New[write_index] = val3;


            // if (write_index == 154) {
                //   printf("\n f %d %d %d %d %d", val5, in_block_offset, in_group_cur_character_left_offset, in_group_smaller_then_cur_character_offset, group_offset);
              //}

            // if (blockIdx.x==1) {
             //    printf("\n %d %d %d %d %d %d",val5, write_index, in_block_offset, in_group_cur_character_left_offset, in_group_smaller_then_cur_character_offset, group_offset);
             //}

        }

    }
    else {
        for (int i = thread_id; i < diff; i += 64) {

            int write_index = l + i;
            S_Rank_New[write_index] = 1e9;

        }

    }

}

__global__ void Reduce_S1(unsigned int* S_Rank, int* Prefix_Sum, int size_S_Sorting, int size_S) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id < size_S_Sorting) {

        unsigned int val2 = S_Rank[thread_id];//curr rank;

        if (val2 != 1e9) {
            Prefix_Sum[thread_id] = 1;
        }
    }
}

__global__ void Reduce_S2(unsigned int* S_Pos, unsigned int* S_Rank, unsigned long long* Cur_Prefix_S, unsigned int* S_Pos_New, unsigned int* S_Rank_New, unsigned long long* Cur_Prefix_S_New, int* Prefix_Sum, int* Prefix_Sum2, int size_S_Sorting, int size_S_new) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id < size_S_Sorting) {

        unsigned int val1 = S_Pos[thread_id];
        unsigned int val2 = S_Rank[thread_id];
        unsigned long long val3 = Cur_Prefix_S[thread_id];



        bool cond2 = (thread_id == size_S_Sorting - 1) || ((thread_id < size_S_Sorting - 1) && val2 != S_Rank[thread_id + 1]);

        if (val2 != 1e9) {
            int write_index = ((thread_id > 0) ? Prefix_Sum[thread_id - 1] : 0);
            S_Pos_New[write_index] = val1;
            S_Rank_New[write_index] = val2;
            Cur_Prefix_S_New[write_index] = val3;

            if (cond2 && write_index + 1 < size_S_new) {
                Prefix_Sum2[write_index + 1] = 1;
                // printf("\n %d %d %d %d", val5, thread_id, size_S_new, S[thread_id + 1 + 4 * size_S_Sorting]);
            }
        }
    }
}






__global__ void Compute_S5(unsigned int* S_Interval_Length_Final, int* Prefix_Sum, int size_S) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id < size_S) {

        Prefix_Sum[thread_id] = S_Interval_Length_Final[thread_id] + 1;
    }


}

__global__ void Compute_S6(unsigned int* S_L, unsigned int* S_Interval_Length, unsigned short* S_Length_Suffix, unsigned char* Dev_Input, unsigned char* Dev_Output, int* Prefix_Sum1, int* Prefix_Sum2, int* D, int size_S, int size_Dev_Input, int size_D) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id < size_S) {


        int index_val12 = S_Length_Suffix[thread_id];
        int l = S_L[thread_id];
        int r = l + S_Interval_Length[thread_id];

        /*
        if (thread_id == 17) {
            printf("s<rgsag");
        }
        */
        int index_val11 = D[l];
        int index_val21 = D[r];
        int index_val22 = index_val12;

        /*
        if (thread_id == 229) {
            for (int i = l; i <= r; i++) {
                int index = 3 * i;
                int pos=D[index];
                int length= D[index+1];
                int internal_pos= ((pos - index_val12 >= 0) ? pos - index_val12 : size_Dev_Input - W - 1);
                char a= Dev_Input[internal_pos];
                printf("%c %d %d\n",a, length,pos);
                for(int j=0;j< length;j++)
                    printf("%c", Dev_Input[pos- length+j+1]);
                printf("\n");
            }
        }
        */
        /*
        if (thread_id == 0) {
            for (int j = 0; j < size_S; j++) {

                int val11 = S[j * 4];
                int val12 = S[j * 4+1];
                printf("\n");
                int index_val111 = ((val11 - val12 >= 0) ? val11 - val12 : size_Dev_Input - W - 1);
                printf("%c ", Dev_Input[index_val111]);
                for (int index_val1 = val11 - val12 + 1; index_val1 <= val11; index_val1++) {
                    char char_val1 = Dev_Input[index_val1];
                    printf("%c", char_val1);
                }
            }

        }
        */

        int offset = ((thread_id > 0) ? Prefix_Sum1[thread_id - 1] : 0);
        /*
        if (offset <= 502 && 502 <= (offset + r - l)) {
            printf("\n # %d %d %d", l, r, offset);

            for (int i = l; i <= r; i++) {
                int index_D = D[i];
                int length_D= D[i+size_D];
                int rank_D= D[i + size_D*2];
                printf("\n %c %d %d %d ", Dev_Input[index_D - length_D],index_D, length_D, rank_D);
                for (int i = index_D + 1; i < index_D + 10; i++) {
                    printf("%c ", Dev_Input[i]);
                }

            }
        }
        */
        index_val11 = ((index_val11 - index_val12 >= 0) ? index_val11 - index_val12 : size_Dev_Input - W - 1);
        index_val21 = ((index_val21 - index_val22 >= 0) ? index_val21 - index_val22 : size_Dev_Input - W - 1);


        /*
        if (index_val21 >= 400005)
            printf("\n %d %d %d %d", thread_id, l, r,size_D);
            */
        unsigned char char_val1 = Dev_Input[index_val11];
        unsigned char char_val2 = Dev_Input[index_val21];


        if (char_val1 == char_val2) {
            //S[thread_id * 2 * 3]=-1;
            r -= l;
            l = offset;
            r += offset;



            for (int i = l; i <= r; i++) {
                Dev_Output[i] = char_val1;


            }
        }
        else {
            //S[thread_id] = offset;
            Prefix_Sum2[thread_id] = 1;
            /*
            r -= l;
            l = offset;
            r += offset;

            if (l <= 408755 && r >= 408755) {
                printf("%d %d %d %d \n",l,r,r-l+1, offset);
            }
            */

        }

    }


}


__global__ void Compute_Suffixes_Without_Same_Preeceding_Char1(unsigned int* S_L, unsigned int* S_Interval_Length, unsigned short* S_Length_Suffix, int* Suffixes_Without_Same_Preeceding_Char, int* Prefix_Sum1, int* Prefix_Sum2, int size_S, int smallest_two_potenz_larger_than_size_Suffixes_Without_Same_Preeceding_Char) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id < size_S) {

        int offset1 = ((thread_id > 0) ? Prefix_Sum1[thread_id - 1] : 0);
        int val = ((thread_id > 0) ? Prefix_Sum1[thread_id] - offset1 : Prefix_Sum1[0]);
        int offset2 = ((thread_id > 0) ? Prefix_Sum2[thread_id - 1] : 0);
        if (val > 0) {
            int l = S_L[thread_id];
            Suffixes_Without_Same_Preeceding_Char[offset1] = l;//l
            Suffixes_Without_Same_Preeceding_Char[offset1 + 1 * smallest_two_potenz_larger_than_size_Suffixes_Without_Same_Preeceding_Char] = l + S_Interval_Length[thread_id];//r
            Suffixes_Without_Same_Preeceding_Char[offset1 + 2 * smallest_two_potenz_larger_than_size_Suffixes_Without_Same_Preeceding_Char] = S_Length_Suffix[thread_id];//length_suffix
            Suffixes_Without_Same_Preeceding_Char[offset1 + 3 * smallest_two_potenz_larger_than_size_Suffixes_Without_Same_Preeceding_Char] = offset2;//offset_bwt_insert

        }

    }


}


__global__ void Bitonic_Sort_Suffixes_Without_Same_Preeceding_Char(int* Suffixes_Without_Same_Preeceding_Char, int j, int k, int size)
{
    unsigned int i, ij;

    i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < size) {

        ij = i ^ j;

        if (ij > i)
        {


            int val11 = Suffixes_Without_Same_Preeceding_Char[i];
            int val12 = Suffixes_Without_Same_Preeceding_Char[i + 1 * size];
            int val13 = Suffixes_Without_Same_Preeceding_Char[i + 2 * size];
            int val14 = Suffixes_Without_Same_Preeceding_Char[i + 3 * size];


            int val21 = Suffixes_Without_Same_Preeceding_Char[ij];
            int val22 = Suffixes_Without_Same_Preeceding_Char[ij + 1 * size];
            int val23 = Suffixes_Without_Same_Preeceding_Char[ij + 2 * size];
            int val24 = Suffixes_Without_Same_Preeceding_Char[ij + 3 * size];


            bool cond = ((i & k) == 0);
            bool val1_greater_then_val2 = false;
            bool val2_greater_then_val1 = false;

            if (val11 != 1e9 && val21 != 1e9) {
                val1_greater_then_val2 = (val12 - val11) > (val22 - val21);
                val2_greater_then_val1 = (val12 - val11) < (val22 - val21);
            }
            else if (val11 == 1e9 && val21 != 1e9)
                val1_greater_then_val2 = true;
            else if (val11 != 1e9 && val21 == 1e9)
                val2_greater_then_val1 = true;

            if (((cond) && (val1_greater_then_val2))
                || ((!cond) && (val2_greater_then_val1))) {

                Suffixes_Without_Same_Preeceding_Char[i] = val21;
                Suffixes_Without_Same_Preeceding_Char[i + 1 * size] = val22;
                Suffixes_Without_Same_Preeceding_Char[i + 2 * size] = val23;
                Suffixes_Without_Same_Preeceding_Char[i + 3 * size] = val24;


                Suffixes_Without_Same_Preeceding_Char[ij] = val11;
                Suffixes_Without_Same_Preeceding_Char[ij + 1 * size] = val12;
                Suffixes_Without_Same_Preeceding_Char[ij + 2 * size] = val13;
                Suffixes_Without_Same_Preeceding_Char[ij + 3 * size] = val14;


            }


        }
    }

}

__global__ void Compute_Cnt_Blocks_Per_Sort1(int* Suffixes_Without_Same_Preeceding_Char, int* Prefix_Sum, int* Res, int size_Suffixes_Without_Same_Preeceding_Char, int smallest_two_potenz_larger_than_size_Suffixes_Without_Same_Preeceding_Char) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id < size_Suffixes_Without_Same_Preeceding_Char) {


        int l = Suffixes_Without_Same_Preeceding_Char[thread_id];
        int r = Suffixes_Without_Same_Preeceding_Char[thread_id + 1 * smallest_two_potenz_larger_than_size_Suffixes_Without_Same_Preeceding_Char];

        int size = r - l + 1;
        if (size <= Max_Sort_in_Shared_Mem) {

            if (thread_id == size_Suffixes_Without_Same_Preeceding_Char - 1)
                Res[0] = -1;
        }
        else {

            Prefix_Sum[thread_id] = size;

            if (thread_id == size_Suffixes_Without_Same_Preeceding_Char - 1)
                Res[0] = size;

        }

    }
}

__global__ void Compute_Cnt_Blocks_Per_Sort2(int* Suffixes_Without_Same_Preeceding_Char, int* Prefix_Sum, int* Prefix_Sum2, int size_Suffixes_Without_Same_Preeceding_Char, int smallest_two_potenz_larger_than_size_Suffixes_Without_Same_Preeceding_Char) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id < size_Suffixes_Without_Same_Preeceding_Char) {


        int l = Suffixes_Without_Same_Preeceding_Char[thread_id];
        int r = Suffixes_Without_Same_Preeceding_Char[thread_id + 1 * smallest_two_potenz_larger_than_size_Suffixes_Without_Same_Preeceding_Char];

        int size = r - l + 1;
        if (size <= Max_Sort_in_Shared_Mem) {
            if (thread_id < size_Suffixes_Without_Same_Preeceding_Char - 1) {
                int l1 = Suffixes_Without_Same_Preeceding_Char[(thread_id + 1)];
                int r1 = Suffixes_Without_Same_Preeceding_Char[(thread_id + 1) + 1 * smallest_two_potenz_larger_than_size_Suffixes_Without_Same_Preeceding_Char];
                int size1 = r1 - l1 + 1;
                int smallest_two_potenz_larger_then_size = smallest_two_potenz_larger_than_k_device(size);
                int smallest_two_potenz_larger_then_size1 = smallest_two_potenz_larger_than_k_device(size1);
                if (smallest_two_potenz_larger_then_size < smallest_two_potenz_larger_then_size1 && smallest_two_potenz_larger_then_size1 >= 65)
                    Prefix_Sum2[thread_id] = 1;

            }
            else
                Prefix_Sum2[thread_id] = 1;
        }
        else {

            if (size < Size_Max_Sort) {

                if (thread_id < size_Suffixes_Without_Same_Preeceding_Char - 1) {
                    int count_blocks_cur = Prefix_Sum[thread_id] / Size_Max_Sort;
                    int count_blocks_next = Prefix_Sum[thread_id + 1] / Size_Max_Sort;
                    int l1 = Suffixes_Without_Same_Preeceding_Char[(thread_id + 1)];
                    int r1 = Suffixes_Without_Same_Preeceding_Char[(thread_id + 1) + 1 * smallest_two_potenz_larger_than_size_Suffixes_Without_Same_Preeceding_Char];
                    int size1 = r1 - l1 + 1;
                    int smallest_two_potenz_larger_then_size = smallest_two_potenz_larger_than_k_device(size);
                    int smallest_two_potenz_larger_then_size1 = smallest_two_potenz_larger_than_k_device(size1);
                    if (count_blocks_cur < count_blocks_next || smallest_two_potenz_larger_then_size < smallest_two_potenz_larger_then_size1)
                        Prefix_Sum2[thread_id] = 1;
                }
                else
                    Prefix_Sum2[thread_id] = 1;
            }
            else
                Prefix_Sum2[thread_id] = 1;


        }

    }
}



__global__ void Compute_Cnt_Blocks_Per_Sort3(int* Suffixes_Without_Same_Preeceding_Char, int* Prefix_Sum, int* Prefix_Sum2, int* Dev_CPU_Blocks_Array, int size_Suffixes_Without_Same_Preeceding_Char, int smallest_two_potenz_larger_than_size_Suffixes_Without_Same_Preeceding_Char) {
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id < size_Suffixes_Without_Same_Preeceding_Char) {
        int prev = (thread_id > 0) ? Prefix_Sum[thread_id - 1] : 0;
        if (Prefix_Sum[thread_id] > prev) {
            int l = Suffixes_Without_Same_Preeceding_Char[thread_id];
            int r = Suffixes_Without_Same_Preeceding_Char[thread_id + 1 * smallest_two_potenz_larger_than_size_Suffixes_Without_Same_Preeceding_Char];

            int size = r - l + 1;
            int smallest_two_potenz_larger_then_size = smallest_two_potenz_larger_than_k_device(size);
            Dev_CPU_Blocks_Array[prev * 3] = thread_id;
            Dev_CPU_Blocks_Array[prev * 3 + 1] = smallest_two_potenz_larger_then_size;
            Dev_CPU_Blocks_Array[prev * 3 + 2] = Prefix_Sum2[thread_id];
        }

    }
}




__global__ void Bitonic_Sort_64_Suffixes_Shared_Mem(int* Block_info, int* D, unsigned char* Dev_Input, unsigned char* Dev_Output, int offset, int size_Dev_Input, int size_D, int size_Block_info)
{

    __shared__ int ranks[64];
    __shared__ unsigned char preeceding_characters[64];
    unsigned int i, ij;

    i = threadIdx.x;

    int block_id = blockIdx.x;
    int index = Block_info[(block_id + offset)] + i;
    int r = Block_info[(block_id + offset) + 1 * size_Block_info];
    int length_suffix = Block_info[(block_id + offset) + 2 * size_Block_info];
    int offset_for_bwt_insert = Block_info[(block_id + offset) + 3 * size_Block_info];


    if (index <= r) {

        int pos_D = D[index];
        int suffix_rank = D[index + 2 * size_D];
        int index_preeceding_characte_suffix = ((pos_D - length_suffix >= 0) ? pos_D - length_suffix : size_Dev_Input - W - 1);
        unsigned char preeceding_characte_suffix = Dev_Input[index_preeceding_characte_suffix];
        ranks[i] = suffix_rank;
        preeceding_characters[i] = preeceding_characte_suffix;
    }
    else {
        ranks[i] = 1e9;
    }

    for (int k = 2; k <= 64; k <<= 1)
    {
        for (int j = k >> 1; j > 0; j = j >> 1)
        {

            __syncthreads();

            ij = i ^ j;

            if (ij > i)
            {
                int idx = i;
                int ijdx = ij;

                int val11 = ranks[idx];
                unsigned char val12 = preeceding_characters[idx];


                int val21 = ranks[ijdx];
                unsigned char val22 = preeceding_characters[ijdx];


                bool cond = ((i & k) == 0);
                bool val1_greater_then_val2 = (val11) > (val21);
                bool val2_greater_then_val1 = (val11) < (val21);


                if (((cond) && (val1_greater_then_val2))
                    || ((!cond) && (val2_greater_then_val1))) {

                    ranks[idx] = val21;
                    preeceding_characters[idx] = val22;


                    ranks[ijdx] = val11;
                    preeceding_characters[ijdx] = val12;

                }


            }

        }
    }
    __syncthreads();
    if (index <= r) {
        Dev_Output[offset_for_bwt_insert + i] = preeceding_characters[i];
    }



}

__global__ void Bitonic_Sort_128_Suffixes_Shared_Mem(int* Block_info, int* D, unsigned char* Dev_Input, unsigned char* Dev_Output, int offset, int size_Dev_Input, int size_D, int size_Block_info)
{

    __shared__ int ranks[128];
    __shared__ unsigned char preeceding_characters[128];
    unsigned int i, ij;

    i = threadIdx.x;

    int block_id = blockIdx.x;
    int index = Block_info[(block_id + offset)] + i;
    int r = Block_info[(block_id + offset) + 1 * size_Block_info];
    int length_suffix = Block_info[(block_id + offset) + 2 * size_Block_info];
    int offset_for_bwt_insert = Block_info[(block_id + offset) + 3 * size_Block_info];
    if (index <= r) {

        int pos_D = D[index];
        int suffix_rank = D[index + 2 * size_D];
        int index_preeceding_characte_suffix = ((pos_D - length_suffix >= 0) ? pos_D - length_suffix : size_Dev_Input - W - 1);
        unsigned char preeceding_characte_suffix = Dev_Input[index_preeceding_characte_suffix];
        ranks[i] = suffix_rank;
        preeceding_characters[i] = preeceding_characte_suffix;
    }
    else {
        ranks[i] = 1e9;
    }

    for (int k = 2; k <= 128; k <<= 1)
    {
        for (int j = k >> 1; j > 0; j = j >> 1)
        {

            __syncthreads();

            ij = i ^ j;

            if (ij > i)
            {
                int idx = i;
                int ijdx = ij;

                int val11 = ranks[idx];
                unsigned char val12 = preeceding_characters[idx];


                int val21 = ranks[ijdx];
                unsigned char val22 = preeceding_characters[ijdx];


                bool cond = ((i & k) == 0);
                bool val1_greater_then_val2 = (val11) > (val21);
                bool val2_greater_then_val1 = (val11) < (val21);


                if (((cond) && (val1_greater_then_val2))
                    || ((!cond) && (val2_greater_then_val1))) {

                    ranks[idx] = val21;
                    preeceding_characters[idx] = val22;


                    ranks[ijdx] = val11;
                    preeceding_characters[ijdx] = val12;

                }


            }

        }
    }
    __syncthreads();
    if (index <= r) {
        Dev_Output[offset_for_bwt_insert + i] = preeceding_characters[i];
    }



}

__global__ void Bitonic_Sort_256_Suffixes_Shared_Mem(int* Block_info, int* D, unsigned char* Dev_Input, unsigned char* Dev_Output, int offset, int size_Dev_Input, int size_D, int size_Block_info)
{

    __shared__ int ranks[256];
    __shared__ unsigned char preeceding_characters[256];
    unsigned int i, ij;

    i = threadIdx.x;

    int block_id = blockIdx.x;
    int index = Block_info[(block_id + offset)] + i;
    int r = Block_info[(block_id + offset) + 1 * size_Block_info];
    int length_suffix = Block_info[(block_id + offset) + 2 * size_Block_info];
    int offset_for_bwt_insert = Block_info[(block_id + offset) + 3 * size_Block_info];
    if (index <= r) {

        int pos_D = D[index];
        int suffix_rank = D[index + 2 * size_D];
        int index_preeceding_characte_suffix = ((pos_D - length_suffix >= 0) ? pos_D - length_suffix : size_Dev_Input - W - 1);
        unsigned char preeceding_characte_suffix = Dev_Input[index_preeceding_characte_suffix];
        ranks[i] = suffix_rank;
        preeceding_characters[i] = preeceding_characte_suffix;
    }
    else {
        ranks[i] = 1e9;
    }

    for (int k = 2; k <= 256; k <<= 1)
    {
        for (int j = k >> 1; j > 0; j = j >> 1)
        {
            __syncthreads();


            ij = i ^ j;

            if (ij > i)
            {
                int idx = i;
                int ijdx = ij;

                int val11 = ranks[idx];
                unsigned char val12 = preeceding_characters[idx];


                int val21 = ranks[ijdx];
                unsigned char val22 = preeceding_characters[ijdx];


                bool cond = ((i & k) == 0);
                bool val1_greater_then_val2 = (val11) > (val21);
                bool val2_greater_then_val1 = (val11) < (val21);


                if (((cond) && (val1_greater_then_val2))
                    || ((!cond) && (val2_greater_then_val1))) {

                    ranks[idx] = val21;
                    preeceding_characters[idx] = val22;


                    ranks[ijdx] = val11;
                    preeceding_characters[ijdx] = val12;

                }


            }

        }
    }
    __syncthreads();
    if (index <= r) {
        Dev_Output[offset_for_bwt_insert + i] = preeceding_characters[i];
    }



}

__global__ void Bitonic_Sort_512_Suffixes_Shared_Mem(int* Block_info, int* D, unsigned char* Dev_Input, unsigned char* Dev_Output, int offset, int size_Dev_Input, int size_D, int size_Block_info)
{

    __shared__ int ranks[512];
    __shared__ unsigned char preeceding_characters[512];
    unsigned int i, ij;

    i = threadIdx.x;

    int block_id = blockIdx.x;
    int index = Block_info[(block_id + offset)] + i;
    int r = Block_info[(block_id + offset) + 1 * size_Block_info];
    int length_suffix = Block_info[(block_id + offset) + 2 * size_Block_info];
    int offset_for_bwt_insert = Block_info[(block_id + offset) + 3 * size_Block_info];
    if (index <= r) {

        int pos_D = D[index];
        int suffix_rank = D[index + 2 * size_D];
        int index_preeceding_characte_suffix = ((pos_D - length_suffix >= 0) ? pos_D - length_suffix : size_Dev_Input - W - 1);
        unsigned char preeceding_characte_suffix = Dev_Input[index_preeceding_characte_suffix];
        ranks[i] = suffix_rank;
        preeceding_characters[i] = preeceding_characte_suffix;
    }
    else {
        ranks[i] = 1e9;
    }

    for (int k = 2; k <= 512; k <<= 1)
    {
        for (int j = k >> 1; j > 0; j = j >> 1)
        {
            __syncthreads();


            ij = i ^ j;

            if (ij > i)
            {
                int idx = i;
                int ijdx = ij;

                int val11 = ranks[idx];
                unsigned char val12 = preeceding_characters[idx];


                int val21 = ranks[ijdx];
                unsigned char val22 = preeceding_characters[ijdx];


                bool cond = ((i & k) == 0);
                bool val1_greater_then_val2 = (val11) > (val21);
                bool val2_greater_then_val1 = (val11) < (val21);


                if (((cond) && (val1_greater_then_val2))
                    || ((!cond) && (val2_greater_then_val1))) {

                    ranks[idx] = val21;
                    preeceding_characters[idx] = val22;


                    ranks[ijdx] = val11;
                    preeceding_characters[ijdx] = val12;

                }


            }

        }
    }
    __syncthreads();
    if (index <= r) {
        Dev_Output[offset_for_bwt_insert + i] = preeceding_characters[i];
    }



}

__global__ void Bitonic_Sort_1024_Suffixes_Shared_Mem(int* Block_info, int* D, unsigned char* Dev_Input, unsigned char* Dev_Output, int offset, int size_Dev_Input, int size_D, int size_Block_info)
{

    __shared__ int ranks[1024];
    __shared__ unsigned char preeceding_characters[1024];
    unsigned int i, ij;

    i = threadIdx.x;

    int block_id = blockIdx.x;
    int index = Block_info[(block_id + offset)] + i;
    int r = Block_info[(block_id + offset) + 1 * size_Block_info];
    int length_suffix = Block_info[(block_id + offset) + 2 * size_Block_info];
    int offset_for_bwt_insert = Block_info[(block_id + offset) + 3 * size_Block_info];
    if (index <= r) {

        int pos_D = D[index];
        int suffix_rank = D[index + 2 * size_D];
        int index_preeceding_characte_suffix = ((pos_D - length_suffix >= 0) ? pos_D - length_suffix : size_Dev_Input - W - 1);
        unsigned char preeceding_characte_suffix = Dev_Input[index_preeceding_characte_suffix];
        ranks[i] = suffix_rank;
        preeceding_characters[i] = preeceding_characte_suffix;
    }
    else {
        ranks[i] = 1e9;
    }

    for (int k = 2; k <= 1024; k <<= 1)
    {
        for (int j = k >> 1; j > 0; j = j >> 1)
        {

            __syncthreads();

            ij = i ^ j;

            if (ij > i)
            {
                int idx = i;
                int ijdx = ij;

                int val11 = ranks[idx];
                unsigned char val12 = preeceding_characters[idx];


                int val21 = ranks[ijdx];
                unsigned char val22 = preeceding_characters[ijdx];


                bool cond = ((i & k) == 0);
                bool val1_greater_then_val2 = (val11) > (val21);
                bool val2_greater_then_val1 = (val11) < (val21);


                if (((cond) && (val1_greater_then_val2))
                    || ((!cond) && (val2_greater_then_val1))) {

                    ranks[idx] = val21;
                    preeceding_characters[idx] = val22;


                    ranks[ijdx] = val11;
                    preeceding_characters[ijdx] = val12;

                }


            }

        }
    }
    __syncthreads();
    if (index <= r) {
        Dev_Output[offset_for_bwt_insert + i] = preeceding_characters[i];
    }



}

__global__ void Bitonic_Sort_2048_Suffixes_Shared_Mem(int* Block_info, int* D, unsigned char* Dev_Input, unsigned char* Dev_Output, int offset, int size_Dev_Input, int size_D, int size_Block_info)
{

    __shared__ int ranks[2048];
    __shared__ unsigned char preeceding_characters[2048];
    unsigned int i, ij;

    i = threadIdx.x;

    int block_id = blockIdx.x;
    int l = Block_info[(block_id + offset)];
    int r = Block_info[(block_id + offset) + 1 * size_Block_info];
    int length_suffix = Block_info[(block_id + offset) + 2 * size_Block_info];
    int offset_for_bwt_insert = Block_info[(block_id + offset) + 3 * size_Block_info];
    for (int index = l + i; index < 2048 + l; index += Max_Threads_Per_SM) {
        if (index <= r) {

            int pos_D = D[index];
            int suffix_rank = D[index + 2 * size_D];
            int index_preeceding_characte_suffix = ((pos_D - length_suffix >= 0) ? pos_D - length_suffix : size_Dev_Input - W - 1);
            unsigned char preeceding_characte_suffix = Dev_Input[index_preeceding_characte_suffix];
            ranks[index - l] = suffix_rank;
            preeceding_characters[index - l] = preeceding_characte_suffix;
            //if (51810 >= suffix_rank)
              //  printf("\n %d", suffix_rank);
        }
        else {
            ranks[index - l] = 1e9;
        }
    }
    for (int k = 2; k <= 2048; k <<= 1)
    {
        for (int j = k >> 1; j > 0; j = j >> 1)
        {

            __syncthreads();
            for (unsigned int index = i; index < 2048; index += Max_Threads_Per_SM) {


                ij = index ^ j;

                if (ij > index)
                {
                    int idx = index;
                    int ijdx = ij;

                    int val11 = ranks[idx];
                    unsigned char val12 = preeceding_characters[idx];


                    int val21 = ranks[ijdx];
                    unsigned char val22 = preeceding_characters[ijdx];


                    bool cond = ((index & k) == 0);
                    bool val1_greater_then_val2 = (val11) > (val21);
                    bool val2_greater_then_val1 = (val11) < (val21);


                    if (((cond) && (val1_greater_then_val2))
                        || ((!cond) && (val2_greater_then_val1))) {

                        ranks[idx] = val21;
                        preeceding_characters[idx] = val22;


                        ranks[ijdx] = val11;
                        preeceding_characters[ijdx] = val12;

                    }


                }
            }

        }
    }
    __syncthreads();
    for (int index = i; index < 2048; index += Max_Threads_Per_SM) {
        /*
        if (51810 >= ranks[index])
              printf("\n %d %d %c", index, ranks[index], preeceding_characters[index]);
              */
        if (index + l <= r) {
            Dev_Output[offset_for_bwt_insert + index] = preeceding_characters[index];
        }

    }

}

__global__ void Bitonic_Sort_4096_Suffixes_Shared_Mem(int* Block_info, int* D, unsigned char* Dev_Input, unsigned char* Dev_Output, int offset, int size_Dev_Input, int size_D, int size_Block_info)
{

    __shared__ int ranks[4096];
    __shared__ unsigned char preeceding_characters[4096];
    unsigned int i, ij;

    i = threadIdx.x;

    int block_id = blockIdx.x;
    int l = Block_info[(block_id + offset)];
    int r = Block_info[(block_id + offset) + 1 * size_Block_info];
    int length_suffix = Block_info[(block_id + offset) + 2 * size_Block_info];
    int offset_for_bwt_insert = Block_info[(block_id + offset) + 3 * size_Block_info];
    for (int index = l + i; index < 4096 + l; index += Max_Threads_Per_SM) {
        if (index <= r) {

            int pos_D = D[index];
            int suffix_rank = D[index + 2 * size_D];
            int index_preeceding_characte_suffix = ((pos_D - length_suffix >= 0) ? pos_D - length_suffix : size_Dev_Input - W - 1);
            unsigned char preeceding_characte_suffix = Dev_Input[index_preeceding_characte_suffix];
            ranks[index - l] = suffix_rank;
            preeceding_characters[index - l] = preeceding_characte_suffix;
        }
        else {
            ranks[index - l] = 1e9;
        }
    }
    for (int k = 2; k <= 4096; k <<= 1)
    {
        for (int j = k >> 1; j > 0; j = j >> 1)
        {
            __syncthreads();
            for (unsigned int index = i; index < 4096; index += Max_Threads_Per_SM) {


                ij = index ^ j;

                if (ij > index)
                {
                    int idx = index;
                    int ijdx = ij;

                    int val11 = ranks[idx];
                    unsigned char val12 = preeceding_characters[idx];


                    int val21 = ranks[ijdx];
                    unsigned char val22 = preeceding_characters[ijdx];


                    bool cond = ((index & k) == 0);
                    bool val1_greater_then_val2 = (val11) > (val21);
                    bool val2_greater_then_val1 = (val11) < (val21);


                    if (((cond) && (val1_greater_then_val2))
                        || ((!cond) && (val2_greater_then_val1))) {

                        ranks[idx] = val21;
                        preeceding_characters[idx] = val22;


                        ranks[ijdx] = val11;
                        preeceding_characters[ijdx] = val12;

                    }


                }
            }

        }
    }

    for (int index = i; index < 4096; index += Max_Threads_Per_SM) {
        if (index + l <= r) {
            Dev_Output[offset_for_bwt_insert + index] = preeceding_characters[index];
        }

    }

}

__global__ void Bitonic_Sort_8192_Suffixes_Shared_Mem(int* Block_info, int* D, unsigned char* Dev_Input, unsigned char* Dev_Output, int offset, int size_Dev_Input, int size_D, int size_Block_info)
{

    __shared__ int ranks[8192];
    __shared__ unsigned char preeceding_characters[8192];
    unsigned int i, ij;

    i = threadIdx.x;

    int block_id = blockIdx.x;
    int l = Block_info[(block_id + offset)];
    int r = Block_info[(block_id + offset) + 1 * size_Block_info];
    int length_suffix = Block_info[(block_id + offset) + 2 * size_Block_info];
    int offset_for_bwt_insert = Block_info[(block_id + offset) + 3 * size_Block_info];
    for (int index = l + i; index < 8192 + l; index += Max_Threads_Per_SM) {
        if (index <= r) {

            int pos_D = D[index];
            int suffix_rank = D[index + 2 * size_D];
            int index_preeceding_characte_suffix = ((pos_D - length_suffix >= 0) ? pos_D - length_suffix : size_Dev_Input - W - 1);
            unsigned char preeceding_characte_suffix = Dev_Input[index_preeceding_characte_suffix];
            ranks[index - l] = suffix_rank;
            preeceding_characters[index - l] = preeceding_characte_suffix;
        }
        else {
            ranks[index - l] = 1e9;
        }
    }
    for (int k = 2; k <= 8192; k <<= 1)
    {
        for (int j = k >> 1; j > 0; j = j >> 1)
        {
            __syncthreads();
            for (unsigned int index = i; index < 8192; index += Max_Threads_Per_SM) {


                ij = index ^ j;

                if (ij > index)
                {
                    int idx = index;
                    int ijdx = ij;

                    int val11 = ranks[idx];
                    unsigned char val12 = preeceding_characters[idx];


                    int val21 = ranks[ijdx];
                    unsigned char val22 = preeceding_characters[ijdx];


                    bool cond = ((index & k) == 0);
                    bool val1_greater_then_val2 = (val11) > (val21);
                    bool val2_greater_then_val1 = (val11) < (val21);


                    if (((cond) && (val1_greater_then_val2))
                        || ((!cond) && (val2_greater_then_val1))) {

                        ranks[idx] = val21;
                        preeceding_characters[idx] = val22;


                        ranks[ijdx] = val11;
                        preeceding_characters[ijdx] = val12;

                    }


                }
            }

        }
    }

    for (int index = i; index < 8192; index += Max_Threads_Per_SM) {
        if (index + l <= r) {
            Dev_Output[offset_for_bwt_insert + index] = preeceding_characters[index];
        }

    }

}

__global__ void Init_Bitonic_Sort_Suffixe_Group_Global(int* Block_info, int* D, unsigned char* Dev_Input, unsigned long long* Ranks, unsigned char* Preeceding_Characters, int* Prefix_Sum, int offset_Block_info, int size_Dev_Input, int blocks_per_sort, int size_D, int size_Block_info, int offset_most_left)

{

    int thread_id = threadIdx.x;


    int block_id = blockIdx.x / blocks_per_sort;
    int rest = blockIdx.x - block_id * blocks_per_sort;


    unsigned int left = ((block_id + offset_Block_info > 0) ? Prefix_Sum[block_id + offset_Block_info - 1] : 0) - offset_most_left;
    int block_offset = left + rest * Max_Threads_Per_SM;
    int l = Block_info[(block_id + offset_Block_info)] + rest * Max_Threads_Per_SM;
    int r = Block_info[(block_id + offset_Block_info) + 1 * size_Block_info];
    int length_suffix = Block_info[(block_id + offset_Block_info) + 2 * size_Block_info];

    int index = thread_id + l;
    if (index <= r) {

        int pos_D = D[index];

        unsigned int suffix_rank = (D[index + 2 * size_D] + 1);
        int index_preeceding_characte_suffix = ((pos_D - length_suffix >= 0) ? pos_D - length_suffix : size_Dev_Input - W - 1);
        unsigned char preeceding_characte_suffix = Dev_Input[index_preeceding_characte_suffix];
        unsigned long long rank = suffix_rank;
        rank |= ((unsigned long long)left) << (32);
        Ranks[thread_id + block_offset] = rank;
        Preeceding_Characters[thread_id + block_offset] = preeceding_characte_suffix;
    }

}



__global__ void Bitonic_Sort_Suffixe_Group_Global(int* Ranks, unsigned char* Preeceding_Characters, int j, int k, int blocks_per_sort)
{




    unsigned int i, ij;
    unsigned int block_id = blockIdx.x / blocks_per_sort;
    unsigned int rest = blockIdx.x - block_id * blocks_per_sort;
    unsigned int offset = block_id * blocks_per_sort * Max_Threads_Per_SM;
    i = rest * Max_Threads_Per_SM + threadIdx.x;

    ij = i ^ j;

    if (ij > i)
    {


        int val11 = Ranks[i + offset];
        unsigned char val12 = Preeceding_Characters[i + offset];


        int val21 = Ranks[ij + offset];
        unsigned char val22 = Preeceding_Characters[ij + offset];


        bool cond = ((i & k) == 0);
        bool val1_greater_then_val2 = (val11) > (val21);
        bool val2_greater_then_val1 = (val11) < (val21);


        if (((cond) && (val1_greater_then_val2))
            || ((!cond) && (val2_greater_then_val1))) {

            Ranks[i + offset] = val21;
            Preeceding_Characters[i + offset] = val22;


            Ranks[ij + offset] = val11;
            Preeceding_Characters[ij + offset] = val12;

        }





    }


}




__global__ void BWT_Fill(int* Block_info, unsigned char* Preeceding_Characters, unsigned char* Dev_Output, int* Prefix_Sum, int offset_Block_info, int size_Dev_Input, int blocks_per_sort, int size_Block_info, int offset_most_left)

{

    int thread_id = threadIdx.x;
    int block_id = blockIdx.x / blocks_per_sort;
    int rest = blockIdx.x - block_id * blocks_per_sort;

    unsigned int left = ((block_id + offset_Block_info > 0) ? Prefix_Sum[block_id + offset_Block_info - 1] : 0) - offset_most_left;
    int block_offset = left + rest * Max_Threads_Per_SM;
    int l = Block_info[(block_id + offset_Block_info)] + rest * Max_Threads_Per_SM;
    int r = Block_info[(block_id + offset_Block_info) + 1 * size_Block_info];

    int offset_for_bwt_insert = Block_info[(block_id + offset_Block_info) + 3 * size_Block_info];
    int index = thread_id + l;
    if (index <= r) {
        Dev_Output[offset_for_bwt_insert + thread_id + rest * Max_Threads_Per_SM] = Preeceding_Characters[thread_id + block_offset];
    }
}




void Compute_S() {

    int size;
    cudaMalloc((void**)&Block_L_Values, (size_Block_L_Values_times_8192) * sizeof(int));

    cudaMalloc((void**)&L_Values, (size_D) * sizeof(short));

    cudaMalloc((void**)&Auxiliar1, maximum1(size_D, size_Block_L_Values_times_8192) * sizeof(int));
    cudaMalloc((void**)&Auxiliar2, maximum1(size_D, size_Block_L_Values_times_8192) * sizeof(int));


    Init_Vector << <num_blocks_size_D, MAX_THREADS_PER_BLOCK >> > (Auxiliar1, size_D, 0);
    Init_Vector << <num_blocks_size_D, MAX_THREADS_PER_BLOCK >> > (Auxiliar2, size_D, 0);

    Compute_S1 << <num_blocks_size_D, MAX_THREADS_PER_BLOCK >> > (D, Dev_Input, Auxiliar1, L_Values, size_D);

    thrust::inclusive_scan(
        thrust::device_pointer_cast(Auxiliar1),
        thrust::device_pointer_cast(Auxiliar1 + size_D),
        thrust::device_pointer_cast(Auxiliar1)
    );

    cudaMemcpy(&size_S, Auxiliar1 + (size_D - 1), sizeof(int), cudaMemcpyDeviceToHost);


    //smallest_two_potenz_larger_than_size_S = smallest_two_potenz_larger_than_k(size_S);
    //num_blocks_smallest_two_potenz_larger_than_size_S = (smallest_two_potenz_larger_than_size_S + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
    //int num_blocks_smallest_two_potenz_larger_than_size_S_2048 = (smallest_two_potenz_larger_than_size_S + 2048 - 1) / 2048;
    //int* S_Copy;
    //cudaMalloc((void**)&S_Copy, (size_S * 4) * sizeof(int));

    cudaMalloc((void**)&S_L, (size_S) * sizeof(unsigned int));
    cudaMalloc((void**)&S_Interval_Length, (size_S) * sizeof(unsigned int));
    cudaMalloc((void**)&S_Length_Suffix, (size_S) * sizeof(unsigned short));
    cudaMalloc((void**)&S_Suffix_End, (size_S) * sizeof(unsigned int));

    cudaMalloc((void**)&S_Pos, (size_S) * sizeof(unsigned int));
    cudaMalloc((void**)&S_Rank, (size_S) * sizeof(unsigned int));
    cudaMalloc((void**)&Cur_Prefix_S, (size_S) * sizeof(unsigned long long));

    cudaMalloc((void**)&S_Pos_New, (size_S) * sizeof(unsigned int));
    cudaMalloc((void**)&S_Rank_New, (size_S) * sizeof(unsigned int));
    cudaMalloc((void**)&Cur_Prefix_S_New, (size_S) * sizeof(unsigned long long));

    cudaMalloc((void**)&S_L_Final, (size_S) * sizeof(unsigned int));
    cudaMalloc((void**)&S_Interval_Length_Final, (size_S) * sizeof(unsigned int));
    cudaMalloc((void**)&S_Length_Suffix_Final, (size_S) * sizeof(unsigned short));





    //cudaMalloc((void**)&Fixed_Length_S_Prefix, (smallest_two_potenz_larger_than_size_S) * sizeof(ulonglong2));

    //if (smallest_two_potenz_larger_than_size_S - size_S > 0)
      //  Init_Vector1 << < (smallest_two_potenz_larger_than_size_S - size_S + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK >> > (S, smallest_two_potenz_larger_than_size_S - size_S, 1e9, size_S);

    num_blocks_size_S = (size_S + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;

    Compute_S2 << <num_blocks_size_D, MAX_THREADS_PER_BLOCK >> > (S_L, S_Length_Suffix, D, Auxiliar1, size_D, size_S);

    Init_Vector << <num_blocks_size_Block_L_Values_times_8192, MAX_THREADS_PER_BLOCK >> > (Auxiliar2, size_Block_L_Values_times_8192, size_D);
    Init_Vector << <num_blocks_size_Block_L_Values_times_8192, MAX_THREADS_PER_BLOCK >> > (Block_L_Values, size_Block_L_Values_times_8192, size_D);

    Compute_S3 << <size_Block_L_Values, 1024 >> > (L_Values, Block_L_Values, size_D);

    Pointer_Jumping_Block_L_Values(Block_L_Values, Auxiliar2, size_Block_L_Values, size_D);

    Compute_S4 << <size_Block_L_Values, 1024 >> > (S_Interval_Length, D, Auxiliar1, L_Values, Block_L_Values, size_D, size_Block_L_Values, size_S);


    cudaFree(Block_L_Values);
    cudaFree(L_Values);

    cudaFree(Auxiliar1);
    cudaFree(Auxiliar2);


    int size_S_Sorting = size_S;
    int size_Block_Info_Radix_Sorting = (size_S_Sorting + 512 - 1) / 512;
    int size_Auxiliar1 = 0;



    cudaMalloc((void**)&Block_Info_Radix_Sorting, (size_Block_Info_Radix_Sorting * 2) * sizeof(int));
    cudaMalloc((void**)&Block_Group_Info, size_Block_Info_Radix_Sorting * sizeof(int));
    cudaMalloc((void**)&Sum_Group_Index, size_Block_Info_Radix_Sorting * sizeof(int));
    cudaMalloc((void**)&Group_Offset, size_Block_Info_Radix_Sorting * sizeof(int));

    Init_Block_Info_Radix_Sorting << < (size_Block_Info_Radix_Sorting + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK >> > (Block_Info_Radix_Sorting, Block_Group_Info, Sum_Group_Index, Group_Offset, size_Block_Info_Radix_Sorting, size_S);

    Init_Vector << < (size_S_Sorting + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK >> > (S_Rank, size_S_Sorting, 0);

    Init_S_Suffix_End_and_S_Pos << <num_blocks_size_S, MAX_THREADS_PER_BLOCK >> > (S_Suffix_End, S_Pos, S_L, D, size_S);

    int it = 0;
    while (size_S_Sorting > 0) {


        int size_Block_Info_Radix_Sorting_256 = size_Block_Info_Radix_Sorting * 256;
        //printf("\n %d %d %d %d", size_S_Sorting, size_Block_Info_Radix_Sorting, size_Block_Info_Radix_Sorting_256,it);
        if (size_Auxiliar1 < maximum1(size_Block_Info_Radix_Sorting_256, size_S_Sorting)) {

            if (size_Auxiliar1 > 0) {
                cudaFree(Auxiliar1);
                cudaFree(Auxiliar2);
                cudaFree(Auxiliar3);
            }
            size_Auxiliar1 = maximum1(size_Block_Info_Radix_Sorting_256, size_S_Sorting);
            cudaMalloc((void**)&Auxiliar1, maximum1(size_Block_Info_Radix_Sorting_256, size_S_Sorting) * sizeof(int));
            cudaMalloc((void**)&Auxiliar2, maximum1(size_Block_Info_Radix_Sorting_256, size_S_Sorting) * sizeof(int));
            cudaMalloc((void**)&Auxiliar3, maximum1(size_Block_Info_Radix_Sorting_256, size_S_Sorting) * sizeof(int));
        }
        int num_blocks_size_Block_Info_Radix_Sorting_256 = (size_Block_Info_Radix_Sorting_256 + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
        int num_blocks_size_S_Sorting = (size_S_Sorting + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;

        Init_Vector << <num_blocks_size_Block_Info_Radix_Sorting_256, MAX_THREADS_PER_BLOCK >> > (Auxiliar1, size_Block_Info_Radix_Sorting_256, 0);
        //Init_Vector << <num_blocks_size_Block_Info_Radix_Sorting_256, MAX_THREADS_PER_BLOCK >> > (Auxiliar2, size_Block_Info_Radix_Sorting_256, 0);
        if (it % 8 == 0)
            Load_8_Byte_Chunk_S << < num_blocks_size_S_Sorting, MAX_THREADS_PER_BLOCK >> > (Cur_Prefix_S, S_Suffix_End, S_Pos, S_Length_Suffix, Dev_Input, size_S_Sorting, it);

        RadixSort_S1 << <size_Block_Info_Radix_Sorting, 64 >> > (S_Pos, S_Rank, Cur_Prefix_S, S_Pos_New, S_Rank_New, Cur_Prefix_S_New, S_L, S_Interval_Length, S_Length_Suffix, S_L_Final, S_Interval_Length_Final, S_Length_Suffix_Final, S_Suffix_End, Dev_Input, Block_Info_Radix_Sorting, Auxiliar1, Block_Group_Info, Sum_Group_Index, size_S_Sorting, it, size_Block_Info_Radix_Sorting, size_S);

        swap(S_Pos, S_Pos_New);
        swap(Cur_Prefix_S, Cur_Prefix_S_New);
        swap(S_Rank, S_Rank_New);
        //Prefix_Sum_Radix_Sort(Auxiliar1, Auxiliar2, Block_Group_Info, Res, size_Block_Info_Radix_Sorting);

        thrust::inclusive_scan(
            thrust::device_pointer_cast(Auxiliar1),
            thrust::device_pointer_cast(Auxiliar1 + size_Block_Info_Radix_Sorting_256),
            thrust::device_pointer_cast(Auxiliar1)
        );

        RadixSort2 << < num_blocks_size_Block_Info_Radix_Sorting_256, MAX_THREADS_PER_BLOCK >> > (Auxiliar1, Auxiliar2, Block_Group_Info, Sum_Group_Index, size_Block_Info_Radix_Sorting);

        RadixSort3 << < num_blocks_size_Block_Info_Radix_Sorting_256, MAX_THREADS_PER_BLOCK >> > (Auxiliar1, Auxiliar2, Sum_Group_Index, size_Block_Info_Radix_Sorting);

        RadixSort_S4 << <size_Block_Info_Radix_Sorting, 64 >> > (S_Pos, S_Rank, Cur_Prefix_S, S_Pos_New, S_Rank_New, Cur_Prefix_S_New, Block_Info_Radix_Sorting, Auxiliar1, Block_Group_Info, Sum_Group_Index, Group_Offset, size_S_Sorting, it, size_Block_Info_Radix_Sorting);

        swap(S_Pos, S_Pos_New);
        swap(Cur_Prefix_S, Cur_Prefix_S_New);
        swap(S_Rank, S_Rank_New);

        Init_Vector << <num_blocks_size_S_Sorting, MAX_THREADS_PER_BLOCK >> > (Auxiliar1, size_S_Sorting, 0);
        //Init_Vector << <num_blocks_size_S_Sorting, MAX_THREADS_PER_BLOCK >> > (Auxiliar2, size_S_Sorting, 0);

        Reduce_S1 << < num_blocks_size_S_Sorting, MAX_THREADS_PER_BLOCK >> > (S_Rank, Auxiliar1, size_S_Sorting, size_S);

        thrust::inclusive_scan(
            thrust::device_pointer_cast(Auxiliar1),
            thrust::device_pointer_cast(Auxiliar1 + size_S_Sorting),
            thrust::device_pointer_cast(Auxiliar1)
        );


        int size_S_New;

        cudaMemcpy(&size_S_New, Auxiliar1 + (size_S_Sorting - 1), sizeof(int), cudaMemcpyDeviceToHost);


        if (size_S_New == 0)
            break;

        Init_Vector << <(size_S_New + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK >> > (Auxiliar2, size_S_New, 0);
        //Init_Vector << <(size_S_New + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK >> > (Auxiliar3, size_S_New, 0);

        Reduce_S2 << < num_blocks_size_S_Sorting, MAX_THREADS_PER_BLOCK >> > (S_Pos, S_Rank, Cur_Prefix_S, S_Pos_New, S_Rank_New, Cur_Prefix_S_New, Auxiliar1, Auxiliar2, size_S_Sorting, size_S_New);

        size_S_Sorting = size_S_New;
        num_blocks_size_S_Sorting = (size_S_Sorting + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;

        swap(S_Pos, S_Pos_New);
        swap(Cur_Prefix_S, Cur_Prefix_S_New);
        swap(S_Rank, S_Rank_New);


        thrust::inclusive_scan(
            thrust::device_pointer_cast(Auxiliar2),
            thrust::device_pointer_cast(Auxiliar2 + size_S_Sorting),
            thrust::device_pointer_cast(Auxiliar2)
        );

        Reduce_3 << < num_blocks_size_S_Sorting, MAX_THREADS_PER_BLOCK >> > (Auxiliar2, Auxiliar1, size_S_Sorting);

        Reduce_4 << < num_blocks_size_S_Sorting, MAX_THREADS_PER_BLOCK >> > (Auxiliar2, Auxiliar1, size_S_Sorting);

        Init_Vector << <num_blocks_size_S_Sorting, MAX_THREADS_PER_BLOCK >> > (Auxiliar1, size_S_Sorting, 0);
        //Init_Vector << <num_blocks_size_S_Sorting, MAX_THREADS_PER_BLOCK >> > (Auxiliar3, size_S_Sorting, 0);

        Reduce_5 << < num_blocks_size_S_Sorting, MAX_THREADS_PER_BLOCK >> > (Auxiliar1, Auxiliar2, size_S_Sorting);

        thrust::inclusive_scan(
            thrust::device_pointer_cast(Auxiliar1),
            thrust::device_pointer_cast(Auxiliar1 + size_S_Sorting),
            thrust::device_pointer_cast(Auxiliar1)
        );

        int size_Block_Info_Radix_Sorting_New;

        cudaMemcpy(&size_Block_Info_Radix_Sorting_New, Auxiliar1 + (size_S_Sorting - 1), sizeof(int), cudaMemcpyDeviceToHost);


        if (size_Block_Info_Radix_Sorting < size_Block_Info_Radix_Sorting_New) {
            cudaFree(Block_Info_Radix_Sorting);
            cudaFree(Block_Group_Info);
            cudaFree(Sum_Group_Index);
            cudaFree(Group_Offset);
            cudaMalloc((void**)&Block_Info_Radix_Sorting, (size_Block_Info_Radix_Sorting_New * 2) * sizeof(int));
            cudaMalloc((void**)&Block_Group_Info, size_Block_Info_Radix_Sorting_New * sizeof(int));
            cudaMalloc((void**)&Sum_Group_Index, size_Block_Info_Radix_Sorting_New * sizeof(int));
            cudaMalloc((void**)&Group_Offset, size_Block_Info_Radix_Sorting_New * sizeof(int));

        }


        size_Block_Info_Radix_Sorting = size_Block_Info_Radix_Sorting_New;
        Reduce_6 << < num_blocks_size_S_Sorting, MAX_THREADS_PER_BLOCK >> > (Auxiliar1, Auxiliar2, Block_Info_Radix_Sorting, Block_Group_Info, Sum_Group_Index, Auxiliar3, size_S_Sorting, size_Block_Info_Radix_Sorting);

        Reduce_7 << <num_blocks_size_S_Sorting, MAX_THREADS_PER_BLOCK >> > (Auxiliar1, Auxiliar2, Auxiliar3, Group_Offset, size_S_Sorting, size_Block_Info_Radix_Sorting);
        it++;

    }






    cudaFree(Auxiliar3);
    cudaFree(Block_Info_Radix_Sorting);
    cudaFree(Block_Group_Info);
    cudaFree(Sum_Group_Index);
    cudaFree(Group_Offset);

    cudaFree(S_L);
    cudaFree(S_Interval_Length);
    cudaFree(S_Length_Suffix);
    cudaFree(S_Suffix_End);

    cudaFree(S_Pos);
    cudaFree(S_Rank);
    cudaFree(Cur_Prefix_S);

    cudaFree(S_Pos_New);
    cudaFree(S_Rank_New);
    cudaFree(Cur_Prefix_S_New);


    cudaFree(Auxiliar1);
    cudaFree(Auxiliar2);
    cudaMalloc((void**)&Auxiliar1, (size_S) * sizeof(int));
    cudaMalloc((void**)&Auxiliar2, (size_S) * sizeof(int));

    S_L = S_L_Final;
    S_Interval_Length = S_Interval_Length_Final;
    S_Length_Suffix = S_Length_Suffix_Final;



    //Check_S << <num_blocks_size_S, MAX_THREADS_PER_BLOCK >> > (S, S_Copy, Dev_Input, size_S);



    Init_Vector << <num_blocks_size_S, MAX_THREADS_PER_BLOCK >> > (Auxiliar1, size_S, 0);
    Init_Vector << <num_blocks_size_S, MAX_THREADS_PER_BLOCK >> > (Auxiliar2, size_S, 0);

    Compute_S5 << <num_blocks_size_S, MAX_THREADS_PER_BLOCK >> > (S_Interval_Length_Final, Auxiliar1, size_S);

    thrust::inclusive_scan(
        thrust::device_pointer_cast(Auxiliar1),
        thrust::device_pointer_cast(Auxiliar1 + size_S),
        thrust::device_pointer_cast(Auxiliar1)
    );



    Compute_S6 << <num_blocks_size_S, MAX_THREADS_PER_BLOCK >> > (S_L, S_Interval_Length, S_Length_Suffix, Dev_Input, Dev_Output, Auxiliar1, Auxiliar2, D, size_S, n_enlarged, size_D);

    //Copy_S << <num_blocks_size_S, MAX_THREADS_PER_BLOCK >> > (S, S_L, S_Interval_Length, S_Length_Suffix, D, size_S);
    /*
    int* S_Local = new int[4 * size_S];

    cudaMemcpy(S_Local, S, 4 * size_S * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size_S; i++) {
        printf("\n %d %d %d %d", S_Local[i], S_Local[i + size_S], S_Local[i + 2 * size_S], S_Local[i + 3 * size_S]);
    }
    */
    thrust::inclusive_scan(
        thrust::device_pointer_cast(Auxiliar2),
        thrust::device_pointer_cast(Auxiliar2 + size_S),
        thrust::device_pointer_cast(Auxiliar2)
    );

    cudaMemcpy(&size_Suffixes_Without_Same_Preeceding_Char, Auxiliar2 + (size_S - 1), sizeof(int), cudaMemcpyDeviceToHost);
    if (size_Suffixes_Without_Same_Preeceding_Char > 0) {
        smallest_two_potenz_larger_than_size_Suffixes_Without_Same_Preeceding_Char = smallest_two_potenz_larger_than_k(size_Suffixes_Without_Same_Preeceding_Char);
        num_blocks_size_Suffixes_Without_Same_Preeceding_Char = (size_Suffixes_Without_Same_Preeceding_Char + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
        num_blocks_smallest_two_potenz_larger_than_size_Suffixes_Without_Same_Preeceding_Char = (smallest_two_potenz_larger_than_size_Suffixes_Without_Same_Preeceding_Char + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;

        cudaMalloc((void**)&Suffixes_Without_Same_Preeceding_Char, (smallest_two_potenz_larger_than_size_Suffixes_Without_Same_Preeceding_Char * 4) * sizeof(int));

        Compute_Suffixes_Without_Same_Preeceding_Char1 << <num_blocks_size_S, MAX_THREADS_PER_BLOCK >> > (S_L, S_Interval_Length, S_Length_Suffix, Suffixes_Without_Same_Preeceding_Char, Auxiliar2, Auxiliar1, size_S, smallest_two_potenz_larger_than_size_Suffixes_Without_Same_Preeceding_Char);


        if (smallest_two_potenz_larger_than_size_Suffixes_Without_Same_Preeceding_Char - size_Suffixes_Without_Same_Preeceding_Char > 0)
            Init_Vector << < (smallest_two_potenz_larger_than_size_Suffixes_Without_Same_Preeceding_Char - size_Suffixes_Without_Same_Preeceding_Char + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK >> > (Suffixes_Without_Same_Preeceding_Char, smallest_two_potenz_larger_than_size_Suffixes_Without_Same_Preeceding_Char - size_Suffixes_Without_Same_Preeceding_Char, 1e9, 1, size_Suffixes_Without_Same_Preeceding_Char);

        cudaFree(S_L);
        cudaFree(S_Interval_Length);
        cudaFree(S_Length_Suffix);

        size = smallest_two_potenz_larger_than_size_Suffixes_Without_Same_Preeceding_Char;
        for (int k = 2; k <= size; k <<= 1)
        {
            for (int j = k >> 1; j > 0; j = j >> 1)
            {

                Bitonic_Sort_Suffixes_Without_Same_Preeceding_Char << <num_blocks_smallest_two_potenz_larger_than_size_Suffixes_Without_Same_Preeceding_Char, MAX_THREADS_PER_BLOCK >> > (Suffixes_Without_Same_Preeceding_Char, j, k, size);

            }


        }

        Init_Vector << <num_blocks_size_Suffixes_Without_Same_Preeceding_Char, MAX_THREADS_PER_BLOCK >> > (Auxiliar1, size_Suffixes_Without_Same_Preeceding_Char, 0);
        Init_Vector << <num_blocks_size_Suffixes_Without_Same_Preeceding_Char, MAX_THREADS_PER_BLOCK >> > (Auxiliar2, size_Suffixes_Without_Same_Preeceding_Char, 0);

        Compute_Cnt_Blocks_Per_Sort1 << <num_blocks_size_Suffixes_Without_Same_Preeceding_Char, MAX_THREADS_PER_BLOCK >> > (Suffixes_Without_Same_Preeceding_Char, Auxiliar1, Res, size_Suffixes_Without_Same_Preeceding_Char, smallest_two_potenz_larger_than_size_Suffixes_Without_Same_Preeceding_Char);

        cudaMemcpy(&size_Ranks, Res, sizeof(int), cudaMemcpyDeviceToHost);

        size_Ranks = ((size_Ranks < Size_Max_Sort * 2) ? Size_Max_Sort * 2 : size_Ranks);

        cudaMalloc((void**)&Ranks, (size_Ranks) * sizeof(unsigned long long));
        cudaMalloc((void**)&Preeceding_Characters, (size_Ranks) * sizeof(char));

        thrust::inclusive_scan(
            thrust::device_pointer_cast(Auxiliar1),
            thrust::device_pointer_cast(Auxiliar1 + size_Suffixes_Without_Same_Preeceding_Char),
            thrust::device_pointer_cast(Auxiliar1)
        );


        //Prefix_Sum(Auxiliar1, Auxiliar2, Res, size_Suffixes_Without_Same_Preeceding_Char);

        Init_Vector << <num_blocks_size_Suffixes_Without_Same_Preeceding_Char, MAX_THREADS_PER_BLOCK >> > (Auxiliar2, size_Suffixes_Without_Same_Preeceding_Char, 0);

        Compute_Cnt_Blocks_Per_Sort2 << <num_blocks_size_Suffixes_Without_Same_Preeceding_Char, MAX_THREADS_PER_BLOCK >> > (Suffixes_Without_Same_Preeceding_Char, Auxiliar1, Auxiliar2, size_Suffixes_Without_Same_Preeceding_Char, smallest_two_potenz_larger_than_size_Suffixes_Without_Same_Preeceding_Char);

        swap(Auxiliar1, Auxiliar2);

        //Prefix_Sum(Auxiliar1, Auxiliar2, Res, size_Suffixes_Without_Same_Preeceding_Char);

        thrust::inclusive_scan(
            thrust::device_pointer_cast(Auxiliar1),
            thrust::device_pointer_cast(Auxiliar1 + size_Suffixes_Without_Same_Preeceding_Char),
            thrust::device_pointer_cast(Auxiliar1)
        );

        cudaMemcpy(&size_CPU_Blocks_Array, Auxiliar1 + (size_Suffixes_Without_Same_Preeceding_Char - 1), sizeof(int), cudaMemcpyDeviceToHost);

        cudaMalloc((void**)&Dev_CPU_Blocks_Array, (size_CPU_Blocks_Array * 3) * sizeof(int));
        CPU_Blocks_Array = new int[size_CPU_Blocks_Array * 3];

        Compute_Cnt_Blocks_Per_Sort3 << <num_blocks_size_Suffixes_Without_Same_Preeceding_Char, MAX_THREADS_PER_BLOCK >> > (Suffixes_Without_Same_Preeceding_Char, Auxiliar1, Auxiliar2, Dev_CPU_Blocks_Array, size_Suffixes_Without_Same_Preeceding_Char, smallest_two_potenz_larger_than_size_Suffixes_Without_Same_Preeceding_Char);

        cudaMemcpy(CPU_Blocks_Array, Dev_CPU_Blocks_Array, (size_CPU_Blocks_Array * 3) * sizeof(int), cudaMemcpyDeviceToHost);

        int offset = 0;
        int* Block_info = Suffixes_Without_Same_Preeceding_Char;
        int offset_left_most = 0;
        for (int i = 0; i < size_CPU_Blocks_Array; i++) {
            int two_potenz = CPU_Blocks_Array[i * 3 + 1];
            int next_offset = CPU_Blocks_Array[i * 3] + 1;
            int next_offset_left_most = CPU_Blocks_Array[i * 3 + 2];
            //printf("\n%d", two_potenz);
            if (two_potenz <= 64) {
                int num_blocks = next_offset - offset;
                Bitonic_Sort_64_Suffixes_Shared_Mem << <num_blocks, 64 >> > (Block_info, D, Dev_Input, Dev_Output, offset, n_enlarged, size_D, smallest_two_potenz_larger_than_size_Suffixes_Without_Same_Preeceding_Char);
            }
            else if (two_potenz <= 128) {
                int num_blocks = next_offset - offset;
                Bitonic_Sort_128_Suffixes_Shared_Mem << <num_blocks, 128 >> > (Block_info, D, Dev_Input, Dev_Output, offset, n_enlarged, size_D, smallest_two_potenz_larger_than_size_Suffixes_Without_Same_Preeceding_Char);
            }
            else if (two_potenz <= 256) {
                int num_blocks = next_offset - offset;
                Bitonic_Sort_256_Suffixes_Shared_Mem << <num_blocks, 256 >> > (Block_info, D, Dev_Input, Dev_Output, offset, n_enlarged, size_D, smallest_two_potenz_larger_than_size_Suffixes_Without_Same_Preeceding_Char);
            }
            else if (two_potenz <= 512) {
                int num_blocks = next_offset - offset;
                Bitonic_Sort_512_Suffixes_Shared_Mem << <num_blocks, 512 >> > (Block_info, D, Dev_Input, Dev_Output, offset, n_enlarged, size_D, smallest_two_potenz_larger_than_size_Suffixes_Without_Same_Preeceding_Char);
            }
            else if (two_potenz <= 1024) {
                int num_blocks = next_offset - offset;
                Bitonic_Sort_1024_Suffixes_Shared_Mem << <num_blocks, 1024 >> > (Block_info, D, Dev_Input, Dev_Output, offset, n_enlarged, size_D, smallest_two_potenz_larger_than_size_Suffixes_Without_Same_Preeceding_Char);
            }
            else if (two_potenz <= 2048) {
                int num_blocks = next_offset - offset;
                Bitonic_Sort_2048_Suffixes_Shared_Mem << <num_blocks, Max_Threads_Per_SM >> > (Block_info, D, Dev_Input, Dev_Output, offset, n_enlarged, size_D, smallest_two_potenz_larger_than_size_Suffixes_Without_Same_Preeceding_Char);
            }
            else if (two_potenz <= 4096) {
                int num_blocks = next_offset - offset;
                Bitonic_Sort_4096_Suffixes_Shared_Mem << <num_blocks, Max_Threads_Per_SM >> > (Block_info, D, Dev_Input, Dev_Output, offset, n_enlarged, size_D, smallest_two_potenz_larger_than_size_Suffixes_Without_Same_Preeceding_Char);
            }
            else if (two_potenz <= 8192) {
                int num_blocks = next_offset - offset;
                Bitonic_Sort_8192_Suffixes_Shared_Mem << <num_blocks, Max_Threads_Per_SM >> > (Block_info, D, Dev_Input, Dev_Output, offset, n_enlarged, size_D, smallest_two_potenz_larger_than_size_Suffixes_Without_Same_Preeceding_Char);
            }
            else {
                int blocks_per_sort = two_potenz / Max_Threads_Per_SM;
                int num_blocks = (next_offset - offset) * blocks_per_sort;

                Init_Bitonic_Sort_Suffixe_Group_Global << <num_blocks, Max_Threads_Per_SM >> > (Block_info, D, Dev_Input, Ranks, Preeceding_Characters, Auxiliar2, offset, n_enlarged, blocks_per_sort, size_D, smallest_two_potenz_larger_than_size_Suffixes_Without_Same_Preeceding_Char, offset_left_most);
                size = next_offset_left_most - offset_left_most;
                /*
                for (int k = 2; k <= size; k <<= 1)
                {
                    for (int j = k >> 1; j > 0; j = j >> 1)
                    {
                        Bitonic_Sort_Suffixe_Group_Global << <num_blocks, Max_Threads_Per_SM >> > (Ranks, Preeceding_Characters, j, k, blocks_per_sort);
                    }
                }
                */

                thrust::sort_by_key(
                    thrust::device_pointer_cast(Ranks),
                    thrust::device_pointer_cast(Ranks + size),
                    thrust::device_pointer_cast(Preeceding_Characters)
                );

                BWT_Fill << <num_blocks, Max_Threads_Per_SM >> > (Block_info, Preeceding_Characters, Dev_Output, Auxiliar2, offset, n_enlarged, blocks_per_sort, smallest_two_potenz_larger_than_size_Suffixes_Without_Same_Preeceding_Char, offset_left_most);
            }

            offset_left_most = next_offset_left_most;
            offset = next_offset;
        }


    }
}
void print_Output() {
    cudaMemcpy(output, Dev_Output, (n + 1) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    
    FILE* f = fopen("out.txt", "w");
    fprintf(f, "%d\n", n);
    for (int i = 0; i < n; i++)
        fprintf(f, "%d ", (output[i]));

        
    /*
    printf("%d\n", n);
    for (int i = 0; i < n; i++)
        printf("%c", (output[i] - 128));
       */ 
}


int main()
{

    int device;
    cudaGetDevice(&device);

    struct cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    Scan_input();

    Compute_Params();

    Malloc_And_Copy_On_GPU();

    Compute_D();

    Compute_S();

    print_Output();

    return 0;
}

