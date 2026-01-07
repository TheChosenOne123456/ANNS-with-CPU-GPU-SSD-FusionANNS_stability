#include "inc/SSDServing/process.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

struct BloomFilter {
    bool* bloom_filter;  // GPU上的位数组
    size_t size;              // 位数组的大小，单位是位
};


__host__ void initBloomFilter(BloomFilter* bf, size_t size) {
    cudaMalloc((void**)&(bf->bloom_filter), size * sizeof(bool));
    cudaMemset(bf->bloom_filter, false, size * sizeof(bool));
    bf->size = size;
}

__device__ size_t hash(int item, int seed, size_t size) {
    size_t hash = item + seed * 31;  // 简单的哈希函数
    return hash % size;
}
__device__ bool checkandset(int *item, bool* bloom_filter, size_t size){
    bool found = true;
        /* code */
        size_t hash_value = hash(*item, 0, size);
        if(!bloom_filter[hash_value]){
            found = false;
        }
        bloom_filter[hash_value] = true;
    return found;
    
}
__global__ void filter(int *item, int numVector, bool* bloom_filter, size_t size){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx == 0)
    {
        for (size_t i = 0; i < numVector; i++)
        {
            /* code */
            bool found = checkandset(&item[i],bloom_filter,size);
            if(found == true){
                item[i] = -1;
            }
        }
    }
}




// struct BloomFilter {
//     unsigned int* bitArray;  // GPU上的位数组
//     size_t size;              // 位数组的大小，单位是位
//     int numHashes;            // 使用的哈希函数数量
// };


// __host__ void initBloomFilter(BloomFilter* bf, size_t size, int numHashes) {
//     size_t numInts = (size + 31) / 32;
//     cudaMalloc((void**)&(bf->bitArray), numInts * sizeof(unsigned int));
//     cudaMemset(bf->bitArray, 0, numInts * sizeof(unsigned int));
//     bf->size = size;
//     bf->numHashes = numHashes;
// }

// __device__ size_t hash(int item, int seed, size_t size) {
//     size_t hash = item + seed * 31;  // 简单的哈希函数
//     return hash % size;
// }

// __global__ void add(int *item, unsigned int* bitArray, size_t size, int numHashes) {
//     int idx = threadIdx.x + blockIdx.x * blockDim.x;
//     if (idx < numHashes) {
//         size_t bitPos = hash(*item, idx, size);
//         unsigned int mask = 1u << (bitPos % 32);
//         atomicOr(reinterpret_cast<unsigned int *>(bitArray) + bitPos / 32, mask);
//     }
// }

// __global__ void query(int *item, unsigned int* bitArray, size_t size, int numHashes, int *allMatch) {
//     int idx = threadIdx.x + blockIdx.x * blockDim.x;
//     if (idx < numHashes) {
//         size_t bitPos = hash(*item, idx, size);
//         unsigned int mask = 1u << (bitPos % 32);
//         unsigned int value = *(reinterpret_cast<unsigned int *>(bitArray) + bitPos / 32);
//         if (!(value & mask)) {
//             atomicExch(allMatch, 1);
//         }
//     }

//     __syncthreads();  // 确保所有线程完成

//     // 在idx == 0的线程中汇总结果
//     if (idx == 0) {
//         if (*allMatch == 0) {  // 检查共享标志
//             *item = -1;  // 假设元素存在
//         }
//     }
// }




__device__ float L2Distance(const uint8_t *pX, const uint8_t *pY, int dim)
{
    float out = 0;
    float *ptr = (float *)pX;
    for (int i = 0; i < dim; i++)
    {
        out += ptr[pY[i]];
        ptr += 256;
    }
    return out;
}

__global__ void Process(void *d_PQVectorSet, int *d_vectorIDs, void *d_table, float *d_dist, int dim, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < count)
    {
        int y = idx / dim;
        if(d_vectorIDs[y] >= 0)
        {
            int x = idx % dim;
            float *ptr = (float *)d_table;
            uint8_t *pX = (uint8_t *)d_PQVectorSet;
            ptr += 256 * x;

            atomicAdd(&d_dist[y], ptr[pX[sizeof(uint8_t) * d_vectorIDs[y] * dim + x]]);
        }
    }
}
void computeDistanceWithGPU(void *d_PQVectorSet, int *d_vectorIDs, void *d_table, float *d_dist, int dim, int count, std::vector<std::int32_t> &h_vectorIDs) 
{
    int block = 256;
    int grid = (count + block - 1) / block;
    Process<<<grid, block>>>(d_PQVectorSet, d_vectorIDs, d_table, d_dist, dim, count);
    cudaDeviceSynchronize();
}

// void computeDistanceWithGPU(void *d_PQVectorSet, int *d_vectorIDs, void *d_table, float *d_dist, int dim, int count, std::vector<int> &h_vectorIDs) 
// {
//     int numVector = count / dim;

//     BloomFilter bf;
//     initBloomFilter(&bf, 299993, 7);

//     int h_allMatch = 0;
//     int *d_allMatch;
//     cudaMalloc((void **)&d_allMatch, sizeof(int) * numVector);
//     cudaMemset(d_allMatch, 0, sizeof(int) * numVector);
//     int noAdd = 0;
//     // std::cout<<"begin noAdd is :"<<noAdd<<std::endl;
//     for (size_t i = 0; i < numVector; i++)
//     {
//         query<<<1, 7>>>(&d_vectorIDs[i], bf.bitArray, bf.size, bf.numHashes, &d_allMatch[i]);
//         // cudaMemcpy(&h_allMatch, &d_allMatch[i], sizeof(int), cudaMemcpyDeviceToHost);
//         // if(h_allMatch == 0)
//         // {
//         //     h_vectorIDs[i] = -1;
//         //     // noAdd ++;
//         // }else{
//         //     add<<<1, 7>>>(&d_vectorIDs[i], bf.bitArray, bf.size, bf.numHashes);
//         //     cudaDeviceSynchronize();
//         // }
//         add<<<1, 7>>>(&d_vectorIDs[i], bf.bitArray, bf.size, bf.numHashes);
//         cudaDeviceSynchronize();
//     }
//     // std::cout<<"noAdd is: "<<noAdd<<std::endl;

//     int block = 256;
//     int grid = (count + block - 1) / block;
//     Process<<<grid, block>>>(d_PQVectorSet, d_vectorIDs, d_table, d_dist, dim, count);
//     cudaDeviceSynchronize();

//     cudaFree(bf.bitArray);
//     cudaFree(d_allMatch);
// }

void processFunction1(void *d_PQVectorSet, int *d_vectorIDs, void *d_table, std::vector<float> &h_dist, int dim, int count)
{
    float *d_dist;
    cudaMalloc((void **)&d_dist, sizeof(float) * count);
    cudaMemset(d_dist, 0, sizeof(float) * count);

    // cudaStream_t stream;
    // cudaStreamCreate(&stream);
    // cudaMemcpyAsync(d_vectorIDs, vectorIDs.data(), sizeof(int) * count, cudaMemcpyHostToDevice, stream);

    int block = 256;
    int grid = (count + block - 1) / block;

    Process<<<grid, block>>>(d_PQVectorSet, d_vectorIDs, d_table, d_dist, dim, count);

    cudaDeviceSynchronize();

    cudaMemcpy(h_dist.data(), d_dist, sizeof(float) * count, cudaMemcpyDeviceToHost);
    cudaFree(d_table);
    cudaFree(d_dist);
    // cudaStreamDestroy(stream);
}
