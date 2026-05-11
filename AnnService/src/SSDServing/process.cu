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

// 在 GPU 上实现标量的 Bit 解压和向量跟 Float 内积的计算引擎
__device__ __forceinline__ uint8_t decode_rabitq_code_at(
    const uint8_t* code, int dim_idx, int bits_per_code)
{
    const int blk = dim_idx >> 4;   // 每 16 维一个打包块
    const int lane = dim_idx & 15;

    if (bits_per_code == 8) {
        return code[dim_idx];
    }
    if (bits_per_code == 4) {
        // 16x4bit -> uint64，布局对齐 CPU UnpackVector
        const uint64_t pack = reinterpret_cast<const uint64_t*>(code)[blk];
        const int half = lane >> 3;        // 0:前8维, 1:后8维
        const int byte = lane & 7;
        const int shift = byte * 8 + half * 4;
        return static_cast<uint8_t>((pack >> shift) & 0x0F);
    }
    if (bits_per_code == 2) {
        // 16x2bit -> uint32，布局对齐 CPU UnpackVector
        const uint32_t pack = reinterpret_cast<const uint32_t*>(code)[blk];
        const int group = lane >> 2;       // 0..3, 对应位平面 0/2/4/6
        const int byte = lane & 3;
        const int shift = byte * 8 + group * 2;
        return static_cast<uint8_t>((pack >> shift) & 0x03);
    }
    if (bits_per_code == 1) {
        // 16x1bit -> uint16
        const uint16_t pack = reinterpret_cast<const uint16_t*>(code)[blk];
        return static_cast<uint8_t>((pack >> lane) & 0x01);
    }

    // 防御分支，正常不会走到
    return 0;
}

__device__ float cuda_rabitq_inner_product(
    const float* d_query, 
    const uint8_t* d_code, 
    int dim, 
    int bits_per_code)
{
    float ip = 0.0f;
    for (int i = 0; i < dim; ++i) {
        const uint8_t q = decode_rabitq_code_at(d_code, i, bits_per_code);
        ip += d_query[i] * static_cast<float>(q);
    }
    return ip;
}

// 这是相当于 rabitqlib::quant::full_est_dist 的 GPU 等效版本
__device__ float cuda_rabitq_full_est_dist(
    const uint8_t* d_code, 
    const float* d_query, 
    int dim, 
    int bits_per_code,
    float f_add, 
    float f_rescale, 
    float g_add, 
    float k1xsumq)
{
    const float ip = cuda_rabitq_inner_product(d_query, d_code, dim, bits_per_code);

    // 与 CPU 标准实现对齐：
    // est = g_add + f_add + f_rescale * (ip + k1xsumq * ((1<<bits)-1))
    const float k = static_cast<float>((1 << bits_per_code) - 1);
    const float inner_val = g_add + f_add + f_rescale * (ip + k1xsumq * k);

    // 所有距离都按平方距离处理，不做 sqrt
    return (inner_val > 0.0f) ? inner_val : 0.0f; 
}

// ----------------------------------------------------
// 我们新的 Kernel 函数
// ----------------------------------------------------
__global__ void ProcessRaBitQ(
    void     *d_QuantizedVectorSet, 
    int      *d_vectorIDs, 
    float    *d_rotated_query, 
    BondMetaMetaGPU bond_meta,  // (注意这里名字定义为了避免与头文件冲突，可以在头文件定义传入)
    float    *d_dist, 
    int       dim, 
    int       bits_per_code, // 【新增传入】告诉 GPU 当前是几 Bit 重建
    int       count,
    float     limitDist)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        int vid = d_vectorIDs[idx];
        if (vid >= 0)
        {
            // RaBitQ 数据结构：
            // 前 5 个 float 是 Meta (delta, vl, f_add, f_rescale, f_error)，占用 20 个字节
            // 后面紧接着才是 Quantized Code。
            int meta_size = 3 * sizeof(float);  // 去掉delta和vl
            // 计算 packed 占用的字节数（向上取整）
            int code_bytes = (dim * bits_per_code + 7) / 8;
            int total_bytes_per_vec = meta_size + code_bytes;

            // 定位到当前这个候选向量在全局大数组 `d_QuantizedVectorSet` 里的物理内存起始位
            uint8_t* pY = ((uint8_t*)d_QuantizedVectorSet) + ((size_t)vid * total_bytes_per_vec);

            // 获取该向量头部的 Meta 数据
            float* meta_ptr = (float*)pY;
            // 去掉delta和vl之后，位置也要减2
            float f_add     = meta_ptr[0];
            float f_rescale = meta_ptr[1];
            float f_error   = meta_ptr[2];

            // Code 区间的起始指针
            uint8_t* code_ptr = pY + meta_size;
            /////////////////////////////////////////////////////////////////////////////////////
            // // 在 ProcessRaBitQ 内部直接展开计算，便于调试
            // float ip = cuda_rabitq_inner_product(
            //     d_rotated_query, code_ptr, dim, bits_per_code
            // );

            // float inner_val = bond_meta.g_add + f_add + bond_meta.k1xsumq + f_rescale * ip;

            // if (idx == 0) {
            //     printf("TEST : inner_val=%f, ip=%f, f_add=%f, f_rescale=%f, g_add=%f, k1xsumq=%f\n",
            //         inner_val, ip, f_add, f_rescale, bond_meta.g_add, bond_meta.k1xsumq);
            // }
            /////////////////////////////////////////////////////////////////////////////////////

            // 1. 调用上方写的 GPU 版全预估距离函数 (等价于 CPU 的 L2DistanceEstimate)
            float estimateDist = cuda_rabitq_full_est_dist(
                code_ptr, d_rotated_query, dim, bits_per_code, 
                f_add, f_rescale, bond_meta.g_add, bond_meta.k1xsumq
            );

            // 2. 计算误差下限 (用于保守剪枝)
            const float err_scale = static_cast<float>(1u << static_cast<unsigned>(bits_per_code - 1));
            float low_dist = estimateDist - (f_error * bond_meta.g_error) / err_scale;

            // 3. 剪枝策略！这是融合进 GPU 的精髓。
            if (low_dist > limitDist) {
                // 如果最理想（最短）都没希望比当前搜出的最远聚类还小，它连去 SSD 重排的资格都没有。
                // 我们直接标记其距离为 3e38f，后续 CPU 收到就会丢弃它
                d_dist[idx] = 3e38f; 
            } else {
                d_dist[idx] = estimateDist;
                // d_dist[idx] = fmaxf(low_dist, 0.0f); // 先用 lower bound 作为入堆分数
            }
        }
        else 
        {
            d_dist[idx] = 3e38f; // 若 id < 0 或者失效的，同理丢弃
        }
    }
}

void computeRaBitQDistanceWithGPU(
    void     *d_QuantizedVectorSet, 
    int      *d_vectorIDs, 
    float    *d_rotated_query, 
    BondMetaMetaGPU bond_meta, 
    float    *d_dist, 
    int       dim, 
    int       bits_per_code, 
    int       count, 
    float     limitDist) 
{
    // 每个线程负责一个候选向量的内算，配置合理的 block 大小
    int block = 256; 
    int grid = (count + block - 1) / block;

    // 清空上次可能的错误
    cudaGetLastError(); 

    ProcessRaBitQ<<<grid, block>>>(
        d_QuantizedVectorSet, 
        d_vectorIDs, 
        d_rotated_query, 
        bond_meta, 
        d_dist, 
        dim, 
        bits_per_code, 
        count, 
        limitDist
    );

    // 同步并捕获 Kernel 自身崩溃的异常
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA KERNEL KILLED! Error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}
