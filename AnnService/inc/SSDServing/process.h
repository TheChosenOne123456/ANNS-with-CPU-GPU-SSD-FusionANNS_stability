#include <vector>
#include <iostream>
#ifndef PROCESS_H
#define PROCESS_H

// struct BondMetaMetaGPU {
//     float g_add;
//     float k1xsumq;
//     float g_error;
// };

void computeDistanceWithGPU(void *d_PQVectorSet, int *d_vectorIDs, void *d_table, float *d_dist, int dim, int count, std::vector<std::int32_t> &h_vectorIDs);
void processFunction1(void *d_PQVectorSet, int *d_vectorIDs, void *d_table, std::vector<float> &h_dist, int dim, int count);

// 新赠 RaBitQ 专属 GPU 调用声明
void computeRaBitQDistanceWithGPU(
    void   *d_QuantizedVectorSet, 
    int    *d_vectorIDs, 
    float  *d_rotated_queries,     // 【修改】传入包含所有质心偏移后的查询集
    float   *d_g_adds,     // 分开传
    float   *d_k1xsumqs,
    float   *d_g_errors,
    float  *d_dist, 
    int     dim, 
    int     bits_per_code, 
    int     count, 
    float   limitDist
);

#endif
