#include <vector>
#include <iostream>
#ifndef PROCESS_H
#define PROCESS_H

void computeDistanceWithGPU(void *d_PQVectorSet, int *d_vectorIDs, void *d_table, float *d_dist, int dim, int count, std::vector<std::int32_t> &h_vectorIDs);
void processFunction1(void *d_PQVectorSet, int *d_vectorIDs, void *d_table, std::vector<float> &h_dist, int dim, int count);
#endif
