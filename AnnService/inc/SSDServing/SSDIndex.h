// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <limits>
#include <future>

#include "inc/Core/Common.h"
#include "inc/Core/Common/DistanceUtils.h"
#include "inc/Core/Common/QueryResultSet.h"
#include "inc/Core/SPANN/Index.h"
#include "inc/Core/SPANN/ExtraFullGraphSearcher.h"
#include "inc/Helper/VectorSetReader.h"
#include "inc/Helper/StringConvert.h"
#include "inc/SSDServing/Utils.h"
#include "cuda_runtime.h"
#include "inc/Core/Common/RaBitQQuantizer.h"

namespace SPTAG
{
    namespace SSDServing
    {
        namespace SSDIndex
        {

            template <typename ValueType>
            ErrorCode OutputResult(const std::string &p_output, std::vector<QueryResult> &p_results, int p_resultNum)
            {
                if (!p_output.empty())
                {
                    auto ptr = f_createIO();
                    if (ptr == nullptr || !ptr->Initialize(p_output.c_str(), std::ios::binary | std::ios::out))
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed create file: %s\n", p_output.c_str());
                        return ErrorCode::FailedCreateFile;
                    }
                    int32_t i32Val = static_cast<int32_t>(p_results.size());
                    if (ptr->WriteBinary(sizeof(i32Val), reinterpret_cast<char *>(&i32Val)) != sizeof(i32Val))
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to write result file!\n");
                        return ErrorCode::DiskIOFail;
                    }
                    i32Val = p_resultNum;
                    if (ptr->WriteBinary(sizeof(i32Val), reinterpret_cast<char *>(&i32Val)) != sizeof(i32Val))
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to write result file!\n");
                        return ErrorCode::DiskIOFail;
                    }

                    float fVal = 0;
                    for (size_t i = 0; i < p_results.size(); ++i)
                    {
                        for (int j = 0; j < p_resultNum; ++j)
                        {
                            i32Val = p_results[i].GetResult(j)->VID;
                            if (ptr->WriteBinary(sizeof(i32Val), reinterpret_cast<char *>(&i32Val)) != sizeof(i32Val))
                            {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to write result file!\n");
                                return ErrorCode::DiskIOFail;
                            }

                            fVal = p_results[i].GetResult(j)->Dist;
                            if (ptr->WriteBinary(sizeof(fVal), reinterpret_cast<char *>(&fVal)) != sizeof(fVal))
                            {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to write result file!\n");
                                return ErrorCode::DiskIOFail;
                            }
                        }
                    }
                }
                return ErrorCode::Success;
            }

            template <typename T, typename V>
            void PrintPercentiles(const std::vector<V> &p_values, std::function<T(const V &)> p_get, const char *p_format)
            {
                double sum = 0;
                std::vector<T> collects;
                collects.reserve(p_values.size());
                for (const auto &v : p_values)
                {
                    T tmp = p_get(v);
                    sum += tmp;
                    collects.push_back(tmp);
                }

                std::sort(collects.begin(), collects.end());

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Avg\t50tiles\t90tiles\t95tiles\t99tiles\t99.9tiles\tMax\n");

                std::string formatStr("%.3lf");
                for (int i = 1; i < 7; ++i)
                {
                    formatStr += '\t';
                    formatStr += p_format;
                }

                formatStr += '\n';

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                             formatStr.c_str(),
                             sum / collects.size(),
                             collects[static_cast<size_t>(collects.size() * 0.50)],
                             collects[static_cast<size_t>(collects.size() * 0.90)],
                             collects[static_cast<size_t>(collects.size() * 0.95)],
                             collects[static_cast<size_t>(collects.size() * 0.99)],
                             collects[static_cast<size_t>(collects.size() * 0.999)],
                             collects[static_cast<size_t>(collects.size() - 1)]);
            }

            template <typename ValueType>
            void SearchSequential(SPANN::Index<ValueType>* p_index,
                int p_numThreads,
                std::vector<QueryResult>& p_results,    // 存internalResultNum 个最近的聚类中心
                std::vector<SPANN::SearchStats>& p_stats,
                int p_maxQueryCount, int p_internalResultNum,
                std::shared_ptr<SPTAG::VectorSet> querySet,
                std::shared_ptr<SPTAG::VectorSet> PQVectorSet,
                std::shared_ptr<SPTAG::VectorSet> rerankVectorSet,
                int QuantizedVectorCount, int QuantizedVectorDim, std::vector<int>& numVecPerPostinglist, std::vector<std::unique_ptr<int[]>>& postinglist, void* d_PQVectorSet,
                uint8_t *d_table, int* d_vectorIDs, float *d_dist, float *h_dist, int totalNumVec, std::shared_ptr<SPTAG::COMMON::IQuantizer> quantizer, bool isWarmup)
            {
                // ========== 1. 初始化查询数量与线程 ==========
                int numQueries = min(static_cast<int>(p_results.size()), p_maxQueryCount);

                std::atomic_size_t queriesSent(0);
                
                std::vector<std::thread> threads;
                threads.reserve(p_numThreads);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Searching: numThread: %d, numQueries: %d.\n", p_numThreads, numQueries);

                SPANN::Options &p_opts = *(p_index->GetOptions());

                // ========== 2. 启动多个工作线程，每个线程不断取查询处理 ==========
                Utils::StopW sw;    // 用于统计整体发送耗时
                for (int i = 0; i < p_numThreads; i++)
                {
                    threads.emplace_back([&, i]()
                                         {
                    NumaStrategy ns = (p_index->GetDiskIndex() != nullptr) ? NumaStrategy::SCATTER : NumaStrategy::LOCAL; // Only for SPANN, we need to avoid IO threads overlap with search threads.
                    Helper::SetThreadAffinity(i, threads[i], ns, OrderStrategy::ASC); 

                    Utils::StopW threadws;  // 单个查询的计时器
                    size_t index = 0;
                    while (true)
                    {
                        // 原子获取下一个查询的编号
                        index = queriesSent.fetch_add(1);
                        if (index < numQueries)
                        {
                            // 每 16384 个查询打印一次进度
                            if ((index & ((1 << 14) - 1)) == 0)
                            {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Sent %.2lf%%...\n", index * 100.0 / numQueries);
                            }

                            // -------- 阶段 A：内存 head 索引搜索（粗排）--------
                            double startTime = threadws.getElapsedMs();
                            p_index->GetMemoryIndex()->SearchIndex(p_results[index]);
                            double endTime = threadws.getElapsedMs();

                            // QueryResult 是贯穿整个检索流程的核心上下文数据结构。它同时包含两方面的信息：
                            // Target（查询自身的信息）：它存储了当前这通 Query 的原始特征向量。
                            // Results（Top-K 中间/最终结果列表）：它维护了一个优先队列（Heap），里面存的是当前搜索出来的候选点的 ID (VID) 和距离 (Dist)

                            // 设置新 PQ 目标，供后续量化距离计算使用
                            if(quantizer->GetQuantizerType() == QuantizerType::PQQuantizer){
                                (*((COMMON::QueryResultSet<ValueType> *)&p_results[index])).Setmy_newPQTarget(quantizer);
                            }
                            std::unordered_set<int> postingIDSet;   // 记录访问过的向量 ID（可能用于去重等）
                            
                            // -------- 阶段 B：量化倒排查找（细排，含 SSD 读取）--------
                            if(p_opts.m_enableGPU)
                            {
                                // GPU 版本：从显存中的 PQ 码计算近似距离
                                p_index->SearchPQIndex_GPU(p_results[index], QuantizedVectorCount, QuantizedVectorDim, numVecPerPostinglist, postinglist, d_PQVectorSet, d_table, d_vectorIDs, d_dist, h_dist, totalNumVec, i, postingIDSet, &(p_stats[index]));
                            }else{
                                // CPU 版本（较少使用）
                                p_index->SearchPQIndex_CPU(p_results[index], PQVectorSet, QuantizedVectorCount, QuantizedVectorDim, numVecPerPostinglist, postinglist, postingIDSet, &(p_stats[index]));
                            }
                            
                            double searchEndTime = threadws.getElapsedMs();

                            // -------- 阶段 C：重排序（rerank）--------
                            if (p_opts.m_enableReorderIndex)
                            {
                                // 利用重排序索引加速的混合重排
                                if (p_opts.m_rerank > 0 && p_opts.m_resultNum > 0) 
                                {
                                    p_index->RerankFullVectorFusion(p_results[index], rerankVectorSet, i, &(p_stats[index]));
                                }
                            }else{
                                // 标准重排序：用原始向量精确计算 top 候选的距离
                                if (p_opts.m_rerank > 0 && p_opts.m_resultNum > 0) 
                                {
                                    p_index->RerankFullVector(p_results[index], rerankVectorSet, i, postingIDSet, &(p_stats[index]));
                                }
                            }
                            
                            // 

                            
                            
                            /*
                            //妯℃嫙ssd寤惰繜
                            if (p_opts.m_rerank > 0) 
                            {
                                int K = p_opts.m_resultNum;
                                for (int j = 0; j < K; j++)
                                {
                                    if (p_results[index].GetResult(j)->VID < 0) continue;
                                    p_results[index].GetResult(j)->Dist = COMMON::DistanceUtils::ComputeDistance((const ValueType*)p_results[index].GetTarget(),
                                        (const ValueType*)rerankVectorSet->GetVector(p_results[index].GetResult(j)->VID), p_opts.m_dim, p_opts.m_distCalcMethod);
                                }

                                // 鍋囪姣忔寮傛璇诲彇鎿嶄綔鐨勬椂闂翠负0.0000002绉掞紝4KB / (20 GB/s) = 0.0000002 绉?
                                double sleepTime = 0.0000002 * K;
                                // 璁＄畻鐫＄湢鏃堕棿锛堝井绉掞級
                                int sleepMicroseconds = sleepTime * 1000000; // 灏嗙杞崲涓哄井绉?
                                // 閫氳繃鐫＄湢鏉ユā鎷熷欢杩燂紙寰绾у埆绮惧害锛?
                                std::this_thread::sleep_for(std::chrono::microseconds(sleepMicroseconds));

                                BasicResult* re = p_results[index].GetResults();
                                std::sort(re, re + p_opts.m_searchInternalResultNum, COMMON::Compare);
                            }
                            */
                            
                            // -------- 阶段 D：统计延时信息 --------
                            double exEndTime = threadws.getElapsedMs();

                            p_stats[index].m_exLatency = searchEndTime - endTime;
                            p_stats[index].m_totalSearchLatency = searchEndTime - startTime;
                            p_stats[index].rerankLatency = exEndTime - searchEndTime;
                            p_stats[index].m_totalLatency = exEndTime - startTime;
                        }
                        else
                        {
                            return; // 所有查询处理完毕，线程退出
                        }
                    } });
                }

                // ========== 3. 等待所有线程结束 ==========
                for (auto &thread : threads)
                {
                    thread.join();
                }

                double sendingCost = sw.getElapsedSec();

                // if(!isWarmup)
                // {
                //     SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                //              "Finish sending in %.3lf seconds, actuallQPS is %.2lf, query count %u.\n",
                //              sendingCost,
                //              numQueries / sendingCost,
                //              static_cast<uint32_t>(numQueries));
                // }

                // 输出整体 QPS
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                        "Finish sending in %.3lf seconds, actuallQPS is %.2lf, query count %u.\n",
                        sendingCost,
                        numQueries / sendingCost,
                        static_cast<uint32_t>(numQueries));
                    
                // ========== 4. 清理量化目标，释放可能的内存 ==========
                for (int i = 0; i < numQueries; i++)
                {
                    p_results[i].CleanQuantizedTarget();
                }
            }

            template <typename ValueType>
            void SearchSequentialRaBitQVersion(
                SPANN::Index<ValueType>* p_index,
                int p_numThreads,
                std::vector<QueryResult>& p_results,    // 存internalResultNum 个最近的聚类中心
                std::vector<SPANN::SearchStats>& p_stats,
                int p_maxQueryCount, 
                int p_internalResultNum,
                std::shared_ptr<SPTAG::VectorSet> querySet,
                std::shared_ptr<SPTAG::VectorSet> QuantizedVectorSet, // 存放量化倒排链的主数据结构
                std::shared_ptr<SPTAG::VectorSet> rerankVectorSet,
                int dim, 
                int bits_per_code,
                std::vector<int>& numVecPerPostinglist, 
                std::vector<std::unique_ptr<int[]>>& postinglist, 
                void* d_QuantizedVectorSet,
                float *d_rotated_query,     // GPU上存放的每条查询单独的 rotated_query，替换掉了 d_table
                int* d_vectorIDs, 
                float *d_dist, 
                float *h_dist, 
                int totalNumVec, 
                std::shared_ptr<SPTAG::COMMON::RaBitQQuantizer<ValueType>> quantizer, 
                bool isWarmup)
            {
                // ========== 1. 初始化查询数量与线程 ==========
                int numQueries = min(static_cast<int>(p_results.size()), p_maxQueryCount);
                std::atomic_size_t queriesSent(0);
                std::vector<std::thread> threads;
                threads.reserve(p_numThreads);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Searching (RaBitQ Version): numThread: %d, numQueries: %d.\n", p_numThreads, numQueries);

                SPANN::Options &p_opts = *(p_index->GetOptions());
                // 提前拿到 padding 后的维数，方便做批处理和显存拷贝
                int paddedDim = quantizer->GetPaddedDim();

                // ========== 2. 启动多个工作线程，每个线程不断取查询处理 ==========
                Utils::StopW sw;    // 用于统计整体发送耗时
                for (int i = 0; i < p_numThreads; i++)
                {
                    threads.emplace_back([&, i]()
                    {
                        NumaStrategy ns = (p_index->GetDiskIndex() != nullptr) ? NumaStrategy::SCATTER : NumaStrategy::LOCAL; 
                        Helper::SetThreadAffinity(i, threads[i], ns, OrderStrategy::ASC); 

                        Utils::StopW threadws;  // 单个查询的计时器
                        size_t index = 0;
                        while (true)
                        {
                            // 原子获取下一个查询的编号
                            index = queriesSent.fetch_add(1);
                            if (index < numQueries)
                            {
                                if ((index & ((1 << 14) - 1)) == 0) {
                                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Sent %.2lf%%...\n", index * 100.0 / numQueries);
                                }

                                // -------- 阶段 A：内存 head 索引搜索（粗排）--------
                                double startTime = threadws.getElapsedMs();
                                p_index->GetMemoryIndex()->SearchIndex(p_results[index]);
                                double endTime = threadws.getElapsedMs();

                                // 我们不在 QueryResultSet 中缓存庞大的 LookUpTable，
                                // 这里我们自己当场通过 RaBitQQuantizer 算出 rotated_query 和 meta。
                                const ValueType* target = reinterpret_cast<const ValueType*>(p_results[index].GetTarget());

                                std::vector<float> rotated_query(paddedDim, 0.0f);
                                
                                quantizer->PreprocessQuery(target, rotated_query.data());
                                typename COMMON::RaBitQQuantizer<ValueType>::BondMeta bond_meta;
                                quantizer->BuildL2EstimateQueryFactors(rotated_query.data(), bond_meta);

                                std::unordered_set<int> postingIDSet;   // 记录访问过的向量 ID

                                // -------- 阶段 B：量化倒排查找（细排，采用 RaBitQ 专属策略）--------
                                if(p_opts.m_enableGPU)
                                {
                                    // 1. 将当前查询算好的 rotated_query 发往 GPU。
                                    // 每个线程通过 threadOrder(`i`) 独占一块 `actualPaddedDim` 长度的显存，互不干扰
                                    float* current_gpu_query = d_rotated_query + i * paddedDim;
                                    cudaMemcpy(current_gpu_query, rotated_query.data(), paddedDim * sizeof(float), cudaMemcpyHostToDevice);

                                    // 2. 调用我们在 SPANNIndex.cpp 写的 SearchRaBitQIndex_GPU 分支
                                    p_index->SearchRaBitQIndex_GPU(
                                        p_results[index], 
                                        dim, 
                                        bits_per_code, 
                                        numVecPerPostinglist, 
                                        postinglist, 
                                        d_QuantizedVectorSet, 
                                        current_gpu_query,      // 纯净的指针
                                        bond_meta.g_add,        // 【修改】：拆成三个 float 传入
                                        bond_meta.k1xsumq,
                                        bond_meta.g_error,
                                        d_vectorIDs, 
                                        d_dist, 
                                        h_dist, 
                                        totalNumVec, 
                                        i,                      // threadOrder
                                        postingIDSet, 
                                        &(p_stats[index])
                                    );
                                }
                                else
                                {
                                    // TODO: 如果你需要支持 CPU，可以在这里补充一个 SearchRaBitQIndex_CPU 调用。
                                }
                                
                                double searchEndTime = threadws.getElapsedMs();

                                // -------- 阶段 C：重排序（rerank，读 SSD）--------
                                if (p_opts.m_enableReorderIndex) {
                                    if (p_opts.m_rerank > 0 && p_opts.m_resultNum > 0) {
                                        p_index->RerankFullVectorFusion(p_results[index], rerankVectorSet, i, &(p_stats[index]));
                                    }
                                } else {
                                    if (p_opts.m_rerank > 0 && p_opts.m_resultNum > 0) {
                                        p_index->RerankFullVector(p_results[index], rerankVectorSet, i, postingIDSet, &(p_stats[index]));
                                    }
                                }
                                
                                // -------- 阶段 D：统计延时信息 --------
                                double exEndTime = threadws.getElapsedMs();
                                p_stats[index].m_exLatency = searchEndTime - endTime;
                                p_stats[index].m_totalSearchLatency = searchEndTime - startTime;
                                p_stats[index].rerankLatency = exEndTime - searchEndTime;
                                p_stats[index].m_totalLatency = exEndTime - startTime;
                            }
                            else
                            {
                                return; // 所有查询处理完毕，线程退出
                            }
                        } 
                    });
                }

                // ========== 3. 等待所有线程结束 ==========
                for (auto &thread : threads)
                {
                    thread.join();
                }

                double sendingCost = sw.getElapsedSec();
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                        "Finish sending in %.3lf seconds, actuallQPS is %.2lf, query count %u.\n",
                        sendingCost,
                        numQueries / sendingCost,
                        static_cast<uint32_t>(numQueries));
            }

            template <typename ValueType>
            void Search(SPANN::Index<ValueType> *p_index)
            {
                // 调试
                std::cout << "Begin Search..." << std::endl;

                SPANN::Options &p_opts = *(p_index->GetOptions());
                std::string outputFile = p_opts.m_searchResult;
                std::string truthFile = p_opts.m_truthPath;
                std::string warmupFile = p_opts.m_warmupPath;

                // 测试结果：p_index->m_pQuantizer是nullptr, 说明没有使用量化器
                if (p_index->m_pQuantizer)
                {
                    // std::cout << "p_index has quantizer" << std::endl;
                    p_index->m_pQuantizer->SetEnableADC(p_opts.m_enableADC);
                }

                if (!p_opts.m_logFile.empty())
                {
                    SetLogger(std::make_shared<Helper::FileLogger>(Helper::LogLevel::LL_Info, p_opts.m_logFile.c_str()));
                }
                int numThreads = p_opts.m_iSSDNumberOfThreads;
                int internalResultNum = p_opts.m_searchInternalResultNum;
                int K = p_opts.m_resultNum; // topK
                int truthK = (p_opts.m_rerank <= 0) ? K : p_opts.m_rerank;

                std::string QuantizervectorFilePath = p_opts.m_quantizerVectorFilePath;
                int count, dim;
                auto ptr_vector = SPTAG::f_createIO();
                if (!ptr_vector->Initialize(QuantizervectorFilePath.c_str(), std::ios::binary | std::ios::in))
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read quantizervector file.\n");
                    return;
                }
                ptr_vector->ReadBinary(sizeof(count), reinterpret_cast<char *>(&(count)));
                ptr_vector->ReadBinary(sizeof(dim), reinterpret_cast<char *>(&(dim)));
                std::shared_ptr<VectorSet> QuantizedVectorSet;
                void *d_QuantizedVectorSet;    // d_表示device，GPU端通常称为device，CPU端通常称为host
                if (!QuantizervectorFilePath.empty() && fileexists(QuantizervectorFilePath.c_str()))
                {
                    std::shared_ptr<Helper::ReaderOptions> vectorOptions(new Helper::ReaderOptions(VectorValueType::UInt8, dim, p_opts.m_vectorType, p_opts.m_vectorDelimiter));
                    auto vectorReader = Helper::VectorSetReader::CreateInstance(vectorOptions);
                    if (ErrorCode::Success == vectorReader->LoadFile(QuantizervectorFilePath))
                    {
                        // 打印Load Vector(1000000000,32)
                        QuantizedVectorSet = vectorReader->GetVectorSet();
                    }
                }

                if(p_opts.m_enableGPU)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Load QuantizedVectorSet to GPU\n");
                    cudaMalloc((void **)&d_QuantizedVectorSet, sizeof(uint8_t) * QuantizedVectorSet->Count() * QuantizedVectorSet->Dimension());
                    cudaMemcpy(d_QuantizedVectorSet, QuantizedVectorSet->GetData(), sizeof(uint8_t) * QuantizedVectorSet->Count() * QuantizedVectorSet->Dimension(), cudaMemcpyHostToDevice);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Load QuantizedVectorSet Finish\n");
                    QuantizedVectorSet.reset();
                }

                std::string PostingListFilePath = p_opts.m_indexDirectory + FolderSep + p_opts.m_postingListIndex;
                std::vector<int> numvec;
                std::vector<std::unique_ptr<int[]>> postinglist;
                int num_postinglist, fullVectorCount, MaxNumVec = 0;
                auto fp_read = SPTAG::f_createIO();
                if (fp_read == nullptr || !fp_read->Initialize(PostingListFilePath.c_str(), std::ios::binary | std::ios::in))
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read quantizervector file.\n");
                    return;
                }
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Beign load postinglist\n");
                if (fp_read->ReadBinary(sizeof(int), reinterpret_cast<char *>(&(num_postinglist))) != sizeof(int))
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read PostingList file!\n");
                    throw std::runtime_error("Failed read file in PostingList");
                }
                if (fp_read->ReadBinary(sizeof(int), reinterpret_cast<char *>(&(fullVectorCount))) != sizeof(int))
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read PostingList file!\n");
                    throw std::runtime_error("Failed read file in PostingList");
                }

                for (int i = 0; i < num_postinglist; i++)
                {
                    int postingListMeta;
                    if (fp_read->ReadBinary(sizeof(int), reinterpret_cast<char *>(&(postingListMeta))) != sizeof(int))
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read PostingList file!\n");
                        throw std::runtime_error("Failed read file in PostingList");
                    }
                    numvec.push_back(postingListMeta);
                    if (postingListMeta > MaxNumVec)  MaxNumVec = postingListMeta;
                }
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "The MaxNumVec of postinglist is %d\n", MaxNumVec);

                for (int i = 0; i < num_postinglist; i++)
                {
                    postinglist.push_back(std::unique_ptr<int[]>(new int[numvec[i]]));
                    if (fp_read->ReadBinary(sizeof(int) * numvec[i], reinterpret_cast<char *>(postinglist[i].get())) != sizeof(int) * numvec[i])
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read PostingList file!\n");
                        throw std::runtime_error("Failed read file in PostingList");
                    }
                }
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Load postinglist finish\n");

                std::shared_ptr<VectorSet> vectorSetRatio;
                if (!p_opts.m_vectorPath.empty() && fileexists(p_opts.m_vectorPath.c_str()))
                {
                    std::shared_ptr<Helper::ReaderOptions> vectorOptions(new Helper::ReaderOptions(p_opts.m_valueType, p_opts.m_dim, p_opts.m_vectorType, p_opts.m_vectorDelimiter));
                    auto vectorReader = Helper::VectorSetReader::CreateInstance(vectorOptions);
                    if (ErrorCode::Success == vectorReader->LoadFile(p_opts.m_vectorPath))
                    {
                        vectorSetRatio = vectorReader->GetVectorSetRatio(p_opts.m_readRatio);
                        if (p_opts.m_distCalcMethod == DistCalcMethod::Cosine)
                            vectorSetRatio->Normalize(numThreads);
                    }
                }

                std::shared_ptr<SPTAG::COMMON::IQuantizer> quantizer;
                std::string QuantizerFilePath = p_opts.m_quantizerPQFilePath;
                auto ptr = SPTAG::f_createIO();
                if (!ptr->Initialize(QuantizerFilePath.c_str(), std::ios::binary | std::ios::in))
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read quantizer file.\n");
                    return;
                }
                // 此处原来调用了PQQuantizer::LoadQuantizer，打印了Loading Quantizer.到Loading quantizer:中间一堆信息
                // 现在改为RaBitQ
                quantizer = SPTAG::COMMON::IQuantizer::LoadIQuantizer(ptr);
                quantizer->SetEnableADC(true);

                int totalNumVec = MaxNumVec * p_opts.m_searchInternalResultNum;
                
                // 根据量化器类型分别声明不同用途的 GPU 内存指针
                uint8_t *d_table = nullptr; 
                float *d_rotated_query = nullptr; 
                std::shared_ptr<SPTAG::COMMON::RaBitQQuantizer<ValueType>> rabitq_quantizer = nullptr;

                // --- 核心改动：直接通过 == 判断类型并分配由于算法不同导致所需的不同显存 ---
                if (quantizer->GetQuantizerType() == QuantizerType::RaBitQQuantizer)
                {
                    rabitq_quantizer = std::dynamic_pointer_cast<SPTAG::COMMON::RaBitQQuantizer<ValueType>>(quantizer);
                    int paddedDim = rabitq_quantizer->GetPaddedDim(); // 使用你新加的 public 接口
                    
                    // RaBitQ 不需要查表，只需要每条线程存放其旋转和补齐后的查询向量
                    cudaMalloc((void **)&d_rotated_query, sizeof(float) * paddedDim * numThreads);
                }
                else
                {
                    // 原始 PQ 模式：存放每个查询的 ADC 查表，分块长度是 256（centroids）× dim
                    cudaMalloc((void **)&d_table, 256 * dim * sizeof(float) * numThreads);
                }

                int *d_vectorIDs;   // 端存放本次要计算的向量 ID 列表（int），由 host 填充后传入 kernel
                cudaMalloc((void **)&d_vectorIDs, sizeof(int) * totalNumVec * numThreads);

                float *d_dist;  // device 端的距离结果数组（float），kernel 对每个向量累加到这里，之后拷回 host
                cudaMalloc((void **)&d_dist, sizeof(float) * totalNumVec * numThreads);

                float *h_dist;  // 用 cudaMallocHost 分配的锁页（pinned）主机内存，用于高效地从 device 拷贝回距离结果并供 CPU 读取
                cudaMallocHost((void **)&h_dist, sizeof(float) * totalNumVec * numThreads);

                bool isWarmup = false;
                
                if (!warmupFile.empty())
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start loading warmup query set...\n");
                    std::shared_ptr<Helper::ReaderOptions> queryOptions(new Helper::ReaderOptions(p_opts.m_valueType, p_opts.m_dim, p_opts.m_warmupType, p_opts.m_warmupDelimiter));
                    auto queryReader = Helper::VectorSetReader::CreateInstance(queryOptions);
                    if (ErrorCode::Success != queryReader->LoadFile(p_opts.m_warmupPath))
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read query file.\n");
                        exit(1);
                    }
                    auto warmupQuerySet = queryReader->GetVectorSet();
                    int warmupNumQueries = warmupQuerySet->Count(); // warmup集的查询数量

                    std::vector<QueryResult> warmupResults(warmupNumQueries, QueryResult(NULL, max(K, internalResultNum), false));
                    std::vector<SPANN::SearchStats> warmpUpStats(warmupNumQueries);
                    for (int i = 0; i < warmupNumQueries; ++i)
                    {
                        (*((COMMON::QueryResultSet<ValueType> *)&warmupResults[i])).SetTarget(reinterpret_cast<ValueType *>(warmupQuerySet->GetVector(i)), p_index->m_pQuantizer);
                        //(*((COMMON::QueryResultSet<ValueType> *)&warmupResults[i])).Setmy_newPQTarget(quantizer);
                        warmupResults[i].Reset();
                    }

                    isWarmup = true;
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start warmup...\n");
                    if (quantizer->GetQuantizerType() == QuantizerType::RaBitQQuantizer) 
                    {
                        SearchSequentialRaBitQVersion(p_index, numThreads, warmupResults, warmpUpStats, p_opts.m_queryCountLimit, internalResultNum, warmupQuerySet, QuantizedVectorSet,
                        vectorSetRatio, dim, rabitq_quantizer->GetBitsPerCode(), numvec, postinglist, d_QuantizedVectorSet, d_rotated_query, d_vectorIDs, d_dist, h_dist, totalNumVec, rabitq_quantizer, isWarmup);
                    }
                    else if (quantizer->GetQuantizerType() == QuantizerType::PQQuantizer)
                    {
                        SearchSequential(p_index, numThreads, warmupResults, warmpUpStats, p_opts.m_queryCountLimit, internalResultNum, warmupQuerySet, QuantizedVectorSet,
                        vectorSetRatio, count, dim, numvec, postinglist, d_QuantizedVectorSet, d_table, d_vectorIDs, d_dist, h_dist, totalNumVec, quantizer, isWarmup);
                    }
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\nFinish warmup...\n");
                    isWarmup = false;
                }

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start loading QuerySet...\n");
                std::shared_ptr<Helper::ReaderOptions> queryOptions(new Helper::ReaderOptions(p_opts.m_valueType, p_opts.m_dim, p_opts.m_queryType, p_opts.m_queryDelimiter));
                auto queryReader = Helper::VectorSetReader::CreateInstance(queryOptions);
                if (ErrorCode::Success != queryReader->LoadFile(p_opts.m_queryPath))
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read query file.\n");
                    exit(1);
                }

                auto querySet = queryReader->GetVectorSet();

                int numQueries = min(querySet->Count(), p_opts.m_queryCountLimit);
                std::vector<QueryResult> results(numQueries, QueryResult(NULL, max(K, internalResultNum), false));
                std::vector<SPANN::SearchStats> stats(numQueries);
                for (int i = 0; i < numQueries; ++i)
                {
                    (*((COMMON::QueryResultSet<ValueType> *)&results[i])).SetTarget(reinterpret_cast<ValueType *>(querySet->GetVector(i)), p_index->m_pQuantizer);
                    //(*((COMMON::QueryResultSet<ValueType> *)&results[i])).Setmy_newPQTarget(quantizer);
                    results[i].Reset();
                }

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start ANN Search...\n");

                if (rabitq_quantizer) 
                {
                    // 测试
                    // std::cout << "Using RaBitQQuantizer for search" << std::endl;
                    SearchSequentialRaBitQVersion(p_index, numThreads, results, stats, p_opts.m_queryCountLimit, internalResultNum, querySet, QuantizedVectorSet,
                    vectorSetRatio, dim, rabitq_quantizer->GetBitsPerCode(), numvec, postinglist, d_QuantizedVectorSet, d_rotated_query, d_vectorIDs, d_dist, h_dist, totalNumVec, rabitq_quantizer, isWarmup);
                }
                else
                {
                    SearchSequential(p_index, numThreads, results, stats, p_opts.m_queryCountLimit, internalResultNum, querySet, QuantizedVectorSet,
                    vectorSetRatio, count, dim, numvec, postinglist, d_QuantizedVectorSet, d_table, d_vectorIDs, d_dist, h_dist, totalNumVec, quantizer, isWarmup);
                }

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\nFinish ANN Search...\n");
                
                // 释放显存
                if (d_table != nullptr) cudaFree(d_table);
                if (d_rotated_query != nullptr) cudaFree(d_rotated_query);

                cudaFree(d_vectorIDs);
                cudaFree(d_dist);
                cudaFreeHost(h_dist);
                vectorSetRatio.reset();

                K = p_opts.m_rerank;

                std::shared_ptr<VectorSet> vectorSet;
                if (!p_opts.m_vectorPath.empty() && fileexists(p_opts.m_vectorPath.c_str()) && p_opts.m_enableCalRecall)
                {
                    std::shared_ptr<Helper::ReaderOptions> vectorOptions(new Helper::ReaderOptions(p_opts.m_valueType, p_opts.m_dim, p_opts.m_vectorType, p_opts.m_vectorDelimiter));
                    auto vectorReader = Helper::VectorSetReader::CreateInstance(vectorOptions);
                    if (ErrorCode::Success == vectorReader->LoadFile(p_opts.m_vectorPath))
                    {
                        vectorSet.reset();
                        auto newVectorSet = vectorReader->GetVectorSet();
                        if (p_opts.m_distCalcMethod == DistCalcMethod::Cosine)
                            newVectorSet->Normalize(numThreads);
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Load VectorSet(%d,%d).\n", newVectorSet->Count(), newVectorSet->Dimension());
                        vectorSet = newVectorSet;
                    }
                }

                float recall = 0, MRR = 0;
                std::vector<std::set<SizeType>> truth;
                if (!truthFile.empty())
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start loading TruthFile...\n");

                    auto ptr = f_createIO();
                    if (ptr == nullptr || !ptr->Initialize(truthFile.c_str(), std::ios::in | std::ios::binary))
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed open truth file: %s\n", truthFile.c_str());
                        exit(1);
                    }
                    int originalK = truthK;
                    COMMON::TruthSet::LoadTruth(ptr, truth, numQueries, originalK, truthK, p_opts.m_truthType);
                    char tmp[4];
                    if (ptr->ReadBinary(4, tmp) == 4)
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Truth number is larger than query number(%d)!\n", numQueries);
                    }

                    recall = COMMON::TruthSet::CalculateRecall<ValueType>((p_index->GetMemoryIndex()).get(), results, truth, K, truthK, querySet, vectorSet, numQueries, nullptr, false, &MRR);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Recall%d@%d: %f MRR@%d: %f\n", truthK, K, recall, K, MRR);
                }

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\nEx Elements Count:\n");
                PrintPercentiles<double, SPANN::SearchStats>(
                    stats,
                    [](const SPANN::SearchStats &ss) -> double
                    {
                        return ss.m_totalListElementsCount;
                    },
                    "%.3lf");

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\nHead Latency Distribution:\n");
                PrintPercentiles<double, SPANN::SearchStats>(
                    stats,
                    [](const SPANN::SearchStats &ss) -> double
                    {
                        return ss.m_totalSearchLatency - ss.m_exLatency;
                    },
                    "%.3lf");

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\nEx Latency Distribution:\n");
                PrintPercentiles<double, SPANN::SearchStats>(
                    stats,
                    [](const SPANN::SearchStats &ss) -> double
                    {
                        return ss.m_exLatency;
                    },
                    "%.3lf");

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\nRerank Latency Distribution:\n");
                PrintPercentiles<double, SPANN::SearchStats>(
                    stats,
                    [](const SPANN::SearchStats &ss) -> double
                    {
                        return ss.rerankLatency;
                    },
                    "%.3lf");

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\nTotal Latency Distribution:\n");
                PrintPercentiles<double, SPANN::SearchStats>(
                    stats,
                    [](const SPANN::SearchStats &ss) -> double
                    {
                        return ss.m_totalLatency;
                    },
                    "%.3lf");

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\nCut Tree Latency Distribution:\n");
                PrintPercentiles<double, SPANN::SearchStats>(
                    stats,
                    [](const SPANN::SearchStats &ss) -> double
                    {
                        return ss.cutTreeLatency;
                    },
                    "%.3lf");

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\nVector Latency0 Distribution:\n");
                PrintPercentiles<double, SPANN::SearchStats>(
                    stats,
                    [](const SPANN::SearchStats &ss) -> double
                    {
                        return ss.vectorLatency0;
                    },
                    "%.3lf");

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\nVector Latency1 Distribution:\n");
                PrintPercentiles<double, SPANN::SearchStats>(
                    stats,
                    [](const SPANN::SearchStats &ss) -> double
                    {
                        return ss.vectorLatency1;
                    },
                    "%.3lf");

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\nFind Table Latency Distribution:\n");
                PrintPercentiles<double, SPANN::SearchStats>(
                    stats,
                    [](const SPANN::SearchStats &ss) -> double
                    {
                        return ss.findTableLatency;
                    },
                    "%.3lf");

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\nAdd Point Latency Distribution:\n");
                PrintPercentiles<double, SPANN::SearchStats>(
                    stats,
                    [](const SPANN::SearchStats &ss) -> double
                    {
                        return ss.addPointLatency;
                    },
                    "%.3lf");

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\nSort Latency Distribution:\n");
                PrintPercentiles<double, SPANN::SearchStats>(
                    stats,
                    [](const SPANN::SearchStats &ss) -> double
                    {
                        return ss.sortLatency;
                    },
                    "%.3lf");

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\nTotal Disk Page Access Distribution:\n");
                PrintPercentiles<int, SPANN::SearchStats>(
                    stats,
                    [](const SPANN::SearchStats &ss) -> int
                    {
                        return ss.m_diskAccessCount;
                    },
                    "%4d");

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\nTotal Disk IO Distribution:\n");
                PrintPercentiles<int, SPANN::SearchStats>(
                    stats,
                    [](const SPANN::SearchStats &ss) -> int
                    {
                        return ss.m_diskIOCount;
                    },
                    "%4d");

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\n");

                if (!outputFile.empty())
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start output to %s\n", outputFile.c_str());
                    OutputResult<ValueType>(outputFile, results, K);
                }

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                             "Recall@%d: %f MRR@%d: %f\n", K, recall, K, MRR);

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "\n");

                if (p_opts.m_recall_analysis)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start recall analysis...\n");

                    std::shared_ptr<VectorIndex> headIndex = p_index->GetMemoryIndex();
                    SizeType sampleSize = numQueries < 100 ? numQueries : 100;
                    SizeType sampleK = headIndex->GetNumSamples() < 1000 ? headIndex->GetNumSamples() : 1000;
                    float sampleE = 1e-6f;

                    std::vector<SizeType> samples(sampleSize, 0);
                    std::vector<float> queryHeadRecalls(sampleSize, 0);
                    std::vector<float> truthRecalls(sampleSize, 0);
                    std::vector<int> shouldSelect(sampleSize, 0);
                    std::vector<int> shouldSelectLong(sampleSize, 0);
                    std::vector<int> nearQueryHeads(sampleSize, 0);
                    std::vector<int> annNotFound(sampleSize, 0);
                    std::vector<int> rngRule(sampleSize, 0);
                    std::vector<int> postingCut(sampleSize, 0);
                    for (int i = 0; i < sampleSize; i++)
                        samples[i] = COMMON::Utils::rand(numQueries);

#pragma omp parallel for schedule(dynamic)
                    for (int i = 0; i < sampleSize; i++)
                    {
                        COMMON::QueryResultSet<ValueType> queryANNHeads((const ValueType *)(querySet->GetVector(samples[i])), max(K, internalResultNum));
                        headIndex->SearchIndex(queryANNHeads);
                        float queryANNHeadsLongestDist = queryANNHeads.GetResult(internalResultNum - 1)->Dist;

                        COMMON::QueryResultSet<ValueType> queryBFHeads((const ValueType *)(querySet->GetVector(samples[i])), max(sampleK, internalResultNum));
                        for (SizeType y = 0; y < headIndex->GetNumSamples(); y++)
                        {
                            float dist = headIndex->ComputeDistance(queryBFHeads.GetQuantizedTarget(), headIndex->GetSample(y));
                            queryBFHeads.AddPoint(y, dist);
                        }
                        queryBFHeads.SortResult();

                        {
                            std::vector<bool> visited(internalResultNum, false);
                            for (SizeType y = 0; y < internalResultNum; y++)
                            {
                                for (SizeType z = 0; z < internalResultNum; z++)
                                {
                                    if (visited[z])
                                        continue;

                                    if (fabs(queryANNHeads.GetResult(z)->Dist - queryBFHeads.GetResult(y)->Dist) < sampleE)
                                    {
                                        queryHeadRecalls[i] += 1;
                                        visited[z] = true;
                                        break;
                                    }
                                }
                            }
                        }

                        std::map<int, std::set<int>> tmpFound; // headID->truths
                        p_index->DebugSearchDiskIndex(queryBFHeads, internalResultNum, sampleK, nullptr, &truth[samples[i]], &tmpFound);

                        for (SizeType z = 0; z < K; z++)
                        {
                            truthRecalls[i] += truth[samples[i]].count(queryBFHeads.GetResult(z)->VID);
                        }

                        for (SizeType z = 0; z < K; z++)
                        {
                            truth[samples[i]].erase(results[samples[i]].GetResult(z)->VID);
                        }

                        for (std::map<int, std::set<int>>::iterator it = tmpFound.begin(); it != tmpFound.end(); it++)
                        {
                            float q2truthposting = headIndex->ComputeDistance(querySet->GetVector(samples[i]), headIndex->GetSample(it->first));
                            for (auto vid : it->second)
                            {
                                if (!truth[samples[i]].count(vid))
                                    continue;

                                if (q2truthposting < queryANNHeadsLongestDist)
                                    shouldSelect[i] += 1;
                                else
                                {
                                    shouldSelectLong[i] += 1;

                                    std::set<int> nearQuerySelectedHeads;
                                    float v2vhead = headIndex->ComputeDistance(vectorSet->GetVector(vid), headIndex->GetSample(it->first));
                                    for (SizeType z = 0; z < internalResultNum; z++)
                                    {
                                        if (queryANNHeads.GetResult(z)->VID < 0)
                                            break;
                                        float v2qhead = headIndex->ComputeDistance(vectorSet->GetVector(vid), headIndex->GetSample(queryANNHeads.GetResult(z)->VID));
                                        if (v2qhead < v2vhead)
                                        {
                                            nearQuerySelectedHeads.insert(queryANNHeads.GetResult(z)->VID);
                                        }
                                    }
                                    if (nearQuerySelectedHeads.size() == 0)
                                        continue;

                                    nearQueryHeads[i] += 1;

                                    COMMON::QueryResultSet<ValueType> annTruthHead((const ValueType *)(vectorSet->GetVector(vid)), p_opts.m_debugBuildInternalResultNum);
                                    headIndex->SearchIndex(annTruthHead);

                                    bool found = false;
                                    for (SizeType z = 0; z < annTruthHead.GetResultNum(); z++)
                                    {
                                        if (nearQuerySelectedHeads.count(annTruthHead.GetResult(z)->VID))
                                        {
                                            found = true;
                                            break;
                                        }
                                    }

                                    if (!found)
                                    {
                                        annNotFound[i] += 1;
                                        continue;
                                    }

                                    // RNG rule and posting cut
                                    std::set<int> replicas;
                                    for (SizeType z = 0; z < annTruthHead.GetResultNum() && replicas.size() < p_opts.m_replicaCount; z++)
                                    {
                                        BasicResult *item = annTruthHead.GetResult(z);
                                        if (item->VID < 0)
                                            break;

                                        bool good = true;
                                        for (auto r : replicas)
                                        {
                                            if (p_opts.m_rngFactor * headIndex->ComputeDistance(headIndex->GetSample(r), headIndex->GetSample(item->VID)) < item->Dist)
                                            {
                                                good = false;
                                                break;
                                            }
                                        }
                                        if (good)
                                            replicas.insert(item->VID);
                                    }

                                    found = false;
                                    for (auto r : nearQuerySelectedHeads)
                                    {
                                        if (replicas.count(r))
                                        {
                                            found = true;
                                            break;
                                        }
                                    }

                                    if (found)
                                        postingCut[i] += 1;
                                    else
                                        rngRule[i] += 1;
                                }
                            }
                        }
                    }
                    float headacc = 0, truthacc = 0, shorter = 0, longer = 0, lost = 0, buildNearQueryHeads = 0, buildAnnNotFound = 0, buildRNGRule = 0, buildPostingCut = 0;
                    for (int i = 0; i < sampleSize; i++)
                    {
                        headacc += queryHeadRecalls[i];
                        truthacc += truthRecalls[i];

                        lost += shouldSelect[i] + shouldSelectLong[i];
                        shorter += shouldSelect[i];
                        longer += shouldSelectLong[i];

                        buildNearQueryHeads += nearQueryHeads[i];
                        buildAnnNotFound += annNotFound[i];
                        buildRNGRule += rngRule[i];
                        buildPostingCut += postingCut[i];
                    }

                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Query head recall @%d:%f.\n", internalResultNum, headacc / sampleSize / internalResultNum);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "BF top %d postings truth recall @%d:%f.\n", sampleK, truthK, truthacc / sampleSize / truthK);

                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                 "Percent of truths in postings have shorter distance than query selected heads: %f percent\n",
                                 shorter / lost * 100);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                 "Percent of truths in postings have longer distance than query selected heads: %f percent\n",
                                 longer / lost * 100);

                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                 "\tPercent of truths no shorter distance in query selected heads: %f percent\n",
                                 (longer - buildNearQueryHeads) / lost * 100);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                 "\tPercent of truths exists shorter distance in query selected heads: %f percent\n",
                                 buildNearQueryHeads / lost * 100);

                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                 "\t\tRNG rule ANN search loss: %f percent\n", buildAnnNotFound / lost * 100);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                 "\t\tPosting cut loss: %f percent\n", buildPostingCut / lost * 100);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                 "\t\tRNG rule loss: %f percent\n", buildRNGRule / lost * 100);
                }
            }
        }
    }
}
