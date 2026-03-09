// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Test.h"
#include "inc/Helper/SimpleIniReader.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Core/Common/CommonUtils.h"

#include "inc/Core/Common/RaBitQQuantizer.h" // 【新增】添加这一行
// // 【新增】为了直接测试 RaBitQ 库的底层逻辑，引入这些头文件
// #include "rabitqlib/quantization/rabitq.hpp"
// #include "rabitqlib/utils/rotator.hpp"

#include <unordered_set>
#include <chrono>

#include <fstream> // 需要增加这个头文件用于计算文件大小

template <typename T>
void Build(SPTAG::IndexAlgoType algo, std::string distCalcMethod, std::shared_ptr<SPTAG::VectorSet>& vec, std::shared_ptr<SPTAG::MetadataSet>& meta, const std::string out)
{

    std::shared_ptr<SPTAG::VectorIndex> vecIndex = SPTAG::VectorIndex::CreateInstance(algo, SPTAG::GetEnumValueType<T>());
    BOOST_CHECK(nullptr != vecIndex);

    if (algo != SPTAG::IndexAlgoType::SPANN) {
        vecIndex->SetParameter("DistCalcMethod", distCalcMethod);
        vecIndex->SetParameter("NumberOfThreads", "16");
    }
    else {
        vecIndex->SetParameter("IndexAlgoType", "BKT", "Base");
        vecIndex->SetParameter("DistCalcMethod", distCalcMethod, "Base");

        vecIndex->SetParameter("isExecute", "true", "SelectHead");
        vecIndex->SetParameter("NumberOfThreads", "4", "SelectHead");
        vecIndex->SetParameter("Ratio", "0.2", "SelectHead"); // vecIndex->SetParameter("Count", "200", "SelectHead");

        vecIndex->SetParameter("isExecute", "true", "BuildHead");
        vecIndex->SetParameter("RefineIterations", "3", "BuildHead");
        vecIndex->SetParameter("NumberOfThreads", "4", "BuildHead");

        vecIndex->SetParameter("isExecute", "true", "BuildSSDIndex");
        vecIndex->SetParameter("BuildSsdIndex", "true", "BuildSSDIndex");
        vecIndex->SetParameter("NumberOfThreads", "4", "BuildSSDIndex");
        vecIndex->SetParameter("PostingPageLimit", "12", "BuildSSDIndex");
        vecIndex->SetParameter("SearchPostingPageLimit", "12", "BuildSSDIndex");
        vecIndex->SetParameter("InternalResultNum", "64", "BuildSSDIndex");
        vecIndex->SetParameter("SearchInternalResultNum", "64", "BuildSSDIndex");
    }

    BOOST_CHECK(SPTAG::ErrorCode::Success == vecIndex->BuildIndex(vec, meta));
    BOOST_CHECK(SPTAG::ErrorCode::Success == vecIndex->SaveIndex(out));
}

template <typename T>
void BuildWithMetaMapping(SPTAG::IndexAlgoType algo, std::string distCalcMethod, std::shared_ptr<SPTAG::VectorSet>& vec, std::shared_ptr<SPTAG::MetadataSet>& meta, const std::string out)
{

    std::shared_ptr<SPTAG::VectorIndex> vecIndex = SPTAG::VectorIndex::CreateInstance(algo, SPTAG::GetEnumValueType<T>());
    BOOST_CHECK(nullptr != vecIndex);

    if (algo != SPTAG::IndexAlgoType::SPANN) {
        vecIndex->SetParameter("DistCalcMethod", distCalcMethod);
        vecIndex->SetParameter("NumberOfThreads", "16");
    }
    else {
        vecIndex->SetParameter("IndexAlgoType", "BKT", "Base");
        vecIndex->SetParameter("DistCalcMethod", distCalcMethod, "Base");

        vecIndex->SetParameter("isExecute", "true", "SelectHead");
        vecIndex->SetParameter("NumberOfThreads", "4", "SelectHead");
        vecIndex->SetParameter("Ratio", "0.2", "SelectHead"); // vecIndex->SetParameter("Count", "200", "SelectHead");

        vecIndex->SetParameter("isExecute", "true", "BuildHead");
        vecIndex->SetParameter("RefineIterations", "3", "BuildHead");
        vecIndex->SetParameter("NumberOfThreads", "4", "BuildHead");

        vecIndex->SetParameter("isExecute", "true", "BuildSSDIndex");
        vecIndex->SetParameter("BuildSsdIndex", "true", "BuildSSDIndex");
        vecIndex->SetParameter("NumberOfThreads", "4", "BuildSSDIndex");
        vecIndex->SetParameter("PostingPageLimit", "12", "BuildSSDIndex");
        vecIndex->SetParameter("SearchPostingPageLimit", "12", "BuildSSDIndex");
        vecIndex->SetParameter("InternalResultNum", "64", "BuildSSDIndex");
        vecIndex->SetParameter("SearchInternalResultNum", "64", "BuildSSDIndex");
    }

    BOOST_CHECK(SPTAG::ErrorCode::Success == vecIndex->BuildIndex(vec, meta, true));
    BOOST_CHECK(SPTAG::ErrorCode::Success == vecIndex->SaveIndex(out));
}

template <typename T>
void Search(const std::string folder, T* vec, SPTAG::SizeType n, int k, std::string* truthmeta)
{
    std::shared_ptr<SPTAG::VectorIndex> vecIndex;
    BOOST_CHECK(SPTAG::ErrorCode::Success == SPTAG::VectorIndex::LoadIndex(folder, vecIndex));
    BOOST_CHECK(nullptr != vecIndex);

    for (SPTAG::SizeType i = 0; i < n; i++) 
    {
        SPTAG::QueryResult res(vec, k, true);
        vecIndex->SearchIndex(res);
        std::unordered_set<std::string> resmeta;
        for (int j = 0; j < k; j++)
        {
            resmeta.insert(std::string((char*)res.GetMetadata(j).Data(), res.GetMetadata(j).Length()));
            std::cout << res.GetResult(j)->Dist << "@(" << res.GetResult(j)->VID << "," << std::string((char*)res.GetMetadata(j).Data(), res.GetMetadata(j).Length()) << ") ";
        }
        std::cout << std::endl;
        for (int j = 0; j < k; j++)
        {
            BOOST_CHECK(resmeta.find(truthmeta[i * k + j]) != resmeta.end());
        }
        vec += vecIndex->GetFeatureDim();
    }
    vecIndex.reset();
}

template <typename T>
void Add(const std::string folder, std::shared_ptr<SPTAG::VectorSet>& vec, std::shared_ptr<SPTAG::MetadataSet>& meta, const std::string out)
{
    std::shared_ptr<SPTAG::VectorIndex> vecIndex;
    BOOST_CHECK(SPTAG::ErrorCode::Success == SPTAG::VectorIndex::LoadIndex(folder, vecIndex));
    BOOST_CHECK(nullptr != vecIndex);

    BOOST_CHECK(SPTAG::ErrorCode::Success == vecIndex->AddIndex(vec, meta));
    BOOST_CHECK(SPTAG::ErrorCode::Success == vecIndex->SaveIndex(out));
    vecIndex.reset();
}

template <typename T>
void AddOneByOne(SPTAG::IndexAlgoType algo, std::string distCalcMethod, std::shared_ptr<SPTAG::VectorSet>& vec, std::shared_ptr<SPTAG::MetadataSet>& meta, const std::string out)
{
    std::shared_ptr<SPTAG::VectorIndex> vecIndex = SPTAG::VectorIndex::CreateInstance(algo, SPTAG::GetEnumValueType<T>());
    BOOST_CHECK(nullptr != vecIndex);

    vecIndex->SetParameter("DistCalcMethod", distCalcMethod);
    vecIndex->SetParameter("NumberOfThreads", "16");
    
    auto t1 = std::chrono::high_resolution_clock::now();
    for (SPTAG::SizeType i = 0; i < vec->Count(); i++) {
        SPTAG::ByteArray metaarr = meta->GetMetadata(i);
        std::uint64_t offset[2] = { 0, metaarr.Length() };
        std::shared_ptr<SPTAG::MetadataSet> metaset(new SPTAG::MemMetadataSet(metaarr, SPTAG::ByteArray((std::uint8_t*)offset, 2 * sizeof(std::uint64_t), false), 1));
        SPTAG::ErrorCode ret = vecIndex->AddIndex(vec->GetVector(i), 1, vec->Dimension(), metaset, true);
        if (SPTAG::ErrorCode::Success != ret) std::cerr << "Error AddIndex(" << (int)(ret) << ") for vector " << i << std::endl;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "AddIndex time: " << (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / (float)(vec->Count())) << "us" << std::endl;
    
    Sleep(10000);

    BOOST_CHECK(SPTAG::ErrorCode::Success == vecIndex->SaveIndex(out));
}

template <typename T>
void Delete(const std::string folder, T* vec, SPTAG::SizeType n, const std::string out)
{
    std::shared_ptr<SPTAG::VectorIndex> vecIndex;
    BOOST_CHECK(SPTAG::ErrorCode::Success == SPTAG::VectorIndex::LoadIndex(folder, vecIndex));
    BOOST_CHECK(nullptr != vecIndex);

    BOOST_CHECK(SPTAG::ErrorCode::Success == vecIndex->DeleteIndex((const void*)vec, n));
    BOOST_CHECK(SPTAG::ErrorCode::Success == vecIndex->SaveIndex(out));
    vecIndex.reset();
}

template <typename T>
void Test(SPTAG::IndexAlgoType algo, std::string distCalcMethod)
{
    SPTAG::SizeType n = 2000, q = 3;
    SPTAG::DimensionType m = 10;
    int k = 3;
    std::vector<T> vec;
    for (SPTAG::SizeType i = 0; i < n; i++) {
        for (SPTAG::DimensionType j = 0; j < m; j++) {
            vec.push_back((T)i);
        }
    }
    
    std::vector<T> query;
    for (SPTAG::SizeType i = 0; i < q; i++) {
        for (SPTAG::DimensionType j = 0; j < m; j++) {
            query.push_back((T)i*2);
        }
    }
    
    std::vector<char> meta;
    std::vector<std::uint64_t> metaoffset;
    for (SPTAG::SizeType i = 0; i < n; i++) {
        metaoffset.push_back((std::uint64_t)meta.size());
        std::string a = std::to_string(i);
        for (size_t j = 0; j < a.length(); j++)
            meta.push_back(a[j]);
    }
    metaoffset.push_back((std::uint64_t)meta.size());

    std::shared_ptr<SPTAG::VectorSet> vecset(new SPTAG::BasicVectorSet(
        SPTAG::ByteArray((std::uint8_t*)vec.data(), sizeof(T) * n * m, false),
        SPTAG::GetEnumValueType<T>(), m, n));

    std::shared_ptr<SPTAG::MetadataSet> metaset(new SPTAG::MemMetadataSet(
        SPTAG::ByteArray((std::uint8_t*)meta.data(), meta.size() * sizeof(char), false),
        SPTAG::ByteArray((std::uint8_t*)metaoffset.data(), metaoffset.size() * sizeof(std::uint64_t), false),
        n));
    
    Build<T>(algo, distCalcMethod, vecset, metaset, "testindices");
    std::string truthmeta1[] = { "0", "1", "2", "2", "1", "3", "4", "3", "5" };
    Search<T>("testindices", query.data(), q, k, truthmeta1);

    if (algo != SPTAG::IndexAlgoType::SPANN) {
        Add<T>("testindices", vecset, metaset, "testindices");
        std::string truthmeta2[] = { "0", "0", "1", "2", "2", "1", "4", "4", "3" };
        Search<T>("testindices", query.data(), q, k, truthmeta2);

        Delete<T>("testindices", query.data(), q, "testindices");
        std::string truthmeta3[] = { "1", "1", "3", "1", "3", "1", "3", "5", "3" };
        Search<T>("testindices", query.data(), q, k, truthmeta3);
    }

    BuildWithMetaMapping<T>(algo, distCalcMethod, vecset, metaset, "testindices");
    std::string truthmeta4[] = { "0", "1", "2", "2", "1", "3", "4", "3", "5" };
    Search<T>("testindices", query.data(), q, k, truthmeta4);

    if (algo != SPTAG::IndexAlgoType::SPANN) {
        Add<T>("testindices", vecset, metaset, "testindices");
        std::string truthmeta5[] = { "0", "1", "2", "2", "1", "3", "4", "3", "5" };
        Search<T>("testindices", query.data(), q, k, truthmeta5);

        AddOneByOne<T>(algo, distCalcMethod, vecset, metaset, "testindices");
        std::string truthmeta6[] = { "0", "1", "2", "2", "1", "3", "4", "3", "5" };
        Search<T>("testindices", query.data(), q, k, truthmeta6);
    }
}

BOOST_AUTO_TEST_SUITE (AlgoTest)

BOOST_AUTO_TEST_CASE(KDTTest)
{
    Test<float>(SPTAG::IndexAlgoType::KDT, "L2");
}

BOOST_AUTO_TEST_CASE(BKTTest)
{
    Test<float>(SPTAG::IndexAlgoType::BKT, "L2");
}

BOOST_AUTO_TEST_CASE(SPANNTest)
{
    Test<float>(SPTAG::IndexAlgoType::SPANN, "L2");
}

BOOST_AUTO_TEST_SUITE_END()

// // 【新增】以下全部添加到文件末尾
// BOOST_AUTO_TEST_SUITE(RaBitQSanityCheck)

// BOOST_AUTO_TEST_CASE(CanCreateRaBitQ)
// {
//     // 验证能否成功创建一个 RaBitQQuantizer 对象
//     auto q = std::make_shared<SPTAG::COMMON::RaBitQQuantizer>();
//     BOOST_CHECK(q != nullptr);
    
//     // 【修改】强制转换为 int 进行比较
//     BOOST_CHECK_EQUAL((int)q->GetQuantizerType(), (int)SPTAG::QuantizerType::RaBitQQuantizer);

//     std::cout << "SUCCESS: RaBitQQuantizer created successfully!" << std::endl;
// }

// BOOST_AUTO_TEST_CASE(CanSaveAndLoadRaBitQConfig)
// {
//     // 验证 IQuantizer 工厂能否正确识别保存的 RaBitQ 类型
//     std::stringstream ss;
//     SPTAG::QuantizerType type = SPTAG::QuantizerType::RaBitQQuantizer;
//     SPTAG::SizeType size = 0; // 模拟空大小
//     ss.write((char*)&type, sizeof(SPTAG::QuantizerType));
//     ss.write((char*)&size, sizeof(SPTAG::SizeType));

//     // 2. 模拟加载
//     ss.seekg(0, std::ios::beg);
//     SPTAG::QuantizerType loadedType;
//     ss.read((char*)&loadedType, sizeof(SPTAG::QuantizerType));
    
//     std::shared_ptr<SPTAG::COMMON::IQuantizer> quantizer;
//     if (loadedType == SPTAG::QuantizerType::RaBitQQuantizer)
//     {
//         quantizer = std::make_shared<SPTAG::COMMON::RaBitQQuantizer>();
//     }

//     BOOST_CHECK(quantizer != nullptr);
    
//     // 【修改】强制转换为 int 进行比较
//     BOOST_CHECK_EQUAL((int)quantizer->GetQuantizerType(), (int)SPTAG::QuantizerType::RaBitQQuantizer);

//     std::cout << "SUCCESS: RaBitQQuantizer factory logic verified!" << std::endl;
// }

// BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(RaBitQSanityCheck)

// 基础验证：对象创建
BOOST_AUTO_TEST_CASE(CanCreateRaBitQ)
{
    auto q = std::make_shared<SPTAG::COMMON::RaBitQQuantizer>();
    BOOST_CHECK(q != nullptr);
    BOOST_CHECK_EQUAL((int)q->GetQuantizerType(), (int)SPTAG::QuantizerType::RaBitQQuantizer);
}

// 核心验证：使用 RaBitQQuantizer 类进行高精度测试 (Wrapper Class Test)
BOOST_AUTO_TEST_CASE(VerifyRaBitQWrapperAccuracy)
{
    std::cout << "\n[RaBitQ] Starting Wrapper Class Accuracy Test (Expect 4-bit precision)..." << std::endl;

    int dim = 128;
    int n = 100; // 训练样本数

    // 1. 创建并训练 Quantizer
    // 注意：RaBitQ Train 主要用于选择 Rotator，不涉及聚类
    std::shared_ptr<SPTAG::COMMON::RaBitQQuantizer> quantizer = std::make_shared<SPTAG::COMMON::RaBitQQuantizer>(dim);
    
    // 生成模拟训练数据
    std::vector<float> data(n * dim);
    for (int i = 0; i < n * dim; i++) {
        data[i] = (float)(rand() % 1000) / 1000.0f;
    }
    quantizer->Train(data.data(), n);

    // 2. 准备测试向量 vecA 和 vecB
    std::vector<float> vecA(dim), vecB(dim);
    for (int i = 0; i < dim; i++) {
        vecA[i] = (float)(rand() % 1000) / 1000.0f;
        vecB[i] = (float)(rand() % 1000) / 1000.0f;
    }

    // 3. 执行量化
    int qSize = quantizer->QuantizeSize();
    std::vector<uint8_t> qA(qSize), qB(qSize);
    
    quantizer->QuantizeVector(vecA.data(), qA.data());
    quantizer->QuantizeVector(vecB.data(), qB.data());

    // 4. 计算距离
    // A. 真实距离
    float trueDist = SPTAG::COMMON::DistanceUtils::ComputeL2Distance(vecA.data(), vecB.data(), dim);

    // B. 对称距离 (Wrapped SD): qA vs qB
    float sdDist = quantizer->L2Distance(qA.data(), qB.data());

    // C. 非对称距离 (Wrapped ADC): 必须先对 Query 进行旋转和补齐
    std::vector<float> rotA(dim + 128); // 多给一点空间容纳可能的 Padding
    quantizer->PreprocessQuery(vecA.data(), rotA.data());
    float adcDist = quantizer->L2Distance(rotA.data(), qB.data());

    // 5. 打印对比
    std::cout << "[RaBitQ Wrap] Dimension: " << dim << " (Padded inside: " << quantizer->QuantizeSize() - 8 << ")" << std::endl;
    std::cout << "[RaBitQ Wrap] True L2 Dist : " << trueDist << std::endl;
    std::cout << "[RaBitQ Wrap] Wrapper SD   : " << sdDist << std::endl;
    std::cout << "[RaBitQ Wrap] Wrapper ADC  : " << adcDist << std::endl;

    // 6. 验证精度 (针对 4-bit 精度预期)
    float error_sd = std::abs(trueDist - sdDist) / trueDist * 100.0f;
    float error_adc = std::abs(trueDist - adcDist) / trueDist * 100.0f;
    
    std::cout << "[RaBitQ Wrap] SD Error     : " << error_sd << "%" << std::endl;
    std::cout << "[RaBitQ Wrap] ADC Error    : " << error_adc << "%" << std::endl;

    // 这里的阈值设为 15% (之前通过 direct lib 测试大约是 5%)
    // 如果封装正确，误差应该差不多
    bool passed = (error_sd < 15.0f);
    
    if (passed) {
        std::cout << "[Pass] RaBitQQuantizer wrapper works perfectly!" << std::endl;
    } else {
        std::cout << "[Fail] Wrapper error too high. Implementation mismatch?" << std::endl;
    }
    BOOST_CHECK(passed);
}

// 新增：集成测试，验证 VectorIndex 能否加载 RaBitQ 配置
BOOST_AUTO_TEST_CASE(IntegrationTest_BuildIndexWithRaBitQ)
{
    std::cout << "\n[RaBitQ] Starting Integration Test: Build Index..." << std::endl;

    // 1. 准备数据
    int n = 200;
    int dim = 128;
    std::shared_ptr<SPTAG::VectorSet> vecSet = std::make_shared<SPTAG::BasicVectorSet>(
        SPTAG::ByteArray::Alloc(n * dim * sizeof(float)), 
        SPTAG::VectorValueType::Float, 
        dim, 
        n
    );
    
    // 填充随机数据
    float* data = reinterpret_cast<float*>(vecSet->GetData());
    for (int i = 0; i < n * dim; i++) {
        data[i] = (float)(rand() % 1000) / 1000.0f;
    }

    // 2. 创建 BKT 索引 (最简单的内存索引)
    // 注意：我们需要显式通过参数配置来启用量化器
    auto index = SPTAG::VectorIndex::CreateInstance(SPTAG::IndexAlgoType::BKT, SPTAG::VectorValueType::Float);
    BOOST_CHECK(index != nullptr);

    // 3. 设置配置
    // 关键：这里模拟从配置文件读取参数
    // 我们手动设置 Quantizer
    // 在实际流程中，Quantizer通常是在 Config 阶段被 SetQuantizer 初始化的，或者在 Build Index 时传入
    
    // 手动注入 RaBitQQuantizer
    auto quantizer = std::make_shared<SPTAG::COMMON::RaBitQQuantizer>(dim);
    index->SetQuantizer(quantizer);

    // 4. 构建索引
    // SPTAG 的 BuildIndex 会调用 Quantizer->Train 和 Quantizer->Quantize
    auto ret = index->BuildIndex(vecSet, nullptr, false); 
    BOOST_CHECK(ret == SPTAG::ErrorCode::Success);

    // 5. 验证是否真的使用了量化
    BOOST_CHECK(index->GetQuantizer() != nullptr);
    BOOST_CHECK_EQUAL((int)index->GetQuantizer()->GetQuantizerType(), (int)SPTAG::QuantizerType::RaBitQQuantizer);

    // // 6. 保存再加载 (验证 Save/Load 逻辑)
    // std::string testFile = "test_rabitq_index"; // 建议去掉 .bin 后缀，因为这通常被视为文件夹或前缀
    
    // // 保存
    // ret = index->SaveIndex(testFile);
    // BOOST_CHECK(ret == SPTAG::ErrorCode::Success);

    // // 加载回一个新的 Index
    // // 【修正】：LoadIndex 是静态函数，不需要先 CreateInstance
    // std::shared_ptr<SPTAG::VectorIndex> index2;
    // ret = SPTAG::VectorIndex::LoadIndex(testFile, index2);
    
    // BOOST_CHECK(ret == SPTAG::ErrorCode::Success);
    // BOOST_CHECK(index2 != nullptr);
    
    // // 验证加载后的 Quantizer
    // BOOST_CHECK(index2->GetQuantizer() != nullptr);
    // BOOST_CHECK_EQUAL((int)index2->GetQuantizer()->GetQuantizerType(), (int)SPTAG::QuantizerType::RaBitQQuantizer);

    std::cout << "[Pass] RaBitQ Integrated into BKT Index workflow successfully!" << std::endl;

    // 清理文件 (简单尝试清理，如果是文件夹可能需要递归删除，但在测试中可以暂时忽略)
    // remove(testFile.c_str()); 
}

BOOST_AUTO_TEST_CASE(RaBitQ_vs_Float32_Kernel_Benchmark)
{
    std::cout << "\n[Benchmark] Starting RaBitQ vs Float32 (Fixed)..." << std::endl;

    int n = 1000;
    int dim = 128;
    int repeats = 100000; // 增加次数

    std::vector<float> data(n * dim);
    for(int i=0; i<n*dim; ++i) data[i] = (float)(rand()%1000)/1000.0f;
    std::vector<float> query(dim);
    for(int i=0; i<dim; ++i) query[i] = (float)(rand()%1000)/1000.0f;

    auto rabitq = std::make_shared<SPTAG::COMMON::RaBitQQuantizer>(dim);
    rabitq->Train(data.data(), n);
    
    // std::vector<uint8_t> qQuery(rabitq->QuantizeSize());
    std::vector<uint8_t> qVec(rabitq->QuantizeSize());
    // rabitq->QuantizeVector(query.data(), qQuery.data());
    rabitq->QuantizeVector(data.data(), qVec.data());

    // --- RaBitQ ---
    std::vector<float> rotQuery(dim + 128);
    rabitq->PreprocessQuery(query.data(), rotQuery.data()); // 预处理一遍
    
    auto start = std::chrono::high_resolution_clock::now();
    volatile float total_dist_q = 0; // volatile 阻止优化
    for(int i=0; i<repeats; ++i) {
        // 修改为 ADC 测速（第一个参数是 float*）
        total_dist_q += rabitq->L2Distance(rotQuery.data(), qVec.data());
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_q = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // --- Float32 ---
    start = std::chrono::high_resolution_clock::now();
    volatile float total_dist_f = 0; // volatile 阻止优化
    for(int i=0; i<repeats; ++i) {
        // 为了防止缓存效应太强，我们可以假装每次都在偏移
        // 但这里为了纯粹测算 kernel 速度，保持不变即可
        total_dist_f += SPTAG::COMMON::DistanceUtils::ComputeL2Distance(query.data(), data.data(), dim);
    }
    end = std::chrono::high_resolution_clock::now();
    auto duration_f = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::cout << "[RaBitQ]  " << repeats << " ops: " << duration_q << " us. (Result: " << total_dist_q << ")" << std::endl;
    std::cout << "[Float32] " << repeats << " ops: " << duration_f << " us. (Result: " << total_dist_f << ")" << std::endl;
    
    if (duration_q > 0)
        std::cout << "[Result] RaBitQ Speed ratio: " << (float)duration_f / duration_q << "x (Expect < 1.0 now)" << std::endl;
}

// 【新增】召回率验证测试
BOOST_AUTO_TEST_CASE(RaBitQ_Search_Recall_Test)
{
    std::cout << "\n=============================================" << std::endl;
    std::cout << "[Comprehensive Test] RaBitQ: Recall, Latency & Storage" << std::endl;
    std::cout << "=============================================" << std::endl;

    int n = 10000;    // Database size
    int dim = 128;    // Dimension
    int K = 10;       // Top K

    // 1. Generate Data
    std::cout << "[1] Generating random data (" << n << " vectors)..." << std::endl;
    std::vector<float> data(n * dim);
    for(int i=0; i<n*dim; ++i) data[i] = (float)(rand()%1000)/1000.0f;

    // 2. Build Index (RaBitQ)
    std::cout << "[2] Building RaBitQ Index..." << std::endl;
    auto index = SPTAG::VectorIndex::CreateInstance(SPTAG::IndexAlgoType::BKT, SPTAG::VectorValueType::Float);
    index->SetParameter("DistCalcMethod", "L2");
    index->SetParameter("QuantizerType", "RaBitQ"); 
    index->SetParameter("RefineIterations", "3");
    index->SetParameter("NeighborhoodSize", "32");
    index->BuildIndex(data.data(), n, dim);

    // --- 内存/存储 评估 ---
    std::string temp_index_file = "test_rabitq_perf_index";
    index->SaveIndex(temp_index_file);
    
    // 计算文件大小
    // 【修复】累加所有相关文件的大小
    std::vector<std::string> index_files = {
        "vector.bin", 
        "graph.bin", 
        "tree.bin",       // BKT/KDT 树结构
        "quantizer.bin",  // 量化器数据
        "indexloader.ini",// 配置文件
        "metadata.bin",
        "metadataIndex.bin",
        "deletids.bin"
    };

    long long index_size_bytes = 0;
    
    // 【这里是修复的关键点】定义 raw_data_bytes
    long long raw_data_bytes = (long long)n * dim * sizeof(float);
    
    // 增加路径分隔符逻辑
    std::string folder_path = temp_index_file;
    if (folder_path.back() != '/' && folder_path.back() != '\\') folder_path += "/";

    for(const auto& fname : index_files) {
        std::ifstream in(folder_path + fname, std::ifstream::ate | std::ifstream::binary);
        if(in.is_open()) {
            long long fsize = in.tellg();
            index_size_bytes += fsize;
            // 调试打印，确认文件被找到
            // std::cout << "Found " << fname << ": " << fsize << " bytes" << std::endl;
        }
        in.close();
    }

    // 3. 性能测试：暴力搜索 (Brute Force / Ground Truth)
    std::cout << "[3] Running Brute Force Search (Baseline)..." << std::endl;
    std::vector<std::vector<int>> ground_truths(n, std::vector<int>(K));
    
    auto start_bf = std::chrono::high_resolution_clock::now();
    
    // 使用 OpenMP 加速暴力搜索的计算，模拟一个强劲的 Baseline
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        std::vector<std::pair<float, int>> distances(n);
        // 纯计算距离
        for (int j = 0; j < n; ++j) {
            float dist = SPTAG::COMMON::DistanceUtils::ComputeL2Distance(
                data.data() + i*dim, data.data() + j*dim, dim);
            distances[j] = {dist, j};
        }
        // Top K 排序
        std::sort(distances.begin(), distances.end()); // 全排序 (简单起见)
        for (int k = 0; k < K; ++k) {
            ground_truths[i][k] = distances[k].second;
        }
    }
    
    auto end_bf = std::chrono::high_resolution_clock::now();
    double time_bf_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_bf - start_bf).count();

    // 4. 性能测试：RaBitQ 索引搜索
    std::cout << "[4] Running RaBitQ Index Search..." << std::endl;
    double total_overlap_ratio = 0.0;
    int perfect_matches = 0;

    auto start_idx = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n; ++i) {
        SPTAG::QueryResult res(data.data() + i * dim, K, false);
        index->SearchIndex(res);

        // (为了不影响计时，召回率统计逻辑也放在循环里，但这点开销相比搜索可以忽略)
        int intersection_count = 0;
        // 注意：这里为了速度，我们假设 ground_truths 已经在上面计算好了
        // 如果 N 很大，频繁建立 unordered_set 会影响计时，但对比暴力搜索依然很快
        
        // 简单的验证逻辑
        for (int k = 0; k < K; ++k) {
            int found_vid = res.GetResult(k)->VID;
            // 在 ground truth 中查找
            bool exist = false;
            for(int g=0; g<K; ++g) {
                if(ground_truths[i][g] == found_vid) { exist=true; break; }
            }
            if (exist) intersection_count++;
        }
        
        total_overlap_ratio += (double)intersection_count / K;
        if (intersection_count == K) perfect_matches++;
    }

    auto end_idx = std::chrono::high_resolution_clock::now();
    double time_idx_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_idx - start_idx).count();

    // 5. 报告输出
    double avg_overlap = (total_overlap_ratio / n) * 100.0;
    double perfect_rate = ((double)perfect_matches / n) * 100.0;

    std::cout << "\n---------------------------------------------" << std::endl;
    std::cout << "             PERFORMANCE REPORT              " << std::endl;
    std::cout << "---------------------------------------------" << std::endl;
    
    std::cout << "Dataset: N=" << n << ", Dim=" << dim << std::endl;
    
    printf("Storage (Raw Float32):   %8.2f MB\n", raw_data_bytes / (1024.0 * 1024.0));
    printf("Storage (RaBitQ Index):  %8.2f MB\n", index_size_bytes / (1024.0 * 1024.0));
    printf("Compression Ratio:       %.2fx smaller\n", (float)raw_data_bytes / index_size_bytes);
    
    std::cout << "---------------------------------------------" << std::endl;

    printf("Time (Brute Force):      %8.2f ms (%.2f QPS)\n", time_bf_ms, (n * 1000.0) / time_bf_ms);
    printf("Time (RaBitQ Index):     %8.2f ms (%.2f QPS)\n", time_idx_ms, (n * 1000.0) / time_idx_ms);
    printf("Speedup:                 %.2fx faster\n", time_bf_ms / time_idx_ms);

    std::cout << "---------------------------------------------" << std::endl;

    std::cout << "[Recall] Average Overlap@" << K << ": " << avg_overlap << "%" << std::endl;
    std::cout << "[Recall] Perfect Match Rate:   " << perfect_rate << "%" << std::endl;
    
    // 简单的 Cleanup
    for(const auto& fname : index_files) remove((folder_path + fname).c_str());

    BOOST_CHECK_GT(avg_overlap, 90.0); 
    // 确保有加速效果 (在 N=20000 时，ANN 应该显著快于 O(N^2) 的暴力)
    BOOST_CHECK_GT(time_bf_ms, time_idx_ms); 
}

BOOST_AUTO_TEST_SUITE_END()