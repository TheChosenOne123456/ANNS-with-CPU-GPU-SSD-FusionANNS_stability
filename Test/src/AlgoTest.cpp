// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Test.h"
#include "inc/Helper/SimpleIniReader.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Core/Common/CommonUtils.h"

#include "inc/Core/Common/RaBitQQuantizer.h" // 【新增】添加这一行
// 【新增】为了直接测试 RaBitQ 库的底层逻辑，引入这些头文件
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/utils/rotator.hpp"

#include <unordered_set>
#include <chrono>

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

    // C. 非对称距离 (Wrapped ADC): vecA vs qB
    float adcDist = quantizer->L2Distance(vecA.data(), qB.data());

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

BOOST_AUTO_TEST_SUITE_END()