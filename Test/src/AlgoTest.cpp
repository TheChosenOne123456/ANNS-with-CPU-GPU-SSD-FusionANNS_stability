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
#include <iomanip>  // 表格对齐

#include <random>
#include <cmath>
#include "inc/Helper/VectorSetReader.h"

#include <set>
#include <queue>

#include "inc/Core/Common/TruthSet.h"

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

namespace {
float ComputeL2U8(const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b)
{
    float s = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float d = static_cast<float>(a[i]) - static_cast<float>(b[i]);
        s += d * d;
    }
    return s;
}

std::string MakeUniqueTestPath(const std::string& prefix)
{
    auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    return prefix + "_" + std::to_string(static_cast<long long>(stamp));
}
} // namespace

BOOST_AUTO_TEST_SUITE(RaBitQSanityCheck)

// 基础验证：模板化后 float / uint8 两种类型都能创建
BOOST_AUTO_TEST_CASE(CanCreateRaBitQ)
{
    auto qf = std::make_shared<SPTAG::COMMON::RaBitQQuantizer<float>>();
    auto qu8 = std::make_shared<SPTAG::COMMON::RaBitQQuantizer<std::uint8_t>>();

    BOOST_REQUIRE(qf != nullptr);
    BOOST_REQUIRE(qu8 != nullptr);

    BOOST_CHECK_EQUAL((int)qf->GetQuantizerType(), (int)SPTAG::QuantizerType::RaBitQQuantizer);
    BOOST_CHECK_EQUAL((int)qu8->GetQuantizerType(), (int)SPTAG::QuantizerType::RaBitQQuantizer);

    BOOST_CHECK_EQUAL((int)qf->GetReconstructType(), (int)SPTAG::VectorValueType::Float);
    BOOST_CHECK_EQUAL((int)qu8->GetReconstructType(), (int)SPTAG::VectorValueType::UInt8);
}

// 核心验证：float wrapper 精度
BOOST_AUTO_TEST_CASE(VerifyRaBitQWrapperAccuracy_Float)
{
    std::cout << "\n[RaBitQ][float] Wrapper Accuracy Test..." << std::endl;

    const int dim = 128;
    const int n = 128;

    auto quantizer = std::make_shared<SPTAG::COMMON::RaBitQQuantizer<float>>(dim);
    BOOST_REQUIRE(quantizer != nullptr);

    // 指定量化bit数
    int bits_per_code = 2;
    quantizer->SetBitsPerCode(bits_per_code);
    std::cout << "BitsPerCode=" << bits_per_code << std::endl;

    std::vector<float> trainData(n * dim);
    for (int i = 0; i < n * dim; ++i) trainData[i] = static_cast<float>(rand() % 1000) / 1000.0f;
    const std::uint8_t centroidCount = 10;
    quantizer->Train(trainData.data(), n, centroidCount);

    // 假设vecA是查询向量，vecB是数据库中的一个向量，我们先计算它们的真实距离，然后通过 RaBitQ 的接口计算近似距离，并比较误差
    std::vector<float> vecA(dim), vecB(dim);
    for (int i = 0; i < dim; ++i) {
        vecA[i] = static_cast<float>(rand() % 1000) / 1000.0f;
        vecB[i] = static_cast<float>(rand() % 1000) / 1000.0f;
    }

    std::vector<std::uint8_t> qB(quantizer->QuantizeSize());
    quantizer->QuantizeVector(vecB.data(), qB.data());
    // 测试
    const auto* meta = reinterpret_cast<const SPTAG::COMMON::RaBitQQuantizer<float>::RaBitQEstimateMeta*>(qB.data());
    std::cout << "meta: delta=" << meta->delta << " vl=" << meta->vl
            << " f_add=" << meta->f_add << " f_rescale=" << meta->f_rescale
            << " f_error=" << meta->f_error << std::endl;

    const float trueDist = SPTAG::COMMON::DistanceUtils::ComputeL2Distance(vecA.data(), vecB.data(), dim);

    const size_t paddedDim = quantizer->GetPaddedDim();
    std::vector<float> rotA(static_cast<size_t>(centroidCount) * paddedDim, 0.0f);
    quantizer->PreprocessQuery(vecA.data(), rotA.data());

    // 计算误差界限信息
    std::vector<SPTAG::COMMON::RaBitQQuantizer<float>::BondMeta> bond_meta(centroidCount);
    quantizer->BuildL2EstimateQueryFactors(rotA.data(), bond_meta.data());

    float lowDist = 0.0f;
    const float estimateDist = quantizer->L2Distance(rotA.data(), qB.data(), bond_meta.data(), &lowDist);
    const float errorBound = estimateDist - lowDist;    // 误差界

    const float denom = std::max(trueDist, 1e-6f);
    const float err = std::abs(trueDist - estimateDist) / denom * 100.0f;

    std::cout << "[float] True=" << trueDist
              << " Estimate=" << estimateDist
              << " err=" << err << "%" << std::endl;
    
    std::cout << "LowerBound=" << lowDist
              << " ErrorBound=" << errorBound
              << " IsSafePruning(" << (lowDist <= trueDist ? "Yes" : "No") << ")" << std::endl;

    BOOST_CHECK_LT(err, 20.0f);
}

// 核心验证：uint8 wrapper 精度（模板类型不再依赖 Auto）
BOOST_AUTO_TEST_CASE(VerifyRaBitQWrapperAccuracy_UInt8)
{
    std::cout << "\n[RaBitQ][uint8] Wrapper Accuracy Test..." << std::endl;

    const int dim = 128;
    const int n = 128;

    auto quantizer = std::make_shared<SPTAG::COMMON::RaBitQQuantizer<std::uint8_t>>(dim);
    BOOST_REQUIRE(quantizer != nullptr);

    // 指定量化bit数
    int bits_per_code = 2;
    quantizer->SetBitsPerCode(bits_per_code);
    std::cout << "BitsPerCode=" << bits_per_code << std::endl;

    std::vector<std::uint8_t> trainData(n * dim);
    for (int i = 0; i < n * dim; ++i) trainData[i] = static_cast<std::uint8_t>(rand() % 256);
    const std::uint8_t centroidCount = 10;
    quantizer->Train(trainData.data(), n, centroidCount);

    std::vector<std::uint8_t> vecA(dim), vecB(dim);
    for (int i = 0; i < dim; ++i) {
        vecA[i] = static_cast<std::uint8_t>(rand() % 256);
        vecB[i] = static_cast<std::uint8_t>(rand() % 256);
    }

    std::vector<std::uint8_t> qB(quantizer->QuantizeSize());
    quantizer->QuantizeVector(vecB.data(), qB.data());

    const float trueDist = ComputeL2U8(vecA, vecB);

    const size_t paddedDim = quantizer->GetPaddedDim();
    std::vector<float> rotA(static_cast<size_t>(centroidCount) * paddedDim, 0.0f);
    quantizer->PreprocessQuery(vecA.data(), rotA.data());

    // 计算误差界限信息
    std::vector<SPTAG::COMMON::RaBitQQuantizer<std::uint8_t>::BondMeta> bond_meta(centroidCount);
    quantizer->BuildL2EstimateQueryFactors(rotA.data(), bond_meta.data());
    
    float lowDist = 0.0f;
    const float estimateDist = quantizer->L2Distance(rotA.data(), qB.data(), bond_meta.data(), &lowDist);
    const float errorBound = estimateDist - lowDist;    // 误差界

    const float denom = std::max(trueDist, 1e-6f);
    const float err = std::abs(trueDist - estimateDist) / denom * 100.0f;

    std::cout << "[uint8] True=" << trueDist
              << " Estimate=" << estimateDist
              << " err=" << err << "%" << std::endl;

    std::cout << "LowerBound=" << lowDist
              << " ErrorBound=" << errorBound
              << " IsSafePruning(" << (lowDist <= trueDist ? "Yes" : "No") << ")" << std::endl;

    BOOST_CHECK_LT(err, 35.0f);
}

// 新增：验证 Save/Load 后 rotator 等信息被正确恢复（同输入得到同量化结果）
BOOST_AUTO_TEST_CASE(SaveLoadRaBitQConfigAndRotator)
{
    const int dim = 128;
    const int n = 128;
    const std::string qFile = MakeUniqueTestPath("test_rabitq_quantizer") + ".bin";

    auto quantizer = std::make_shared<SPTAG::COMMON::RaBitQQuantizer<float>>(dim);
    BOOST_REQUIRE(quantizer != nullptr);

    std::vector<float> trainData(n * dim);
    for (int i = 0; i < n * dim; ++i) trainData[i] = static_cast<float>(rand() % 1000) / 1000.0f;
    quantizer->Train(trainData.data(), n, 10);

    std::vector<float> vec(dim);
    for (int i = 0; i < dim; ++i) vec[i] = static_cast<float>(rand() % 1000) / 1000.0f;

    std::vector<std::uint8_t> qBefore(quantizer->QuantizeSize());
    quantizer->QuantizeVector(vec.data(), qBefore.data());

    {
        auto out = SPTAG::f_createIO();
        BOOST_REQUIRE(out != nullptr);
        BOOST_REQUIRE(out->Initialize(qFile.c_str(), std::ios::binary | std::ios::out));
        BOOST_REQUIRE(SPTAG::ErrorCode::Success == quantizer->SaveQuantizer(out));
        out->ShutDown();
    }

    std::shared_ptr<SPTAG::COMMON::IQuantizer> loadedBase;
    {
        auto in = SPTAG::f_createIO();
        BOOST_REQUIRE(in != nullptr);
        BOOST_REQUIRE(in->Initialize(qFile.c_str(), std::ios::binary | std::ios::in));
        loadedBase = SPTAG::COMMON::IQuantizer::LoadIQuantizer(in);
        in->ShutDown();
    }

    BOOST_REQUIRE(loadedBase != nullptr);
    auto loaded = std::dynamic_pointer_cast<SPTAG::COMMON::RaBitQQuantizer<float>>(loadedBase);
    BOOST_REQUIRE(loaded != nullptr);

    std::vector<std::uint8_t> qAfter(loaded->QuantizeSize());
    loaded->QuantizeVector(vec.data(), qAfter.data());

    BOOST_CHECK_EQUAL(qBefore.size(), qAfter.size());
    BOOST_CHECK_EQUAL_COLLECTIONS(qBefore.begin(), qBefore.end(), qAfter.begin(), qAfter.end());

    std::remove(qFile.c_str());
}

// 集成测试：VectorIndex + RaBitQ(float) 的 Build/Save/Load 流程
BOOST_AUTO_TEST_CASE(IntegrationTest_BuildIndexWithRaBitQ)
{
    std::cout << "\n[RaBitQ] Integration Test: Build/Save/Load..." << std::endl;

    const int n = 512;
    const int dim = 128;

    auto vecSet = std::make_shared<SPTAG::BasicVectorSet>(
        SPTAG::ByteArray::Alloc(n * dim * sizeof(float)),
        SPTAG::VectorValueType::Float,
        dim,
        n
    );
    BOOST_REQUIRE(vecSet != nullptr);

    float* data = reinterpret_cast<float*>(vecSet->GetData());
    for (int i = 0; i < n * dim; ++i) data[i] = static_cast<float>(rand() % 1000) / 1000.0f;

    auto index = SPTAG::VectorIndex::CreateInstance(SPTAG::IndexAlgoType::BKT, SPTAG::VectorValueType::Float);
    BOOST_REQUIRE(index != nullptr);

    index->SetParameter("DistCalcMethod", "L2");
    index->SetParameter("RefineIterations", "3");
    index->SetParameter("NeighborhoodSize", "32");

    auto quantizer = std::make_shared<SPTAG::COMMON::RaBitQQuantizer<float>>(dim);
    BOOST_REQUIRE(quantizer != nullptr);
    index->SetQuantizer(quantizer);

    BOOST_REQUIRE(SPTAG::ErrorCode::Success == index->BuildIndex(vecSet, nullptr, false));
    BOOST_REQUIRE(index->GetQuantizer() != nullptr);
    BOOST_CHECK_EQUAL((int)index->GetQuantizer()->GetQuantizerType(), (int)SPTAG::QuantizerType::RaBitQQuantizer);

    const std::string indexDir = MakeUniqueTestPath("test_rabitq_index");
    BOOST_REQUIRE(SPTAG::ErrorCode::Success == index->SaveIndex(indexDir));

    std::shared_ptr<SPTAG::VectorIndex> loaded;
    BOOST_REQUIRE(SPTAG::ErrorCode::Success == SPTAG::VectorIndex::LoadIndex(indexDir, loaded));
    BOOST_REQUIRE(loaded != nullptr);
    BOOST_REQUIRE(loaded->GetQuantizer() != nullptr);

    BOOST_CHECK_EQUAL((int)loaded->GetQuantizer()->GetQuantizerType(), (int)SPTAG::QuantizerType::RaBitQQuantizer);
    BOOST_CHECK_EQUAL((int)loaded->GetQuantizer()->GetReconstructType(), (int)SPTAG::VectorValueType::Float);

    SPTAG::QueryResult qr(data, 5, false);
    BOOST_CHECK(SPTAG::ErrorCode::Success == loaded->SearchIndex(qr));

    std::cout << std::endl;
}

// 性能冒烟：确保路径可运行，不做强约束速度断言
BOOST_AUTO_TEST_CASE(RaBitQ_vs_Float32_Kernel_Benchmark)
{
    std::cout << "\n[Benchmark] RaBitQ vs Float32..." << std::endl;

    const int n = 1000000;
    const int dim = 128;
    const int repeats = 100;

    std::vector<float> data(n * dim);
    for (int i = 0; i < n * dim; ++i) data[i] = static_cast<float>(rand() % 1000) / 1000.0f;

    std::vector<float> query(dim);
    for (int i = 0; i < dim; ++i) query[i] = static_cast<float>(rand() % 1000) / 1000.0f;

    auto rabitq = std::make_shared<SPTAG::COMMON::RaBitQQuantizer<float>>(dim);
    BOOST_REQUIRE(rabitq != nullptr);

    // 指定量化bit数
    int bits_per_code = 2;
    rabitq->SetBitsPerCode(bits_per_code);
    std::cout << "BitsPerCode=" << bits_per_code << std::endl;
    const std::uint8_t centroidCount = 10;
    rabitq->Train(data.data(), n, centroidCount);

    // 此处应该分别量化所有数据向量
    std::vector<std::uint8_t> qData(n * rabitq->QuantizeSize());
    for (int i = 0; i < n; ++i) {
        rabitq->QuantizeVector(data.data() + i * dim, qData.data() + i * rabitq->QuantizeSize());
    }

    const size_t paddedDim = rabitq->GetPaddedDim();
    std::vector<float> rotQuery(static_cast<size_t>(centroidCount) * paddedDim, 0.0f);
    std::vector<SPTAG::COMMON::RaBitQQuantizer<float>::BondMeta> bond_meta(centroidCount);

    auto start = std::chrono::high_resolution_clock::now();
    volatile float totalQ = 0;
    for (int i = 0; i < repeats; ++i) {
        rabitq->PreprocessQuery(query.data(), rotQuery.data());
        rabitq->BuildL2EstimateQueryFactors(rotQuery.data(), bond_meta.data());

        for (int j = 0; j < n; ++j) {
            totalQ += rabitq->L2Distance(rotQuery.data(),
                                        qData.data() + j * rabitq->QuantizeSize(),
                                        bond_meta.data());
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto tQ = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    volatile float totalF = 0;
    for (int i = 0; i < repeats; ++i) {
        for (int j = 0; j < n; ++j) {
            totalF += SPTAG::COMMON::DistanceUtils::ComputeL2Distance(query.data(), data.data() + j * dim, dim);
        }
    }

    end = std::chrono::high_resolution_clock::now();
    auto tF = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << "[RaBitQ]  " << repeats << " ops: " << tQ << " us, sum=" << totalQ << std::endl;
    std::cout << "[Float32] " << repeats << " ops: " << tF << " us, sum=" << totalF << std::endl;
    BOOST_CHECK_GT(tQ, 0);
    BOOST_CHECK_GT(tF, 0);
}

// 性能冒烟：确保路径可运行，不做强约束速度断言
BOOST_AUTO_TEST_CASE(RaBitQ_vs_Uint8_Kernel_Benchmark)
{
    std::cout << "\n[Benchmark] RaBitQ vs Uint8..." << std::endl;

    const int n = 1000000;
    const int dim = 128;
    const int repeats = 100;

    std::vector<std::uint8_t> data(n * dim);
    for (int i = 0; i < n * dim; ++i) data[i] = static_cast<std::uint8_t>(rand() % 256);

    std::vector<std::uint8_t> query(dim);
    for (int i = 0; i < dim; ++i) query[i] = static_cast<std::uint8_t>(rand() % 256);

    auto rabitq = std::make_shared<SPTAG::COMMON::RaBitQQuantizer<std::uint8_t>>(dim);
    BOOST_REQUIRE(rabitq != nullptr);

    // 指定量化bit数
    int bits_per_code = 2;
    rabitq->SetBitsPerCode(bits_per_code);
    std::cout << "BitsPerCode=" << bits_per_code << std::endl;
    const std::uint8_t centroidCount = 10;
    rabitq->Train(data.data(), n, centroidCount);

    // 此处应该分别量化所有数据向量
    std::vector<std::uint8_t> qData(n * rabitq->QuantizeSize());
    for (int i = 0; i < n; ++i) {
        rabitq->QuantizeVector(data.data() + i * dim, qData.data() + i * rabitq->QuantizeSize());
    }

    const size_t paddedDim = rabitq->GetPaddedDim();
    std::vector<float> rotQuery(static_cast<size_t>(centroidCount) * paddedDim, 0.0f);
    std::vector<SPTAG::COMMON::RaBitQQuantizer<std::uint8_t>::BondMeta> bond_meta(centroidCount);

    auto start = std::chrono::high_resolution_clock::now();
    volatile float totalQ = 0;
    for (int i = 0; i < repeats; ++i) {
        rabitq->PreprocessQuery(query.data(), rotQuery.data());
        rabitq->BuildL2EstimateQueryFactors(rotQuery.data(), bond_meta.data());

        for (int j = 0; j < n; ++j) {
            totalQ += rabitq->L2Distance(rotQuery.data(),
                                        qData.data() + j * rabitq->QuantizeSize(),
                                        bond_meta.data());
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto tQ = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    volatile float totalI = 0;
    for (int i = 0; i < repeats; ++i) {
        for (int j = 0; j < n; ++j) {
            totalI += ComputeL2U8(query, std::vector<std::uint8_t>(data.data() + j * dim, data.data() + (j+1) * dim));
        }
    }

    end = std::chrono::high_resolution_clock::now();
    auto tI = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << "[RaBitQ]  " << repeats << " ops: " << tQ << " us, sum=" << totalQ << std::endl;
    std::cout << "[Uint8] " << repeats << " ops: " << tI << " us, sum=" << totalI << std::endl;
    BOOST_CHECK_GT(tQ, 0);
    BOOST_CHECK_GT(tI, 0);
}

// // 召回率/延迟/存储（缩小规模，保证单测可跑通）
// BOOST_AUTO_TEST_CASE(RaBitQ_Search_Recall_Test)
// {
//     std::cout << "\n[Comprehensive Test] RaBitQ: Recall / Latency / Storage" << std::endl;

//     const int n = 50000;
//     const int dim = 128;
//     const int q = 100;
//     const int K = 5;

//     std::vector<float> data(n * dim);
//     for (int i = 0; i < n * dim; ++i) data[i] = static_cast<float>(rand() % 1000) / 1000.0f;

//     auto vecSet = std::make_shared<SPTAG::BasicVectorSet>(
//         SPTAG::ByteArray(reinterpret_cast<std::uint8_t*>(data.data()), sizeof(float) * data.size(), false),
//         SPTAG::VectorValueType::Float,
//         dim,
//         n
//     );
//     BOOST_REQUIRE(vecSet != nullptr);

//     auto index = SPTAG::VectorIndex::CreateInstance(SPTAG::IndexAlgoType::BKT, SPTAG::VectorValueType::Float);
//     BOOST_REQUIRE(index != nullptr);

//     index->SetParameter("DistCalcMethod", "L2");
//     index->SetParameter("RefineIterations", "3");
//     index->SetParameter("NeighborhoodSize", "32");

//     auto quantizer = std::make_shared<SPTAG::COMMON::RaBitQQuantizer<float>>(dim);
//     BOOST_REQUIRE(quantizer != nullptr);
//     // 指定量化bit数
//     int bits_per_code = 2;
//     quantizer->SetBitsPerCode(bits_per_code);
//     std::cout << "BitsPerCode=" << bits_per_code << std::endl;
//     index->SetQuantizer(quantizer);

//     BOOST_REQUIRE(SPTAG::ErrorCode::Success == index->BuildIndex(vecSet, nullptr, false));

//     const std::string outDir = MakeUniqueTestPath("test_rabitq_perf_index");
//     BOOST_REQUIRE(SPTAG::ErrorCode::Success == index->SaveIndex(outDir));

//     long long indexSizeBytes = 0;
//     const std::vector<std::string> indexFiles = {
//         "vector.bin", "graph.bin", "tree.bin", "quantizer.bin",
//         "indexloader.ini", "metadata.bin", "metadataIndex.bin", "deletids.bin"
//     };

//     std::string folder = outDir;
//     if (folder.back() != '/' && folder.back() != '\\') folder += "/";

//     for (const auto& f : indexFiles) {
//         std::ifstream in(folder + f, std::ifstream::ate | std::ifstream::binary);
//         if (in.is_open()) indexSizeBytes += static_cast<long long>(in.tellg());
//     }
//     BOOST_CHECK_GT(indexSizeBytes, 0);

//     std::vector<std::vector<int>> gt(q, std::vector<int>(K, -1));
//     auto startBF = std::chrono::high_resolution_clock::now();
//     for (int qi = 0; qi < q; ++qi) {
//         const float* qv = data.data() + qi * dim;
//         std::vector<std::pair<float, int>> dists;
//         dists.reserve(n);
//         for (int j = 0; j < n; ++j) {
//             float d = SPTAG::COMMON::DistanceUtils::ComputeL2Distance(qv, data.data() + j * dim, dim);
//             dists.emplace_back(d, j);
//         }
//         std::partial_sort(dists.begin(), dists.begin() + K, dists.end(),
//             [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
//                 return a.first < b.first;
//             });
//         for (int k = 0; k < K; ++k) gt[qi][k] = dists[k].second;
//     }
//     auto endBF = std::chrono::high_resolution_clock::now();
//     double bfMs = std::chrono::duration_cast<std::chrono::milliseconds>(endBF - startBF).count();

//     double totalOverlap = 0.0;
//     int perfect = 0;
//     auto startIdx = std::chrono::high_resolution_clock::now();
//     for (int qi = 0; qi < q; ++qi) {
//         SPTAG::QueryResult res(data.data() + qi * dim, K, false);
//         BOOST_REQUIRE(SPTAG::ErrorCode::Success == index->SearchIndex(res));

//         int hit = 0;
//         for (int k = 0; k < K; ++k) {
//             auto r = res.GetResult(k);
//             if (r == nullptr) continue;
//             int vid = r->VID;
//             for (int g = 0; g < K; ++g) {
//                 if (gt[qi][g] == vid) {
//                     ++hit;
//                     break;
//                 }
//             }
//         }
//         totalOverlap += static_cast<double>(hit) / K;
//         if (hit == K) ++perfect;
//     }
//     auto endIdx = std::chrono::high_resolution_clock::now();
//     double idxMs = std::chrono::duration_cast<std::chrono::milliseconds>(endIdx - startIdx).count();

//     double avgOverlap = totalOverlap / q * 100.0;
//     double perfectRate = static_cast<double>(perfect) / q * 100.0;
//     long long rawBytes = static_cast<long long>(n) * dim * sizeof(float);

//     std::cout << "Raw size: " << (rawBytes / (1024.0 * 1024.0)) << " MB, "
//               << "Index size: " << (indexSizeBytes / (1024.0 * 1024.0)) << " MB" << std::endl;
//     std::cout << "Brute-force: " << bfMs << " ms, Index search: " << idxMs << " ms" << std::endl;
//     std::cout << "Average Overlap@" << K << ": " << avgOverlap
//               << "%, Perfect rate: " << perfectRate << "%" << std::endl;

//     BOOST_CHECK_GT(avgOverlap, 60.0);
// }

BOOST_AUTO_TEST_CASE(RaBitQ_RealData_EstimateVsTrue)
{
    const std::string root = "/home/ANNS_SSD/tyh/SIFT100M_data/";
    const std::string quantizerPath = root + "2bits_rabitq_quantizer.100M";
    const std::string codePath = root + "learn_2bits_rabitq.100M.u8bin";
    const std::string basePath = root + "learn.100M.u8bin";

    const int Q = 10000; // 查询向量数（外层）
    const int K = 10000; // 每次查询随机比较的量化向量数（内层）
    const int poolN = std::max(Q, K);

    auto ptr_vector = SPTAG::f_createIO();
    BOOST_REQUIRE(ptr_vector != nullptr);
    BOOST_REQUIRE(ptr_vector->Initialize(codePath.c_str(), std::ios::binary | std::ios::in));
    int qcount = 0, qdim = 0;
    BOOST_REQUIRE_EQUAL(ptr_vector->ReadBinary(sizeof(qcount), reinterpret_cast<char*>(&qcount)), sizeof(qcount));
    BOOST_REQUIRE_EQUAL(ptr_vector->ReadBinary(sizeof(qdim), reinterpret_cast<char*>(&qdim)), sizeof(qdim));
    ptr_vector.reset();

    std::shared_ptr<SPTAG::VectorSet> qSet;
    {
        auto vectorOptions = std::make_shared<SPTAG::Helper::ReaderOptions>(
            SPTAG::VectorValueType::UInt8,
            qdim,
            SPTAG::VectorFileType::DEFAULT);
        auto vectorReader = SPTAG::Helper::VectorSetReader::CreateInstance(vectorOptions);
        BOOST_REQUIRE(vectorReader != nullptr);
        BOOST_REQUIRE(SPTAG::ErrorCode::Success == vectorReader->LoadFile(codePath));
        qSet = vectorReader->GetVectorSet(0, poolN);
        BOOST_REQUIRE(qSet != nullptr);
    }
    int actualQCount = static_cast<int>(std::min<int>(poolN, (int)qSet->Count()));

    auto ptr_base = SPTAG::f_createIO();
    BOOST_REQUIRE(ptr_base != nullptr);
    BOOST_REQUIRE(ptr_base->Initialize(basePath.c_str(), std::ios::binary | std::ios::in));
    int bcount = 0, bdim = 0;
    BOOST_REQUIRE_EQUAL(ptr_base->ReadBinary(sizeof(bcount), reinterpret_cast<char*>(&bcount)), sizeof(bcount));
    BOOST_REQUIRE_EQUAL(ptr_base->ReadBinary(sizeof(bdim), reinterpret_cast<char*>(&bdim)), sizeof(bdim));
    ptr_base.reset();

    std::shared_ptr<SPTAG::VectorSet> baseSet;
    {
        auto baseOptions = std::make_shared<SPTAG::Helper::ReaderOptions>(
            SPTAG::VectorValueType::UInt8,
            bdim,
            SPTAG::VectorFileType::DEFAULT);
        auto baseReader = SPTAG::Helper::VectorSetReader::CreateInstance(baseOptions);
        BOOST_REQUIRE(baseReader != nullptr);
        BOOST_REQUIRE(SPTAG::ErrorCode::Success == baseReader->LoadFile(basePath));
        baseSet = baseReader->GetVectorSet(0, poolN);
        BOOST_REQUIRE(baseSet != nullptr);
    }
    int actualBCount = static_cast<int>(std::min<int>(poolN, (int)baseSet->Count()));
    BOOST_REQUIRE_EQUAL(bdim, bdim); // noop, keep type checks visible

    auto ptr_q = SPTAG::f_createIO();
    BOOST_REQUIRE(ptr_q != nullptr);
    BOOST_REQUIRE(ptr_q->Initialize(quantizerPath.c_str(), std::ios::binary | std::ios::in));
    auto iquant = SPTAG::COMMON::IQuantizer::LoadIQuantizer(ptr_q);
    BOOST_REQUIRE(iquant != nullptr);
    auto rtype = iquant->GetReconstructType();

    std::mt19937 rng(123456);
    std::uniform_int_distribution<int> distIdx(0, std::min(actualQCount, actualBCount) - 1);

    double totalEstimateTimeUs = 0.0, totalTrueTimeUs = 0.0;
    double sumRelErr = 0.0;
    uint64_t totalComparisons = 0;
    double sumErrorBoundOverEst = 0.0;  // errorBound在est中的比例
    uint64_t violations_below_lower = 0;
    uint64_t violations_above_upper = 0;
    uint64_t very_large_error_cases = 0;

    const float EPS = 1e-6f;

    if (rtype == SPTAG::VectorValueType::UInt8)
    {
        auto rq = std::dynamic_pointer_cast<SPTAG::COMMON::RaBitQQuantizer<std::uint8_t>>(iquant);
        BOOST_REQUIRE(rq != nullptr);

        const size_t paddedDim = rq->GetPaddedDim();
        const size_t centroidCount = rq->GetCentroidCount();
        std::vector<float> rotBuf(centroidCount * paddedDim, 0.0f);
        std::vector<SPTAG::COMMON::RaBitQQuantizer<std::uint8_t>::BondMeta> bond(centroidCount);

        for (int qi = 0; qi < Q; ++qi)
        {
            int qIdx = qi % actualBCount;
            const std::uint8_t* queryVec = reinterpret_cast<const std::uint8_t*>(baseSet->GetVector(qIdx));

            // PreprocessQuery + build bond_meta (not timed into estimate loop)
            rq->PreprocessQuery(queryVec, rotBuf.data());
            rq->BuildL2EstimateQueryFactors(rotBuf.data(), bond.data());

            // 先采样 K 个索引（保证估计与真实距离分别计时）
            std::vector<int> sampledIdx(K);
            for (int ki = 0; ki < K; ++ki) sampledIdx[ki] = distIdx(rng) % actualBCount;

            // 估计循环（单独计时）
            std::vector<float> estimates(K);
            std::vector<float> lowBounds(K);
            auto t_est_start = std::chrono::high_resolution_clock::now();
            for (int ki = 0; ki < K; ++ki)
            {
                const uint8_t* codePtr = reinterpret_cast<const uint8_t*>(qSet->GetVector(sampledIdx[ki]));
                float lowBound = 0.0f;
                float est = rq->L2Distance(rotBuf.data(), codePtr, bond.data(), &lowBound);
                estimates[ki] = est;
                lowBounds[ki] = lowBound;
            }
            auto t_est_end = std::chrono::high_resolution_clock::now();
            totalEstimateTimeUs += std::chrono::duration_cast<std::chrono::microseconds>(t_est_end - t_est_start).count();

            // 真实距离循环（单独计时并统计违例）
            auto t_true_start = std::chrono::high_resolution_clock::now();
            for (int ki = 0; ki < K; ++ki)
            {
                int dbIdx = sampledIdx[ki];
                const uint8_t* trueVec = reinterpret_cast<const uint8_t*>(baseSet->GetVector(dbIdx));

                float trueDist = 0.0f;
                for (int d = 0; d < bdim; ++d)
                {
                    float diff = static_cast<float>(queryVec[d]) - static_cast<float>(trueVec[d]);
                    trueDist += diff * diff;
                }

                if (trueDist > 0.0){
                    float est = estimates[ki];
                    float lowBound = lowBounds[ki];
                    // lowBound = 2 * lowBound - est; // 临时降低，改为两个errorBound
                    float errorBound = est - lowBound;
                    // std::cout << "Est=" << est << ", LowBound=" << lowBound
                    //           << ", ErrorBound=" << errorBound << std::endl;
                    float boundRatio = (est > 1e-6f) ? (errorBound / est) : 0.0f;
                    sumErrorBoundOverEst += boundRatio;
                    if(trueDist <= 8000.0 && trueDist > 0.0) {
                        std::cout << "small dist : " << "Est=" << est << ", LowBound=" << lowBound << ", TrueDist = " << trueDist << std::endl;
                    }
                    float upperBound = est + errorBound; // 对称近似上界用于检验是否存在上界

                    if (trueDist + 1e-6f < lowBound) ++violations_below_lower;
                    if (trueDist > upperBound + 1e-6f) ++violations_above_upper;
                    if ((est / trueDist > 8.0f) && (trueDist > 0.0)) ++very_large_error_cases;

                    float rel = (trueDist > EPS) ? (std::abs(est - trueDist) / trueDist) : 0.0f;
                    sumRelErr += std::min(rel, 1.0f);
                    ++totalComparisons;
                }
            }
            auto t_true_end = std::chrono::high_resolution_clock::now();
            totalTrueTimeUs += std::chrono::duration_cast<std::chrono::microseconds>(t_true_end - t_true_start).count();
        }
    }
    else if (rtype == SPTAG::VectorValueType::Float)
    {
        auto rqf = std::dynamic_pointer_cast<SPTAG::COMMON::RaBitQQuantizer<float>>(iquant);
        BOOST_REQUIRE(rqf != nullptr);

        const size_t paddedDim = rqf->GetPaddedDim();
        const size_t centroidCount = rqf->GetCentroidCount();
        std::vector<float> rotBuf(centroidCount * paddedDim, 0.0f);
        std::vector<SPTAG::COMMON::RaBitQQuantizer<float>::BondMeta> bond(centroidCount);

        for (int qi = 0; qi < Q; ++qi)
        {
            int qIdx = qi % actualBCount;
            const std::uint8_t* rawQueryU8 = reinterpret_cast<const std::uint8_t*>(baseSet->GetVector(qIdx));
            std::vector<float> queryF(bdim);
            for (int d = 0; d < bdim; ++d) queryF[d] = static_cast<float>(rawQueryU8[d]);

            rqf->PreprocessQuery(queryF.data(), rotBuf.data());
            rqf->BuildL2EstimateQueryFactors(rotBuf.data(), bond.data());

            std::vector<int> sampledIdx(K);
            for (int ki = 0; ki < K; ++ki) sampledIdx[ki] = distIdx(rng) % actualBCount;

            std::vector<float> estimates(K);
            std::vector<float> lowBounds(K);
            auto t_est_start = std::chrono::high_resolution_clock::now();
            for (int ki = 0; ki < K; ++ki)
            {
                const uint8_t* codePtr = reinterpret_cast<const uint8_t*>(qSet->GetVector(sampledIdx[ki]));
                float lowBound = 0.0f;
                float est = rqf->L2Distance(rotBuf.data(), codePtr, bond.data(), &lowBound);
                estimates[ki] = est;
                lowBounds[ki] = lowBound;
            }
            auto t_est_end = std::chrono::high_resolution_clock::now();
            totalEstimateTimeUs += std::chrono::duration_cast<std::chrono::microseconds>(t_est_end - t_est_start).count();

            auto t_true_start = std::chrono::high_resolution_clock::now();
            for (int ki = 0; ki < K; ++ki)
            {
                int dbIdx = sampledIdx[ki];
                const uint8_t* trueVecU8 = reinterpret_cast<const uint8_t*>(baseSet->GetVector(dbIdx));

                float trueDist = 0.0f;
                for (int d = 0; d < bdim; ++d)
                {
                    float diff = queryF[d] - static_cast<float>(trueVecU8[d]);
                    trueDist += diff * diff;
                }

                if (trueDist > 0.0){
                    float est = estimates[ki];
                    float lowBound = lowBounds[ki];
                    // lowBound = 2 * lowBound - est; // 临时降低，改为两个errorBound
                    float errorBound = est - lowBound;
                    float boundRatio = (est > 1e-6f) ? (errorBound / est) : 0.0f;
                    sumErrorBoundOverEst += boundRatio;
                    float upperBound = est + errorBound;

                    if (trueDist + 1e-6f < lowBound) ++violations_below_lower;
                    if (trueDist > upperBound + 1e-6f) ++violations_above_upper;
                    if ((est / trueDist > 8.0f) && (trueDist > 0)) ++very_large_error_cases;

                    float rel = (trueDist > EPS) ? (std::abs(est - trueDist) / trueDist) : 0.0f;
                    sumRelErr += std::min(rel, 1.0f);
                    ++totalComparisons;
                }
            }
            auto t_true_end = std::chrono::high_resolution_clock::now();
            totalTrueTimeUs += std::chrono::duration_cast<std::chrono::microseconds>(t_true_end - t_true_start).count();
        }
    }
    else
    {
        BOOST_FAIL("Unsupported quantizer reconstruct type for this test.");
    }

    double avgRelErr = (totalComparisons > 0) ? (sumRelErr / totalComparisons) : 0.0;
    double avgEstimateTimePerCompUs = (totalComparisons > 0) ? (totalEstimateTimeUs / totalComparisons) : 0.0;
    double avgTrueTimePerCompUs = (totalComparisons > 0) ? (totalTrueTimeUs / totalComparisons) : 0.0;
    double avgErrorBoundRatio = (totalComparisons > 0) ? (sumErrorBoundOverEst / totalComparisons) : 0.0;

    uint64_t countViolations = violations_below_lower + violations_above_upper;

    std::cout << "Total comparisons: " << totalComparisons << std::endl;
    std::cout << "Violations below lower bound: " << violations_below_lower << std::endl;
    std::cout << "Violations above upper bound: " << violations_above_upper << std::endl;
    std::cout << "very large error cases: " << very_large_error_cases << std::endl;
    std::cout << "Violations (sum): " << countViolations << std::endl;
    std::cout << "Average errorBound/est ratio: " << avgErrorBoundRatio << std::endl;
    std::cout << "Average relative error (capped at 1): " << avgRelErr << std::endl;
    std::cout << "Avg estimate time per comp (us): " << avgEstimateTimePerCompUs << std::endl;
    std::cout << "Avg true L2 time per comp (us): " << avgTrueTimePerCompUs << std::endl;

    // 断言：无过多违例（阈值可调，保持和之前一致）
    BOOST_CHECK_LT(static_cast<double>(countViolations) / std::max<uint64_t>(1, totalComparisons), 0.05); // 违例比例 < 5%
}

BOOST_AUTO_TEST_CASE(RaBitQ_RealData_Recall_Top50)
{
    const std::string root = "/home/ANNS_SSD/tyh/SIFT100M_data/";
    const std::string quantizerPath = root + "2bits_rabitq_quantizer.100M";
    const std::string codePath = root + "learn_2bits_rabitq.100M.u8bin";
    const std::string basePath = root + "learn.100M.u8bin";
    const std::string queryPath = root + "query.public.10K.u8bin";
    const std::string truthPath = root + "sift_100M_gt100";

    auto qio = SPTAG::f_createIO();
    BOOST_REQUIRE(qio && qio->Initialize(quantizerPath.c_str(), std::ios::binary | std::ios::in));
    auto iquant = SPTAG::COMMON::IQuantizer::LoadIQuantizer(qio);
    auto rq = std::dynamic_pointer_cast<SPTAG::COMMON::RaBitQQuantizer<std::uint8_t>>(iquant);
    BOOST_REQUIRE(rq != nullptr);

    int qcount = 0, qdim = 0;
    {
        auto ptr = SPTAG::f_createIO();
        BOOST_REQUIRE(ptr && ptr->Initialize(codePath.c_str(), std::ios::binary | std::ios::in));
        BOOST_REQUIRE_EQUAL(ptr->ReadBinary(sizeof(qcount), (char*)&qcount), sizeof(qcount));
        BOOST_REQUIRE_EQUAL(ptr->ReadBinary(sizeof(qdim), (char*)&qdim), sizeof(qdim));
    }

    std::shared_ptr<SPTAG::VectorSet> qSet;
    {
        auto opts = std::make_shared<SPTAG::Helper::ReaderOptions>(
            SPTAG::VectorValueType::UInt8, qdim, SPTAG::VectorFileType::DEFAULT);
        auto reader = SPTAG::Helper::VectorSetReader::CreateInstance(opts);
        BOOST_REQUIRE(reader && reader->LoadFile(codePath) == SPTAG::ErrorCode::Success);
        qSet = reader->GetVectorSet(0, qcount);
    }

    int bcount = 0, bdim = 0;
    {
        auto ptr = SPTAG::f_createIO();
        BOOST_REQUIRE(ptr && ptr->Initialize(basePath.c_str(), std::ios::binary | std::ios::in));
        BOOST_REQUIRE_EQUAL(ptr->ReadBinary(sizeof(bcount), (char*)&bcount), sizeof(bcount));
        BOOST_REQUIRE_EQUAL(ptr->ReadBinary(sizeof(bdim), (char*)&bdim), sizeof(bdim));
    }

    std::shared_ptr<SPTAG::VectorSet> baseSet;
    {
        auto opts = std::make_shared<SPTAG::Helper::ReaderOptions>(
            SPTAG::VectorValueType::UInt8, bdim, SPTAG::VectorFileType::DEFAULT);
        auto reader = SPTAG::Helper::VectorSetReader::CreateInstance(opts);
        BOOST_REQUIRE(reader && reader->LoadFile(basePath) == SPTAG::ErrorCode::Success);
        baseSet = reader->GetVectorSet(0, bcount);
    }

    std::shared_ptr<SPTAG::VectorSet> querySet;
    {
        auto opts = std::make_shared<SPTAG::Helper::ReaderOptions>(
            SPTAG::VectorValueType::UInt8, bdim, SPTAG::VectorFileType::DEFAULT);
        auto reader = SPTAG::Helper::VectorSetReader::CreateInstance(opts);
        BOOST_REQUIRE(reader && reader->LoadFile(queryPath) == SPTAG::ErrorCode::Success);
        querySet = reader->GetVectorSet();
    }

    std::vector<std::set<SPTAG::SizeType>> truth;
    {
        auto ptr = SPTAG::f_createIO();
        BOOST_REQUIRE(ptr && ptr->Initialize(truthPath.c_str(), std::ios::binary | std::ios::in));

        SPTAG::SizeType numQueries = static_cast<SPTAG::SizeType>(querySet->Count());
        int originalK = 0;
        const int K = 50;

        SPTAG::COMMON::TruthSet::LoadTruth(
            ptr, truth, numQueries, originalK, K, SPTAG::TruthFileType::DEFAULT);
    }

    const int K = 50;
    const SPTAG::SizeType maxQ = std::min<SPTAG::SizeType>(
        100, std::min<SPTAG::SizeType>(querySet->Count(), static_cast<SPTAG::SizeType>(truth.size())));

    size_t centroidCount = rq->GetCentroidCount();
    if (centroidCount == 0) centroidCount = 1;
    const size_t paddedDim = rq->GetPaddedDim();

    std::vector<float> rot(centroidCount * paddedDim, 0.0f);
    std::vector<SPTAG::COMMON::RaBitQQuantizer<std::uint8_t>::BondMeta> bond(centroidCount);

    struct Item { float dist; SPTAG::SizeType id; };
    struct Row  { SPTAG::SizeType id; float trueDist; float estDist; float lowBound; };

    for (SPTAG::SizeType qi = 0; qi < maxQ; ++qi)
    {
        rq->PreprocessQuery(querySet->GetVector(qi), rot.data());
        rq->BuildL2EstimateQueryFactors(rot.data(), bond.data());

        auto cmp = [](const Item& a, const Item& b){ return a.dist < b.dist; };
        std::priority_queue<Item, std::vector<Item>, decltype(cmp)> heap(cmp);

        for (SPTAG::SizeType i = 0; i < qSet->Count(); ++i) {
            const std::uint8_t* code = reinterpret_cast<const std::uint8_t*>(qSet->GetVector(i));
            float d = rq->L2Distance(rot.data(), code, bond.data());
            if (static_cast<int>(heap.size()) < K) heap.push({d, i});
            else if (d < heap.top().dist) { heap.pop(); heap.push({d, i}); }
        }

        std::vector<Item> topk;
        while (!heap.empty()) { topk.push_back(heap.top()); heap.pop(); }
        std::sort(topk.begin(), topk.end(), [](const Item& a, const Item& b){ return a.dist < b.dist; });

        std::unordered_set<SPTAG::SizeType> topkIds;
        topkIds.reserve(topk.size());
        for (const auto& it : topk) topkIds.insert(it.id);

        int hit = 0;
        for (auto& it : topk) if (truth[qi].count(it.id)) ++hit;
        float recall = static_cast<float>(hit) / K;

        std::vector<Row> rows;
        rows.reserve(truth[qi].size());

        const std::uint8_t* qv = reinterpret_cast<const std::uint8_t*>(querySet->GetVector(qi));
        for (auto id : truth[qi]) {
            if (id >= baseSet->Count() || id >= qSet->Count()) continue;

            const std::uint8_t* bv = reinterpret_cast<const std::uint8_t*>(baseSet->GetVector(id));
            float gtDist = 0.0f;
            for (int d = 0; d < bdim; ++d) {
                float diff = static_cast<float>(qv[d]) - static_cast<float>(bv[d]);
                gtDist += diff * diff;
            }

            const std::uint8_t* code = reinterpret_cast<const std::uint8_t*>(qSet->GetVector(id));
            float lowBound = 0.0f;
            float estDist = rq->L2Distance(rot.data(), code, bond.data(), &lowBound);

            rows.push_back({id, gtDist, estDist, lowBound});
        }

        std::sort(rows.begin(), rows.end(),
                  [](const Row& a, const Row& b) { return a.trueDist < b.trueDist; });

        const size_t outN = std::min<size_t>(K, rows.size());

        size_t lbGtCount = 0;
        double sumRelErr = 0.0;
        for (size_t i = 0; i < outN; ++i) {
            const auto& r = rows[i];
            if (r.lowBound > r.trueDist) ++lbGtCount;
            if (r.trueDist > 0.0f) {
                sumRelErr += std::abs(r.estDist - r.trueDist) / r.trueDist;
            }
        }
        double lbGtRate = outN ? (100.0 * static_cast<double>(lbGtCount) / outN) : 0.0;
        double avgRelErr = outN ? (sumRelErr / outN) : 0.0;

        size_t top10in50 = 0;
        size_t top10N = std::min<size_t>(10, rows.size());
        for (size_t i = 0; i < top10N; ++i) {
            if (topkIds.count(rows[i].id)) ++top10in50;
        }

        auto oldFlags = std::cout.flags();
        auto oldPrec = std::cout.precision();
        std::cout << "Q" << qi << " recall@50=" << recall
                  << std::fixed << std::setprecision(6)
                  << " lb>gt%=" << lbGtRate
                  << " avgRelErr=" << avgRelErr
                  << " top10in50=" << top10in50
                  << "\n";
        std::cout.flags(oldFlags);
        std::cout.precision(oldPrec);

        std::cout << std::left
                  << std::setw(12) << "ID"
                  << std::setw(18) << "TrueDist"
                  << std::setw(18) << "EstDist"
                  << std::setw(18) << "LowBound"
                  << "\n";

        std::cout << std::fixed << std::setprecision(6);
        for (size_t i = 0; i < outN; ++i) {
            const auto& r = rows[i];
            std::cout << std::left
                      << std::setw(12) << r.id
                      << std::setw(18) << r.trueDist
                      << std::setw(18) << r.estDist
                      << std::setw(18) << r.lowBound
                      << "\n";
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()