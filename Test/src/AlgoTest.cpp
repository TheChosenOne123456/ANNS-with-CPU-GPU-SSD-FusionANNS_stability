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
    int bits_per_code = 4;
    quantizer->SetBitsPerCode(bits_per_code);
    std::cout << "BitsPerCode=" << bits_per_code << std::endl;

    std::vector<float> trainData(n * dim);
    for (int i = 0; i < n * dim; ++i) trainData[i] = static_cast<float>(rand() % 1000) / 1000.0f;
    quantizer->Train(trainData.data(), n);

    // 假设vecA是查询向量，vecB是数据库中的一个向量，我们先计算它们的真实距离，然后通过 RaBitQ 的接口计算近似距离，并比较误差
    std::vector<float> vecA(dim), vecB(dim);
    for (int i = 0; i < dim; ++i) {
        vecA[i] = static_cast<float>(rand() % 1000) / 1000.0f;
        vecB[i] = static_cast<float>(rand() % 1000) / 1000.0f;
    }

    std::vector<std::uint8_t> qB(quantizer->QuantizeSize());
    quantizer->QuantizeVector(vecB.data(), qB.data());

    const float trueDist = SPTAG::COMMON::DistanceUtils::ComputeL2Distance(vecA.data(), vecB.data(), dim);

    std::vector<float> rotA(dim + 128, 0.0f);
    quantizer->PreprocessQuery(vecA.data(), rotA.data());

    // 计算误差界限信息
    SPTAG::COMMON::RaBitQQuantizer<float>::BondMeta bond_meta;
    quantizer->BuildL2EstimateQueryFactors(rotA.data(), bond_meta);

    float lowDist = 0.0f;
    const float estimateDist = quantizer->L2Distance(rotA.data(), qB.data(), bond_meta, &lowDist);
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
    int bits_per_code = 4;
    quantizer->SetBitsPerCode(bits_per_code);
    std::cout << "BitsPerCode=" << bits_per_code << std::endl;

    std::vector<std::uint8_t> trainData(n * dim);
    for (int i = 0; i < n * dim; ++i) trainData[i] = static_cast<std::uint8_t>(rand() % 256);
    quantizer->Train(trainData.data(), n);

    std::vector<std::uint8_t> vecA(dim), vecB(dim);
    for (int i = 0; i < dim; ++i) {
        vecA[i] = static_cast<std::uint8_t>(rand() % 256);
        vecB[i] = static_cast<std::uint8_t>(rand() % 256);
    }

    std::vector<std::uint8_t> qB(quantizer->QuantizeSize());
    quantizer->QuantizeVector(vecB.data(), qB.data());

    const float trueDist = ComputeL2U8(vecA, vecB);

    std::vector<float> rotA(dim + 128, 0.0f);
    quantizer->PreprocessQuery(vecA.data(), rotA.data());

    // 计算误差界限信息
    SPTAG::COMMON::RaBitQQuantizer<std::uint8_t>::BondMeta bond_meta;
    quantizer->BuildL2EstimateQueryFactors(rotA.data(), bond_meta);
    
    float lowDist = 0.0f;
    const float estimateDist = quantizer->L2Distance(rotA.data(), qB.data(), bond_meta, &lowDist);
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
    quantizer->Train(trainData.data(), n);

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
    rabitq->Train(data.data(), n);

    // 此处应该分别量化所有数据向量
    std::vector<std::uint8_t> qData(n * rabitq->QuantizeSize());
    for (int i = 0; i < n; ++i) {
        rabitq->QuantizeVector(data.data() + i * dim, qData.data() + i * rabitq->QuantizeSize());
    }

    auto start = std::chrono::high_resolution_clock::now();
    volatile float totalQ = 0;
    for (int i = 0; i < repeats; ++i) {
        std::vector<float> rotQuery(dim + 128, 0.0f);
        rabitq->PreprocessQuery(query.data(), rotQuery.data());

        SPTAG::COMMON::RaBitQQuantizer<float>::BondMeta bond_meta;
        rabitq->BuildL2EstimateQueryFactors(rotQuery.data(), bond_meta);

        for (int j = 0; j < n; ++j) {
            totalQ += rabitq->L2Distance(rotQuery.data(), qData.data() + j * rabitq->QuantizeSize(), bond_meta);
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
    rabitq->Train(data.data(), n);

    // 此处应该分别量化所有数据向量
    std::vector<std::uint8_t> qData(n * rabitq->QuantizeSize());
    for (int i = 0; i < n; ++i) {
        rabitq->QuantizeVector(data.data() + i * dim, qData.data() + i * rabitq->QuantizeSize());
    }

    auto start = std::chrono::high_resolution_clock::now();
    volatile float totalQ = 0;
    for (int i = 0; i < repeats; ++i) {
        std::vector<float> rotQuery(dim + 128, 0.0f);
        rabitq->PreprocessQuery(query.data(), rotQuery.data());

        SPTAG::COMMON::RaBitQQuantizer<std::uint8_t>::BondMeta bond_meta;
        rabitq->BuildL2EstimateQueryFactors(rotQuery.data(), bond_meta);

        for (int j = 0; j < n; ++j) {
            totalQ += rabitq->L2Distance(rotQuery.data(), qData.data() + j * rabitq->QuantizeSize(), bond_meta);
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

// 召回率/延迟/存储（缩小规模，保证单测可跑通）
BOOST_AUTO_TEST_CASE(RaBitQ_Search_Recall_Test)
{
    std::cout << "\n[Comprehensive Test] RaBitQ: Recall / Latency / Storage" << std::endl;

    const int n = 50000;
    const int dim = 128;
    const int q = 100;
    const int K = 5;

    std::vector<float> data(n * dim);
    for (int i = 0; i < n * dim; ++i) data[i] = static_cast<float>(rand() % 1000) / 1000.0f;

    auto vecSet = std::make_shared<SPTAG::BasicVectorSet>(
        SPTAG::ByteArray(reinterpret_cast<std::uint8_t*>(data.data()), sizeof(float) * data.size(), false),
        SPTAG::VectorValueType::Float,
        dim,
        n
    );
    BOOST_REQUIRE(vecSet != nullptr);

    auto index = SPTAG::VectorIndex::CreateInstance(SPTAG::IndexAlgoType::BKT, SPTAG::VectorValueType::Float);
    BOOST_REQUIRE(index != nullptr);

    index->SetParameter("DistCalcMethod", "L2");
    index->SetParameter("RefineIterations", "3");
    index->SetParameter("NeighborhoodSize", "32");

    auto quantizer = std::make_shared<SPTAG::COMMON::RaBitQQuantizer<float>>(dim);
    BOOST_REQUIRE(quantizer != nullptr);
    // 指定量化bit数
    int bits_per_code = 2;
    quantizer->SetBitsPerCode(bits_per_code);
    std::cout << "BitsPerCode=" << bits_per_code << std::endl;
    index->SetQuantizer(quantizer);

    BOOST_REQUIRE(SPTAG::ErrorCode::Success == index->BuildIndex(vecSet, nullptr, false));

    const std::string outDir = MakeUniqueTestPath("test_rabitq_perf_index");
    BOOST_REQUIRE(SPTAG::ErrorCode::Success == index->SaveIndex(outDir));

    long long indexSizeBytes = 0;
    const std::vector<std::string> indexFiles = {
        "vector.bin", "graph.bin", "tree.bin", "quantizer.bin",
        "indexloader.ini", "metadata.bin", "metadataIndex.bin", "deletids.bin"
    };

    std::string folder = outDir;
    if (folder.back() != '/' && folder.back() != '\\') folder += "/";

    for (const auto& f : indexFiles) {
        std::ifstream in(folder + f, std::ifstream::ate | std::ifstream::binary);
        if (in.is_open()) indexSizeBytes += static_cast<long long>(in.tellg());
    }
    BOOST_CHECK_GT(indexSizeBytes, 0);

    std::vector<std::vector<int>> gt(q, std::vector<int>(K, -1));
    auto startBF = std::chrono::high_resolution_clock::now();
    for (int qi = 0; qi < q; ++qi) {
        const float* qv = data.data() + qi * dim;
        std::vector<std::pair<float, int>> dists;
        dists.reserve(n);
        for (int j = 0; j < n; ++j) {
            float d = SPTAG::COMMON::DistanceUtils::ComputeL2Distance(qv, data.data() + j * dim, dim);
            dists.emplace_back(d, j);
        }
        std::partial_sort(dists.begin(), dists.begin() + K, dists.end(),
            [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                return a.first < b.first;
            });
        for (int k = 0; k < K; ++k) gt[qi][k] = dists[k].second;
    }
    auto endBF = std::chrono::high_resolution_clock::now();
    double bfMs = std::chrono::duration_cast<std::chrono::milliseconds>(endBF - startBF).count();

    double totalOverlap = 0.0;
    int perfect = 0;
    auto startIdx = std::chrono::high_resolution_clock::now();
    for (int qi = 0; qi < q; ++qi) {
        SPTAG::QueryResult res(data.data() + qi * dim, K, false);
        BOOST_REQUIRE(SPTAG::ErrorCode::Success == index->SearchIndex(res));

        int hit = 0;
        for (int k = 0; k < K; ++k) {
            auto r = res.GetResult(k);
            if (r == nullptr) continue;
            int vid = r->VID;
            for (int g = 0; g < K; ++g) {
                if (gt[qi][g] == vid) {
                    ++hit;
                    break;
                }
            }
        }
        totalOverlap += static_cast<double>(hit) / K;
        if (hit == K) ++perfect;
    }
    auto endIdx = std::chrono::high_resolution_clock::now();
    double idxMs = std::chrono::duration_cast<std::chrono::milliseconds>(endIdx - startIdx).count();

    double avgOverlap = totalOverlap / q * 100.0;
    double perfectRate = static_cast<double>(perfect) / q * 100.0;
    long long rawBytes = static_cast<long long>(n) * dim * sizeof(float);

    std::cout << "Raw size: " << (rawBytes / (1024.0 * 1024.0)) << " MB, "
              << "Index size: " << (indexSizeBytes / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "Brute-force: " << bfMs << " ms, Index search: " << idxMs << " ms" << std::endl;
    std::cout << "Average Overlap@" << K << ": " << avgOverlap
              << "%, Perfect rate: " << perfectRate << "%" << std::endl;

    BOOST_CHECK_GT(avgOverlap, 60.0);
}

BOOST_AUTO_TEST_SUITE_END()