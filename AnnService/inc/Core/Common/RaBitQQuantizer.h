/************************************************************
 *  ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
 *  ★        RaBitQ EMBEDDED MODULE (Dual Input)           ★
 *  ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
 ************************************************************/

#ifndef _SPTAG_COMMON_RABITQQUANTIZER_H_
#define _SPTAG_COMMON_RABITQQUANTIZER_H_

#include "CommonUtils.h"
#include "DistanceUtils.h"
#include "IQuantizer.h"
#include <vector>
#include <cstring>
#include <cmath>
#include <memory> 
#include <iostream>
#include <limits>
#include <algorithm>
#include <cstdint>

#include <type_traits>
#include <fstream>
#include <cstdio>
#include <unistd.h>

// RaBitQ Includes
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/index/estimator.hpp"
// 引入底层的位打包与专用内积函数
#include "rabitqlib/quantization/pack_excode.hpp"
#include "rabitqlib/utils/space.hpp"

#ifndef __CUDACC__
#include <immintrin.h> 
#endif

namespace SPTAG
{
    namespace COMMON
    {
        // 封装 RaBitQ 并支持 float/uint8 两种原始输入类型
        template<typename T = float>
        class RaBitQQuantizer : public IQuantizer
        {
            // 防止报错
            // static_assert(std::is_same<T, float>::value || std::is_same<T, std::uint8_t>::value, "RaBitQQuantizer only supports float or uint8 input types.");
        public:
            // 标识量化器落盘格式版本，给 Save/Load 做向后兼容分支
            enum class PersistVersion : std::uint32_t { V1 = 1, V2 = 2 };
            // 标识“旋转器是按哪种形式存储/恢复”的类型标签
            enum class RotatorStorageType : std::uint8_t { FhtKac = 0, Matrix = 1 };

            // 默认构造并初始化基础状态
            RaBitQQuantizer()
                : m_Dim(0),
                  m_PaddedDim(0),
                  m_CodeSize(0),
                  m_MetaSize(sizeof(RaBitQEstimateMeta)),
                  m_QuantizedSize(0),
                  m_Rotator(nullptr),
                  m_EnableADC(false),
                  m_BitsPerCode(4),
                  m_RotatorType(RotatorStorageType::FhtKac),
                  m_FormatVersion(PersistVersion::V2),
                  m_hasCentroid(0),
                  m_Centroid(nullptr)
            {
            }

            // 按维度构造并计算量化存储大小
            RaBitQQuantizer(DimensionType dim)
                : m_Dim(dim),
                  m_Rotator(nullptr),
                  m_EnableADC(false),
                  m_BitsPerCode(4),
                  m_RotatorType(RotatorStorageType::FhtKac),
                  m_FormatVersion(PersistVersion::V2)
            {
                RecalcSizes();
            }

            // 析构时释放旋转器资源
            virtual ~RaBitQQuantizer()
            {
                if (m_Rotator) delete m_Rotator;
            }

            // 返回量化器类型标识
            virtual QuantizerType GetQuantizerType() const { return QuantizerType::RaBitQQuantizer; }

            // 这里返回输入类型，与 PQ 逻辑一致，用于工厂分发具体模板实例
            virtual VectorValueType GetReconstructType() const { return GetEnumValueType<T>(); }

            // 返回单向量量化后的字节数
            virtual SizeType QuantizeSize() const { return m_QuantizedSize; }

            // 返回当前量化器的基数
            virtual int GetBase() const { return 1; }

            // 返回子向量数量（该实现为单子向量）
            virtual DimensionType GetNumSubvectors() const { return 1; }

            // 返回是否启用 ADC 模式
            virtual bool GetEnableADC() const { return m_EnableADC; }

            // 设置是否启用 ADC 模式
            virtual void SetEnableADC(bool enableADC) { m_EnableADC = enableADC; }

            // 返回 L2 查找表指针（此量化器不使用查找表）
            virtual float* GetL2DistanceTables() { return nullptr; }

            // 返回 codebook 指针（此量化器不使用 codebook）
            template<typename CodebookT> CodebookT* GetCodebooks() { return nullptr; }

            void SetBitsPerCode(SizeType bits)
            {
                // 当前这版距离核按“每维1字节”读取，建议先限制在 1/2/4/8
                if (bits != 1 && bits != 2 && bits != 4 && bits != 8) bits = 4;
                m_BitsPerCode = bits;
                RecalcSizes();
            }

            SizeType GetBitsPerCode() const { return m_BitsPerCode; }
            
            // 设置旋转器存储类型（影响 Save/Load 行为）
            void SetRotatorType(RotatorStorageType t) { m_RotatorType = t; }

            // 数据向量的元信息
            struct RaBitQEstimateMeta {
                float delta;
                float vl;
                float f_add;
                float f_rescale;
                float f_error;
            };

            // 查询向量的元信息，和理论误差界限的计算相关
            struct BondMeta {
                float g_add;
                float k1xsumq;
                float g_error;
            };

            // 构建查询侧估算因子（对应论文公式里的 g_add / k1xsumq / g_error）
            inline void BuildL2EstimateQueryFactors(
                const float* rotated_query,
                BondMeta& bond_meta
            ) const
            {
                float sumq = 0.0f;
                float norm2 = 0.0f;
                for (DimensionType i = 0; i < m_PaddedDim; ++i)  {
                    float v = rotated_query[i];
                    sumq += v;
                    norm2 += v * v;
                }

                // L2 下 g_add = ||q||^2
                bond_meta.g_add = norm2;

                // 与 rabitqlib::query.hpp 的 c1 一致: c1 = -((1<<1)-1)/2 = -0.5
                bond_meta.k1xsumq = -0.5f * sumq;

                // 误差界项里会乘 g_error（L2 下为 ||q||）
                bond_meta.g_error = std::sqrt(std::max(0.0f, norm2));
            }

            // full_est_dist 需要的内积函数指针（float query vs uint8 code）
            inline static float IPFloatU8(const float* q, const uint8_t* c, size_t dim)
            {
                return rabitqlib::excode_ipimpl::ip_fxi<float, uint8_t>(q, c, dim);
            }

            // 新的估算距离核心：返回估算距离；可选返回 lower bound
            inline float L2DistanceEstimate(
                const float* rotated_query,
                const std::uint8_t* pY,
                float* out_low_dist = nullptr
            ) const
            {
                const auto* meta = reinterpret_cast<const RaBitQEstimateMeta*>(pY);
                const uint8_t* code = reinterpret_cast<const uint8_t*>(pY + m_MetaSize);

                // float g_add = 0.0f;
                // float k1xsumq = 0.0f;
                // float g_error = 0.0f;
                BondMeta bond_meta;
                BuildL2EstimateQueryFactors(rotated_query, bond_meta);

                float est = rabitqlib::quant::full_est_dist<float, uint8_t>(
                    code,
                    rotated_query,
                    m_PackedIpFunc, // 使用位压缩专用 SIMD 内核替换普通的 IPFloatU8
                    m_PaddedDim,
                    m_BitsPerCode,
                    meta->f_add,
                    meta->f_rescale,
                    bond_meta.g_add,
                    bond_meta.k1xsumq
                );

                // 论文式 lower bound（保守剪枝）
                if (out_low_dist != nullptr) {
                    *out_low_dist = est - meta->f_error * bond_meta.g_error;
                }

                return est;
            }

            // 重载，额外接收查询元信息以避免重复计算（建议使用）
            inline float L2DistanceEstimate(
                const float* rotated_query,
                const std::uint8_t* pY,
                const BondMeta bond_meta,
                float* out_low_dist = nullptr
            ) const
            {
                const auto* meta = reinterpret_cast<const RaBitQEstimateMeta*>(pY);
                const uint8_t* code = reinterpret_cast<const uint8_t*>(pY + m_MetaSize);

                float g_add = bond_meta.g_add;
                float k1xsumq = bond_meta.k1xsumq;
                float g_error = bond_meta.g_error;

                float est = rabitqlib::quant::full_est_dist<float, uint8_t>(
                    code,
                    rotated_query,
                    m_PackedIpFunc,
                    m_PaddedDim,
                    m_BitsPerCode,
                    meta->f_add,
                    meta->f_rescale,
                    g_add,
                    k1xsumq
                );

                // 论文式 lower bound（保守剪枝）
                if (out_low_dist != nullptr) {
                    *out_low_dist = est - meta->f_error * g_error;
                }

                return est;
            }

            // 训练阶段仅用于选择旋转器和更新尺寸
            void Train(const void* data, SizeType num, bool useCentroid = false)
            {
                if (!m_Rotator) {
                    if (m_RotatorType == RotatorStorageType::Matrix) {
                        m_Rotator = rabitqlib::choose_rotator<float>(
                            m_Dim, rabitqlib::RotatorType::MatrixRotator
                        );
                    } else {
                        m_Rotator = rabitqlib::choose_rotator<float>(
                            m_Dim, rabitqlib::RotatorType::FhtKacRotator
                        );
                    }
                    // m_Rotator = rabitqlib::choose_rotator<float>(
                    //         m_Dim, rabitqlib::RotatorType::FhtKacRotator
                    //     );
                    RecalcSizes();
                }
                if (!useCentroid || data == nullptr || num == 0) {
                    m_Centroid.reset();
                    m_hasCentroid = 0;
                    return;
                }

                m_hasCentroid = 1;
                m_Centroid.reset(new std::vector<float>(m_Dim, 0.0f));
                const T* p = reinterpret_cast<const T*>(data);

                for (SizeType n = 0; n < num; ++n) {
                    const T* v = p + static_cast<size_t>(n) * m_Dim;
                    for (DimensionType d = 0; d < m_Dim; ++d) {
                        (*m_Centroid)[d] += static_cast<float>(v[d]);
                    }
                }

                const float inv = 1.0f / static_cast<float>(num);
                for (DimensionType d = 0; d < m_Dim; ++d) {
                    (*m_Centroid)[d] *= inv;
                }
            }

            // 将原始向量量化为指定码值并写入 meta 信息
            virtual void QuantizeVector(const void* vec, std::uint8_t* vecout, bool ADC = true) const
            {
                (void)ADC;
                std::vector<float> fvec(m_Dim);
                ConvertInputToFloat(reinterpret_cast<const T*>(vec), fvec.data());

                if (m_Centroid != nullptr && m_hasCentroid == 1) {
                    // 减质心
                    for (DimensionType i = 0; i < m_Dim; ++i) {
                        fvec[i] -= (*m_Centroid)[i];
                    }
                }

                std::vector<float> rotated_vec(m_PaddedDim, 0.0f);
                if (m_Rotator) {
                    m_Rotator->rotate(fvec.data(), rotated_vec.data());
                } else {
                    std::memcpy(rotated_vec.data(), fvec.data(), m_Dim * sizeof(float));
                }

                auto* meta = reinterpret_cast<RaBitQEstimateMeta*>(vecout);
                uint8_t* bin_ptr = reinterpret_cast<uint8_t*>(vecout + m_MetaSize);

                // delta vl
                float delta = 0.0f;
                float vl = 0.0f;

                // 2) 额外计算论文估算需要的 f_add/f_rescale/f_error
                float f_add = 0.0f;
                float f_rescale = 0.0f;
                float f_error = 0.0f;

                if (m_BitsPerCode == 1) {
                    // 1) 先算 delta/vl（total_bits=1）
                    std::vector<uint8_t> temp_scalar_code(m_PaddedDim, 0);
                    rabitqlib::quant::quantize_scalar(
                        rotated_vec.data(),
                        m_PaddedDim,
                        static_cast<size_t>(1),
                        temp_scalar_code.data(),
                        delta,
                        vl
                    );

                    // 对于 1-bit：直接调用 rabitq 的 compact 接口（它会写入压缩的二进制码并返回 f_*）
                    rabitqlib::quant::quantize_compact_one_bit(
                        rotated_vec.data(),
                        m_PaddedDim,
                        bin_ptr,      // 直接写入压缩位码（padded_dim/8 bytes）
                        f_add,
                        f_rescale,
                        f_error,
                        rabitqlib::METRIC_L2
                    );

                    // 将 meta 写回（delta/vl 已计算）
                    meta->delta = delta;
                    meta->vl = vl;
                    meta->f_add = f_add;
                    meta->f_rescale = f_rescale;
                    meta->f_error = f_error;
                    return;
                }

                // 多 bit 路径（保持原逻辑）：先得到 full byte 格式，再 packing 到 bin_ptr
                std::vector<uint8_t> temp_scalar_code(m_PaddedDim, 0);
                std::vector<uint8_t> temp_full_code(m_PaddedDim, 0);

                rabitqlib::quant::quantize_scalar(
                    rotated_vec.data(),
                    m_PaddedDim,
                    m_BitsPerCode,
                    temp_scalar_code.data(),
                    delta,
                    vl
                );

                rabitqlib::quant::quantize_full_single<float, uint8_t>(
                    rotated_vec.data(),
                    m_PaddedDim,
                    m_BitsPerCode,
                    temp_full_code.data(),
                    f_add,
                    f_rescale,
                    f_error,
                    rabitqlib::METRIC_L2
                );

#ifndef NDEBUG
                // 两条路径理论上应产生一致码字；调试期做一致性检查
                if (std::memcmp(temp_full_code.data(), temp_scalar_code.data(), m_PaddedDim) != 0) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "RaBitQ scalar/full code mismatch detected.\n");
                }
#endif

                // 【新增】将 1-byte 宽度的临时量化码压缩成我们设定的比特位，压实到 bin_ptr 中
                rabitqlib::quant::rabitq_impl::ex_bits::packing_rabitqplus_code(
                    temp_full_code.data(), 
                    bin_ptr, 
                    m_PaddedDim, 
                    m_BitsPerCode
                );

                meta->delta = delta;
                meta->vl = vl;
                meta->f_add = f_add;
                meta->f_rescale = f_rescale;
                meta->f_error = f_error;
            }

            // 将量化向量反量化重建为原始 float 向量
            virtual void ReconstructVector(const std::uint8_t* qvec, void* vecout) const
            {
                std::vector<float> rotated_reconst(m_PaddedDim);

                const float* meta_ptr = reinterpret_cast<const float*>(qvec);
                const uint8_t* bin_ptr = reinterpret_cast<const uint8_t*>(qvec + m_MetaSize);

                float delta = meta_ptr[0];
                float vl = meta_ptr[1];

                // 解压：先把紧凑存放的位数提取回一维一字节的格式进行后续常规重建
                std::vector<uint8_t> raw_code(m_PaddedDim, 0);
                UnpackVector(bin_ptr, raw_code.data());

                rabitqlib::quant::reconstruct_vec(
                    raw_code.data(),
                    delta,
                    vl,
                    m_PaddedDim,
                    rotated_reconst.data()
                );

                std::vector<float> out_float(m_Dim, 0.0f);
                if (m_Rotator) {
                    m_Rotator->rotate(rotated_reconst.data(), out_float.data());
                } else {
                    std::memcpy(out_float.data(), rotated_reconst.data(), m_Dim * sizeof(float));
                }

                // 如果启用了中心化，重建回原空间时需要把质心加回来
                if (m_hasCentroid == 1 &&
                    m_Centroid != nullptr &&
                    m_Centroid->size() == static_cast<size_t>(m_Dim)) {
                    for (DimensionType i = 0; i < m_Dim; ++i) {
                        out_float[i] += (*m_Centroid)[i];
                    }
                }

                // 重建结果的类型落地转换，uint8 版本需要做范围裁剪和四舍五入
                if constexpr (std::is_same<T, float>::value) {
                    std::memcpy(vecout, out_float.data(), m_Dim * sizeof(float));
                } else {
                    T* out_T = reinterpret_cast<T*>(vecout);
                    for (DimensionType i = 0; i < m_Dim; ++i) {
                        float v = out_float[i];
                        // 取值范围裁剪：适配 uint8_t/int8_t/int16_t
                        float max_val = static_cast<float>(std::numeric_limits<T>::max());
                        float min_val = static_cast<float>(std::numeric_limits<T>::lowest());
                        if (v < min_val) v = min_val;
                        if (v > max_val) v = max_val;
                        out_T[i] = static_cast<T>(v + (v >= 0 ? 0.5f : -0.5f));
                    }
                }
            }

            // 返回重建向量所需字节数
            virtual SizeType ReconstructSize() const { return m_Dim * sizeof(T); }

            // 返回重建向量维度
            virtual DimensionType ReconstructDim() const { return m_Dim; }

            virtual DimensionType GetPaddedDim() const { return m_PaddedDim; }

            // 返回序列化量化器头信息大小
            virtual std::uint64_t BufferSize() const
            {
                // qtype + rtype + version + dim + padded + bits + rotType + magic + blobSize + blob
                std::uint64_t base = sizeof(QuantizerType) + sizeof(VectorValueType) + sizeof(std::uint32_t) +
                       sizeof(DimensionType) + sizeof(DimensionType) + sizeof(SizeType) +
                       sizeof(std::uint8_t) + sizeof(std::uint32_t) + sizeof(std::uint64_t) +
                       RotatorBlobBytes();
                base += sizeof(std::uint8_t); // m_hasCentroid
                if (m_Centroid != nullptr && m_hasCentroid == 1) {
                    base += static_cast<std::uint64_t>(m_Dim) * sizeof(float);
                }
                return base;
            }

            // 保存量化器配置到磁盘流
            virtual ErrorCode SaveQuantizer(std::shared_ptr<Helper::DiskIO> p_out) const 
            {
                // 【核心修复】：必须先写入类型头票据，才能被通用 LoadIQuantizer 解码映射！
                QuantizerType qtype = QuantizerType::RaBitQQuantizer;
                VectorValueType rtype = GetEnumValueType<T>();

                IOBINARY(p_out, WriteBinary, sizeof(QuantizerType), (char*)&qtype);
                IOBINARY(p_out, WriteBinary, sizeof(VectorValueType), (char*)&rtype);

                std::uint32_t version = static_cast<std::uint32_t>(m_FormatVersion);
                IOBINARY(p_out, WriteBinary, sizeof(std::uint32_t), (char*)&version);

                IOBINARY(p_out, WriteBinary, sizeof(DimensionType), (char*)&m_Dim);
                IOBINARY(p_out, WriteBinary, sizeof(DimensionType), (char*)&m_PaddedDim);
                IOBINARY(p_out, WriteBinary, sizeof(SizeType), (char*)&m_BitsPerCode);

                std::uint8_t rotType = static_cast<std::uint8_t>(m_RotatorType);
                IOBINARY(p_out, WriteBinary, sizeof(std::uint8_t), (char*)&rotType);

                std::uint32_t magic = 0x52425451;
                IOBINARY(p_out, WriteBinary, sizeof(std::uint32_t), (char*)&magic);

                std::vector<std::uint8_t> rotBlob;
                if (ErrorCode::Success != SerializeRotator(rotBlob)) return ErrorCode::DiskIOFail;

                std::uint64_t blobBytes = static_cast<std::uint64_t>(rotBlob.size());
                IOBINARY(p_out, WriteBinary, sizeof(std::uint64_t), (char*)&blobBytes);
                if (blobBytes > 0) {
                    IOBINARY(p_out, WriteBinary, blobBytes, (char*)rotBlob.data());
                }
                IOBINARY(p_out, WriteBinary, sizeof(std::uint8_t), (char*)&m_hasCentroid);
                if (m_Centroid != nullptr && m_hasCentroid == 1) {
                    IOBINARY(p_out, WriteBinary, sizeof(float) * m_Dim, (char*)m_Centroid->data());
                }
                return ErrorCode::Success;
            }

            // 从磁盘流加载量化器配置
            virtual ErrorCode LoadQuantizer(std::shared_ptr<Helper::DiskIO> p_in)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Loading RaBitQQuantizer.\n");
                std::uint32_t version = 0;
                IOBINARY(p_in, ReadBinary, sizeof(std::uint32_t), (char*)&version);

                if (version == static_cast<std::uint32_t>(PersistVersion::V1)) {
                    // 兼容旧格式: [dim][magic]
                    IOBINARY(p_in, ReadBinary, sizeof(DimensionType), (char*)&m_Dim);
                    std::uint32_t magicV1 = 0;
                    IOBINARY(p_in, ReadBinary, sizeof(std::uint32_t), (char*)&magicV1);
                    if (magicV1 != 0x52425451) return ErrorCode::Fail;

                    m_BitsPerCode = 4;
                    m_RotatorType = RotatorStorageType::FhtKac;
                    RebuildRotator(m_Dim, 0, m_RotatorType);
                    RecalcSizes();
                    return ErrorCode::Success;
                }

                if (version != static_cast<std::uint32_t>(PersistVersion::V2)) return ErrorCode::Fail;

                IOBINARY(p_in, ReadBinary, sizeof(DimensionType), (char*)&m_Dim);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "After read dim: %s.\n", std::to_string(m_Dim).c_str());
                IOBINARY(p_in, ReadBinary, sizeof(DimensionType), (char*)&m_PaddedDim);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "After read PaddedDim: %s.\n", std::to_string(m_PaddedDim).c_str());
                IOBINARY(p_in, ReadBinary, sizeof(SizeType), (char*)&m_BitsPerCode);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "After read BitsPerCode: %s.\n", std::to_string(m_BitsPerCode).c_str());

                std::uint8_t rotType = 0;
                IOBINARY(p_in, ReadBinary, sizeof(std::uint8_t), (char*)&rotType);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "After read rotType: %s.\n", std::to_string(rotType).c_str());
                m_RotatorType = static_cast<RotatorStorageType>(rotType);

                std::uint32_t magic = 0;
                IOBINARY(p_in, ReadBinary, sizeof(std::uint32_t), (char*)&magic);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "After read magic: %s.\n", std::to_string(magic).c_str());
                if (magic != 0x52425451) return ErrorCode::Fail;

                std::uint64_t blobBytes = 0;    // 序列化的旋转器（Rotator）对象的二进制字节数，用于保存和恢复 RaBitQ 量化所依赖的正交随机旋转矩阵。
                IOBINARY(p_in, ReadBinary, sizeof(std::uint64_t), (char*)&blobBytes);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "After read blobBytes: %s.\n", std::to_string(blobBytes).c_str());

                std::vector<std::uint8_t> rotBlob(blobBytes);
                if (blobBytes > 0) {
                    IOBINARY(p_in, ReadBinary, blobBytes, (char*)rotBlob.data());
                }

                if (ErrorCode::Success != RebuildRotator(m_Dim, m_PaddedDim, m_RotatorType)) return ErrorCode::Fail;
                if (ErrorCode::Success != DeserializeRotator(rotBlob)) return ErrorCode::Fail;

                IOBINARY(p_in, ReadBinary, sizeof(std::uint8_t), (char*)&m_hasCentroid);
                if (m_hasCentroid == 1) {
                    m_Centroid.reset(new std::vector<float>(m_Dim));
                    IOBINARY(p_in, ReadBinary, sizeof(float) * m_Dim, (char*)m_Centroid->data());
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "After read using centroid.\n");
                } else {
                    m_Centroid.reset();
                }

                RecalcSizes();
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Loading RaBitQQuantizer finished.\n");
                return ErrorCode::Success;
            }

            // 从内存缓冲区加载量化器配置
            virtual ErrorCode LoadQuantizer(std::uint8_t* ptr)
            {
                // 内存映射版同上: 假定 V2
                std::uint32_t version = *(reinterpret_cast<std::uint32_t*>(ptr));
                ptr += sizeof(std::uint32_t);

                if (version != static_cast<std::uint32_t>(PersistVersion::V2)) return ErrorCode::Fail;

                m_Dim = *(reinterpret_cast<DimensionType*>(ptr));
                ptr += sizeof(DimensionType);

                m_PaddedDim = *(reinterpret_cast<DimensionType*>(ptr));
                ptr += sizeof(DimensionType);

                m_BitsPerCode = *(reinterpret_cast<SizeType*>(ptr));
                ptr += sizeof(SizeType);

                m_RotatorType = static_cast<RotatorStorageType>(*ptr);
                ptr += sizeof(std::uint8_t);

                std::uint32_t magic = *(reinterpret_cast<std::uint32_t*>(ptr));
                ptr += sizeof(std::uint32_t);
                if (magic != 0x52425451) return ErrorCode::Fail;

                std::uint64_t blobBytes = *(reinterpret_cast<std::uint64_t*>(ptr));
                ptr += sizeof(std::uint64_t);

                std::vector<std::uint8_t> rotBlob(blobBytes);
                if (blobBytes > 0) {
                    std::memcpy(rotBlob.data(), ptr, blobBytes);
                }
                ptr += blobBytes;

                if (ErrorCode::Success != RebuildRotator(m_Dim, m_PaddedDim, m_RotatorType)) return ErrorCode::Fail;
                if (ErrorCode::Success != DeserializeRotator(rotBlob)) return ErrorCode::Fail;

                m_hasCentroid = *ptr; 
                ptr += sizeof(std::uint8_t);
                if (m_hasCentroid == 1) {
                    m_Centroid.reset(new std::vector<float>(m_Dim));
                    std::memcpy(m_Centroid->data(), ptr, sizeof(float) * m_Dim);
                    ptr += sizeof(float) * m_Dim;
                } else {
                    m_Centroid.reset();
                }
                
                RecalcSizes();
                return ErrorCode::Success;
            }

            // 复用 L2 距离作为余弦接口占位实现
            virtual float CosineDistance(const std::uint8_t* pX, const std::uint8_t* pY) const
            {
                return L2Distance(pX, pY);
            }

            // 对查询向量做旋转与 padding 预处理
            void PreprocessQuery(const void* in_query, float* out_rotated) const
            {
                std::vector<float> temp(m_Dim, 0.0f);
                ConvertInputToFloat(reinterpret_cast<const T*>(in_query), temp.data());

                if (m_Centroid != nullptr && m_Centroid->size() == m_Dim) {
                    // 1) 查询向量也必须做同样的中心化
                    for (DimensionType i = 0; i < m_Dim; ++i) {
                        temp[i] -= (*m_Centroid)[i];
                    }
                }

                if (m_Rotator) {
                    m_Rotator->rotate(temp.data(), out_rotated);
                } else {
                    std::memcpy(out_rotated, temp.data(), m_Dim * sizeof(float));
                    if (m_PaddedDim > m_Dim) {
                        std::memset(out_rotated + m_Dim, 0, (m_PaddedDim - m_Dim) * sizeof(float));
                    }
                }
            }

#ifndef __CUDACC__
            // 计算两个量化向量之间的对称 L2 距离（SD）
            inline float L2DistanceByDequantization(
                const std::uint8_t* pX,
                const std::uint8_t* pY
            ) const
            {
                const float* metaX = reinterpret_cast<const float*>(pX);
                const float* metaY = reinterpret_cast<const float*>(pY);

                __m512 vDeltaX = _mm512_set1_ps(metaX[0]);
                __m512 vVLX    = _mm512_set1_ps(metaX[1]);
                __m512 vDeltaY = _mm512_set1_ps(metaY[0]);
                __m512 vVLY    = _mm512_set1_ps(metaY[1]);

                const uint8_t* binX = reinterpret_cast<const uint8_t*>(pX + m_MetaSize);
                const uint8_t* binY = reinterpret_cast<const uint8_t*>(pY + m_MetaSize);

                // 【新增解压操作】：
                std::vector<uint8_t> rawX(m_PaddedDim);
                std::vector<uint8_t> rawY(m_PaddedDim);
                UnpackVector(binX, rawX.data());
                UnpackVector(binY, rawY.data());
                const uint8_t* u_binX = rawX.data();
                const uint8_t* u_binY = rawY.data();

                __m512 vDist = _mm512_setzero_ps();
                DimensionType dim = m_PaddedDim;
                DimensionType i = 0;

                // AVX-512 Loop
                for (; i + 15 < dim; i += 16) {
                    __m512i intX = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(u_binX + i)));
                    __m512i intY = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(u_binY + i)));

                    __m512 fX = _mm512_cvtepi32_ps(intX);
                    __m512 fY = _mm512_cvtepi32_ps(intY);

                    // Correct Reconstruction: val = code * delta + vl
                    fX = _mm512_fmadd_ps(fX, vDeltaX, vVLX);
                    fY = _mm512_fmadd_ps(fY, vDeltaY, vVLY);

                    __m512 diff = _mm512_sub_ps(fX, fY);
                    vDist = _mm512_fmadd_ps(diff, diff, vDist);
                }

                float dist = _mm512_reduce_add_ps(vDist);

                // Scalar Tail
                for (; i < dim; ++i) {
                    float valX = static_cast<float>(u_binX[i]) * metaX[0] + metaX[1];
                    float valY = static_cast<float>(u_binY[i]) * metaY[0] + metaY[1];
                    float diff = valX - valY;
                    dist += diff * diff;
                }

                return dist;
            }

            // ADC 不再旋转Query向量，默认传入的 rotated_query 已经是旋转后的结果（调用PreprocessQuery）
            inline float L2DistanceByDequantization(
                const float* rotated_query,
                const std::uint8_t* pY
            ) const
            {
                const float* metaY = reinterpret_cast<const float*>(pY);
                __m512 vDeltaY = _mm512_set1_ps(metaY[0]);
                __m512 vVLY    = _mm512_set1_ps(metaY[1]);
                const uint8_t* binY = reinterpret_cast<const uint8_t*>(pY + m_MetaSize);

                __m512 vDist = _mm512_setzero_ps();
                DimensionType dim = m_PaddedDim;
                DimensionType i = 0;

                for (; i + 15 < dim; i += 16) {
                    __m512 vX = _mm512_loadu_ps(rotated_query + i);
                    __m512i intY = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(binY + i)));
                    __m512 fY = _mm512_cvtepi32_ps(intY);
                    fY = _mm512_fmadd_ps(fY, vDeltaY, vVLY);
                    __m512 diff = _mm512_sub_ps(vX, fY);
                    vDist = _mm512_fmadd_ps(diff, diff, vDist);
                }

                float dist = _mm512_reduce_add_ps(vDist);

                for (; i < dim; ++i) {
                    float valY = static_cast<float>(binY[i]) * metaY[0] + metaY[1];
                    float diff = rotated_query[i] - valY;
                    dist += diff * diff;
                }

                return dist;
            }
#else
            // 给 NVCC 提供一个空的 dummy，防止链接失败或编译报错
            inline float L2DistanceByDequantization(const std::uint8_t* pX, const std::uint8_t* pY) const { return 0.0f; }
            inline float L2DistanceByDequantization(const float* rotated_query, const std::uint8_t* pY) const { return 0.0f; }
#endif

            // IQuantizer 强制的对称接口：暂时保持旧语义（论文估算主要用于 query->db）
            virtual float L2Distance(const std::uint8_t* pX, const std::uint8_t* pY) const
            {
                return L2DistanceByDequantization(pX, pY);
            }

            // 查询到库向量：切换为论文式估算
            virtual float L2Distance(const float* rotated_query, const std::uint8_t* pY, float* out_low_dist = nullptr) const
            {
                return L2DistanceEstimate(rotated_query, pY, out_low_dist);
            }

            // 重载，额外接收查询元信息以避免重复计算
            virtual float L2Distance(const float* rotated_query, const std::uint8_t* pY, const BondMeta bond_meta, float* out_low_dist = nullptr) const
            {
                return L2DistanceEstimate(rotated_query, pY, bond_meta, out_low_dist);
            }

            // 兼容入口：先预处理查询，再走估算
            virtual float L2Distance(const void* pX, const std::uint8_t* pY) const
            {
                std::vector<float> temp_rot(m_PaddedDim);
                PreprocessQuery(reinterpret_cast<const T*>(pX), temp_rot.data());
                return L2Distance(temp_rot.data(), pY);
            }

        private:
            // 将任意输入向量转换为 float 向量以统一后续计算
            inline void ConvertInputToFloat(const T* vec, float* out_float) const
            {
                if constexpr (std::is_same<T, float>::value) {
                    std::memcpy(out_float, vec, m_Dim * sizeof(float));
                } else {
                    for (DimensionType i = 0; i < m_Dim; ++i) {
                        out_float[i] = static_cast<float>(vec[i]);
                    }
                }
            }

            // 还原解压函数（内部调试）
            inline void UnpackVector(const uint8_t* in_compact, uint8_t* out_raw) const {
                size_t dim = m_PaddedDim;
                size_t bits = m_BitsPerCode;

                if (bits == 8) {
                    std::memcpy(out_raw, in_compact, dim);
                } else if (bits == 4) {
                    for (size_t j = 0; j < dim; j += 16) {
                        uint64_t compact = *reinterpret_cast<const uint64_t*>(in_compact);
                        *reinterpret_cast<uint64_t*>(out_raw) = compact & 0x0F0F0F0F0F0F0F0F;
                        *reinterpret_cast<uint64_t*>(out_raw + 8) = (compact >> 4) & 0x0F0F0F0F0F0F0F0F;
                        in_compact += 8;
                        out_raw += 16;
                    }
                } else if (bits == 2) {
                    for (size_t j = 0; j < dim; j += 16) {
                        uint32_t compact = *reinterpret_cast<const uint32_t*>(in_compact);
                        *reinterpret_cast<uint32_t*>(out_raw) = compact & 0x03030303;
                        *reinterpret_cast<uint32_t*>(out_raw + 4) = (compact >> 2) & 0x03030303;
                        *reinterpret_cast<uint32_t*>(out_raw + 8) = (compact >> 4) & 0x03030303;
                        *reinterpret_cast<uint32_t*>(out_raw + 12) = (compact >> 6) & 0x03030303;
                        in_compact += 4;
                        out_raw += 16;
                    }
                } else if (bits == 1) {
                    for (size_t j = 0; j < dim; j += 16) {
                        uint16_t compact = *reinterpret_cast<const uint16_t*>(in_compact);
                        for (size_t i = 0; i < 16; ++i) {
                            out_raw[i] = (compact >> i) & 1;
                        }
                        in_compact += 2;
                        out_raw += 16;
                    }
                } else {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Unsupported BitsPerCode: %d\n", bits);
                }
            }

            // 根据当前维度和旋转器状态更新内部尺寸参数
            inline void RecalcSizes()
            {
                if (m_Rotator == nullptr && m_Dim > 0) {
                    if (m_RotatorType == RotatorStorageType::Matrix) {
                        auto* temp = rabitqlib::choose_rotator<float>(
                            m_Dim, rabitqlib::RotatorType::MatrixRotator
                        );
                        m_PaddedDim = temp->size();
                        delete temp;
                    } else {
                        auto* temp = rabitqlib::choose_rotator<float>(
                            m_Dim, rabitqlib::RotatorType::FhtKacRotator
                        );
                        m_PaddedDim = temp->size();
                        delete temp;
                    }
                } else if (m_Rotator) {
                    m_PaddedDim = m_Rotator->size();
                } else {
                    m_PaddedDim = (m_Dim + 127) / 128 * 128;
                }

                m_CodeSize = (m_PaddedDim * m_BitsPerCode + 7) / 8;
                m_MetaSize = sizeof(RaBitQEstimateMeta);
                m_QuantizedSize = m_MetaSize + m_CodeSize;

                // 根据底层位宽分配对应的专属解包 SIMD 内核
                m_PackedIpFunc = rabitqlib::select_excode_ipfunc(m_BitsPerCode);
            }

            // 提前算一下：如果要保存当前的旋转器，到底需要多少个字节的空间
            std::uint64_t RotatorBlobBytes() const
            {
                DimensionType pd = m_PaddedDim;
                if (pd == 0 && m_Dim > 0) {
                    auto* temp = rabitqlib::choose_rotator<float>(m_Dim);
                    pd = static_cast<DimensionType>(temp->size());
                    delete temp;
                }

                if (m_RotatorType == RotatorStorageType::Matrix) {
                    return static_cast<std::uint64_t>(sizeof(float)) * m_Dim * pd;
                }

                // FhtKacRotator::flip_ size = 4 * padded_dim / 8
                return static_cast<std::uint64_t>(4) * pd / 8;
            }

            // 在系统的 tmp 目录下随机生成一个临时文件的路径，保证它不和别人的文件撞名
            static std::string MakeTempFilePath()
            {
                char path[] = "/tmp/sptag_rabitq_rot_XXXXXX";
                int fd = mkstemp(path);
                if (fd >= 0) close(fd);
                return std::string(path);
            }

            // 把旋转器塞进字节数组里
            ErrorCode SerializeRotator(std::vector<std::uint8_t>& out) const
            {
                if (m_Rotator == nullptr) {
                    out.clear();
                    return ErrorCode::Success;
                }

                std::string path = MakeTempFilePath();
                {
                    std::ofstream ofs(path, std::ios::binary | std::ios::out);
                    if (!ofs.is_open()) return ErrorCode::DiskIOFail;
                    m_Rotator->save(ofs);
                }

                std::uint64_t bytes = RotatorBlobBytes();
                out.resize(bytes);

                {
                    std::ifstream ifs(path, std::ios::binary | std::ios::in);
                    if (!ifs.is_open()) {
                        std::remove(path.c_str());
                        return ErrorCode::DiskIOFail;
                    }
                    ifs.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
                }

                std::remove(path.c_str());
                return ErrorCode::Success;
            }

            // 用字节数组恢复出旋转器对象 (打包 / 序列化)
            ErrorCode DeserializeRotator(const std::vector<std::uint8_t>& blob)
            {
                if (m_Rotator == nullptr) return ErrorCode::Fail;
                if (blob.empty()) return ErrorCode::Success;

                std::string path = MakeTempFilePath();
                {
                    std::ofstream ofs(path, std::ios::binary | std::ios::out);
                    if (!ofs.is_open()) return ErrorCode::DiskIOFail;
                    ofs.write(reinterpret_cast<const char*>(blob.data()), static_cast<std::streamsize>(blob.size()));
                }

                {
                    std::ifstream ifs(path, std::ios::binary | std::ios::in);
                    if (!ifs.is_open()) {
                        std::remove(path.c_str());
                        return ErrorCode::DiskIOFail;
                    }
                    m_Rotator->load(ifs);
                }

                std::remove(path.c_str());
                return ErrorCode::Success;
            }

            // 清理掉旧的旋转器，根据给定的维度、类型，在内存里重新“New”一个干干净净的旋转器壳子，准备等着后面被赋值或者训练
            ErrorCode RebuildRotator(
                DimensionType dim,
                DimensionType padded,
                RotatorStorageType type
            ) {
                if (m_Rotator) {
                    delete m_Rotator;
                    m_Rotator = nullptr;
                }

                rabitqlib::RotatorType rt =
                    (type == RotatorStorageType::Matrix)
                        ? rabitqlib::RotatorType::MatrixRotator
                        : rabitqlib::RotatorType::FhtKacRotator;

                m_Rotator = rabitqlib::choose_rotator<float>(dim, rt, static_cast<size_t>(padded));
                if (m_Rotator == nullptr) return ErrorCode::Fail;
                return ErrorCode::Success;
            }

            DimensionType m_Dim;    // 原始向量维度（未补齐前）
            DimensionType m_PaddedDim;  // 经过 rotator 或对齐规则后的维度（通常用于 SIMD/批处理）
            SizeType m_CodeSize;    // 单条向量的量化码区字节数（不含 meta）
            SizeType m_MetaSize;    // 单条向量元信息字节数（当前是 delta 和 vl 两个 float）
            SizeType m_QuantizedSize;   // 单条向量总的量化后字节数（meta + code）
            SizeType m_BitsPerCode = 4; // 每个维度的量化位数（当前固定为 4）
            rabitqlib::ex_ipfunc m_PackedIpFunc = nullptr; // 保存当前位数对应的 AVX512 最优解包内积函数
            mutable rabitqlib::Rotator<float>* m_Rotator;   // 旋转器指针（训练时选择，量化时使用）
            bool m_EnableADC;   // 是否启用 ADC 模式（影响距离计算接口行为）
            
            RotatorStorageType m_RotatorType;
            PersistVersion m_FormatVersion;

            std::uint8_t m_hasCentroid = 0; // 0 表示未启用质心，1 表示启用质心
            std::unique_ptr<std::vector<float>> m_Centroid = nullptr; // 为空表示未启用质心
        };
    }
}

#endif // _SPTAG_COMMON_RABITQQUANTIZER_H_