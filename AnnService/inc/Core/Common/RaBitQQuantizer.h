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

#include <immintrin.h> 

namespace SPTAG
{
    namespace COMMON
    {
        // 封装 RaBitQ 并支持 float/uint8 两种原始输入类型
        template<typename T = float>
        class RaBitQQuantizer : public IQuantizer
        {
            static_assert(std::is_same<T, float>::value || std::is_same<T, std::uint8_t>::value, "RaBitQQuantizer only supports float or uint8 input types.");
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
                  m_MetaSize(2 * sizeof(float)),
                  m_QuantizedSize(0),
                  m_Rotator(nullptr),
                  m_EnableADC(false),
                  m_BitsPerCode(4),
                  m_RotatorType(RotatorStorageType::FhtKac),
                  m_FormatVersion(PersistVersion::V2)
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
                // 当前这版距离核按“每维1字节”读取，建议先限制在 4/8
                if (bits != 4 && bits != 8) bits = 4;
                m_BitsPerCode = bits;
                RecalcSizes();
            }

            SizeType GetBitsPerCode() const { return m_BitsPerCode; }
            
            // 设置旋转器存储类型（影响 Save/Load 行为）
            void SetRotatorType(RotatorStorageType t) { m_RotatorType = t; }

            // 训练阶段仅用于选择旋转器和更新尺寸
            void Train(const void* data, SizeType num)
            {
                (void)data;
                (void)num;
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
                    RecalcSizes();
                }
            }

            // 将原始向量量化为 4bit 码值并写入 meta 信息
            virtual void QuantizeVector(const void* vec, std::uint8_t* vecout, bool ADC = true) const
            {
                (void)ADC;
                std::vector<float> fvec(m_Dim);
                ConvertInputToFloat(reinterpret_cast<const T*>(vec), fvec.data());

                std::vector<float> rotated_vec(m_PaddedDim, 0.0f);
                if (m_Rotator) {
                    m_Rotator->rotate(fvec.data(), rotated_vec.data());
                } else {
                    std::memcpy(rotated_vec.data(), fvec.data(), m_Dim * sizeof(float));
                }

                float* meta_ptr = reinterpret_cast<float*>(vecout);
                uint8_t* bin_ptr = reinterpret_cast<uint8_t*>(vecout + m_MetaSize);

                float delta = 0;
                float vl = 0;

                rabitqlib::quant::quantize_scalar(
                    rotated_vec.data(),
                    m_PaddedDim,
                    m_BitsPerCode,
                    bin_ptr,
                    delta,
                    vl
                );

                meta_ptr[0] = delta;
                meta_ptr[1] = vl;
            }

            // 将量化向量反量化重建为原始 float 向量
            virtual void ReconstructVector(const std::uint8_t* qvec, void* vecout) const
            {
                std::vector<float> rotated_reconst(m_PaddedDim);

                const float* meta_ptr = reinterpret_cast<const float*>(qvec);
                const uint8_t* bin_ptr = reinterpret_cast<const uint8_t*>(qvec + m_MetaSize);

                float delta = meta_ptr[0];
                float vl = meta_ptr[1];

                rabitqlib::quant::reconstruct_vec(
                    const_cast<uint8_t*>(bin_ptr),
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

                // 重建结果的类型落地转换，uint8 版本需要做范围裁剪和四舍五入
                if constexpr (std::is_same<T, float>::value) {
                    std::memcpy(vecout, out_float.data(), m_Dim * sizeof(float));
                } else {
                    std::uint8_t* out_u8 = reinterpret_cast<std::uint8_t*>(vecout);
                    for (DimensionType i = 0; i < m_Dim; ++i) {
                        float v = out_float[i];
                        if (v < 0.0f) v = 0.0f;
                        if (v > 255.0f) v = 255.0f;
                        out_u8[i] = static_cast<std::uint8_t>(v + 0.5f);
                    }
                }
            }

            // 返回重建向量所需字节数
            virtual SizeType ReconstructSize() const { return m_Dim * sizeof(T); }

            // 返回重建向量维度
            virtual DimensionType ReconstructDim() const { return m_Dim; }

            // 返回序列化量化器头信息大小
            virtual std::uint64_t BufferSize() const
            {
                // qtype + rtype + version + dim + padded + bits + rotType + magic + blobSize + blob
                return sizeof(QuantizerType) + sizeof(VectorValueType) + sizeof(std::uint32_t) +
                       sizeof(DimensionType) + sizeof(DimensionType) + sizeof(SizeType) +
                       sizeof(std::uint8_t) + sizeof(std::uint32_t) + sizeof(std::uint64_t) +
                       RotatorBlobBytes();
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
                return ErrorCode::Success;
            }

            // 从磁盘流加载量化器配置
            virtual ErrorCode LoadQuantizer(std::shared_ptr<Helper::DiskIO> p_in)
            {
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
                IOBINARY(p_in, ReadBinary, sizeof(DimensionType), (char*)&m_PaddedDim);
                IOBINARY(p_in, ReadBinary, sizeof(SizeType), (char*)&m_BitsPerCode);

                std::uint8_t rotType = 0;
                IOBINARY(p_in, ReadBinary, sizeof(std::uint8_t), (char*)&rotType);
                m_RotatorType = static_cast<RotatorStorageType>(rotType);

                std::uint32_t magic = 0;
                IOBINARY(p_in, ReadBinary, sizeof(std::uint32_t), (char*)&magic);
                if (magic != 0x52425451) return ErrorCode::Fail;

                std::uint64_t blobBytes = 0;
                IOBINARY(p_in, ReadBinary, sizeof(std::uint64_t), (char*)&blobBytes);

                std::vector<std::uint8_t> rotBlob(blobBytes);
                if (blobBytes > 0) {
                    IOBINARY(p_in, ReadBinary, blobBytes, (char*)rotBlob.data());
                }

                if (ErrorCode::Success != RebuildRotator(m_Dim, m_PaddedDim, m_RotatorType)) return ErrorCode::Fail;
                if (ErrorCode::Success != DeserializeRotator(rotBlob)) return ErrorCode::Fail;

                RecalcSizes();
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

                if (ErrorCode::Success != RebuildRotator(m_Dim, m_PaddedDim, m_RotatorType)) return ErrorCode::Fail;
                if (ErrorCode::Success != DeserializeRotator(rotBlob)) return ErrorCode::Fail;

                RecalcSizes();
                return ErrorCode::Success;
            }

            // 复用 L2 距离作为余弦接口占位实现
            virtual float CosineDistance(const std::uint8_t* pX, const std::uint8_t* pY) const
            {
                return L2Distance(pX, pY);
            }

            // 对查询向量做旋转与 padding 预处理
            void PreprocessQuery(const T* in_query, float* out_rotated) const
            {
                std::vector<float> temp(m_Dim, 0.0f);
                ConvertInputToFloat(in_query, temp.data());

                if (m_Rotator) {
                    m_Rotator->rotate(temp.data(), out_rotated);
                } else {
                    std::memcpy(out_rotated, temp.data(), m_Dim * sizeof(float));
                    if (m_PaddedDim > m_Dim) {
                        std::memset(out_rotated + m_Dim, 0, (m_PaddedDim - m_Dim) * sizeof(float));
                    }
                }
            }

            // 计算两个量化向量之间的对称 L2 距离（SD）
            virtual float L2Distance(const std::uint8_t* pX, const std::uint8_t* pY) const
            {
                const float* metaX = reinterpret_cast<const float*>(pX);
                const float* metaY = reinterpret_cast<const float*>(pY);

                __m512 vDeltaX = _mm512_set1_ps(metaX[0]);
                __m512 vVLX    = _mm512_set1_ps(metaX[1]);
                __m512 vDeltaY = _mm512_set1_ps(metaY[0]);
                __m512 vVLY    = _mm512_set1_ps(metaY[1]);

                const uint8_t* binX = reinterpret_cast<const uint8_t*>(pX + m_MetaSize);
                const uint8_t* binY = reinterpret_cast<const uint8_t*>(pY + m_MetaSize);

                __m512 vDist = _mm512_setzero_ps();
                DimensionType dim = m_PaddedDim;
                DimensionType i = 0;

                // AVX-512 Loop
                for (; i + 15 < dim; i += 16) {
                    __m512i intX = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(binX + i)));
                    __m512i intY = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(binY + i)));

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
                    float valX = static_cast<float>(binX[i]) * metaX[0] + metaX[1];
                    float valY = static_cast<float>(binY[i]) * metaY[0] + metaY[1];
                    float diff = valX - valY;
                    dist += diff * diff;
                }

                return dist;
            }

            // ADC 不再旋转Query向量，默认传入的 rotated_query 已经是旋转后的结果（调用PreprocessQuery）
            virtual float L2Distance(const float* rotated_query, const std::uint8_t* pY) const
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

            // 兼容入口并根据输入模式自动做查询预处理后计算 ADC 距离
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
                if constexpr (std::is_same<T, std::uint8_t>::value) {
                    for (DimensionType i = 0; i < m_Dim; ++i) {
                        out_float[i] = static_cast<float>(vec[i]);
                    }
                } else {
                    std::memcpy(out_float, vec, m_Dim * sizeof(float));
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

                m_CodeSize = m_PaddedDim * 1;
                m_MetaSize = 2 * sizeof(float);
                m_QuantizedSize = m_MetaSize + m_CodeSize;
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
            mutable rabitqlib::Rotator<float>* m_Rotator;   // 旋转器指针（训练时选择，量化时使用）
            bool m_EnableADC;   // 是否启用 ADC 模式（影响距离计算接口行为）
            
            RotatorStorageType m_RotatorType;
            PersistVersion m_FormatVersion;
        };
    }
}

#endif // _SPTAG_COMMON_RABITQQUANTIZER_H_