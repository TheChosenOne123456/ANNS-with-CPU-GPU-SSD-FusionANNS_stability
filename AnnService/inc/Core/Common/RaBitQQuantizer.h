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
        class RaBitQQuantizer : public IQuantizer
        {
        public:
            // 定义输入原始向量的数据模式
            enum class InputDataMode : std::uint8_t { Auto = 0, Float32 = 1, UInt8 = 2 };

            // 默认构造并初始化基础状态
            RaBitQQuantizer()
                : m_Dim(0),
                  m_PaddedDim(0),
                  m_CodeSize(0),
                  m_MetaSize(2 * sizeof(float)),
                  m_QuantizedSize(0),
                  m_Rotator(nullptr),
                  m_EnableADC(false),
                  m_InputMode(InputDataMode::Auto),
                  m_BitsPerCode(4)
            {
            }

            // 按维度构造并计算量化存储大小
            RaBitQQuantizer(DimensionType dim)
                : m_Dim(dim),
                  m_Rotator(nullptr),
                  m_EnableADC(false),
                  m_InputMode(InputDataMode::Auto),
                  m_BitsPerCode(4)
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

            // 返回重建向量类型为 float
            virtual VectorValueType GetReconstructType() const { return VectorValueType::Float; }

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
            template<typename T> T* GetCodebooks() { return nullptr; }

            // 设置原始输入数据模式以避免自动判断误判
            void SetInputDataMode(InputDataMode mode) { m_InputMode = mode; }

            // 获取当前原始输入数据模式
            InputDataMode GetInputDataMode() const { return m_InputMode; }

            void SetBitsPerCode(SizeType bits)
            {
                // 当前这版距离核按“每维1字节”读取，建议先限制在 4/8
                if (bits != 4 && bits != 8) bits = 4;
                m_BitsPerCode = bits;
                RecalcSizes();
            }

            SizeType GetBitsPerCode() const { return m_BitsPerCode; }

            // 训练阶段仅用于选择旋转器和更新尺寸
            void Train(const void* data, SizeType num)
            {
                (void)data;
                (void)num;
                if (!m_Rotator) {
                    m_Rotator = rabitqlib::choose_rotator<float>(m_Dim);
                    RecalcSizes();
                }
            }

            // 将原始向量量化为 4bit 码值并写入 meta 信息
            virtual void QuantizeVector(const void* vec, std::uint8_t* vecout, bool ADC = true) const
            {
                (void)ADC;
                std::vector<float> fvec(m_Dim);
                ConvertInputToFloat(vec, fvec.data());

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
                float* out_vec_raw = reinterpret_cast<float*>(vecout);
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

                if (m_Rotator) {
                    m_Rotator->rotate(rotated_reconst.data(), out_vec_raw);
                } else {
                    std::memcpy(out_vec_raw, rotated_reconst.data(), m_Dim * sizeof(float));
                }
            }

            // 返回重建向量所需字节数
            virtual SizeType ReconstructSize() const { return m_Dim * sizeof(float); }

            // 返回重建向量维度
            virtual DimensionType ReconstructDim() const { return m_Dim; }

            // 返回序列化量化器头信息大小
            virtual std::uint64_t BufferSize() const
            {
                return sizeof(QuantizerType) + sizeof(VectorValueType) + sizeof(DimensionType) + sizeof(int);
            }

            // 保存量化器配置到磁盘流
            virtual ErrorCode SaveQuantizer(std::shared_ptr<Helper::DiskIO> p_out) const 
            {
                // 【核心修复】：必须先写入类型头票据，才能被通用 LoadIQuantizer 解码映射！
                QuantizerType qtype = QuantizerType::RaBitQQuantizer;
                VectorValueType rtype = VectorValueType::Float;
                IOBINARY(p_out, WriteBinary, sizeof(QuantizerType), (char*)&qtype);
                IOBINARY(p_out, WriteBinary, sizeof(VectorValueType), (char*)&rtype);

                // 核心业务参数
                IOBINARY(p_out, WriteBinary, sizeof(DimensionType), (char*)&m_Dim);
                int magic = 0x52425451;
                IOBINARY(p_out, WriteBinary, sizeof(int), (char*)&magic);
                return ErrorCode::Success;
            }

            // 从磁盘流加载量化器配置
            virtual ErrorCode LoadQuantizer(std::shared_ptr<Helper::DiskIO> p_in) 
            {
                DimensionType dim;
                IOBINARY(p_in, ReadBinary, sizeof(DimensionType), (char*)&dim);
                m_Dim = dim;
                int magic;
                IOBINARY(p_in, ReadBinary, sizeof(int), (char*)&magic);

                if (m_Rotator) delete m_Rotator;
                m_Rotator = rabitqlib::choose_rotator<float>(m_Dim);
                RecalcSizes();
                return ErrorCode::Success;
            }

            // 从内存缓冲区加载量化器配置
            virtual ErrorCode LoadQuantizer(std::uint8_t* ptr)
            {
                m_Dim = *(reinterpret_cast<DimensionType*>(ptr));
                ptr += sizeof(DimensionType);
                ptr += sizeof(int);

                if (m_Rotator) delete m_Rotator;
                m_Rotator = rabitqlib::choose_rotator<float>(m_Dim); 
                RecalcSizes();
                return ErrorCode::Success;
            }

            // 复用 L2 距离作为余弦接口占位实现
            virtual float CosineDistance(const std::uint8_t* pX, const std::uint8_t* pY) const
            {
                return L2Distance(pX, pY);
            }

            // 对 float 查询向量做旋转与 padding 预处理
            void PreprocessQuery(const float* in_query, float* out_rotated) const
            {
                if (m_Rotator) {
                    m_Rotator->rotate(const_cast<float*>(in_query), out_rotated);
                } else {
                    std::memcpy(out_rotated, in_query, m_Dim * sizeof(float));
                    if (m_PaddedDim > m_Dim) std::memset(out_rotated + m_Dim, 0, (m_PaddedDim - m_Dim) * sizeof(float));
                }
            }

            // 对 uint8 查询向量做转换后旋转与 padding 预处理
            void PreprocessQuery(const std::uint8_t* in_query, float* out_rotated) const
            {
                std::vector<float> temp(m_Dim);
                for (DimensionType i = 0; i < m_Dim; ++i) temp[i] = static_cast<float>(in_query[i]);
                PreprocessQuery(temp.data(), out_rotated);
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

                if (ResolveInputMode(pX) == InputDataMode::UInt8) {
                    PreprocessQuery(reinterpret_cast<const std::uint8_t*>(pX), temp_rot.data());
                } else {
                    PreprocessQuery(reinterpret_cast<const float*>(pX), temp_rot.data());
                }

                return L2Distance(temp_rot.data(), pY);
            }

        private:
            // 将任意输入向量转换为 float 向量以统一后续计算
            inline void ConvertInputToFloat(const void* vec, float* out_float) const
            {
                InputDataMode mode = ResolveInputMode(vec);
                if (mode == InputDataMode::UInt8) {
                    const std::uint8_t* u8 = reinterpret_cast<const std::uint8_t*>(vec);
                    for (DimensionType i = 0; i < m_Dim; ++i) out_float[i] = static_cast<float>(u8[i]);
                } else {
                    const float* f = reinterpret_cast<const float*>(vec);
                    std::memcpy(out_float, f, m_Dim * sizeof(float));
                }
            }

            // 在 Auto 模式下通过简单启发式判断输入更像 float 还是 uint8
            inline InputDataMode ResolveInputMode(const void* vec) const
            {
                if (m_InputMode == InputDataMode::Float32 || m_InputMode == InputDataMode::UInt8) return m_InputMode;

                const float* f = reinterpret_cast<const float*>(vec);
                int checkN = static_cast<int>(std::min<DimensionType>(m_Dim, 8));
                int finiteCnt = 0;
                int inRangeCnt = 0;
                for (int i = 0; i < checkN; ++i) {
                    float v = f[i];
                    if (std::isfinite(v)) {
                        ++finiteCnt;
                        if (std::fabs(v) < 1e6f) ++inRangeCnt;
                    }
                }

                if (finiteCnt == checkN && inRangeCnt >= checkN - 1) return InputDataMode::Float32;
                return InputDataMode::UInt8;
            }

            // 直接把量化向量重建到旋转空间（工具函数）
            inline void ReconstructRotated(const std::uint8_t* qvec, float* out_rot) const
            {
                const float* meta_ptr = reinterpret_cast<const float*>(qvec);
                const uint8_t* bin_ptr = reinterpret_cast<const uint8_t*>(qvec + m_MetaSize);

                float delta = meta_ptr[0];
                float vl = meta_ptr[1];

                rabitqlib::quant::reconstruct_vec(
                    const_cast<uint8_t*>(bin_ptr),
                    delta,
                    vl,
                    m_PaddedDim,
                    out_rot
                );
            }

            // 根据当前维度和旋转器状态更新内部尺寸参数
            inline void RecalcSizes()
            {
                if (m_Rotator == nullptr && m_Dim > 0) {
                    auto* temp = rabitqlib::choose_rotator<float>(m_Dim);
                    m_PaddedDim = temp->size();
                    delete temp;
                } else if (m_Rotator) {
                    m_PaddedDim = m_Rotator->size();
                } else {
                    m_PaddedDim = (m_Dim + 127) / 128 * 128;
                }

                m_CodeSize = m_PaddedDim * 1;
                m_MetaSize = 2 * sizeof(float);
                m_QuantizedSize = m_MetaSize + m_CodeSize;
            }

            DimensionType m_Dim;    // 原始向量维度（未补齐前）
            DimensionType m_PaddedDim;  // 经过 rotator 或对齐规则后的维度（通常用于 SIMD/批处理）
            SizeType m_CodeSize;    // 单条向量的量化码区字节数（不含 meta）
            SizeType m_MetaSize;    // 单条向量元信息字节数（当前是 delta 和 vl 两个 float）
            SizeType m_QuantizedSize;   // 单条向量总的量化后字节数（meta + code）
            SizeType m_BitsPerCode = 4; // 每个维度的量化位数（当前固定为 4）
            mutable rabitqlib::Rotator<float>* m_Rotator;   // 旋转器指针（训练时选择，量化时使用）
            bool m_EnableADC;   // 是否启用 ADC 模式（影响距离计算接口行为）
            InputDataMode m_InputMode;  // 输入数据模式（Auto/Float32/UInt8），影响预处理和距离计算的行为
        };
    }
}

#endif // _SPTAG_COMMON_RABITQQUANTIZER_H_