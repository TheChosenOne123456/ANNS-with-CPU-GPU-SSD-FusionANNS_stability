/************************************************************
 *  ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
 *  ★        RaBitQ EMBEDDED MODULE (Fixed)                 ★
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

// RaBitQ Includes
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/index/estimator.hpp"

#include <immintrin.h> 

namespace SPTAG
{
    namespace COMMON
    {
        // Wrapper for RaBitQ
        class RaBitQQuantizer : public IQuantizer
        {
        public:
            RaBitQQuantizer() : m_Dim(0), m_PaddedDim(0), m_CodeSize(0), m_MetaSize(2 * sizeof(float)), m_QuantizedSize(0), m_Rotator(nullptr), m_EnableADC(false)
            {
            }

            RaBitQQuantizer(DimensionType dim) : m_Dim(dim), m_Rotator(nullptr), m_EnableADC(false)
            {
                RecalcSizes();
            }

            virtual ~RaBitQQuantizer() 
            {
                if (m_Rotator) delete m_Rotator;
            }

            virtual QuantizerType GetQuantizerType() const { return QuantizerType::RaBitQQuantizer; }
            virtual VectorValueType GetReconstructType() const { return VectorValueType::Float; }
            
            // 必须实现
            virtual SizeType QuantizeSize() const { return m_QuantizedSize; }
            virtual int GetBase() const { return 1; }
            virtual DimensionType GetNumSubvectors() const { return 1; }
            
            virtual bool GetEnableADC() const { return m_EnableADC; } 
            virtual void SetEnableADC(bool enableADC) { m_EnableADC = enableADC; }
            
            virtual float* GetL2DistanceTables() { return nullptr; }
            template<typename T> T* GetCodebooks() { return nullptr; }

            void Train(const void* data, SizeType num) 
            {
                 if (!m_Rotator) {
                     m_Rotator = rabitqlib::choose_rotator<float>(m_Dim);
                     RecalcSizes();
                 }
            }

            // 【基于库函数实现的 4-bit 量化】
            virtual void QuantizeVector(const void* vec, std::uint8_t* vecout, bool ADC = true) const
            {
                const float* fvec = reinterpret_cast<const float*>(vec);
                std::vector<float> rotated_vec(m_PaddedDim, 0.0f);
                
                if (m_Rotator) {
                    m_Rotator->rotate(const_cast<float*>(fvec), rotated_vec.data());
                } else {
                    std::memcpy(rotated_vec.data(), fvec, m_Dim * sizeof(float));
                }

                float* meta_ptr = reinterpret_cast<float*>(vecout);
                uint8_t* bin_ptr = reinterpret_cast<uint8_t*>(vecout + m_MetaSize);
                
                float delta = 0;
                float vl = 0;
                
                // 使用库函数进行 4-bit 量化
                // 注意：bin_ptr 必须有 padded_dim 大小的空间
                // m_CodeSize 在 RecalcSizes 里已经调整为 padded_dim
                rabitqlib::quant::quantize_scalar(
                    rotated_vec.data(), 
                    m_PaddedDim, 
                    4, // Bits
                    bin_ptr, 
                    delta, 
                    vl
                );
                
                meta_ptr[0] = delta;
                meta_ptr[1] = vl;
            }

            virtual void ReconstructVector(const std::uint8_t* qvec, void* vecout) const 
            {
                float* out_vec_raw = reinterpret_cast<float*>(vecout);
                
                std::vector<float> rotated_reconst(m_PaddedDim);
                
                const float* meta_ptr = reinterpret_cast<const float*>(qvec);
                const uint8_t* bin_ptr = reinterpret_cast<const uint8_t*>(qvec + m_MetaSize);
                
                float delta = meta_ptr[0];
                float vl = meta_ptr[1];

                // 使用库函数重建
                // 注意：这里需要移除 const_cast，因为库函数入参可能没写 const
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
            
            virtual SizeType ReconstructSize() const { return m_Dim * sizeof(float); }
            virtual DimensionType ReconstructDim() const { return m_Dim; }
            virtual std::uint64_t BufferSize() const { return sizeof(DimensionType); }

            virtual ErrorCode SaveQuantizer(std::shared_ptr<Helper::DiskIO> p_out) const 
            {
                IOBINARY(p_out, WriteBinary, sizeof(DimensionType), (char*)&m_Dim);
                int magic = 0x52425451;
                IOBINARY(p_out, WriteBinary, sizeof(int), (char*)&magic);
                return ErrorCode::Success;
            }

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

            virtual float CosineDistance(const std::uint8_t* pX, const std::uint8_t* pY) const
            {
                return L2Distance(pX, pY);
            }

            // [极速融合版] SD Distance
            virtual float L2Distance(const std::uint8_t* pX, const std::uint8_t* pY) const 
            {
                // 1. 获取量化参数
                const float* metaX = reinterpret_cast<const float*>(pX);
                const float* metaY = reinterpret_cast<const float*>(pY);
                
                // 广播标量参数
                __m512 vScaleX = _mm512_set1_ps((2.0f * metaX[1]) / 15.0f);
                __m512 vBiasX  = _mm512_set1_ps(metaX[0] - metaX[1]);
                __m512 vScaleY = _mm512_set1_ps((2.0f * metaY[1]) / 15.0f);
                __m512 vBiasY  = _mm512_set1_ps(metaY[0] - metaY[1]);

                const uint8_t* binX = reinterpret_cast<const uint8_t*>(pX + m_MetaSize);
                const uint8_t* binY = reinterpret_cast<const uint8_t*>(pY + m_MetaSize);
                
                __m512 vDist = _mm512_setzero_ps();
                DimensionType dim = m_PaddedDim;
                DimensionType i = 0;

                // 2. AVX-512 主循环 (步进 16)
                for (; i + 15 < dim; i += 16) {
                    // 加载 16 个 uint8 (128-bit)
                    __m128i rawX = _mm_loadu_si128((const __m128i*)(binX + i));
                    __m128i rawY = _mm_loadu_si128((const __m128i*)(binY + i));

                    // 扩展 uint8 -> int32 -> float (512-bit)
                    // _mm512_cvtepu8_epi32: 零扩展 8-bit 到 32-bit
                    __m512i intX = _mm512_cvtepu8_epi32(rawX);
                    __m512i intY = _mm512_cvtepu8_epi32(rawY);

                    // 转换为 float
                    __m512 fX = _mm512_cvtepi32_ps(intX);
                    __m512 fY = _mm512_cvtepi32_ps(intY);

                    // 线性变换: val = code * scale + bias
                    fX = _mm512_fmadd_ps(fX, vScaleX, vBiasX);
                    fY = _mm512_fmadd_ps(fY, vScaleY, vBiasY);

                    // 计算欧氏距离: dist += (x-y)^2
                    __m512 diff = _mm512_sub_ps(fX, fY);
                    vDist = _mm512_fmadd_ps(diff, diff, vDist);
                }

                // 3. 水平求和
                float dist = _mm512_reduce_add_ps(vDist);

                // 4. 处理剩余部分
                float scaleX = (2.0f * metaX[1]) / 15.0f;
                float biasX  = metaX[0] - metaX[1];
                float scaleY = (2.0f * metaY[1]) / 15.0f;
                float biasY  = metaY[0] - metaY[1];

                for (; i < dim; ++i) {
                    float valX = (float)binX[i] * scaleX + biasX;
                    float valY = (float)binY[i] * scaleY + biasY;
                    float diff = valX - valY;
                    dist += diff * diff;
                }

                return dist;
            }

            // [ADC 融合版 - 回归正确逻辑但优化内存]
            virtual float L2Distance(const void* pX, const std::uint8_t* pY) const
            {
                // 1. 使用 thread_local 缓存 Query (X) 旋转后的结果
                // 目的：避免 std::vector X_rot(dim) 的 malloc 开销
                static thread_local std::vector<float> tls_X_rot;
                if (tls_X_rot.size() < m_PaddedDim) tls_X_rot.resize(m_PaddedDim);
                
                // 清零 (为了 Rotator 正确性，必须保留)
                std::memset(tls_X_rot.data(), 0, m_PaddedDim * sizeof(float));
                float* X_ptr = tls_X_rot.data();

                // 2. 旋转 Query X
                const float* rawX = reinterpret_cast<const float*>(pX);
                if (m_Rotator) {
                    m_Rotator->rotate(const_cast<float*>(rawX), X_ptr);
                } else {
                    std::memcpy(X_ptr, rawX, m_Dim * sizeof(float));
                }

                // 3. 【回归点】不手动解压 Y，而是使用 thread_local 缓存 + ReconstructRotated
                // 既然 ReconstructRotated 的逻辑是绝对正确的，我们就调用它
                // 只是把输出目标从堆上新建的 vector 改为 TLS 复用的 vector
                static thread_local std::vector<float> tls_Y_rot;
                if (tls_Y_rot.size() < m_PaddedDim) tls_Y_rot.resize(m_PaddedDim);
                
                float* Y_ptr = tls_Y_rot.data();
                ReconstructRotated(pY, Y_ptr); // <--- 调用已知正确的解压函数

                // 4. 计算距离 (Float vs Float)
                // 使用 SPTAG 高度优化的 SIMD 距离函数
                return SPTAG::COMMON::DistanceUtils::ComputeL2Distance(X_ptr, Y_ptr, m_PaddedDim);
            }

        private:
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

                // 【重要修改】
                // 如果使用 scalar quantizer，每个维度占用 1 Byte (即使是 4-bit)
                // 压缩比：32x -> 4x
                m_CodeSize = m_PaddedDim * 1; 
                
                m_MetaSize = 2 * sizeof(float);
                m_QuantizedSize = m_MetaSize + m_CodeSize;
            }

            DimensionType m_Dim;
            DimensionType m_PaddedDim;
            SizeType m_CodeSize;
            SizeType m_MetaSize;
            SizeType m_QuantizedSize;
            mutable rabitqlib::Rotator<float>* m_Rotator;
            bool m_EnableADC;
        };
    }
}

#endif // _SPTAG_COMMON_RABITQQUANTIZER_H_