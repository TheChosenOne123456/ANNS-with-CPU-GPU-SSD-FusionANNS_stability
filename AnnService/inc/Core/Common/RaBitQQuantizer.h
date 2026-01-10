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

            virtual float L2Distance(const std::uint8_t* pX, const std::uint8_t* pY) const 
            {
                // 对称距离：先重建，再算距离
                // 为了性能，实际生产中应该写针对 4-bit 的查表法 Simd
                // 这里为了正确性，先走完整重建流程
                std::vector<float> X_rot(m_PaddedDim);
                std::vector<float> Y_rot(m_PaddedDim);
                
                // 我们只需要重建到 rotated space 即可，不需要 inverse rotate
                ReconstructRotated(pX, X_rot.data());
                ReconstructRotated(pY, Y_rot.data());

                return SPTAG::COMMON::DistanceUtils::ComputeL2Distance(X_rot.data(), Y_rot.data(), m_PaddedDim);
            }

            virtual float L2Distance(const void* pX, const std::uint8_t* pY) const
            {
                std::vector<float> X_rot(m_PaddedDim);
                const float* rawX = reinterpret_cast<const float*>(pX);

                if (m_Rotator) {
                    m_Rotator->rotate(const_cast<float*>(rawX), X_rot.data());
                } else {
                    std::memcpy(X_rot.data(), rawX, m_Dim * sizeof(float));
                    if (m_PaddedDim > m_Dim) std::memset(X_rot.data() + m_Dim, 0, (m_PaddedDim - m_Dim) * sizeof(float));
                }

                std::vector<float> Y_rot(m_PaddedDim);
                ReconstructRotated(pY, Y_rot.data());

                return SPTAG::COMMON::DistanceUtils::ComputeL2Distance(X_rot.data(), Y_rot.data(), m_PaddedDim);
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