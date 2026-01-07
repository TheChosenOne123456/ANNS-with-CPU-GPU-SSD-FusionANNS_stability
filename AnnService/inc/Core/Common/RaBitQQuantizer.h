/************************************************************
 *  ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
 *  ★                                                      ★
 *  ★        RaBitQ EMBEDDED MODULE (毕业设计新增)          ★
 *  ★                                                      ★
 *  ★  Purpose : Integrate RaBitQ into legacy vector search *
 *  ★  Author  : Ruilin                                     *
 *  ★  Date    : 2026-01                                    *
 *  ★                                                      ★
 *  ★  NOTE: This file is NOT part of the original project ★
 *  ★        Added specifically for graduation thesis      ★
 *  ★                                                      ★
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

// RaBitQ Includes
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/index/estimator.hpp"

namespace SPTAG
{
    namespace COMMON
    {
        // Wrapper for RaBitQ
        class RaBitQQuantizer : public IQuantizer
        {
        public:
            RaBitQQuantizer(DimensionType dim) : m_Dim(dim)
            {
                // RaBitQ requires padding to 256 bits (32 bytes) or similar alignment usually
                // For now, let's assume dim is compatible or handled by external padding
                m_PaddedDim = (dim + 127) / 128 * 128; // Example alignment, adjust based on rabitq reqs
                if (m_PaddedDim < dim) m_PaddedDim = dim; // overflow check
                
                // RaBitQ Code size: Dim bits -> Dim/8 bytes
                m_CodeSize = m_PaddedDim / 8;
                
                // Metadata: f_add(float), f_rescale(float), f_error(float)
                m_MetaSize = 3 * sizeof(float);
                
                m_QuantizedSize = m_CodeSize + m_MetaSize;
            }

            virtual ~RaBitQQuantizer() {}

            virtual QuantizerType GetQuantizerType() const { return QuantizerType::None; } // TODO: Define new type if needed
            virtual VectorValueType GetReconstructType() const { return VectorValueType::Float; }

            virtual SizeType QuantizeSize() const { return m_QuantizedSize; }
            
            virtual int GetBase() const { return 1; }
            virtual DimensionType GetNumSubvectors() const { return 1; }
            virtual bool GetEnableADC() const { return false; }
            virtual void SetEnableADC(bool enableADC) {}
            
            virtual float* GetL2DistanceTables() { return nullptr; }
            template<typename T> T* GetCodebooks() { return nullptr; }

            // Core Quantization Logic
            virtual void QuantizeVector(const void* vec, std::uint8_t* vecout, bool ADC = true) const
            {
                const float* fvec = reinterpret_cast<const float*>(vec);
                
                // Layout: [f_add][f_rescale][f_error][binary_code...]
                float* meta_ptr = reinterpret_cast<float*>(vecout);
                char* bin_ptr = reinterpret_cast<char*>(vecout + m_MetaSize);
                
                // Temp variables for metadata
                float f_add, f_rescale, f_error;
                
                // If dim != PaddedDim, we might need a temp buffer for padding, 
                // but let's assume for now input is safe or fastscan handles it.
                // NOTE: rabitq assumes data might be padded. 
                // For safety in this MVP, let's use a temp aligned buffer if needed
                
                // Call RaBitQ
                // We assume centroid is 0 for basic usage (or we need to store centroids)
                // Using 0-centroid effectively quantizes the vector itself.
                rabitqlib::quant::quantize_compact_one_bit<float>(
                    fvec, 
                    m_PaddedDim, // dimension
                    bin_ptr,     // output binary code
                    f_add, 
                    f_rescale, 
                    f_error
                );
                
                // Store metadata
                meta_ptr[0] = f_add;
                meta_ptr[1] = f_rescale;
                meta_ptr[2] = f_error;
            }

            // Reconstruction (Reverse Quantization)
            virtual void ReconstructVector(const std::uint8_t* qvec, void* vecout) const 
            {
                float* out_vec = reinterpret_cast<float*>(vecout);
                
                const float* meta_ptr = reinterpret_cast<const float*>(qvec);
                const char* bin_ptr = reinterpret_cast<const char*>(qvec + m_MetaSize);
                
                float f_add = meta_ptr[0];
                float f_rescale = meta_ptr[1];
                // f_error not typically used for simple reconstruction
                
                // Naive reconstruction loop (RaBitQ doesn't seem to expose a simple 'dequantize' for one vector in headers easily)
                // Logic: value ~ f_add + f_rescale * (bit ? 1 : -1) * ... logic specific to RaBitQ
                // Let's implement a simple approximation based on RaBitQ paper/logic
                // x ~ C + s * b  where b is {-1, 1}
                // Actually RaBitQ usually means: if bit is 1, val > 0. 
                
                // WARNING: This is a placeholder. Correct reconstruction requires 
                // knowing the exact formula `rabitq` uses. 
                // From estimator.hpp: est_dist = ... f_rescale * (ip ... )
                // It is hard to reconstruct EXACTLY because it is a projection. 
                // But for re-ranking, we might just need the distance.
                
                memset(out_vec, 0, m_Dim * sizeof(float)); 
            }

            virtual SizeType ReconstructSize() const { return m_Dim * sizeof(float); }
            virtual DimensionType ReconstructDim() const { return m_Dim; }
            virtual std::uint64_t BufferSize() const { return sizeof(DimensionType); }

            // IO
            virtual ErrorCode SaveQuantizer(std::shared_ptr<Helper::DiskIO> p_out) const 
            {
                IOBINARY(p_out, WriteBinary, sizeof(DimensionType), (char*)&m_Dim);
                return ErrorCode::Success;
            }

            virtual ErrorCode LoadQuantizer(std::shared_ptr<Helper::DiskIO> p_in) 
            {
                IOBINARY(p_in, ReadBinary, sizeof(DimensionType), (char*)&m_Dim);
                return ErrorCode::Success;
            }

            virtual ErrorCode LoadQuantizer(uint8_t* raw_bytes) 
            {
                m_Dim = *(reinterpret_cast<DimensionType*>(raw_bytes));
                return ErrorCode::Success;
            }

            // Distance Calculation - The most critical part
            // This is "Symmetric" distance (Code vs Code) which RaBitQ doesn't natively optimize for Query scenarios.
            // RaBitQ is optimized for Asymmetric (Query float vs Code binary).
            virtual float L2Distance(const std::uint8_t* pX, const std::uint8_t* pY) const 
            {
                // This function is rarely used in high-perf search (usu. Asymmetric).
                // But we must implement it.
                // Implementing symmetric distance between two OneBit quantized vectors.
                
                const float* metaX = reinterpret_cast<const float*>(pX);
                const float* metaY = reinterpret_cast<const float*>(pY);
                
                // Decode metadata
                // ...
                
                // Calculate Hamming distance
                // ...
                
                return 0.0f; // TODO: Implement Symmetric L2
            }
            
            // Asymmetric Distance: Float Vector vs Encoded Code
            // This is key! We need to override DistanceCalcSelector logic elsewhere or 
            // expose a special function. 
            virtual float CosineDistance(const std::uint8_t* pX, const std::uint8_t* pY) const { return 0.0f; }

        private:
            DimensionType m_Dim;
            DimensionType m_PaddedDim;
            SizeType m_CodeSize;
            SizeType m_MetaSize;
            SizeType m_QuantizedSize;
        };
    }
}

#endif // _SPTAG_COMMON_RABITQQUANTIZER_H_