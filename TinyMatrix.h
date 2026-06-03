/*
 * ===========================================================================
 * TinyMatrix - A lightweight 16-bit half-float & integer matrix engine
 * ===========================================================================
 * * HOW TO USE THIS LIBRARY:
 * * In EXACTLY ONE C++ file, define TINYMATRIX_IMPLEMENTATION before including
 * this header to create the implementation.
 * * Example:
 * #define TINYMATRIX_IMPLEMENTATION
 * #include "TinyMatrix.h"
 * * ===========================================================================
 */

#ifndef TINY_MATRIX_H
#define TINY_MATRIX_H

#include <assert.h>
#include <iostream>
#include <cmath>
#include <initializer_list>
#include <cstring> 
#include <immintrin.h>

typedef uint16_t HALF;

struct mHalf {
    union {
        HALF uVal;
        struct {
            HALF Frac : 10;
            HALF Exp : 5;
            HALF Sign : 1;
        }parts{0};
    };
    mHalf(const uint16_t& t) {
        this->uVal = t;
    }
};

struct nRet {
    float val;
    nRet(const float& _f) {
        this->val = _f;
    };
    nRet() {
        this->val = 0.0f;
    }
    operator float() {
        return val;
    };
    operator int() {
        return (int)val;
    };
    nRet& operator +=(const float& o) {
        this->val += o;
        return *this;
    };
};

struct mFloat {
    union {
        float fVal;
        struct {
            uint32_t Frac : 23;
            uint32_t Exp : 8;
            uint32_t Sign : 1;
        }parts{0};
    };
    mFloat(const float& t) {
        this->fVal = t;
    }
};

class TinyMatrix {
private:
    int rows;
    int cols;
    int size;       // Logical bytes (for backward compatibility)
    int stride;     // Physical columns in RAM
    int cap_rows;   // Physical rows in RAM
    unsigned char* data;
    bool isFloat = false;

    static thread_local TinyMatrix* _scratch;
    static thread_local TinyMatrix* _operandScratch;
    static thread_local size_t _scratchCapacity;
    static thread_local size_t _operandScratchCapacity;
    static thread_local float* _accumScratch;
    static thread_local size_t _accumCapacity;
    static thread_local int32_t* _accumScratchInt;
    static thread_local size_t _accumIntCapacity;
    static thread_local TinyMatrix* _transposeScratch;
    static thread_local size_t _transposeScratchCapacity;

public:

    void Reserve(int max_r, int max_c);
    void SetLogicalBounds(int new_rows, int new_cols);
    void ShrinkToFit();

    void WriteRaw(std::ostream& out) {
        ShrinkToFit(); // Guarantee tightly packed memory before writing to disk
        out.write((char*)&this->isFloat, sizeof(bool));
        out.write((char*)this->data, this->size);
    }

    void ReadRaw(std::istream& in) {
        ShrinkToFit(); // Ensure we are writing into a perfectly sized matrix
        in.read((char*)&this->isFloat, sizeof(bool));
        in.read((char*)this->data, this->size);
    }

    int Rows() {
        return this->rows;
    }
    int Cols() {
        return this->cols;
    }
    int Size() {
        return this->size;
    }
    bool IsFloat() {
        return this->isFloat;
    }

    TinyMatrix();
    TinyMatrix(int r, int c);
    TinyMatrix(int r, int c, std::initializer_list<double> nums);
    TinyMatrix(int r, int c, std::initializer_list<int> nums);

    void init(int r, int c);
    ~TinyMatrix();

    TinyMatrix(const TinyMatrix& source);
    TinyMatrix& operator=(const TinyMatrix& source);

    float operator()(int r, int c);
    TinyMatrix& Shape(int r, int c, bool absolute = false);

    void operator()(int r, int c, const int v);
    void operator()(int r, int c, const float v);
    void operator()(int r, int c, const double v);

    TinyMatrix& add(int s);
    TinyMatrix& sub(int s);
    TinyMatrix& add(float s);
    TinyMatrix& sub(float s);
    TinyMatrix& add(double s);
    TinyMatrix& sub(double s);
    TinyMatrix& add(const TinyMatrix& a, bool reshape = false);
    TinyMatrix& sub(const TinyMatrix& a, bool reshape = false);
    TinyMatrix& multiply(int s);
    TinyMatrix& multiply(float s);
    TinyMatrix& multiply(double s);
    TinyMatrix& Relu();
    TinyMatrix& Sigmoid();
    TinyMatrix& D_Relu();
    TinyMatrix& D_Sigmoid();
    TinyMatrix& hadamard(const TinyMatrix& a);
    TinyMatrix& QuantizeQ88();
    TinyMatrix& DequantizeQ88();
    TinyMatrix& fixed_dot(TinyMatrix& a, TinyMatrix& b);
    TinyMatrix& fixed_hadamard(const TinyMatrix& a);
    TinyMatrix& Tanh();
    TinyMatrix& D_Tanh();

    static int16_t tanh_lut[1024];
    static void InitLUT();
    static void CleanupEngine() {
        delete _scratch;
        _scratch = nullptr;
        delete _operandScratch;
        _operandScratch = nullptr;
        delete[] _accumScratch;
        _accumScratch = nullptr;
        delete[] _accumScratchInt;
        _accumScratchInt = nullptr;
        delete _transposeScratch;
        _transposeScratch = nullptr;
    }

    TinyMatrix& fixed_tanh();
    void print(std::string extra = "");

    TinyMatrix& transpose();
    TinyMatrix& dot(TinyMatrix& a, TinyMatrix& b);
    TinyMatrix& dot(TinyMatrix& a);
    TinyMatrix& Ints();
    TinyMatrix& Floats();
    TinyMatrix& Randomize(float min = -1.0f, float max = 1.0f);
    nRet sum();

    // Move Constructor
    TinyMatrix(TinyMatrix&& source) noexcept {
        this->rows = source.rows;
        this->cols = source.cols;
        this->stride = source.stride;
        this->cap_rows = source.cap_rows;
        this->size = source.size;
        this->data = source.data;
        this->isFloat = source.isFloat;

        // Nullify the source so its destructor doesn't free the memory
        source.data = nullptr;
    }

    // Move Assignment
    TinyMatrix& operator=(TinyMatrix&& source) noexcept {
        if(&source == this) return *this;

        delete[] this->data; // Free existing memory

        this->rows = source.rows;
        this->cols = source.cols;
        this->stride = source.stride;
        this->cap_rows = source.cap_rows;
        this->size = source.size;
        this->data = source.data;
        this->isFloat = source.isFloat;

        source.data = nullptr;
        return *this;
    }

    template <typename Func>
    TinyMatrix& mapInline(Func operation, bool forceFloat = false) {
        bool origFloat = this->isFloat;
        bool targetFloat = origFloat || forceFloat;
        this->isFloat = targetFloat;

        for(int i = 0; i < this->rows; i++) {
            // Jump to the physical start of the row
            int pos = (i * this->stride) * 2;

            for(int j = 0; j < this->cols; j++) {
                float val = origFloat ? halfToFloat(*(uint16_t*)(this->data + pos))
                    : (float)*(int16_t*)(this->data + pos);

                float result = operation(val, i, j);

                if(targetFloat) {
                    *(uint16_t*)(this->data + pos) = floatToHalf(result);
                } else {
                    *(int16_t*)(this->data + pos) = (int16_t)result;
                }
                pos += 2;
            }
        }
        return *this;
    }

    template <typename Func>
    TinyMatrix& mapInline(TinyMatrix& other, Func operation, bool forceFloat = false) {
        assert(((this->rows == other.rows && this->cols == other.cols) ||
            (other.rows == 1 && other.cols == this->cols) ||
            (other.cols == 1 && other.rows == this->rows) ||
            (other.rows == 1 && other.cols == 1)) &&
            "Matrix dimensions must match or be compatible for broadcasting!");

        bool origFloat = this->isFloat;
        bool otherFloat = other.IsFloat();
        bool targetFloat = origFloat || otherFloat || forceFloat;
        this->isFloat = targetFloat;

        bool same_shape = (this->rows == other.rows && this->cols == other.cols);

        for(int i = 0; i < this->rows; i++) {
            int other_i = (other.rows == 1) ? 0 : i;

            int pos = (i * this->stride) * 2;
            int other_pos = (other_i * other.stride) * 2;

            for(int j = 0; j < this->cols; j++) {
                if(!same_shape && other.cols == 1) {
                    other_pos = (other_i * other.stride + 0) * 2;
                }

                float val1 = origFloat ? halfToFloat(*(uint16_t*)(this->data + pos))
                    : (float)*(int16_t*)(this->data + pos);

                float val2 = otherFloat ? halfToFloat(*(uint16_t*)(other.data + other_pos))
                    : (float)*(int16_t*)(other.data + other_pos);

                float result = operation(val1, val2, i, j);

                if(targetFloat) {
                    *(uint16_t*)(this->data + pos) = floatToHalf(result);
                } else {
                    *(int16_t*)(this->data + pos) = (int16_t)result;
                }

                pos += 2;
                if(same_shape || other.cols != 1) {
                    other_pos += 2;
                }
            }
        }
        return *this;
    }
private:
    static HALF floatToHalf(mFloat i);
    static float halfToFloat(mHalf y);
    void _snapshotToRegister(TinyMatrix*& dest, const TinyMatrix* src, size_t& cap);
};
#endif

#ifdef TINYMATRIX_IMPLEMENTATION
thread_local TinyMatrix* TinyMatrix::_scratch = nullptr;
thread_local TinyMatrix* TinyMatrix::_operandScratch = nullptr;
thread_local size_t TinyMatrix::_scratchCapacity = 0;
thread_local size_t TinyMatrix::_operandScratchCapacity = 0;
thread_local float* TinyMatrix::_accumScratch = nullptr;
thread_local size_t TinyMatrix::_accumCapacity = 0;
thread_local int* TinyMatrix::_accumScratchInt = nullptr;
thread_local size_t TinyMatrix::_accumIntCapacity = 0;
thread_local TinyMatrix* TinyMatrix::_transposeScratch = nullptr;
thread_local size_t TinyMatrix::_transposeScratchCapacity = 0;

constexpr auto makeEven(int a) {
    return (a + 1 ^ (~a & 1));
}

int16_t TinyMatrix::tanh_lut[1024];

void TinyMatrix::InitLUT() {
    for(int i = 0; i < 1024; i++) {
        float float_val = (float)i / 256.0f;
        float t_val = std::tanh(float_val);
        tanh_lut[i] = (int16_t)(t_val * 256.0f);
    }
}

// =====================================================================
// CAPACITY & STRIDE MANAGEMENT
// =====================================================================

void TinyMatrix::Reserve(int max_r, int max_c) {
    if(max_r <= cap_rows && max_c <= stride) return;

    int new_cap_rows = std::max(cap_rows, max_r);
    int new_stride = std::max(stride, max_c);
    int physical_size = makeEven(new_cap_rows * new_stride) * 2;

    unsigned char* new_data = new unsigned char[physical_size]();

    if(data) {
        for(int r = 0; r < rows; r++) {
            for(int c = 0; c < cols; c++) {
                int old_pos = (r * stride + c) * 2;
                int new_pos = (r * new_stride + c) * 2;
                new_data[new_pos] = data[old_pos];
                new_data[new_pos + 1] = data[old_pos + 1];
            }
        }
        delete[] data;
    }

    data = new_data;
    cap_rows = new_cap_rows;
    stride = new_stride;
}

void TinyMatrix::SetLogicalBounds(int new_rows, int new_cols) {
    // 1. Auto-Reserve Safety Net: If they ask for more than physical RAM, allocate it safely!
    if(new_rows > cap_rows || new_cols > stride) {
        Reserve(std::max(cap_rows, new_rows), std::max(stride, new_cols));
    }

    // 2. Fast memory zeroing of ONLY the newly exposed area (Safely ignores shrinking)
    for(int r = 0; r < new_rows; r++) {
        for(int c = 0; c < new_cols; c++) {
            if(r >= rows || c >= cols) {
                int pos = (r * stride + c) * 2;
                data[pos] = 0;
                data[pos + 1] = 0;
            }
        }
    }

    // 3. Update the boundaries (This must ALWAYS run, even when shrinking!)
    rows = new_rows;
    cols = new_cols;
    this->size = makeEven(rows * cols) * 2; // Maintain legacy tracking size
}

void TinyMatrix::ShrinkToFit() {
    if(rows == cap_rows && cols == stride) return;

    int tight_size = makeEven(rows * cols) * 2;
    unsigned char* tight_data = new unsigned char[tight_size]();

    for(int r = 0; r < rows; r++) {
        for(int c = 0; c < cols; c++) {
            int old_pos = (r * stride + c) * 2;
            int new_pos = (r * cols + c) * 2;
            tight_data[new_pos] = data[old_pos];
            tight_data[new_pos + 1] = data[old_pos + 1];
        }
    }

    delete[] data;
    data = tight_data;
    cap_rows = rows;
    stride = cols;
    this->size = tight_size;
}

// =====================================================================

TinyMatrix::TinyMatrix() {
    this->rows = 1; this->cols = 1; this->size = 4;
    this->stride = 1; this->cap_rows = 1;
    this->data = new unsigned char[this->size]();
}

TinyMatrix::TinyMatrix(int r, int c) {
    this->rows = r; this->cols = c;
    this->stride = c; this->cap_rows = r;
    this->size = (makeEven((r * c)) * 2);
    this->data = new unsigned char[this->size]();
}

TinyMatrix::TinyMatrix(int r, int c, std::initializer_list<double> nums) {
    this->rows = r; this->cols = c;
    this->stride = c; this->cap_rows = r;
    this->size = (makeEven((r * c)) * 2);
    this->data = new unsigned char[this->size]();
    this->isFloat = true;

    int i = 0;
    for(double val : nums) {
        if(i >= (this->size / 2)) break;
        *(uint16_t*)(data + i * 2) = floatToHalf((float)val);
        i++;
    }
}

TinyMatrix::TinyMatrix(int r, int c, std::initializer_list<int> nums) {
    this->rows = r; this->cols = c;
    this->stride = c; this->cap_rows = r;
    this->size = (makeEven((r * c)) * 2);
    this->data = new unsigned char[this->size]();
    this->isFloat = false;

    int i = 0;
    for(int val : nums) {
        if(i >= (this->size / 2)) break;
        *(int16_t*)(data + i * 2) = (int16_t)val;
        i++;
    }
}

void TinyMatrix::init(int r, int c) {
    if(this == nullptr) return;
    delete[] this->data;
    this->rows = r; this->cols = c;
    this->stride = c; this->cap_rows = r;
    this->size = (makeEven((r * c)) * 2);
    this->data = new unsigned char[this->size]();
}

TinyMatrix::~TinyMatrix() {
    delete[] this->data;
}

TinyMatrix::TinyMatrix(const TinyMatrix& source) {
    this->rows = source.rows;
    this->cols = source.cols;
    this->stride = source.cols;
    this->cap_rows = source.rows;
    this->size = source.size;
    this->data = new unsigned char[this->size]();
    this->isFloat = source.isFloat;

    for(int i = 0; i < this->rows; i++) {
        for(int j = 0; j < this->cols; j++) {
            int s_pos = (i * source.stride + j) * 2;
            int d_pos = (i * this->stride + j) * 2;
            this->data[d_pos] = source.data[s_pos];
            this->data[d_pos + 1] = source.data[s_pos + 1];
        }
    }
}

TinyMatrix& TinyMatrix::operator=(const TinyMatrix& source) {
    if(&source == this) return *this;
    this->Shape(source.rows, source.cols, true);
    this->isFloat = source.isFloat;

    for(int i = 0; i < this->rows; i++) {
        for(int j = 0; j < this->cols; j++) {
            int s_pos = (i * source.stride + j) * 2;
            int d_pos = (i * this->stride + j) * 2;
            this->data[d_pos] = source.data[s_pos];
            this->data[d_pos + 1] = source.data[s_pos + 1];
        }
    }
    return *this;
}

float TinyMatrix::operator()(int r, int c) {
    assert(r >= 0 && r < this->rows && c >= 0 && c < this->cols && "Matrix Read Out of Bounds!");
    int pos = (r * this->stride + c) * 2;
    return (this->isFloat ? halfToFloat(*(uint16_t*)(this->data + pos)) : (float)*(int16_t*)(this->data + pos));
}

TinyMatrix& TinyMatrix::Shape(int r, int c, bool absolute) {
    if(this->rows == r && this->cols == c) return *this;

    // Mode 1: The "Absolute" structural reshape
    if(absolute) {
        ShrinkToFit();
        int tight_size = makeEven(r * c) * 2;
        unsigned char* new_data = new unsigned char[tight_size]();

        for(int i = 0; i < std::min(rows, r); i++) {
            for(int j = 0; j < std::min(cols, c); j++) {
                int old_pos = (i * stride + j) * 2;
                int new_pos = (i * c + j) * 2;
                new_data[new_pos] = data[old_pos];
                new_data[new_pos + 1] = data[old_pos + 1];
            }
        }
        delete[] data;
        data = new_data;
        rows = r; cols = c;
        cap_rows = r; stride = c;
        this->size = tight_size;
        return *this;
    }

    // Mode 2: 1D Flattening / Exact Volume Reshape
    // Fixes TestUnusualUses! If the total element count is identical,
    // tightly pack the memory to remove 2D stride gaps, then adjust bounds.
    if(r * c == this->rows * this->cols) {
        ShrinkToFit();
        this->rows = r; this->cols = c;
        this->cap_rows = r; this->stride = c;
        this->size = makeEven(r * c) * 2;
        return *this;
    }

    // Mode 3: Fast In-Place Capacity Expansion
    // Used by SnakeBrain. Expands logic boundaries without allocating memory.
    if(r <= cap_rows && c <= stride) {
        SetLogicalBounds(r, c);
        return *this;
    }

    // Mode 4: Legacy 1D Allocation & Copy
    // Fallback for weird structural wraps to maintain 100% backward compatibility
    ShrinkToFit();
    int tight_size = makeEven(r * c) * 2;
    unsigned char* new_data = new unsigned char[tight_size]();
    std::copy(data, data + std::min(this->size, tight_size), new_data);
    delete[] data;
    data = new_data;
    rows = r; cols = c;
    cap_rows = r; stride = c;
    this->size = tight_size;
    return *this;
}

void TinyMatrix::operator()(int r, int c, const int v) {
    assert(r >= 0 && r < this->rows && c >= 0 && c < this->cols && "Matrix Int Write Out of Bounds!");
    if(this->isFloat) {
        (*this)(r, c, (float)v); return;
    }
    int pos = (r * this->stride + c) * 2;
    *(int16_t*)(this->data + pos) = (int16_t)v;
}

void TinyMatrix::operator()(int r, int c, const float v) {
    assert(r >= 0 && r < this->rows && c >= 0 && c < this->cols && "Matrix Float Write Out of Bounds!");
    this->isFloat = true;
    int pos = (r * this->stride + c) * 2;
    *(uint16_t*)(this->data + pos) = floatToHalf(v);
}

void TinyMatrix::operator()(int r, int c, const double v) {
    TinyMatrix::operator()(r, c, (float)v);
}

float TinyMatrix::halfToFloat(mHalf y) {
    __m128i v_half = _mm_cvtsi32_si128(y.uVal);
    __m128 v_float = _mm_cvtph_ps(v_half);
    return _mm_cvtss_f32(v_float);
}

HALF TinyMatrix::floatToHalf(mFloat i) {
    __m128 v_float = _mm_set_ss(i.fVal);
    __m128i v_half = _mm_cvtps_ph(v_float, 0);
    return (HALF)_mm_extract_epi16(v_half, 0);
}

TinyMatrix& TinyMatrix::Ints() {
    if(!this->isFloat) return *this;

    for(int i = 0; i < this->Rows(); i++) {
        for(int j = 0; j < this->Cols(); j++) {
            float val = (*this)(i, j);
            int pos = (i * this->stride + j) * 2;
            *(int16_t*)(this->data + pos) = (int16_t)val;
        }
    }
    this->isFloat = false;
    return *this;
}

TinyMatrix& TinyMatrix::add(float s) {
    bool promote = (!this->isFloat && s != (float)(int)s);
    return this->mapInline([s](float val, int r, int c) { return val + s; }, promote);
}

TinyMatrix& TinyMatrix::sub(float s) {
    bool promote = (!this->isFloat && s != (float)(int)s);
    return this->mapInline([s](float val, int r, int c) { return val - s; }, promote);
}

TinyMatrix& TinyMatrix::multiply(float s) {
    bool promote = (!this->isFloat && s != (float)(int)s);
    return this->mapInline([s](float val, int r, int c) { return val * s; }, promote);
}

TinyMatrix& TinyMatrix::add(int s) {
    return this->add((float)s);
}
TinyMatrix& TinyMatrix::sub(int s) {
    return this->sub((float)s);
}
TinyMatrix& TinyMatrix::multiply(int s) {
    return this->multiply((float)s);
}
TinyMatrix& TinyMatrix::add(double s) {
    return this->add((float)s);
}
TinyMatrix& TinyMatrix::sub(double s) {
    return this->sub((float)s);
}
TinyMatrix& TinyMatrix::multiply(double s) {
    return this->multiply((float)s);
}

TinyMatrix& TinyMatrix::add(const TinyMatrix& a, bool reshape) {
    if(reshape) this->Shape(((TinyMatrix&)a).Rows(), ((TinyMatrix&)a).Cols(), true);
    return this->mapInline((TinyMatrix&)a, [](float v1, float v2, int r, int c) { return v1 + v2; });
}

TinyMatrix& TinyMatrix::sub(const TinyMatrix& a, bool reshape) {
    if(reshape) this->Shape(((TinyMatrix&)a).Rows(), ((TinyMatrix&)a).Cols(), true);
    return this->mapInline((TinyMatrix&)a, [](float v1, float v2, int r, int c) { return v1 - v2; });
}

TinyMatrix& TinyMatrix::hadamard(const TinyMatrix& a) {
    return this->mapInline((TinyMatrix&)a, [](float v1, float v2, int r, int c) { return v1 * v2; });
}

TinyMatrix& TinyMatrix::fixed_hadamard(const TinyMatrix& a) {
    return this->mapInline((TinyMatrix&)a, [](float v1, float v2, int r, int c) {
        int32_t x = (int16_t)v1;
        int32_t y = (int16_t)v2;
        return (float)(int16_t)((x * y) >> 8);
        });
}

TinyMatrix& TinyMatrix::Relu() {
    return this->mapInline([](float val, int r, int c) { return val > 0.0001f ? val : 0.0f; }, true);
}
TinyMatrix& TinyMatrix::D_Relu() {
    return this->mapInline([](float val, int r, int c) { return val > 0.0001f ? 1.0f : 0.0f; }, true);
}
TinyMatrix& TinyMatrix::Sigmoid() {
    return this->mapInline([](float val, int r, int c) { return 1.0f / (1.0f + std::exp(-val)); }, true);
}
TinyMatrix& TinyMatrix::D_Sigmoid() {
    return this->mapInline([](float val, int r, int c) { return val * (1.0f - val); }, true);
}
TinyMatrix& TinyMatrix::Tanh() {
    return this->mapInline([](float val, int r, int c) { return std::tanh(val); }, true);
}
TinyMatrix& TinyMatrix::D_Tanh() {
    return this->mapInline([](float val, int r, int c) { return 1.0f - (val * val); }, true);
}

TinyMatrix& TinyMatrix::fixed_tanh() {
    return this->mapInline([](float val, int r, int c) {
        int16_t x = (int16_t)val;
        bool is_negative = (x < 0);
        if(is_negative) x = -x;

        int16_t result = (x >= 1024) ? 256 : TinyMatrix::tanh_lut[x];
        if(is_negative) result = -result;

        return (float)result;
        });
}

TinyMatrix& TinyMatrix::dot(TinyMatrix& a, TinyMatrix& b) {
    _snapshotToRegister(_scratch, &a, _scratchCapacity);
    TinyMatrix* left = _scratch;
    TinyMatrix* right = &b;

    if(right == this) {
        _snapshotToRegister(_operandScratch, right, _operandScratchCapacity);
        right = _operandScratch;
    }

    int outRows = left->Rows();
    int outCols = right->Cols();
    int innerDim = left->Cols();

    this->Shape(outRows, outCols);

    bool leftFloat = left->IsFloat();
    bool rightFloat = right->IsFloat();
    this->isFloat = leftFloat || rightFloat;

    size_t neededAccum = (size_t)(outRows * outCols);
    if(_accumScratch == nullptr || neededAccum > _accumCapacity) {
        delete[] _accumScratch;
        _accumScratch = new float[neededAccum];
        _accumCapacity = neededAccum;
    }

    std::memset(_accumScratch, 0, neededAccum * sizeof(float));
    float* accum = _accumScratch;

    unsigned char* left_data = left->data;
    unsigned char* right_data = right->data;
    int l_stride = left->stride;
    int r_stride = right->stride;

    for(int i = 0; i < outRows; i++) {
        for(int k = 0; k < innerDim; k++) {
            int left_pos = (i * l_stride + k) * 2;
            float left_val = leftFloat ? halfToFloat(*(uint16_t*)(left_data + left_pos))
                : (float)*(int16_t*)(left_data + left_pos);

            for(int j = 0; j < outCols; j++) {
                int right_pos = (k * r_stride + j) * 2;
                float right_val = rightFloat ? halfToFloat(*(uint16_t*)(right_data + right_pos))
                    : (float)*(int16_t*)(right_data + right_pos);

                accum[i * outCols + j] += left_val * right_val;
            }
        }
    }

    unsigned char* target_data = this->data;
    int t_stride = this->stride;

    if(this->isFloat) {
        for(int i = 0; i < outRows; i++) {
            for(int j = 0; j < outCols; j++) {
                *(uint16_t*)(target_data + (i * t_stride + j) * 2) = floatToHalf(accum[i * outCols + j]);
            }
        }
    } else {
        for(int i = 0; i < outRows; i++) {
            for(int j = 0; j < outCols; j++) {
                *(int16_t*)(target_data + (i * t_stride + j) * 2) = (int16_t)accum[i * outCols + j];
            }
        }
    }
    return *this;
}

TinyMatrix& TinyMatrix::dot(TinyMatrix& a) {
    return this->dot(*this, a);
}

TinyMatrix& TinyMatrix::fixed_dot(TinyMatrix& a, TinyMatrix& b) {
    _snapshotToRegister(_scratch, &a, _scratchCapacity);
    TinyMatrix* left = _scratch;
    TinyMatrix* right = &b;
    if(&b == this) {
        _snapshotToRegister(_operandScratch, &b, _operandScratchCapacity);
        right = _operandScratch;
    }

    int outRows = left->Rows();
    int outCols = right->Cols();
    int innerDim = left->Cols();

    this->Shape(outRows, outCols);
    this->isFloat = false;

    size_t neededAccum = (size_t)(outRows * outCols);
    if(_accumScratchInt == nullptr || neededAccum > _accumIntCapacity) {
        delete[] _accumScratchInt;
        _accumScratchInt = new int32_t[neededAccum];
        _accumIntCapacity = neededAccum;
    }

    std::memset(_accumScratchInt, 0, neededAccum * sizeof(*_accumScratchInt));
    int32_t* accum = _accumScratchInt;

    unsigned char* left_data = left->data;
    unsigned char* right_data = right->data;
    int l_stride = left->stride;
    int r_stride = right->stride;

    for(int i = 0; i < outRows; i++) {
        for(int k = 0; k < innerDim; k++) {
            int32_t left_val = (int16_t) * (int16_t*)(left_data + (i * l_stride + k) * 2);

            for(int j = 0; j < outCols; j++) {
                int32_t right_val = (int16_t) * (int16_t*)(right_data + (k * r_stride + j) * 2);
                accum[i * outCols + j] += (left_val * right_val) >> 8;
            }
        }
    }

    unsigned char* target_data = this->data;
    int t_stride = this->stride;
    for(int i = 0; i < outRows; i++) {
        for(int j = 0; j < outCols; j++) {
            *(int16_t*)(target_data + (i * t_stride + j) * 2) = (int16_t)accum[i * outCols + j];
        }
    }
    return *this;
}

TinyMatrix& TinyMatrix::QuantizeQ88() {
    this->multiply(256.0f);
    this->Ints();
    return *this;
}

TinyMatrix& TinyMatrix::DequantizeQ88() {
    this->multiply(1.0f / 256.0f);
    return *this;
}

void TinyMatrix::print(std::string extra) {
    for(int i = 0; i < this->Rows(); i++) {
        for(int j = 0; j < this->Cols(); j++) {
            if(this->isFloat) {
                printf("%.3f%s", (*this)(i, j), (j == this->Cols() - 1 ? "\n" : " "));
            } else {
                printf("%d%s", (int16_t)(*this)(i, j), (j == this->Cols() - 1 ? "\n" : " "));
            }
        }
    }
    printf("%s", extra.c_str());
}

TinyMatrix& TinyMatrix::transpose() {
    _snapshotToRegister(_transposeScratch, this, _transposeScratchCapacity);
    this->Shape(_transposeScratch->Cols(), _transposeScratch->Rows());

    for(int i = 0; i < this->Rows(); i++) {
        for(int j = 0; j < this->Cols(); j++) {
            (*this)(i, j, (*_transposeScratch)(j, i));
        }
    }
    return *this;
}

nRet TinyMatrix::sum() {
    float total = 0.0f;
    for(int i = 0; i < this->Rows(); i++) {
        for(int j = 0; j < this->Cols(); j++) {
            total += (*this)(i, j);
        }
    }
    return nRet(total);
}

TinyMatrix& TinyMatrix::Floats() {
    this->isFloat = true;
    return *this;
}

TinyMatrix& TinyMatrix::Randomize(float min, float max) {
    for(int i = 0; i < this->rows; i++) {
        for(int j = 0; j < this->cols; j++) {
            float r = min + ((float)rand() / (RAND_MAX)) * (max - min);

            int pos = (i * this->stride + j) * 2;
            if(this->isFloat) {
                *(uint16_t*)(this->data + pos) = floatToHalf(r);
            } else {
                *(int16_t*)(this->data + pos) = (int16_t)r;
            }
        }
    }
    return *this;
}

void TinyMatrix::_snapshotToRegister(TinyMatrix*& dest, const TinyMatrix* src, size_t& cap) {
    if(!src) return;

    if(dest == nullptr) {
        dest = new TinyMatrix(src->rows, src->cols);
        cap = dest->size;
    }

    // Use the new capacity-aware tools to prep the scratchpad!
    dest->Reserve(src->rows, src->cols);
    dest->SetLogicalBounds(src->rows, src->cols);
    dest->isFloat = src->isFloat;

    for(int r = 0; r < src->rows; r++) {
        for(int c = 0; c < src->cols; c++) {
            int d_pos = (r * dest->stride + c) * 2;
            int s_pos = (r * src->stride + c) * 2;
            dest->data[d_pos] = src->data[s_pos];
            dest->data[d_pos + 1] = src->data[s_pos + 1];
        }
    }
}

#endif