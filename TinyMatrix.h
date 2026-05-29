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
#include <cstring>   // for std::memcpy in _snapshotToRegister


enum mapFuncs {
    TPOS, PRINT, ADDS, SUBS, MULS, DOT, SUM, SUB, ADD, 
    ADDR, SUBR, INTS, MULFS, RELU, SIGMOID, D_RELU, 
    D_SIGMOID, HADAMARD, FIXED_DOT, FIXED_HADAMARD,
    TANH, D_TANH, FIXED_TANH
};

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
    int size;
    unsigned char* data;
    bool isFloat = false;

    static thread_local TinyMatrix* _scratch;
    static thread_local TinyMatrix* _operandScratch;
    static thread_local size_t _scratchCapacity;
    static thread_local size_t _operandScratchCapacity;

public:
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

    // Modern initializer lists replace the old unsafe variadics
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
    }
    TinyMatrix& fixed_tanh();
    void print(std::string extra = "");

    TinyMatrix& transpose();
    TinyMatrix& dot(TinyMatrix& a, TinyMatrix& b);
    TinyMatrix& dot(TinyMatrix& a);
    TinyMatrix& Ints();
    TinyMatrix& Randomize(float min = -1.0f, float max = 1.0f);
    nRet sum();

private:
    unsigned char* operator[](const int p);
    void map(int n, void* o2_ret = nullptr, TinyMatrix* other = nullptr);
    static HALF floatToHalf(mFloat i);
    static float halfToFloat(mHalf y);

    // Helpers for cheap, capacity-aware snapshots into the persistent scratch registers.
    void _ensureScratchCapacity(TinyMatrix*& reg, size_t& capacity, int rows, int cols);
    void _snapshotToRegister(TinyMatrix*& dest, const TinyMatrix* src, size_t& cap);

};
#endif

#ifdef TINYMATRIX_IMPLEMENTATION
thread_local TinyMatrix* TinyMatrix::_scratch = nullptr;
thread_local TinyMatrix* TinyMatrix::_operandScratch = nullptr;
thread_local size_t TinyMatrix::_scratchCapacity = 0;
thread_local size_t TinyMatrix::_operandScratchCapacity = 0;

constexpr auto makeEven(int a) {
    return (a + 1 ^ (~a & 1));
}
constexpr auto NOT_VOID = 1;
constexpr auto NEEDS_COPY = 2;
constexpr auto NEEDS_VAL = 4;
constexpr auto NEEDS_OTHER = 8;
constexpr auto RESHAPE = 16;
constexpr auto AS_FLOAT = 32;
// Allocate the 2KB of RAM for the Lookup Table
int16_t TinyMatrix::tanh_lut[1024];

// Run this ONCE at boot to pre-calculate the Q8.8 Tanh curve
void TinyMatrix::InitLUT() {
    for(int i = 0; i < 1024; i++) {
        // Convert Q8.8 integer index to true float, run FPU Tanh, and convert back!
        float float_val = (float)i / 256.0f;
        float t_val = std::tanh(float_val);
        tanh_lut[i] = (int16_t)(t_val * 256.0f);
    }
}

float print(float val, int r, int c, TinyMatrix* _this) {
    if(_this->IsFloat()) {
        printf("%.3f%s", val, ((c == 0 && r == 0) ? "\n" : (c == 0 ? "\n" : " ")));
    } else {
        printf("%d%s", (int16_t)val, ((c == 0 && r == 0) ? "\n" : (c == 0 ? "\n" : " ")));
    }
    return 0.0f;
};


float tpos(float val, float r, int c, TinyMatrix* d) {
    //int pos = (r * (d->Cols()) + c);
    return (*d)(c, (int)r);
}

float add(float val, float r, int c, TinyMatrix* _notused) {
    return val + r;
};

float add2(float val, float r, int c, TinyMatrix* d) {
    return val + (float)(*d)((int)r, c);
};

float sub(float val, float s, int _notused, TinyMatrix* _notused2) {
    return val - s;
};

float sub2(float val, float r, int c, TinyMatrix* d) {
    return (float)(*d)((int)r, c) - val;
};

float mul(float val, float s, int _notused, TinyMatrix* _notused2) {
    return val * s;
};

float relu(float val, float _notused, int _notused2, TinyMatrix* _notused3) {
    // Add epsilon to prevent floating point noise from activating dead neurons
    return val > 0.0001f ? val : 0.0f;
}

float d_relu(float val, float _notused, int _notused2, TinyMatrix* _notused3) {
    // Add epsilon to gradient calculations
    return val > 0.0001f ? 1.0f : 0.0f;
}

float sigmoid(float val, float _notused, int _notused2, TinyMatrix* _notused3) {
    return 1.0f / (1.0f + std::exp(-val));
}

// Sigmoid Derivative: output * (1 - output)
float d_sigmoid(float val, float _notused, int _notused2, TinyMatrix* _notused3) {
    return val * (1.0f - val);
}

// Element-wise Matrix Multiplication
float hadamard(float val, float r, int c, TinyMatrix* d) {
    return val * (float)(*d)((int)r, c);
};

// Q8.8 Fixed-Point Scalar/Dot Multiplication
float fixed_mul(float val, float s, int _notused, TinyMatrix* _notused2) {
    int32_t a = (int16_t)val;
    int32_t b = (int16_t)s;
    return (float)(int16_t)((a * b) >> 8);
}

// Q8.8 Fixed-Point Hadamard
float fixed_hadamard_op(float val, float r, int c, TinyMatrix* d) {
    int32_t a = (int16_t)val;
    int32_t b = (int16_t)(*d)((int)r, c);
    return (float)(int16_t)((a * b) >> 8);
}

float tanh_op(float val, float _notused, int _notused2, TinyMatrix* _notused3) {
    return std::tanh(val);
}

// Tanh Derivative: 1 - output^2
float d_tanh_op(float val, float _notused, int _notused2, TinyMatrix* _notused3) {
    return 1.0f - (val * val);
}

// Fast O(1) Integer Array Lookup
float fixed_tanh_op(float val, float _notused, int _notused2, TinyMatrix* _notused3) {
    int16_t x = (int16_t)val;

    // Tanh is symmetrical, so we only store positive numbers!
    bool is_negative = (x < 0);
    if(is_negative) x = -x;

    int16_t result;
    if(x >= 1024) {
        result = 256; // If input > 4.0, it's fully saturated to 1.0 (256 in Q8.8)
    } else {
        result = TinyMatrix::tanh_lut[x]; // Grab pre-calculated value from RAM
    }

    if(is_negative) result = -result; // Restore the sign

    return (float)result;
}

void* mFuncs[23] = {
    &tpos,              // TPOS
    &print,             // PRINT 
    &add,               // ADDS
    &sub,               // SUBS
    &mul,               // MULS
    &mul,               // DOT
    &mul,               // SUM
    &sub2,              // SUB
    &add2,              // ADD
    &add2,              // ADDR
    &sub2,              // SUBR
    &add2,              // INTS
    &mul,               // MULFS
    &relu,              // RELU
    &sigmoid,           // SIGMOID
    &d_relu,            // D_RELU
    &d_sigmoid,         // D_SIGMOID
    &hadamard,          // HADAMARD
    &fixed_mul,         // FIXED_DOT
    &fixed_hadamard_op, // FIXED_HADAMARD
    &tanh_op,           // TANH
    &d_tanh_op,         // D_TANH
    &fixed_tanh_op      //FIXED_TANH
};
char  mFuncs_t[23] = {
    (NOT_VOID | NEEDS_COPY),                         //Transpose
    (NEEDS_VAL),                                     //Print
    (NOT_VOID | NEEDS_VAL),                          //Add scalar ADDS
    (NOT_VOID | NEEDS_VAL),                          //Subtract scalar
    (NOT_VOID | NEEDS_VAL),                          //Multiply scalar
    (NOT_VOID | NEEDS_OTHER),                        //Dot Product of 2 matricies or 1 matrix (autotransposes 2nd matrix or self if only 1 provided)
    (NEEDS_VAL),                                     //Summation of matrix
    (NOT_VOID | NEEDS_COPY | NEEDS_OTHER),           //Subtract 1 matrix from self
    (NOT_VOID | NEEDS_COPY | NEEDS_OTHER),           //add matrix to self
    (NOT_VOID | NEEDS_COPY | NEEDS_OTHER | RESHAPE), //Subtract 1 matrix from self (reshape fill empty with 0s)
    (NOT_VOID | NEEDS_COPY | NEEDS_OTHER | RESHAPE), //add matrix to self (reshape fill empty with 0s)
    (NOT_VOID | NEEDS_COPY),                         //converts half float matrix back to int16_t matrix
    (NOT_VOID | NEEDS_VAL | AS_FLOAT),               //multiply by float scalar...
    (NOT_VOID | NEEDS_VAL),                          // RELU (Just reads the val and overwrites it)
    (NOT_VOID | NEEDS_VAL),                          // SIGMOID
    (NOT_VOID | NEEDS_VAL),                          // D_RELU
    (NOT_VOID | NEEDS_VAL),                          // D_SIGMOID
    (NOT_VOID | NEEDS_COPY | NEEDS_OTHER),           // HADAMARD
    (NOT_VOID | NEEDS_OTHER),                        // FIXED_DOT (Copy whatever flag your standard DOT uses here!)
    (NOT_VOID | NEEDS_COPY | NEEDS_OTHER),           // FIXED_HADAMARD
    (NOT_VOID | NEEDS_VAL),                          // TANH
    (NOT_VOID | NEEDS_VAL),                          // D_TANH
    (NOT_VOID | NEEDS_VAL)                           // FIXED_TANH
};



TinyMatrix::TinyMatrix() {
    this->rows = 1; this->cols = 1; this->size = 4;
    this->data = new unsigned char[this->size] { 0 };
}

TinyMatrix::TinyMatrix(int r, int c) {
    this->rows = r; this->cols = c;
    this->size = (makeEven((r * c)) * 2);
    this->data = new unsigned char[this->size] { 0 };
}

TinyMatrix::TinyMatrix(int r, int c, std::initializer_list<double> nums) {
    this->rows = r; this->cols = c;
    this->size = (makeEven((r * c)) * 2);
    this->data = new unsigned char[this->size] { 0 };
    this->isFloat = true;

    int i = 0;
    for(double val : nums) {
        if(i >= (this->size / 2)) break;
        *(uint16_t*)(*this)[i] = floatToHalf((float)val);
        i++;
    }
}

TinyMatrix::TinyMatrix(int r, int c, std::initializer_list<int> nums) {
    this->rows = r; this->cols = c;
    this->size = (makeEven((r * c)) * 2);
    this->data = new unsigned char[this->size] { 0 };
    this->isFloat = false;

    int i = 0;
    for(int val : nums) {
        if(i >= (this->size / 2)) break;
        *(int16_t*)(*this)[i] = (int16_t)val;
        i++;
    }
}

void TinyMatrix::init(int r, int c) {
    if(this == nullptr) return;
    delete[] this->data;                 // safe: all constructed objects own a valid buffer
    this->rows = r; this->cols = c;
    this->size = (makeEven((r * c)) * 2);
    this->data = new unsigned char[this->size] { 0 };
}

TinyMatrix::~TinyMatrix() {
    delete[] this->data;
}

TinyMatrix::TinyMatrix(const TinyMatrix& source) {
    this->rows = source.rows;
    this->cols = source.cols;
    this->size = source.size;
    this->data = new unsigned char[this->size] { 0 };
    this->isFloat = source.isFloat;
    std::copy(source.data, source.data + source.size, this->data);
}

TinyMatrix& TinyMatrix::operator=(const TinyMatrix& source) {
    if(&source == this) return *this;

    if(source.rows == this->rows && source.cols == this->cols) {
        // Same dimensions: reuse existing buffer (fast path, no allocation).
        // Only copy the payload and type flag.
        this->isFloat = source.isFloat;
        std::copy(source.data, source.data + source.size, this->data);
        return *this;
    }

    // Size change: clean up old memory, then allocate fresh buffer.
    delete[] this->data;
    this->rows = source.rows;
    this->cols = source.cols;
    this->size = source.size;
    this->isFloat = source.isFloat;
    this->data = new unsigned char[this->size] { 0 };
    std::copy(source.data, source.data + source.size, this->data);
    return *this;
}

float TinyMatrix::operator()(int r, int c) {
    int pos = (r * (this->cols) + c);
    return (this->isFloat ? halfToFloat((uint16_t) * (uint16_t*)((*this)[pos])) : (float)*(int16_t*)((*this)[pos]));
}

TinyMatrix& TinyMatrix::Shape(int r, int c, bool absolute) {
    if(this->rows == r && this->cols == c) return *this;

    int old_r = this->rows; int old_c = this->cols; int old_size = this->size;
    this->rows = r; this->cols = c;
    unsigned char* old_data = &this->data[0];
    this->size = (makeEven((r * c)) * 2);

    bool olim = (old_size <= this->size);
    if(absolute) {
        this->data = new unsigned char[this->size] { 0 };
        for(int rt = 0; rt < (olim ? old_r : this->rows); rt++) {
            for(int ct = 0; ct < (olim ? old_c : this->cols); ct++) {
                int np = ((rt * this->cols) + ct) * 2;
                int op = ((rt * old_c) + ct) * 2;

                if(np < this->size - 1 && op < old_size - 1) {
                    this->data[np] = old_data[op];
                    this->data[np + 1] = old_data[op + 1];
                } else if(np < this->size - 1) {
                    this->data[np] = 0;
                    this->data[np + 1] = 0;
                }
            }
        }
        delete[] old_data; // FIXED!
    } else {
        this->data = new unsigned char[this->size] { 0 };
        std::copy(old_data, old_data + (old_size < this->size ? old_size : this->size), this->data);
        delete[] old_data; // FIXED!
    }
    return *this;
}

void TinyMatrix::operator()(int r, int c, const int v) {
    if(this->isFloat) {
        (*this)(r, c, (float)v); return;
    }
    int pos = (r * (this->cols) + c);
    *(int16_t*)(*this)[pos] = (int16_t)v;
}

void TinyMatrix::operator()(int r, int c, const double v) {
    TinyMatrix::operator()(r, c, (float)v);
}

void TinyMatrix::operator()(int r, int c, const float v) {
    this->isFloat = true;
    int pos = (r * (this->cols) + c);
    *(uint16_t*)(*this)[pos] = floatToHalf(v);
}

float TinyMatrix::halfToFloat(mHalf y) {

    mFloat ret(0.0f);
    ret.parts.Sign = y.parts.Sign;

    if(!y.parts.Exp) {
        if(!y.parts.Frac) {
            ret.parts.Exp = 0;
            ret.parts.Frac = 0;
        } else {
            const float half_denorm = (1.0f / 16384.0f);
            float mantissa = ((float)y.parts.Frac) / 1024.0f;
            float sgn = (((float)y.parts.Sign) * 2.0f) - 1.0f;
            ret.fVal = sgn * mantissa * half_denorm;
        }
    } else if(y.parts.Exp == 31) {
        ret.parts.Exp = 0xff;
        ret.parts.Frac = (y.parts.Frac != 0) ? 0x400000 : 0;
    } else {

        ret.parts.Exp = (uint32_t)y.parts.Exp + (127 - 15);
        ret.parts.Frac = (uint32_t)y.parts.Frac << 13;
    }

    return ret.fVal;
}

HALF TinyMatrix::floatToHalf(mFloat i) {
    if(i.fVal == 0.0f || i.fVal == -0.0f) {
        mHalf zero(0);
        zero.parts.Sign = i.parts.Sign;
        return zero.uVal;
    }

    if(i.parts.Exp == 0xff) {
        mHalf special(0);
        special.parts.Sign = i.parts.Sign;
        special.parts.Exp = 31;
        // If it was NaN (Frac != 0), keep the Frac non-zero. Otherwise 0 for Inf.
        special.parts.Frac = (i.parts.Frac != 0) ? 1 : 0;
        return special.uVal;
    }

    mHalf ret(0);
    ret.parts.Sign = i.parts.Sign;
    int _3precision = (int)floor((i.fVal * 1000.0f));
    float nf = _3precision / 999.875f;
    i.fVal = (nf + 0.00006103515625f);

    register int e = i.parts.Exp - 112;
    register int f = i.parts.Frac;

    if(e < 0) {
        if(e < -10)
            return ret.uVal;

        f = (f | 0x00800000) >> (1 - e);
        ret.parts.Frac = (f >> 13);
        return ret.uVal;

    } else if(e == 0xff - (112)) {
        ret.parts.Frac = (i.parts.Frac != 0);
        ret.parts.Exp = 31;
        return ret.uVal;

    } else {
        if(e > 30) {
            ret.parts.Exp = 31;
            return ret.uVal;
        }

        ret.parts.Exp = e;
        ret.parts.Frac = (f >> 13);
        return ret.uVal;
    }
}

TinyMatrix& TinyMatrix::Ints() {
    if(!this->isFloat)
        return *this;

    this->map(mapFuncs::INTS);
    return *this;
}

TinyMatrix& TinyMatrix::add(int s) {
    float as = (float)s;
    this->map(mapFuncs::ADDS, &as);
    return *this;
}

TinyMatrix& TinyMatrix::sub(int s) {
    float as = (float)s;
    this->map(mapFuncs::SUBS, &as);
    return *this;
}

TinyMatrix& TinyMatrix::add(float s) {
    float as = s;
    this->map(mapFuncs::ADDS, &as);
    return *this;
}

TinyMatrix& TinyMatrix::sub(float s) {
    float as = s;
    this->map(mapFuncs::SUBS, &as);
    return *this;
}

TinyMatrix& TinyMatrix::add(double s) {
    this->add((float)s);
    return *this;
}

TinyMatrix& TinyMatrix::sub(double s) {
    this->sub((float)s);
    return *this;
}

TinyMatrix& TinyMatrix::add(const TinyMatrix& a, bool reshape) {
    if(reshape) {
        this->map(mapFuncs::ADDR, (void*)&a);
    } else {
        this->map(mapFuncs::ADD, (void*)&a);
    }
    return *this;
}

TinyMatrix& TinyMatrix::sub(const TinyMatrix& a, bool reshape) {
    if(reshape) {
        this->map(mapFuncs::SUBR, (void*)&a);
    } else {
        this->map(mapFuncs::SUB, (void*)&a);
    }
    return *this;
}

TinyMatrix& TinyMatrix::multiply(int s) {
    float as = (float)s;
    this->map(mapFuncs::MULS, &as);
    return *this;
}

TinyMatrix& TinyMatrix::multiply(float s) {
    float as = (float)s;
    if(this->isFloat) {
        this->map(mapFuncs::MULS, &as);
    } else {
        this->map(mapFuncs::MULFS, &as);
    }
    return *this;
}

TinyMatrix& TinyMatrix::multiply(double s) {
    this->multiply((float)s);
    return *this;
}

TinyMatrix& TinyMatrix::dot(TinyMatrix& a, TinyMatrix& b) {

    this->map(mapFuncs::DOT, &a, &b);
    return *this;
}

TinyMatrix& TinyMatrix::dot(TinyMatrix& a) {

    this->map(mapFuncs::DOT, &a, nullptr);
    return *this;
}

TinyMatrix& TinyMatrix::QuantizeQ88() {
    this->multiply(256.0f); // Shift float left 8 bits mathematically
    this->Ints();           // Strip float state and truncate to int16_t
    return *this;
}

TinyMatrix& TinyMatrix::DequantizeQ88() {
    // Multiplying an Int matrix by a float automatically promotes it back to fp16!
    this->multiply(1.0f / 256.0f);
    return *this;
}

TinyMatrix& TinyMatrix::fixed_dot(TinyMatrix& a, TinyMatrix& b) {
    this->map(mapFuncs::FIXED_DOT, (void*)&a, &b);
    return *this;
}

TinyMatrix& TinyMatrix::fixed_hadamard(const TinyMatrix& a) {
    this->map(mapFuncs::FIXED_HADAMARD, (void*)&a);
    return *this;
}

TinyMatrix& TinyMatrix::fixed_tanh() {
    this->map(mapFuncs::FIXED_TANH);
    return *this;
}

void TinyMatrix::print(std::string extra) {
    this->map(mapFuncs::PRINT);
    printf("%s", extra.c_str());
}

TinyMatrix& TinyMatrix::Relu() {
    this->map(mapFuncs::RELU);
    return *this;
}

TinyMatrix& TinyMatrix::Sigmoid() {
    this->map(mapFuncs::SIGMOID);
    return *this;
}

TinyMatrix& TinyMatrix::transpose() {
    this->map(mapFuncs::TPOS);
    return *this;
}
TinyMatrix& TinyMatrix::D_Relu() {
    this->map(mapFuncs::D_RELU);
    return *this;
}

TinyMatrix& TinyMatrix::D_Sigmoid() {
    this->map(mapFuncs::D_SIGMOID);
    return *this;
}

TinyMatrix& TinyMatrix::hadamard(const TinyMatrix& a) {
    this->map(mapFuncs::HADAMARD, (void*)&a);
    return *this;
}

TinyMatrix& TinyMatrix::Tanh() {
    this->map(mapFuncs::TANH);
    return *this;
}

TinyMatrix& TinyMatrix::D_Tanh() {
    this->map(mapFuncs::D_TANH);
    return *this;
}

nRet TinyMatrix::sum() {
    nRet as;
    this->map(mapFuncs::SUM, &as);
    return as;
}

TinyMatrix& TinyMatrix::Randomize(float min, float max) {
    for(int i = 0; i < this->rows; i++) {
        for(int j = 0; j < this->cols; j++) {
            // Generate a random float between min and max
            float r = min + ((float)rand() / (RAND_MAX)) * (max - min);

            if(this->isFloat) {
                (*this)(i, j, r);
            } else {
                (*this)(i, j, (int16_t)r);
            }
        }
    }
    return *this;
}

void TinyMatrix::map(int n, void* o2_ret, TinyMatrix* other) {
    if (_scratch == nullptr) {
        _scratch = new TinyMatrix(1, 1);
        _scratchCapacity = _scratch->size;
    }
    if (_operandScratch == nullptr) {
        _operandScratch = new TinyMatrix(1, 1);
        _operandScratchCapacity = _operandScratch->size;
    }

    TinyMatrix* scratchNib = nullptr;
    TinyMatrix* o2_copy = nullptr;
    float val = 0;
    float sum = 0;

    // Captured intended isFloat for the three scratch/"operand" buffers
    // at the exact moment they were snapshotted or aliased. These are the
    // only reliable values to pass to fastGet when reading those buffers.
    bool scratch_snapshot_is_float = false;
    bool other_snapshot_is_float   = false;
    bool left_snapshot_is_float    = false;

    float(*foo)(float, float, int, TinyMatrix*);
    foo = (float(*)(float, float, int, TinyMatrix*))mFuncs[n];

    // Fast direct buffer access helpers — defined early so they are available everywhere in map()
    auto fastGet = [](unsigned char* base, int cols, bool isF, int r, int c) -> float {
        int pos = (r * cols + c) * 2;
        return isF ? halfToFloat(*(uint16_t*)(base + pos))
                   : (float)*(int16_t*)(base + pos);
    };

    auto fastSet = [](unsigned char* base, int cols, bool isF, int r, int c, float v) {
        int pos = (r * cols + c) * 2;
        if (isF) {
            *(uint16_t*)(base + pos) = floatToHalf(v);
        } else {
            *(int16_t*)(base + pos) = (int16_t)v;
        }
    };


    if(mFuncs_t[n] & (NEEDS_COPY | NEEDS_OTHER)) {
        if(mFuncs_t[n] & NEEDS_COPY) {
            _snapshotToRegister(_scratch, this, _scratchCapacity);
            scratchNib = _scratch;
            scratch_snapshot_is_float = this->isFloat;
        }
        if(mFuncs_t[n] & NEEDS_OTHER && !(mFuncs_t[n] & NEEDS_COPY)) {
            assert(other != nullptr || o2_ret != nullptr);
            if(o2_ret != nullptr && other == nullptr)
                other = (TinyMatrix*)o2_ret;
            _snapshotToRegister(_scratch, other, _scratchCapacity);
            scratchNib = _scratch;
            scratch_snapshot_is_float = other->isFloat;
            if(n == mapFuncs::DOT || n == mapFuncs::FIXED_DOT) {
                if(scratchNib->rows != ((TinyMatrix*)o2_ret)->cols && scratchNib->cols == ((TinyMatrix*)o2_ret)->cols) {
                    //other matrix was provided but not transposed or DOT with self
                    scratchNib->map(mapFuncs::TPOS);
                }
            }
        }
        if(n == mapFuncs::TPOS) {
            this->Shape(this->cols, this->rows);
        }
        if(n == mapFuncs::INTS) {
            this->isFloat = false;
        }
        assert(scratchNib != nullptr);
    }
    TinyMatrix* leftOperand = nullptr;

    if(n == mapFuncs::DOT || n == mapFuncs::FIXED_DOT) {
        assert(o2_ret != nullptr);
        TinyMatrix* o2_matrix = (TinyMatrix*)o2_ret;

        if(this == o2_matrix) {
            _snapshotToRegister(_operandScratch, o2_matrix, _operandScratchCapacity);
            leftOperand = _operandScratch;
            left_snapshot_is_float = o2_matrix->isFloat;
        } else {
            leftOperand = o2_matrix;
            left_snapshot_is_float = o2_matrix->isFloat;
        }

        if(this->cols != scratchNib->cols || this->rows != leftOperand->rows) {
            this->Shape(leftOperand->rows, scratchNib->cols);
        }

        if(scratchNib->isFloat || leftOperand->isFloat) {
            this->isFloat = true;
        }
    }

    // Decide once (before the hot loop) whether we need a reshaped copy of the "other" operand.
    // This logic was previously re-evaluated on every element for binary matrix ops.
    // Guarded strictly to only the opcodes that use o2_copy / fromMatrix.
    if (n == mapFuncs::ADD || n == mapFuncs::SUB || n == mapFuncs::SUBR ||
        n == mapFuncs::ADDR || n == mapFuncs::HADAMARD || n == mapFuncs::FIXED_HADAMARD) {

        if (scratchNib != nullptr) {
            assert(o2_ret != nullptr);

            bool need_reshaped_o2 = (o2_copy == nullptr) &&
                ((mFuncs_t[n] & RESHAPE) ||
                 (((TinyMatrix*)o2_ret)->rows != scratchNib->rows ||
                  ((TinyMatrix*)o2_ret)->cols != scratchNib->cols));

            if (need_reshaped_o2) {
                _snapshotToRegister(_operandScratch, (TinyMatrix*)o2_ret, _operandScratchCapacity);
                _operandScratch->Shape(scratchNib->rows, scratchNib->cols, true);
                o2_copy = _operandScratch;
                other_snapshot_is_float = ((TinyMatrix*)o2_ret)->isFloat;
            } else if (o2_copy == nullptr) {
                o2_copy = (TinyMatrix*)o2_ret;
                other_snapshot_is_float = ((TinyMatrix*)o2_ret)->isFloat;
            }
            assert(o2_copy->rows == scratchNib->rows && o2_copy->cols == scratchNib->cols);
        }
    }

    for(int i = 0; i < this->rows; i++) {
        for(int j = 0; j < this->cols; j++) {
            if(mFuncs_t[n] & NEEDS_VAL) {
                if(mFuncs_t[n] & AS_FLOAT) this->isFloat = false;
                val = fastGet(this->data, this->cols, this->isFloat, i, j);
            }
            if(mFuncs_t[n] & NOT_VOID) {
                if(scratchNib != nullptr) {
                    switch(n) {
                    case mapFuncs::DOT:
                    case mapFuncs::FIXED_DOT:
                        sum = 0;
                        // Fast direct access using captured snapshot isFloat (avoids fragile object state)
                        for(int a = 0; a < leftOperand->cols; a++) {
                            float sVal = fastGet(scratchNib->data, scratchNib->cols, scratch_snapshot_is_float, a, j);
                            float lVal = fastGet(leftOperand->data, leftOperand->cols, left_snapshot_is_float, i, a);
                            sum += foo(sVal, lVal, 0, nullptr);
                        }

                        if(this->isFloat) {
                            fastSet(this->data, this->cols, this->isFloat, i, j, sum);
                        } else {
                            fastSet(this->data, this->cols, false, i, j, sum);
                        }
                        break;
                    case mapFuncs::SUB:
                    case mapFuncs::ADD:
                    case mapFuncs::SUBR:
                    case mapFuncs::ADDR:
                    case mapFuncs::HADAMARD:
                    case mapFuncs::FIXED_HADAMARD:
                        // o2_copy decision hoisted before the loop (see above)
                        assert(o2_copy && o2_copy->rows == scratchNib->rows && o2_copy->cols == scratchNib->cols);
                    default:
                        TinyMatrix* fromMatrix = (o2_copy != nullptr ? o2_copy : scratchNib);
                        if((mFuncs_t[n] & NEEDS_COPY) && (mFuncs_t[n] & NEEDS_OTHER))
                            val = fastGet(fromMatrix->data, fromMatrix->cols, other_snapshot_is_float, i, j);

                        // Fast direct access using captured snapshot isFloat for the "this" side
                        float this_val = fastGet(scratchNib->data, scratchNib->cols, scratch_snapshot_is_float, i, j);

                        float result;
                        if (n == mapFuncs::ADD || n == mapFuncs::ADDR) {
                            result = val + this_val;
                        } else if (n == mapFuncs::SUB) {
                            result = this_val - val;
                        } else if (n == mapFuncs::SUBR) {
                            result = val - this_val;
                        } else if (n == mapFuncs::HADAMARD) {
                            result = val * this_val;
                        } else if (n == mapFuncs::FIXED_HADAMARD) {
                            result = foo(val, (float)i, j, scratchNib);  // keep fixed-point integer math
                        } else {
                            result = foo(val, (float)i, j, scratchNib);
                        }

                        bool write_as_float = this->isFloat || fromMatrix->isFloat && n != mapFuncs::INTS;
                        if (write_as_float) {
                            this->isFloat = true;
                        }
                        fastSet(this->data, this->cols, this->isFloat, i, j, result);
                        break;
                    }
                } else {
                    switch(n) {
                        // --- SCALAR MATH ---
                    case mapFuncs::ADDS:
                    case mapFuncs::SUBS:
                    case mapFuncs::MULS:
                    case mapFuncs::MULFS:
                        if(this->isFloat || (mFuncs_t[n] & AS_FLOAT)) {
                            if (mFuncs_t[n] & AS_FLOAT) {
                                this->isFloat = true;
                            }
                            float r = foo(val, *(float*)o2_ret, 0, nullptr);
                            fastSet(this->data, this->cols, this->isFloat, i, j, r);
                        } else {
                            float r = foo(val, (int16_t)(*(float*)o2_ret), 0, nullptr);
                            fastSet(this->data, this->cols, false, i, j, r);
                        }
                        break;

                        // --- NEURAL NETWORK ACTIVATIONS ---
                    case mapFuncs::RELU:
                    case mapFuncs::SIGMOID:
                    case mapFuncs::TANH:
                    case mapFuncs::D_RELU:
                    case mapFuncs::D_SIGMOID:
                    case mapFuncs::D_TANH:
                    case mapFuncs::FIXED_TANH:
                        if(this->isFloat || (mFuncs_t[n] & AS_FLOAT)) {
                            float r = foo(val, 0.0f, 0, nullptr);
                            fastSet(this->data, this->cols, this->isFloat, i, j, r);
                        } else {
                            float r = foo(val, 0.0f, 0, nullptr);
                            fastSet(this->data, this->cols, false, i, j, r);
                        }
                        break;

                    default:
                        break;
                    }
                }
            } else {
                if(n == mapFuncs::PRINT)
                    foo(val, (float)(this->rows - i - 1), (this->cols - j - 1), this);

                if(n == mapFuncs::SUM)
                    *(nRet*)o2_ret += val;
            }
        }
    }

}

unsigned char* TinyMatrix::operator[](const int p) {
    //return &this->data[(p)];
    return &this->data[(p * 2)];
}

// =====================================================================
// Capacity-aware scratch register helpers (allocation savings)
// =====================================================================

void TinyMatrix::_ensureScratchCapacity(TinyMatrix*& reg, size_t& capacity, int rows, int cols) {
    size_t needed = size_t(makeEven(rows * cols)) * 2;

    if (reg == nullptr || needed > capacity) {
        delete reg;
        reg = new TinyMatrix(rows, cols);
        capacity = reg->size;
    } else {
        // Buffer is big enough — just update logical dimensions, no allocation
        reg->rows = rows;
        reg->cols = cols;
        reg->size = (int)needed;
    }
}

void TinyMatrix::_snapshotToRegister(TinyMatrix*& dest, const TinyMatrix* src, size_t& cap) {
    if (!src) return;

    _ensureScratchCapacity(dest, cap, src->rows, src->cols);

    if (dest && src && dest->data && src->data && dest->size >= src->size) {
        std::memcpy(dest->data, src->data, src->size);
        dest->isFloat = src->isFloat;
    }
}

#endif