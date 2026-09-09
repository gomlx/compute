// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64

#include "textflag.h"

// ----------------------------------------------------------------------------
// Macros for horizontal vector reductions
// ----------------------------------------------------------------------------

// REDUCE_Z_F32: reduces a 16-element float32 ZMM register (z) to a single scalar float in x using scratch registers y, x, and X24.
#define REDUCE_Z_F32(z, y, x) \
	VEXTRACTF32X8 $1, z, Y24 \
	VADDPS Y24, y, y \
	VEXTRACTF32X4 $1, y, X24 \
	VADDPS X24, x, x \
	VMOVSHDUP x, X24 \
	VADDPS X24, x, x \
	VPERMILPS $0x02, x, X24 \
	VADDSS X24, x, x

// REDUCE_Z_F64: reduces an 8-element float64 ZMM register (z) to a single scalar double in x using scratch registers y, x, and X24.
#define REDUCE_Z_F64(z, y, x) \
	VEXTRACTF64X4 $1, z, Y24 \
	VADDPD Y24, y, y \
	VEXTRACTF64X2 $1, y, X24 \
	VADDPD X24, x, x \
	VPERMILPD $0x01, x, X24 \
	VADDSD X24, x, x

// REDUCE_Z_I64: reduces an 8-element int64 ZMM register (z) to a 64-bit scalar integer in general-purpose register r using scratch registers y, x, and X24.
#define REDUCE_Z_I64(z, y, x, r) \
	VEXTRACTI64X4 $1, z, Y24 \
	VPADDQ Y24, y, y \
	VEXTRACTI64X2 $1, y, X24 \
	VPADDQ X24, x, x \
	VPERMILPD $0x01, x, X24 \
	VPADDQ X24, x, x \
	VMOVQ x, r

// ----------------------------------------------------------------------------
// Float32: reduceTrailingSumFloat32AVX512(in, out unsafe.Pointer, A, B int)
// ----------------------------------------------------------------------------
TEXT ·reduceTrailingSumFloat32AVX512(SB), NOSPLIT, $0-32
	MOVQ in+0(FP), SI
	MOVQ out+8(FP), DI
	MOVQ A+16(FP), R8
	MOVQ B+24(FP), R9

	TESTQ R8, R8
	JLE done_f32
	TESTQ R9, R9
	JLE done_f32

	MOVQ R9, R10
	SHLQ $2, R10             // R10 = row stride in bytes = B * 4

	MOVQ R9, R11
	SHRQ $6, R11             // R11 = B / 64 (unroll by 4 ZMM = 64 floats)

	XORQ AX, AX

row_loop_f32:
	CMPQ AX, R8
	JGE done_f32

	VXORPS Z0, Z0, Z0
	VXORPS Z1, Z1, Z1
	VXORPS Z2, Z2, Z2
	VXORPS Z3, Z3, Z3

	MOVQ SI, BX
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail16_f32

loop64_f32:
	VADDPS (BX), Z0, Z0
	VADDPS 64(BX), Z1, Z1
	VADDPS 128(BX), Z2, Z2
	VADDPS 192(BX), Z3, Z3
	ADDQ $256, BX
	DECQ CX
	JNZ loop64_f32

tail16_f32:
	VADDPS Z2, Z0, Z0
	VADDPS Z3, Z1, Z1
	VADDPS Z1, Z0, Z0        // Z0 has accumulated 16-lane sum

	MOVQ R9, CX
	ANDQ $63, CX
	SHRQ $4, CX              // CX = (B % 64) / 16
	JZ tail_scalar_f32

loop16_f32:
	VADDPS (BX), Z0, Z0
	ADDQ $64, BX
	DECQ CX
	JNZ loop16_f32

tail_scalar_f32:
	REDUCE_Z_F32(Z0, Y0, X0)

	MOVQ R9, CX
	ANDQ $15, CX
	JZ store_f32

scalar_loop_f32:
	VADDSS (BX), X0, X0
	ADDQ $4, BX
	DECQ CX
	JNZ scalar_loop_f32

store_f32:
	VMOVSS X0, (DI)
	ADDQ $4, DI
	ADDQ R10, SI
	INCQ AX
	JMP row_loop_f32

done_f32:
	VZEROUPPER
	RET

// ----------------------------------------------------------------------------
// Float64: reduceTrailingSumFloat64AVX512(in, out unsafe.Pointer, A, B int)
// ----------------------------------------------------------------------------
TEXT ·reduceTrailingSumFloat64AVX512(SB), NOSPLIT, $0-32
	MOVQ in+0(FP), SI
	MOVQ out+8(FP), DI
	MOVQ A+16(FP), R8
	MOVQ B+24(FP), R9

	TESTQ R8, R8
	JLE done_f64
	TESTQ R9, R9
	JLE done_f64

	MOVQ R9, R10
	SHLQ $3, R10             // R10 = row stride in bytes = B * 8

	MOVQ R9, R11
	SHRQ $5, R11             // R11 = B / 32 (unroll by 4 ZMM = 32 doubles)

	XORQ AX, AX

row_loop_f64:
	CMPQ AX, R8
	JGE done_f64

	VXORPD Z0, Z0, Z0
	VXORPD Z1, Z1, Z1
	VXORPD Z2, Z2, Z2
	VXORPD Z3, Z3, Z3

	MOVQ SI, BX
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail8_f64

loop32_f64:
	VADDPD (BX), Z0, Z0
	VADDPD 64(BX), Z1, Z1
	VADDPD 128(BX), Z2, Z2
	VADDPD 192(BX), Z3, Z3
	ADDQ $256, BX
	DECQ CX
	JNZ loop32_f64

tail8_f64:
	VADDPD Z2, Z0, Z0
	VADDPD Z3, Z1, Z1
	VADDPD Z1, Z0, Z0

	MOVQ R9, CX
	ANDQ $31, CX
	SHRQ $3, CX              // (B % 32) / 8
	JZ tail_scalar_f64

loop8_f64:
	VADDPD (BX), Z0, Z0
	ADDQ $64, BX
	DECQ CX
	JNZ loop8_f64

tail_scalar_f64:
	REDUCE_Z_F64(Z0, Y0, X0)

	MOVQ R9, CX
	ANDQ $7, CX
	JZ store_f64

scalar_loop_f64:
	VADDSD (BX), X0, X0
	ADDQ $8, BX
	DECQ CX
	JNZ scalar_loop_f64

store_f64:
	VMOVSD X0, (DI)
	ADDQ $8, DI
	ADDQ R10, SI
	INCQ AX
	JMP row_loop_f64

done_f64:
	VZEROUPPER
	RET

// ----------------------------------------------------------------------------
// Float16: reduceTrailingSumFloat16AVX512(in, out unsafe.Pointer, A, B int)
// VCVTPH2PS (BX), Z4 converts 16 FP16 (32 bytes) to 16 FP32 (64 bytes in ZMM)
// ----------------------------------------------------------------------------
TEXT ·reduceTrailingSumFloat16AVX512(SB), NOSPLIT, $0-32
	MOVQ in+0(FP), SI
	MOVQ out+8(FP), DI
	MOVQ A+16(FP), R8
	MOVQ B+24(FP), R9

	TESTQ R8, R8
	JLE done_f16
	TESTQ R9, R9
	JLE done_f16

	MOVQ R9, R10
	SHLQ $1, R10             // R10 = row stride in bytes = B * 2

	MOVQ R9, R11
	SHRQ $6, R11             // R11 = B / 64

	XORQ AX, AX

row_loop_f16:
	CMPQ AX, R8
	JGE done_f16

	VXORPS Z0, Z0, Z0
	VXORPS Z1, Z1, Z1
	VXORPS Z2, Z2, Z2
	VXORPS Z3, Z3, Z3

	MOVQ SI, BX
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail16_f16

loop64_f16:
	VCVTPH2PS (BX), Z4
	VCVTPH2PS 32(BX), Z5
	VCVTPH2PS 64(BX), Z6
	VCVTPH2PS 96(BX), Z7
	VADDPS Z4, Z0, Z0
	VADDPS Z5, Z1, Z1
	VADDPS Z6, Z2, Z2
	VADDPS Z7, Z3, Z3
	ADDQ $128, BX
	DECQ CX
	JNZ loop64_f16

tail16_f16:
	VADDPS Z2, Z0, Z0
	VADDPS Z3, Z1, Z1
	VADDPS Z1, Z0, Z0

	MOVQ R9, CX
	ANDQ $63, CX
	SHRQ $4, CX              // (B % 64) / 16
	JZ tail_scalar_f16

loop16_f16:
	VCVTPH2PS (BX), Z4
	VADDPS Z4, Z0, Z0
	ADDQ $32, BX
	DECQ CX
	JNZ loop16_f16

tail_scalar_f16:
	REDUCE_Z_F32(Z0, Y0, X0) // X0 has float32 sum

	MOVQ R9, CX
	ANDQ $15, CX
	JZ store_f16

scalar_loop_f16:
	MOVWLZX (BX), DX
	VMOVD DX, X2
	VCVTPH2PS X2, X2
	VADDSS X2, X0, X0
	ADDQ $2, BX
	DECQ CX
	JNZ scalar_loop_f16

store_f16:
	VCVTPS2PH $0, X0, X1
	VMOVD X1, DX
	MOVW DX, (DI)
	ADDQ $2, DI
	ADDQ R10, SI
	INCQ AX
	JMP row_loop_f16

done_f16:
	VZEROUPPER
	RET

// ----------------------------------------------------------------------------
// BFloat16: reduceTrailingSumBFloat16AVX512(in, out unsafe.Pointer, A, B int)
// VPMOVZXWD (BX), Z4 + VPSLLD $16, Z4, Z4 converts 16 BF16 to 16 FP32 in ZMM
// ----------------------------------------------------------------------------
TEXT ·reduceTrailingSumBFloat16AVX512(SB), NOSPLIT, $0-32
	MOVQ in+0(FP), SI
	MOVQ out+8(FP), DI
	MOVQ A+16(FP), R8
	MOVQ B+24(FP), R9

	TESTQ R8, R8
	JLE done_bf16
	TESTQ R9, R9
	JLE done_bf16

	MOVQ R9, R10
	SHLQ $1, R10

	MOVQ R9, R11
	SHRQ $6, R11             // B / 64

	XORQ AX, AX

row_loop_bf16:
	CMPQ AX, R8
	JGE done_bf16

	VXORPS Z0, Z0, Z0
	VXORPS Z1, Z1, Z1
	VXORPS Z2, Z2, Z2
	VXORPS Z3, Z3, Z3

	MOVQ SI, BX
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail16_bf16

loop64_bf16:
	VPMOVZXWD (BX), Z4
	VPSLLD $16, Z4, Z4
	VPMOVZXWD 32(BX), Z5
	VPSLLD $16, Z5, Z5
	VPMOVZXWD 64(BX), Z6
	VPSLLD $16, Z6, Z6
	VPMOVZXWD 96(BX), Z7
	VPSLLD $16, Z7, Z7

	VADDPS Z4, Z0, Z0
	VADDPS Z5, Z1, Z1
	VADDPS Z6, Z2, Z2
	VADDPS Z7, Z3, Z3
	ADDQ $128, BX
	DECQ CX
	JNZ loop64_bf16

tail16_bf16:
	VADDPS Z2, Z0, Z0
	VADDPS Z3, Z1, Z1
	VADDPS Z1, Z0, Z0

	MOVQ R9, CX
	ANDQ $63, CX
	SHRQ $4, CX              // (B % 64) / 16
	JZ tail_scalar_bf16

loop16_bf16:
	VPMOVZXWD (BX), Z4
	VPSLLD $16, Z4, Z4
	VADDPS Z4, Z0, Z0
	ADDQ $32, BX
	DECQ CX
	JNZ loop16_bf16

tail_scalar_bf16:
	REDUCE_Z_F32(Z0, Y0, X0)

	MOVQ R9, CX
	ANDQ $15, CX
	JZ store_bf16

scalar_loop_bf16:
	MOVWLZX (BX), DX
	SHLL $16, DX
	VMOVD DX, X2
	VADDSS X2, X0, X0
	ADDQ $2, BX
	DECQ CX
	JNZ scalar_loop_bf16

store_bf16:
	VMOVD X0, DX
	MOVL DX, R12
	SHRL $16, R12
	ANDL $1, R12
	ADDL $0x7FFF, R12
	ADDL R12, DX
	SHRL $16, DX
	MOVW DX, (DI)
	ADDQ $2, DI
	ADDQ R10, SI
	INCQ AX
	JMP row_loop_bf16

done_bf16:
	VZEROUPPER
	RET

// ----------------------------------------------------------------------------
// Int32: reduceTrailingSumInt32AVX512(in, out unsafe.Pointer, A, B int)
// ----------------------------------------------------------------------------
TEXT ·reduceTrailingSumInt32AVX512(SB), NOSPLIT, $0-32
	MOVQ in+0(FP), SI
	MOVQ out+8(FP), DI
	MOVQ A+16(FP), R8
	MOVQ B+24(FP), R9

	TESTQ R8, R8
	JLE done_i32
	TESTQ R9, R9
	JLE done_i32

	MOVQ R9, R10
	SHLQ $2, R10

	MOVQ R9, R11
	SHRQ $6, R11             // B / 64

	XORQ AX, AX

row_loop_i32:
	CMPQ AX, R8
	JGE done_i32

	VPXORD Z0, Z0, Z0
	VPXORD Z1, Z1, Z1
	VPXORD Z2, Z2, Z2
	VPXORD Z3, Z3, Z3

	MOVQ SI, BX
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail16_i32

loop64_i32:
	VPADDD (BX), Z0, Z0
	VPADDD 64(BX), Z1, Z1
	VPADDD 128(BX), Z2, Z2
	VPADDD 192(BX), Z3, Z3
	ADDQ $256, BX
	DECQ CX
	JNZ loop64_i32

tail16_i32:
	VPADDD Z2, Z0, Z0
	VPADDD Z3, Z1, Z1
	VPADDD Z1, Z0, Z0

	MOVQ R9, CX
	ANDQ $63, CX
	SHRQ $4, CX
	JZ tail_scalar_i32

loop16_i32:
	VPADDD (BX), Z0, Z0
	ADDQ $64, BX
	DECQ CX
	JNZ loop16_i32

tail_scalar_i32:
	// Horizontal sum of Z0 (16 int32s) -> DX
	VEXTRACTI32X8 $1, Z0, Y1
	VPADDD Y1, Y0, Y0        // 8 int32s in Y0
	VEXTRACTI128 $1, Y0, X1
	VPADDD X1, X0, X0        // 4 int32s in X0
	VPSHUFD $0x4E, X0, X1
	VPADDD X1, X0, X0
	VPSHUFD $0xE5, X0, X1
	VPADDD X1, X0, X0
	VMOVD X0, DX

	MOVQ R9, CX
	ANDQ $15, CX
	JZ store_i32

scalar_loop_i32:
	ADDL (BX), DX
	ADDQ $4, BX
	DECQ CX
	JNZ scalar_loop_i32

store_i32:
	MOVL DX, (DI)
	ADDQ $4, DI
	ADDQ R10, SI
	INCQ AX
	JMP row_loop_i32

done_i32:
	VZEROUPPER
	RET

// ----------------------------------------------------------------------------
// Uint32: reduceTrailingSumUint32AVX512(in, out unsafe.Pointer, A, B int)
// ----------------------------------------------------------------------------
TEXT ·reduceTrailingSumUint32AVX512(SB), NOSPLIT, $0-32
	JMP ·reduceTrailingSumInt32AVX512(SB)

// ----------------------------------------------------------------------------
// Int16: reduceTrailingSumInt16AVX512(in, out unsafe.Pointer, A, B int)
// ----------------------------------------------------------------------------
TEXT ·reduceTrailingSumInt16AVX512(SB), NOSPLIT, $0-32
	MOVQ in+0(FP), SI
	MOVQ out+8(FP), DI
	MOVQ A+16(FP), R8
	MOVQ B+24(FP), R9

	TESTQ R8, R8
	JLE done_i16
	TESTQ R9, R9
	JLE done_i16

	MOVQ R9, R10
	SHLQ $1, R10

	MOVQ R9, R11
	SHRQ $7, R11             // B / 128 (unroll by 4 ZMM = 128 int16)

	XORQ AX, AX

row_loop_i16:
	CMPQ AX, R8
	JGE done_i16

	VPXORD Z0, Z0, Z0
	VPXORD Z1, Z1, Z1
	VPXORD Z2, Z2, Z2
	VPXORD Z3, Z3, Z3

	MOVQ SI, BX
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail32_i16

loop128_i16:
	VPADDW (BX), Z0, Z0
	VPADDW 64(BX), Z1, Z1
	VPADDW 128(BX), Z2, Z2
	VPADDW 192(BX), Z3, Z3
	ADDQ $256, BX
	DECQ CX
	JNZ loop128_i16

tail32_i16:
	VPADDW Z2, Z0, Z0
	VPADDW Z3, Z1, Z1
	VPADDW Z1, Z0, Z0

	MOVQ R9, CX
	ANDQ $127, CX
	SHRQ $5, CX              // (B % 128) / 32
	JZ tail_scalar_i16

loop32_i16:
	VPADDW (BX), Z0, Z0
	ADDQ $64, BX
	DECQ CX
	JNZ loop32_i16

tail_scalar_i16:
	// Horizontal sum of Z0 (32 int16s) -> DX
	VEXTRACTI32X8 $1, Z0, Y1
	VPADDW Y1, Y0, Y0        // 16 int16s in Y0
	VEXTRACTI128 $1, Y0, X1
	VPADDW X1, X0, X0        // 8 int16s in X0
	VPSHUFD $0x4E, X0, X1
	VPADDW X1, X0, X0
	VPSHUFD $0x01, X0, X1
	VPADDW X1, X0, X0
	VPSRLDQ $2, X0, X1
	VPADDW X1, X0, X0
	VMOVD X0, DX
	MOVW DX, DX

	MOVQ R9, CX
	ANDQ $31, CX
	JZ store_i16

scalar_loop_i16:
	ADDW (BX), DX
	ADDQ $2, BX
	DECQ CX
	JNZ scalar_loop_i16

store_i16:
	MOVW DX, (DI)
	ADDQ $2, DI
	ADDQ R10, SI
	INCQ AX
	JMP row_loop_i16

done_i16:
	VZEROUPPER
	RET

// ----------------------------------------------------------------------------
// Uint16: reduceTrailingSumUint16AVX512(in, out unsafe.Pointer, A, B int)
// ----------------------------------------------------------------------------
TEXT ·reduceTrailingSumUint16AVX512(SB), NOSPLIT, $0-32
	JMP ·reduceTrailingSumInt16AVX512(SB)

// ----------------------------------------------------------------------------
// Int8: reduceTrailingSumInt8AVX512(in, out unsafe.Pointer, A, B int)
// ----------------------------------------------------------------------------
TEXT ·reduceTrailingSumInt8AVX512(SB), NOSPLIT, $0-32
	MOVQ in+0(FP), SI
	MOVQ out+8(FP), DI
	MOVQ A+16(FP), R8
	MOVQ B+24(FP), R9

	TESTQ R8, R8
	JLE done_i8
	TESTQ R9, R9
	JLE done_i8

	MOVQ R9, R10             // row stride = B

	MOVQ R9, R11
	SHRQ $8, R11             // B / 256 (unroll by 4 ZMM = 256 int8)

	XORQ AX, AX

row_loop_i8:
	CMPQ AX, R8
	JGE done_i8

	VPXORD Z0, Z0, Z0
	VPXORD Z1, Z1, Z1
	VPXORD Z2, Z2, Z2
	VPXORD Z3, Z3, Z3

	MOVQ SI, BX
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail64_i8

loop256_i8:
	VPADDB (BX), Z0, Z0
	VPADDB 64(BX), Z1, Z1
	VPADDB 128(BX), Z2, Z2
	VPADDB 192(BX), Z3, Z3
	ADDQ $256, BX
	DECQ CX
	JNZ loop256_i8

tail64_i8:
	VPADDB Z2, Z0, Z0
	VPADDB Z3, Z1, Z1
	VPADDB Z1, Z0, Z0

	MOVQ R9, CX
	ANDQ $255, CX
	SHRQ $6, CX              // (B % 256) / 64
	JZ tail_scalar_i8

loop64_i8:
	VPADDB (BX), Z0, Z0
	ADDQ $64, BX
	DECQ CX
	JNZ loop64_i8

tail_scalar_i8:
	// Horizontal sum of Z0 (64 int8s) -> DX
	VEXTRACTI32X8 $1, Z0, Y1
	VPADDB Y1, Y0, Y0        // 32 int8s in Y0
	VEXTRACTI128 $1, Y0, X1
	VPADDB X1, X0, X0        // 16 int8s in X0
	VPSRLDQ $8, X0, X1
	VPADDB X1, X0, X0
	VPSRLDQ $4, X0, X1
	VPADDB X1, X0, X0
	VPSRLDQ $2, X0, X1
	VPADDB X1, X0, X0
	VPSRLDQ $1, X0, X1
	VPADDB X1, X0, X0
	VMOVD X0, DX
	MOVB DX, DX

	MOVQ R9, CX
	ANDQ $63, CX
	JZ store_i8

scalar_loop_i8:
	ADDB (BX), DX
	INCQ BX
	DECQ CX
	JNZ scalar_loop_i8

store_i8:
	MOVB DX, (DI)
	INCQ DI
	ADDQ R10, SI
	INCQ AX
	JMP row_loop_i8

done_i8:
	VZEROUPPER
	RET

// ----------------------------------------------------------------------------
// Uint8: reduceTrailingSumUint8AVX512(in, out unsafe.Pointer, A, B int)
// ----------------------------------------------------------------------------
TEXT ·reduceTrailingSumUint8AVX512(SB), NOSPLIT, $0-32
	JMP ·reduceTrailingSumInt8AVX512(SB)

// ----------------------------------------------------------------------------
// Int64: reduceTrailingSumInt64AVX512(in, out unsafe.Pointer, A, B int)
// ----------------------------------------------------------------------------
TEXT ·reduceTrailingSumInt64AVX512(SB), NOSPLIT, $0-32
	MOVQ in+0(FP), SI
	MOVQ out+8(FP), DI
	MOVQ A+16(FP), R8
	MOVQ B+24(FP), R9

	TESTQ R8, R8
	JLE done_i64
	TESTQ R9, R9
	JLE done_i64

	MOVQ R9, R10
	SHLQ $3, R10             // R10 = row stride in bytes = B * 8

	MOVQ R9, R11
	SHRQ $5, R11             // R11 = B / 32 (unroll by 4 ZMM = 32 int64s)

	XORQ AX, AX

row_loop_i64:
	CMPQ AX, R8
	JGE done_i64

	VPXORD Z0, Z0, Z0
	VPXORD Z1, Z1, Z1
	VPXORD Z2, Z2, Z2
	VPXORD Z3, Z3, Z3

	MOVQ SI, BX
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail8_i64

loop32_i64:
	VPADDQ (BX), Z0, Z0
	VPADDQ 64(BX), Z1, Z1
	VPADDQ 128(BX), Z2, Z2
	VPADDQ 192(BX), Z3, Z3
	ADDQ $256, BX
	DECQ CX
	JNZ loop32_i64

tail8_i64:
	VPADDQ Z2, Z0, Z0
	VPADDQ Z3, Z1, Z1
	VPADDQ Z1, Z0, Z0

	MOVQ R9, CX
	ANDQ $31, CX
	SHRQ $3, CX              // (B % 32) / 8
	JZ tail_scalar_i64

loop8_i64:
	VPADDQ (BX), Z0, Z0
	ADDQ $64, BX
	DECQ CX
	JNZ loop8_i64

tail_scalar_i64:
	REDUCE_Z_I64(Z0, Y0, X0, R12)

	MOVQ R9, CX
	ANDQ $7, CX
	JZ store_i64

scalar_loop_i64:
	ADDQ (BX), R12
	ADDQ $8, BX
	DECQ CX
	JNZ scalar_loop_i64

store_i64:
	MOVQ R12, (DI)
	ADDQ $8, DI
	ADDQ R10, SI
	INCQ AX
	JMP row_loop_i64

done_i64:
	VZEROUPPER
	RET

// ----------------------------------------------------------------------------
// Uint64: reduceTrailingSumUint64AVX512(in, out unsafe.Pointer, A, B int)
// ----------------------------------------------------------------------------
TEXT ·reduceTrailingSumUint64AVX512(SB), NOSPLIT, $0-32
	JMP ·reduceTrailingSumInt64AVX512(SB)
