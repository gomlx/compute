// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64

#include "textflag.h"

// ----------------------------------------------------------------------------
// Float32: reduceTrailingSumFloat32AVX2(in, out unsafe.Pointer, A, B int)
// ----------------------------------------------------------------------------
TEXT ·reduceTrailingSumFloat32AVX2(SB), NOSPLIT, $0-32
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
	SHRQ $5, R11             // R11 = B / 32 (unroll by 4 YMM)

	XORQ AX, AX              // AX = row idx = 0

row_loop_f32:
	CMPQ AX, R8
	JGE done_f32

	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3

	MOVQ SI, BX              // BX = row ptr
	MOVQ R11, CX             // CX = count of 32-element chunks
	TESTQ CX, CX
	JZ tail8_f32

loop32_f32:
	VADDPS (BX), Y0, Y0
	VADDPS 32(BX), Y1, Y1
	VADDPS 64(BX), Y2, Y2
	VADDPS 96(BX), Y3, Y3
	ADDQ $128, BX
	DECQ CX
	JNZ loop32_f32

tail8_f32:
	VADDPS Y2, Y0, Y0
	VADDPS Y3, Y1, Y1
	VADDPS Y1, Y0, Y0        // Y0 has accumulated 8-lane sum

	// Remaining 8-float chunks
	MOVQ R9, CX
	ANDQ $31, CX             // CX = B % 32
	SHRQ $3, CX              // CX = (B % 32) / 8
	JZ tail_scalar_f32

loop8_f32:
	VADDPS (BX), Y0, Y0
	ADDQ $32, BX
	DECQ CX
	JNZ loop8_f32

tail_scalar_f32:
	// Horizontal sum of Y0 -> X0 scalar
	VEXTRACTF128 $1, Y0, X1  // upper 128 bits
	VADDPS X1, X0, X0        // 4 floats
	VPERMILPS $0xEE, X0, X1
	VADDPS X1, X0, X0        // 2 floats
	VPERMILPS $0x01, X0, X1
	VADDSS X1, X0, X0        // 1 scalar float in X0

	// Remaining 0..7 scalar elements
	MOVQ R9, CX
	ANDQ $7, CX
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
// Float64: reduceTrailingSumFloat64AVX2(in, out unsafe.Pointer, A, B int)
// ----------------------------------------------------------------------------
TEXT ·reduceTrailingSumFloat64AVX2(SB), NOSPLIT, $0-32
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
	SHRQ $4, R11             // R11 = B / 16 (unroll by 4 YMM = 16 float64)

	XORQ AX, AX

row_loop_f64:
	CMPQ AX, R8
	JGE done_f64

	VXORPD Y0, Y0, Y0
	VXORPD Y1, Y1, Y1
	VXORPD Y2, Y2, Y2
	VXORPD Y3, Y3, Y3

	MOVQ SI, BX
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail4_f64

loop16_f64:
	VADDPD (BX), Y0, Y0
	VADDPD 32(BX), Y1, Y1
	VADDPD 64(BX), Y2, Y2
	VADDPD 96(BX), Y3, Y3
	ADDQ $128, BX
	DECQ CX
	JNZ loop16_f64

tail4_f64:
	VADDPD Y2, Y0, Y0
	VADDPD Y3, Y1, Y1
	VADDPD Y1, Y0, Y0

	MOVQ R9, CX
	ANDQ $15, CX
	SHRQ $2, CX              // CX = (B % 16) / 4
	JZ tail_scalar_f64

loop4_f64:
	VADDPD (BX), Y0, Y0
	ADDQ $32, BX
	DECQ CX
	JNZ loop4_f64

tail_scalar_f64:
	// Horizontal sum of Y0 -> X0
	VEXTRACTF128 $1, Y0, X1
	VADDPD X1, X0, X0        // 2 doubles in X0
	VUNPCKHPD X0, X0, X1     // high double
	VADDSD X1, X0, X0        // 1 scalar double in X0

	MOVQ R9, CX
	ANDQ $3, CX
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
// Float16: reduceTrailingSumFloat16AVX2(in, out unsafe.Pointer, A, B int)
// Input is Float16 (2 bytes each), converted to Float32 via VCVTPH2PS, summed in FP32, converted back via VCVTPS2PH
// ----------------------------------------------------------------------------
TEXT ·reduceTrailingSumFloat16AVX2(SB), NOSPLIT, $0-32
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
	SHRQ $5, R11             // R11 = B / 32

	XORQ AX, AX

row_loop_f16:
	CMPQ AX, R8
	JGE done_f16

	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3

	MOVQ SI, BX
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail8_f16

loop32_f16:
	VCVTPH2PS (BX), Y4
	VCVTPH2PS 16(BX), Y5
	VCVTPH2PS 32(BX), Y6
	VCVTPH2PS 48(BX), Y7
	VADDPS Y4, Y0, Y0
	VADDPS Y5, Y1, Y1
	VADDPS Y6, Y2, Y2
	VADDPS Y7, Y3, Y3
	ADDQ $64, BX
	DECQ CX
	JNZ loop32_f16

tail8_f16:
	VADDPS Y2, Y0, Y0
	VADDPS Y3, Y1, Y1
	VADDPS Y1, Y0, Y0

	MOVQ R9, CX
	ANDQ $31, CX
	SHRQ $3, CX              // (B % 32) / 8
	JZ tail_scalar_f16

loop8_f16:
	VCVTPH2PS (BX), Y4
	VADDPS Y4, Y0, Y0
	ADDQ $16, BX
	DECQ CX
	JNZ loop8_f16

tail_scalar_f16:
	// Horizontal sum of Y0 -> X0
	VEXTRACTF128 $1, Y0, X1
	VADDPS X1, X0, X0
	VPERMILPS $0xEE, X0, X1
	VADDPS X1, X0, X0
	VPERMILPS $0x01, X0, X1
	VADDSS X1, X0, X0        // X0 has accumulated sum in float32

	MOVQ R9, CX
	ANDQ $7, CX
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
	VCVTPS2PH $0, X0, X1     // round to nearest even
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
// BFloat16: reduceTrailingSumBFloat16AVX2(in, out unsafe.Pointer, A, B int)
// Input is BFloat16 (2 bytes each), converted to Float32 via VPMOVZXWD + VPSLLD $16, summed in FP32
// BFloat16 conversion back: add rounding bias (0x7FFF + bit 16) and shift right 16
// ----------------------------------------------------------------------------
TEXT ·reduceTrailingSumBFloat16AVX2(SB), NOSPLIT, $0-32
	MOVQ in+0(FP), SI
	MOVQ out+8(FP), DI
	MOVQ A+16(FP), R8
	MOVQ B+24(FP), R9

	TESTQ R8, R8
	JLE done_bf16
	TESTQ R9, R9
	JLE done_bf16

	MOVQ R9, R10
	SHLQ $1, R10             // R10 = row stride in bytes = B * 2

	MOVQ R9, R11
	SHRQ $5, R11             // R11 = B / 32

	XORQ AX, AX

row_loop_bf16:
	CMPQ AX, R8
	JGE done_bf16

	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3

	MOVQ SI, BX
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail8_bf16

loop32_bf16:
	VPMOVZXWD (BX), Y4
	VPSLLD $16, Y4, Y4
	VPMOVZXWD 16(BX), Y5
	VPSLLD $16, Y5, Y5
	VPMOVZXWD 32(BX), Y6
	VPSLLD $16, Y6, Y6
	VPMOVZXWD 48(BX), Y7
	VPSLLD $16, Y7, Y7

	VADDPS Y4, Y0, Y0
	VADDPS Y5, Y1, Y1
	VADDPS Y6, Y2, Y2
	VADDPS Y7, Y3, Y3
	ADDQ $64, BX
	DECQ CX
	JNZ loop32_bf16

tail8_bf16:
	VADDPS Y2, Y0, Y0
	VADDPS Y3, Y1, Y1
	VADDPS Y1, Y0, Y0

	MOVQ R9, CX
	ANDQ $31, CX
	SHRQ $3, CX
	JZ tail_scalar_bf16

loop8_bf16:
	VPMOVZXWD (BX), Y4
	VPSLLD $16, Y4, Y4
	VADDPS Y4, Y0, Y0
	ADDQ $16, BX
	DECQ CX
	JNZ loop8_bf16

tail_scalar_bf16:
	// Horizontal sum of Y0 -> X0
	VEXTRACTF128 $1, Y0, X1
	VADDPS X1, X0, X0
	VPERMILPS $0xEE, X0, X1
	VADDPS X1, X0, X0
	VPERMILPS $0x01, X0, X1
	VADDSS X1, X0, X0        // X0 has float32 sum

	MOVQ R9, CX
	ANDQ $7, CX
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
	// Float32 in X0 -> BFloat16 with round-to-nearest-even
	VMOVD X0, DX             // DX = uint32 bits of float32
	MOVL DX, R12
	SHRL $16, R12
	ANDL $1, R12             // R12 = lsb
	ADDL $0x7FFF, R12        // R12 = bias
	ADDL R12, DX             // DX += bias
	SHRL $16, DX             // DX = uint16 bf16
	MOVW DX, (DI)
	ADDQ $2, DI
	ADDQ R10, SI
	INCQ AX
	JMP row_loop_bf16

done_bf16:
	VZEROUPPER
	RET

// ----------------------------------------------------------------------------
// Int32: reduceTrailingSumInt32AVX2(in, out unsafe.Pointer, A, B int)
// ----------------------------------------------------------------------------
TEXT ·reduceTrailingSumInt32AVX2(SB), NOSPLIT, $0-32
	MOVQ in+0(FP), SI
	MOVQ out+8(FP), DI
	MOVQ A+16(FP), R8
	MOVQ B+24(FP), R9

	TESTQ R8, R8
	JLE done_i32
	TESTQ R9, R9
	JLE done_i32

	MOVQ R9, R10
	SHLQ $2, R10             // row stride in bytes = B * 4

	MOVQ R9, R11
	SHRQ $5, R11             // B / 32

	XORQ AX, AX

row_loop_i32:
	CMPQ AX, R8
	JGE done_i32

	VPXOR Y0, Y0, Y0
	VPXOR Y1, Y1, Y1
	VPXOR Y2, Y2, Y2
	VPXOR Y3, Y3, Y3

	MOVQ SI, BX
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail8_i32

loop32_i32:
	VPADDD (BX), Y0, Y0
	VPADDD 32(BX), Y1, Y1
	VPADDD 64(BX), Y2, Y2
	VPADDD 96(BX), Y3, Y3
	ADDQ $128, BX
	DECQ CX
	JNZ loop32_i32

tail8_i32:
	VPADDD Y2, Y0, Y0
	VPADDD Y3, Y1, Y1
	VPADDD Y1, Y0, Y0

	MOVQ R9, CX
	ANDQ $31, CX
	SHRQ $3, CX
	JZ tail_scalar_i32

loop8_i32:
	VPADDD (BX), Y0, Y0
	ADDQ $32, BX
	DECQ CX
	JNZ loop8_i32

tail_scalar_i32:
	// Horizontal sum of Y0 (8 int32s) -> DX
	VEXTRACTI128 $1, Y0, X1
	VPADDD X1, X0, X0        // 4 int32s in X0
	VPSHUFD $0x4E, X0, X1    // swap pairs (2 and 3 with 0 and 1)
	VPADDD X1, X0, X0
	VPSHUFD $0xE5, X0, X1    // element 1 to 0
	VPADDD X1, X0, X0
	VMOVD X0, DX             // DX = scalar sum

	MOVQ R9, CX
	ANDQ $7, CX
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
// Uint32: reduceTrailingSumUint32AVX2(in, out unsafe.Pointer, A, B int)
// ----------------------------------------------------------------------------
TEXT ·reduceTrailingSumUint32AVX2(SB), NOSPLIT, $0-32
	JMP ·reduceTrailingSumInt32AVX2(SB)

// ----------------------------------------------------------------------------
// Int16: reduceTrailingSumInt16AVX2(in, out unsafe.Pointer, A, B int)
// ----------------------------------------------------------------------------
TEXT ·reduceTrailingSumInt16AVX2(SB), NOSPLIT, $0-32
	MOVQ in+0(FP), SI
	MOVQ out+8(FP), DI
	MOVQ A+16(FP), R8
	MOVQ B+24(FP), R9

	TESTQ R8, R8
	JLE done_i16
	TESTQ R9, R9
	JLE done_i16

	MOVQ R9, R10
	SHLQ $1, R10             // row stride in bytes = B * 2

	MOVQ R9, R11
	SHRQ $6, R11             // B / 64 (unroll by 4 YMM = 64 int16)

	XORQ AX, AX

row_loop_i16:
	CMPQ AX, R8
	JGE done_i16

	VPXOR Y0, Y0, Y0
	VPXOR Y1, Y1, Y1
	VPXOR Y2, Y2, Y2
	VPXOR Y3, Y3, Y3

	MOVQ SI, BX
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail16_i16

loop64_i16:
	VPADDW (BX), Y0, Y0
	VPADDW 32(BX), Y1, Y1
	VPADDW 64(BX), Y2, Y2
	VPADDW 96(BX), Y3, Y3
	ADDQ $128, BX
	DECQ CX
	JNZ loop64_i16

tail16_i16:
	VPADDW Y2, Y0, Y0
	VPADDW Y3, Y1, Y1
	VPADDW Y1, Y0, Y0

	MOVQ R9, CX
	ANDQ $63, CX
	SHRQ $4, CX              // (B % 64) / 16
	JZ tail_scalar_i16

loop16_i16:
	VPADDW (BX), Y0, Y0
	ADDQ $32, BX
	DECQ CX
	JNZ loop16_i16

tail_scalar_i16:
	// Horizontal sum of Y0 (16 int16s) -> DX
	VEXTRACTI128 $1, Y0, X1
	VPADDW X1, X0, X0        // 8 int16s in X0
	VPSHUFD $0x4E, X0, X1
	VPADDW X1, X0, X0        // 4 int16s (lanes 0..3)
	VPSHUFD $0x01, X0, X1
	VPADDW X1, X0, X0        // 2 int16s (lanes 0..1)
	VPSRLDQ $2, X0, X1
	VPADDW X1, X0, X0
	VMOVD X0, DX
	MOVW DX, DX              // DX = int16 sum

	MOVQ R9, CX
	ANDQ $15, CX
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
// Uint16: reduceTrailingSumUint16AVX2(in, out unsafe.Pointer, A, B int)
// ----------------------------------------------------------------------------
TEXT ·reduceTrailingSumUint16AVX2(SB), NOSPLIT, $0-32
	JMP ·reduceTrailingSumInt16AVX2(SB)

// ----------------------------------------------------------------------------
// Int8: reduceTrailingSumInt8AVX2(in, out unsafe.Pointer, A, B int)
// ----------------------------------------------------------------------------
TEXT ·reduceTrailingSumInt8AVX2(SB), NOSPLIT, $0-32
	MOVQ in+0(FP), SI
	MOVQ out+8(FP), DI
	MOVQ A+16(FP), R8
	MOVQ B+24(FP), R9

	TESTQ R8, R8
	JLE done_i8
	TESTQ R9, R9
	JLE done_i8

	MOVQ R9, R10             // row stride in bytes = B

	MOVQ R9, R11
	SHRQ $7, R11             // B / 128 (unroll by 4 YMM = 128 int8)

	XORQ AX, AX

row_loop_i8:
	CMPQ AX, R8
	JGE done_i8

	VPXOR Y0, Y0, Y0
	VPXOR Y1, Y1, Y1
	VPXOR Y2, Y2, Y2
	VPXOR Y3, Y3, Y3

	MOVQ SI, BX
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail32_i8

loop128_i8:
	VPADDB (BX), Y0, Y0
	VPADDB 32(BX), Y1, Y1
	VPADDB 64(BX), Y2, Y2
	VPADDB 96(BX), Y3, Y3
	ADDQ $128, BX
	DECQ CX
	JNZ loop128_i8

tail32_i8:
	VPADDB Y2, Y0, Y0
	VPADDB Y3, Y1, Y1
	VPADDB Y1, Y0, Y0

	MOVQ R9, CX
	ANDQ $127, CX
	SHRQ $5, CX              // (B % 128) / 32
	JZ tail_scalar_i8

loop32_i8:
	VPADDB (BX), Y0, Y0
	ADDQ $32, BX
	DECQ CX
	JNZ loop32_i8

tail_scalar_i8:
	// Horizontal sum of Y0 (32 int8s) -> DX
	VEXTRACTI128 $1, Y0, X1
	VPADDB X1, X0, X0        // 16 int8s in X0
	VPSRLDQ $8, X0, X1
	VPADDB X1, X0, X0        // 8 int8s
	VPSRLDQ $4, X0, X1
	VPADDB X1, X0, X0        // 4 int8s
	VPSRLDQ $2, X0, X1
	VPADDB X1, X0, X0        // 2 int8s
	VPSRLDQ $1, X0, X1
	VPADDB X1, X0, X0        // 1 int8 in lowest byte
	VMOVD X0, DX
	MOVB DX, DX

	MOVQ R9, CX
	ANDQ $31, CX
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
// Uint8: reduceTrailingSumUint8AVX2(in, out unsafe.Pointer, A, B int)
// ----------------------------------------------------------------------------
TEXT ·reduceTrailingSumUint8AVX2(SB), NOSPLIT, $0-32
	JMP ·reduceTrailingSumInt8AVX2(SB)

// ----------------------------------------------------------------------------
// Int64: reduceTrailingSumInt64AVX2(in, out unsafe.Pointer, A, B int)
// ----------------------------------------------------------------------------
TEXT ·reduceTrailingSumInt64AVX2(SB), NOSPLIT, $0-32
	MOVQ in+0(FP), SI
	MOVQ out+8(FP), DI
	MOVQ A+16(FP), R8
	MOVQ B+24(FP), R9

	TESTQ R8, R8
	JLE done_i64
	TESTQ R9, R9
	JLE done_i64

	MOVQ R9, R10
	SHLQ $3, R10             // row stride in bytes = B * 8

	MOVQ R9, R11
	SHRQ $4, R11             // B / 16 (unroll by 4 YMM = 16 int64s)

	XORQ AX, AX

row_loop_i64:
	CMPQ AX, R8
	JGE done_i64

	VPXOR Y0, Y0, Y0
	VPXOR Y1, Y1, Y1
	VPXOR Y2, Y2, Y2
	VPXOR Y3, Y3, Y3

	MOVQ SI, BX
	MOVQ R11, CX
	TESTQ CX, CX
	JZ tail4_i64

loop16_i64:
	VPADDQ (BX), Y0, Y0
	VPADDQ 32(BX), Y1, Y1
	VPADDQ 64(BX), Y2, Y2
	VPADDQ 96(BX), Y3, Y3
	ADDQ $128, BX
	DECQ CX
	JNZ loop16_i64

tail4_i64:
	VPADDQ Y2, Y0, Y0
	VPADDQ Y3, Y1, Y1
	VPADDQ Y1, Y0, Y0

	MOVQ R9, CX
	ANDQ $15, CX
	SHRQ $2, CX              // (B % 16) / 4
	JZ tail_scalar_i64

loop4_i64:
	VPADDQ (BX), Y0, Y0
	ADDQ $32, BX
	DECQ CX
	JNZ loop4_i64

tail_scalar_i64:
	// Horizontal sum of 4 64-bit ints in Y0 -> scalar int64 in R12
	VEXTRACTI128 $1, Y0, X1
	VPADDQ X1, X0, X0
	VPERMILPD $1, X0, X1
	VPADDQ X1, X0, X0
	VMOVQ X0, R12

	MOVQ R9, CX
	ANDQ $3, CX
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
// Uint64: reduceTrailingSumUint64AVX2(in, out unsafe.Pointer, A, B int)
// ----------------------------------------------------------------------------
TEXT ·reduceTrailingSumUint64AVX2(SB), NOSPLIT, $0-32
	JMP ·reduceTrailingSumInt64AVX2(SB)
