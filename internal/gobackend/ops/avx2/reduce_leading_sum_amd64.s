// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build amd64

#include "textflag.h"

// ----------------------------------------------------------------------------
// Float32: reduceLeadingSumFloat32AVX2(in, out unsafe.Pointer, A, B int)
// ----------------------------------------------------------------------------
TEXT ·reduceLeadingSumFloat32AVX2(SB), NOSPLIT, $0-32
	MOVQ in+0(FP), SI
	MOVQ out+8(FP), DI
	MOVQ A+16(FP), R8
	MOVQ B+24(FP), R9

	TESTQ R8, R8
	JLE done_f32
	TESTQ R9, R9
	JLE done_f32

	CMPQ R8, $1
	JNE start_f32

	// A == 1: copy row 0 (B floats = B*4 bytes) to out
	MOVQ R9, CX
	SHLQ $2, CX
copy_loop_f32:
	CMPQ CX, $32
	JL copy_tail_f32
	VMOVUPS (SI), Y0
	VMOVUPS Y0, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $32, CX
	JMP copy_loop_f32
copy_tail_f32:
	TESTQ CX, CX
	JZ done_f32
	MOVB (SI), AL
	MOVB AL, (DI)
	INCQ SI
	INCQ DI
	DECQ CX
	JNZ copy_tail_f32
	JMP done_f32

start_f32:
	MOVQ R9, R10
	SHLQ $2, R10             // R10 = stride = B * 4 bytes
	MOVQ R8, R11
	DECQ R11                 // R11 = A - 1
	MOVQ R9, R12             // R12 = remaining cols

col32_loop_f32:
	CMPQ R12, $32
	JL col8_f32

	MOVQ SI, R13
	VMOVUPS (R13), Y0
	VMOVUPS 32(R13), Y1
	VMOVUPS 64(R13), Y2
	VMOVUPS 96(R13), Y3

	MOVQ R11, CX
	ADDQ R10, R13

row32_loop_f32:
	VADDPS (R13), Y0, Y0
	VADDPS 32(R13), Y1, Y1
	VADDPS 64(R13), Y2, Y2
	VADDPS 96(R13), Y3, Y3
	ADDQ R10, R13
	DECQ CX
	JNZ row32_loop_f32

	VMOVUPS Y0, (DI)
	VMOVUPS Y1, 32(DI)
	VMOVUPS Y2, 64(DI)
	VMOVUPS Y3, 96(DI)
	ADDQ $128, DI
	ADDQ $128, SI
	SUBQ $32, R12
	JMP col32_loop_f32

col8_f32:
	CMPQ R12, $8
	JL col4_f32

	MOVQ SI, R13
	VMOVUPS (R13), Y0
	MOVQ R11, CX
	ADDQ R10, R13

row8_loop_f32:
	VADDPS (R13), Y0, Y0
	ADDQ R10, R13
	DECQ CX
	JNZ row8_loop_f32

	VMOVUPS Y0, (DI)
	ADDQ $32, DI
	ADDQ $32, SI
	SUBQ $8, R12
	JMP col8_f32

col4_f32:
	CMPQ R12, $4
	JL col_scalar_f32

	MOVQ SI, R13
	VMOVUPS (R13), X0
	MOVQ R11, CX
	ADDQ R10, R13

row4_loop_f32:
	VADDPS (R13), X0, X0
	ADDQ R10, R13
	DECQ CX
	JNZ row4_loop_f32

	VMOVUPS X0, (DI)
	ADDQ $16, DI
	ADDQ $16, SI
	SUBQ $4, R12

col_scalar_f32:
	TESTQ R12, R12
	JZ done_f32

scalar_col_loop_f32:
	MOVQ SI, R13
	VMOVSS (R13), X0
	MOVQ R11, CX
	ADDQ R10, R13

scalar_row_loop_f32:
	VADDSS (R13), X0, X0
	ADDQ R10, R13
	DECQ CX
	JNZ scalar_row_loop_f32

	VMOVSS X0, (DI)
	ADDQ $4, DI
	ADDQ $4, SI
	DECQ R12
	JNZ scalar_col_loop_f32

done_f32:
	VZEROUPPER
	RET

// ----------------------------------------------------------------------------
// Float64: reduceLeadingSumFloat64AVX2(in, out unsafe.Pointer, A, B int)
// ----------------------------------------------------------------------------
TEXT ·reduceLeadingSumFloat64AVX2(SB), NOSPLIT, $0-32
	MOVQ in+0(FP), SI
	MOVQ out+8(FP), DI
	MOVQ A+16(FP), R8
	MOVQ B+24(FP), R9

	TESTQ R8, R8
	JLE done_f64
	TESTQ R9, R9
	JLE done_f64

	CMPQ R8, $1
	JNE start_f64

	// A == 1: copy row 0
	MOVQ R9, CX
	SHLQ $3, CX
copy_loop_f64:
	CMPQ CX, $32
	JL copy_tail_f64
	VMOVUPD (SI), Y0
	VMOVUPD Y0, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $32, CX
	JMP copy_loop_f64
copy_tail_f64:
	TESTQ CX, CX
	JZ done_f64
	MOVB (SI), AL
	MOVB AL, (DI)
	INCQ SI
	INCQ DI
	DECQ CX
	JNZ copy_tail_f64
	JMP done_f64

start_f64:
	MOVQ R9, R10
	SHLQ $3, R10             // R10 = stride = B * 8 bytes
	MOVQ R8, R11
	DECQ R11                 // R11 = A - 1
	MOVQ R9, R12             // R12 = remaining cols

col16_loop_f64:
	CMPQ R12, $16
	JL col4_f64

	MOVQ SI, R13
	VMOVUPD (R13), Y0
	VMOVUPD 32(R13), Y1
	VMOVUPD 64(R13), Y2
	VMOVUPD 96(R13), Y3

	MOVQ R11, CX
	ADDQ R10, R13

row16_loop_f64:
	VADDPD (R13), Y0, Y0
	VADDPD 32(R13), Y1, Y1
	VADDPD 64(R13), Y2, Y2
	VADDPD 96(R13), Y3, Y3
	ADDQ R10, R13
	DECQ CX
	JNZ row16_loop_f64

	VMOVUPD Y0, (DI)
	VMOVUPD Y1, 32(DI)
	VMOVUPD Y2, 64(DI)
	VMOVUPD Y3, 96(DI)
	ADDQ $128, DI
	ADDQ $128, SI
	SUBQ $16, R12
	JMP col16_loop_f64

col4_f64:
	CMPQ R12, $4
	JL col2_f64

	MOVQ SI, R13
	VMOVUPD (R13), Y0
	MOVQ R11, CX
	ADDQ R10, R13

row4_loop_f64:
	VADDPD (R13), Y0, Y0
	ADDQ R10, R13
	DECQ CX
	JNZ row4_loop_f64

	VMOVUPD Y0, (DI)
	ADDQ $32, DI
	ADDQ $32, SI
	SUBQ $4, R12
	JMP col4_f64

col2_f64:
	CMPQ R12, $2
	JL col_scalar_f64

	MOVQ SI, R13
	VMOVUPD (R13), X0
	MOVQ R11, CX
	ADDQ R10, R13

row2_loop_f64:
	VADDPD (R13), X0, X0
	ADDQ R10, R13
	DECQ CX
	JNZ row2_loop_f64

	VMOVUPD X0, (DI)
	ADDQ $16, DI
	ADDQ $16, SI
	SUBQ $2, R12

col_scalar_f64:
	TESTQ R12, R12
	JZ done_f64

scalar_col_loop_f64:
	MOVQ SI, R13
	VMOVSD (R13), X0
	MOVQ R11, CX
	ADDQ R10, R13

scalar_row_loop_f64:
	VADDSD (R13), X0, X0
	ADDQ R10, R13
	DECQ CX
	JNZ scalar_row_loop_f64

	VMOVSD X0, (DI)
	ADDQ $8, DI
	ADDQ $8, SI
	DECQ R12
	JNZ scalar_col_loop_f64

done_f64:
	VZEROUPPER
	RET

// ----------------------------------------------------------------------------
// Int32: reduceLeadingSumInt32AVX2(in, out unsafe.Pointer, A, B int)
// ----------------------------------------------------------------------------
TEXT ·reduceLeadingSumInt32AVX2(SB), NOSPLIT, $0-32
	MOVQ in+0(FP), SI
	MOVQ out+8(FP), DI
	MOVQ A+16(FP), R8
	MOVQ B+24(FP), R9

	TESTQ R8, R8
	JLE done_i32
	TESTQ R9, R9
	JLE done_i32

	CMPQ R8, $1
	JNE start_i32

	MOVQ R9, CX
	SHLQ $2, CX
copy_loop_i32:
	CMPQ CX, $32
	JL copy_tail_i32
	VMOVDQU (SI), Y0
	VMOVDQU Y0, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $32, CX
	JMP copy_loop_i32
copy_tail_i32:
	TESTQ CX, CX
	JZ done_i32
	MOVB (SI), AL
	MOVB AL, (DI)
	INCQ SI
	INCQ DI
	DECQ CX
	JNZ copy_tail_i32
	JMP done_i32

start_i32:
	MOVQ R9, R10
	SHLQ $2, R10             // R10 = stride = B * 4 bytes
	MOVQ R8, R11
	DECQ R11                 // R11 = A - 1
	MOVQ R9, R12             // R12 = remaining cols

col32_loop_i32:
	CMPQ R12, $32
	JL col8_i32

	MOVQ SI, R13
	VMOVDQU (R13), Y0
	VMOVDQU 32(R13), Y1
	VMOVDQU 64(R13), Y2
	VMOVDQU 96(R13), Y3

	MOVQ R11, CX
	ADDQ R10, R13

row32_loop_i32:
	VPADDD (R13), Y0, Y0
	VPADDD 32(R13), Y1, Y1
	VPADDD 64(R13), Y2, Y2
	VPADDD 96(R13), Y3, Y3
	ADDQ R10, R13
	DECQ CX
	JNZ row32_loop_i32

	VMOVDQU Y0, (DI)
	VMOVDQU Y1, 32(DI)
	VMOVDQU Y2, 64(DI)
	VMOVDQU Y3, 96(DI)
	ADDQ $128, DI
	ADDQ $128, SI
	SUBQ $32, R12
	JMP col32_loop_i32

col8_i32:
	CMPQ R12, $8
	JL col4_i32

	MOVQ SI, R13
	VMOVDQU (R13), Y0
	MOVQ R11, CX
	ADDQ R10, R13

row8_loop_i32:
	VPADDD (R13), Y0, Y0
	ADDQ R10, R13
	DECQ CX
	JNZ row8_loop_i32

	VMOVDQU Y0, (DI)
	ADDQ $32, DI
	ADDQ $32, SI
	SUBQ $8, R12
	JMP col8_i32

col4_i32:
	CMPQ R12, $4
	JL col_scalar_i32

	MOVQ SI, R13
	VMOVDQU (R13), X0
	MOVQ R11, CX
	ADDQ R10, R13

row4_loop_i32:
	VPADDD (R13), X0, X0
	ADDQ R10, R13
	DECQ CX
	JNZ row4_loop_i32

	VMOVDQU X0, (DI)
	ADDQ $16, DI
	ADDQ $16, SI
	SUBQ $4, R12

col_scalar_i32:
	TESTQ R12, R12
	JZ done_i32

scalar_col_loop_i32:
	MOVQ SI, R13
	MOVL (R13), AX
	MOVQ R11, CX
	ADDQ R10, R13

scalar_row_loop_i32:
	ADDL (R13), AX
	ADDQ R10, R13
	DECQ CX
	JNZ scalar_row_loop_i32

	MOVL AX, (DI)
	ADDQ $4, DI
	ADDQ $4, SI
	DECQ R12
	JNZ scalar_col_loop_i32

done_i32:
	VZEROUPPER
	RET

// ----------------------------------------------------------------------------
// Uint32: reduceLeadingSumUint32AVX2(in, out unsafe.Pointer, A, B int)
// ----------------------------------------------------------------------------
TEXT ·reduceLeadingSumUint32AVX2(SB), NOSPLIT, $0-32
	JMP ·reduceLeadingSumInt32AVX2(SB)

// ----------------------------------------------------------------------------
// Int64: reduceLeadingSumInt64AVX2(in, out unsafe.Pointer, A, B int)
// ----------------------------------------------------------------------------
TEXT ·reduceLeadingSumInt64AVX2(SB), NOSPLIT, $0-32
	MOVQ in+0(FP), SI
	MOVQ out+8(FP), DI
	MOVQ A+16(FP), R8
	MOVQ B+24(FP), R9

	TESTQ R8, R8
	JLE done_i64
	TESTQ R9, R9
	JLE done_i64

	CMPQ R8, $1
	JNE start_i64

	MOVQ R9, CX
	SHLQ $3, CX
copy_loop_i64:
	CMPQ CX, $32
	JL copy_tail_i64
	VMOVDQU (SI), Y0
	VMOVDQU Y0, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $32, CX
	JMP copy_loop_i64
copy_tail_i64:
	TESTQ CX, CX
	JZ done_i64
	MOVB (SI), AL
	MOVB AL, (DI)
	INCQ SI
	INCQ DI
	DECQ CX
	JNZ copy_tail_i64
	JMP done_i64

start_i64:
	MOVQ R9, R10
	SHLQ $3, R10             // R10 = stride = B * 8 bytes
	MOVQ R8, R11
	DECQ R11                 // R11 = A - 1
	MOVQ R9, R12             // R12 = remaining cols

col16_loop_i64:
	CMPQ R12, $16
	JL col4_i64

	MOVQ SI, R13
	VMOVDQU (R13), Y0
	VMOVDQU 32(R13), Y1
	VMOVDQU 64(R13), Y2
	VMOVDQU 96(R13), Y3

	MOVQ R11, CX
	ADDQ R10, R13

row16_loop_i64:
	VPADDQ (R13), Y0, Y0
	VPADDQ 32(R13), Y1, Y1
	VPADDQ 64(R13), Y2, Y2
	VPADDQ 96(R13), Y3, Y3
	ADDQ R10, R13
	DECQ CX
	JNZ row16_loop_i64

	VMOVDQU Y0, (DI)
	VMOVDQU Y1, 32(DI)
	VMOVDQU Y2, 64(DI)
	VMOVDQU Y3, 96(DI)
	ADDQ $128, DI
	ADDQ $128, SI
	SUBQ $16, R12
	JMP col16_loop_i64

col4_i64:
	CMPQ R12, $4
	JL col2_i64

	MOVQ SI, R13
	VMOVDQU (R13), Y0
	MOVQ R11, CX
	ADDQ R10, R13

row4_loop_i64:
	VPADDQ (R13), Y0, Y0
	ADDQ R10, R13
	DECQ CX
	JNZ row4_loop_i64

	VMOVDQU Y0, (DI)
	ADDQ $32, DI
	ADDQ $32, SI
	SUBQ $4, R12
	JMP col4_i64

col2_i64:
	CMPQ R12, $2
	JL col_scalar_i64

	MOVQ SI, R13
	VMOVDQU (R13), X0
	MOVQ R11, CX
	ADDQ R10, R13

row2_loop_i64:
	VPADDQ (R13), X0, X0
	ADDQ R10, R13
	DECQ CX
	JNZ row2_loop_i64

	VMOVDQU X0, (DI)
	ADDQ $16, DI
	ADDQ $16, SI
	SUBQ $2, R12

col_scalar_i64:
	TESTQ R12, R12
	JZ done_i64

scalar_col_loop_i64:
	MOVQ SI, R13
	MOVQ (R13), AX
	MOVQ R11, CX
	ADDQ R10, R13

scalar_row_loop_i64:
	ADDQ (R13), AX
	ADDQ R10, R13
	DECQ CX
	JNZ scalar_row_loop_i64

	MOVQ AX, (DI)
	ADDQ $8, DI
	ADDQ $8, SI
	DECQ R12
	JNZ scalar_col_loop_i64

done_i64:
	VZEROUPPER
	RET

// ----------------------------------------------------------------------------
// Uint64: reduceLeadingSumUint64AVX2(in, out unsafe.Pointer, A, B int)
// ----------------------------------------------------------------------------
TEXT ·reduceLeadingSumUint64AVX2(SB), NOSPLIT, $0-32
	JMP ·reduceLeadingSumInt64AVX2(SB)

// ----------------------------------------------------------------------------
// Int16: reduceLeadingSumInt16AVX2(in, out unsafe.Pointer, A, B int)
// ----------------------------------------------------------------------------
TEXT ·reduceLeadingSumInt16AVX2(SB), NOSPLIT, $0-32
	MOVQ in+0(FP), SI
	MOVQ out+8(FP), DI
	MOVQ A+16(FP), R8
	MOVQ B+24(FP), R9

	TESTQ R8, R8
	JLE done_i16
	TESTQ R9, R9
	JLE done_i16

	CMPQ R8, $1
	JNE start_i16

	MOVQ R9, CX
	SHLQ $1, CX
copy_loop_i16:
	CMPQ CX, $32
	JL copy_tail_i16
	VMOVDQU (SI), Y0
	VMOVDQU Y0, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $32, CX
	JMP copy_loop_i16
copy_tail_i16:
	TESTQ CX, CX
	JZ done_i16
	MOVB (SI), AL
	MOVB AL, (DI)
	INCQ SI
	INCQ DI
	DECQ CX
	JNZ copy_tail_i16
	JMP done_i16

start_i16:
	MOVQ R9, R10
	SHLQ $1, R10             // stride = B * 2
	MOVQ R8, R11
	DECQ R11
	MOVQ R9, R12

col64_loop_i16:
	CMPQ R12, $64
	JL col16_i16

	MOVQ SI, R13
	VMOVDQU (R13), Y0
	VMOVDQU 32(R13), Y1
	VMOVDQU 64(R13), Y2
	VMOVDQU 96(R13), Y3

	MOVQ R11, CX
	ADDQ R10, R13

row64_loop_i16:
	VPADDW (R13), Y0, Y0
	VPADDW 32(R13), Y1, Y1
	VPADDW 64(R13), Y2, Y2
	VPADDW 96(R13), Y3, Y3
	ADDQ R10, R13
	DECQ CX
	JNZ row64_loop_i16

	VMOVDQU Y0, (DI)
	VMOVDQU Y1, 32(DI)
	VMOVDQU Y2, 64(DI)
	VMOVDQU Y3, 96(DI)
	ADDQ $128, DI
	ADDQ $128, SI
	SUBQ $64, R12
	JMP col64_loop_i16

col16_i16:
	CMPQ R12, $16
	JL col8_i16

	MOVQ SI, R13
	VMOVDQU (R13), Y0
	MOVQ R11, CX
	ADDQ R10, R13

row16_loop_i16:
	VPADDW (R13), Y0, Y0
	ADDQ R10, R13
	DECQ CX
	JNZ row16_loop_i16

	VMOVDQU Y0, (DI)
	ADDQ $32, DI
	ADDQ $32, SI
	SUBQ $16, R12
	JMP col16_i16

col8_i16:
	CMPQ R12, $8
	JL col_scalar_i16

	MOVQ SI, R13
	VMOVDQU (R13), X0
	MOVQ R11, CX
	ADDQ R10, R13

row8_loop_i16:
	VPADDW (R13), X0, X0
	ADDQ R10, R13
	DECQ CX
	JNZ row8_loop_i16

	VMOVDQU X0, (DI)
	ADDQ $16, DI
	ADDQ $16, SI
	SUBQ $8, R12

col_scalar_i16:
	TESTQ R12, R12
	JZ done_i16

scalar_col_loop_i16:
	MOVQ SI, R13
	MOVW (R13), AX
	MOVQ R11, CX
	ADDQ R10, R13

scalar_row_loop_i16:
	ADDW (R13), AX
	ADDQ R10, R13
	DECQ CX
	JNZ scalar_row_loop_i16

	MOVW AX, (DI)
	ADDQ $2, DI
	ADDQ $2, SI
	DECQ R12
	JNZ scalar_col_loop_i16

done_i16:
	VZEROUPPER
	RET

// ----------------------------------------------------------------------------
// Uint16: reduceLeadingSumUint16AVX2(in, out unsafe.Pointer, A, B int)
// ----------------------------------------------------------------------------
TEXT ·reduceLeadingSumUint16AVX2(SB), NOSPLIT, $0-32
	JMP ·reduceLeadingSumInt16AVX2(SB)

// ----------------------------------------------------------------------------
// Int8: reduceLeadingSumInt8AVX2(in, out unsafe.Pointer, A, B int)
// ----------------------------------------------------------------------------
TEXT ·reduceLeadingSumInt8AVX2(SB), NOSPLIT, $0-32
	MOVQ in+0(FP), SI
	MOVQ out+8(FP), DI
	MOVQ A+16(FP), R8
	MOVQ B+24(FP), R9

	TESTQ R8, R8
	JLE done_i8
	TESTQ R9, R9
	JLE done_i8

	CMPQ R8, $1
	JNE start_i8

	MOVQ R9, CX
copy_loop_i8:
	CMPQ CX, $32
	JL copy_tail_i8
	VMOVDQU (SI), Y0
	VMOVDQU Y0, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $32, CX
	JMP copy_loop_i8
copy_tail_i8:
	TESTQ CX, CX
	JZ done_i8
	MOVB (SI), AL
	MOVB AL, (DI)
	INCQ SI
	INCQ DI
	DECQ CX
	JNZ copy_tail_i8
	JMP done_i8

start_i8:
	MOVQ R9, R10             // stride = B
	MOVQ R8, R11
	DECQ R11
	MOVQ R9, R12

col128_loop_i8:
	CMPQ R12, $128
	JL col32_i8

	MOVQ SI, R13
	VMOVDQU (R13), Y0
	VMOVDQU 32(R13), Y1
	VMOVDQU 64(R13), Y2
	VMOVDQU 96(R13), Y3

	MOVQ R11, CX
	ADDQ R10, R13

row128_loop_i8:
	VPADDB (R13), Y0, Y0
	VPADDB 32(R13), Y1, Y1
	VPADDB 64(R13), Y2, Y2
	VPADDB 96(R13), Y3, Y3
	ADDQ R10, R13
	DECQ CX
	JNZ row128_loop_i8

	VMOVDQU Y0, (DI)
	VMOVDQU Y1, 32(DI)
	VMOVDQU Y2, 64(DI)
	VMOVDQU Y3, 96(DI)
	ADDQ $128, DI
	ADDQ $128, SI
	SUBQ $128, R12
	JMP col128_loop_i8

col32_i8:
	CMPQ R12, $32
	JL col16_i8

	MOVQ SI, R13
	VMOVDQU (R13), Y0
	MOVQ R11, CX
	ADDQ R10, R13

row32_loop_i8:
	VPADDB (R13), Y0, Y0
	ADDQ R10, R13
	DECQ CX
	JNZ row32_loop_i8

	VMOVDQU Y0, (DI)
	ADDQ $32, DI
	ADDQ $32, SI
	SUBQ $32, R12
	JMP col32_i8

col16_i8:
	CMPQ R12, $16
	JL col_scalar_i8

	MOVQ SI, R13
	VMOVDQU (R13), X0
	MOVQ R11, CX
	ADDQ R10, R13

row16_loop_i8:
	VPADDB (R13), X0, X0
	ADDQ R10, R13
	DECQ CX
	JNZ row16_loop_i8

	VMOVDQU X0, (DI)
	ADDQ $16, DI
	ADDQ $16, SI
	SUBQ $16, R12

col_scalar_i8:
	TESTQ R12, R12
	JZ done_i8

scalar_col_loop_i8:
	MOVQ SI, R13
	MOVB (R13), AL
	MOVQ R11, CX
	ADDQ R10, R13

scalar_row_loop_i8:
	ADDB (R13), AL
	ADDQ R10, R13
	DECQ CX
	JNZ scalar_row_loop_i8

	MOVB AL, (DI)
	INCQ DI
	INCQ SI
	DECQ R12
	JNZ scalar_col_loop_i8

done_i8:
	VZEROUPPER
	RET

// ----------------------------------------------------------------------------
// Uint8: reduceLeadingSumUint8AVX2(in, out unsafe.Pointer, A, B int)
// ----------------------------------------------------------------------------
TEXT ·reduceLeadingSumUint8AVX2(SB), NOSPLIT, $0-32
	JMP ·reduceLeadingSumInt8AVX2(SB)

// ----------------------------------------------------------------------------
// Float16: reduceLeadingSumFloat16AVX2(in, out unsafe.Pointer, A, B int)
// ----------------------------------------------------------------------------
TEXT ·reduceLeadingSumFloat16AVX2(SB), NOSPLIT, $0-32
	MOVQ in+0(FP), SI
	MOVQ out+8(FP), DI
	MOVQ A+16(FP), R8
	MOVQ B+24(FP), R9

	TESTQ R8, R8
	JLE done_f16
	TESTQ R9, R9
	JLE done_f16

	CMPQ R8, $1
	JNE start_f16

	MOVQ R9, CX
	SHLQ $1, CX
copy_loop_f16:
	CMPQ CX, $32
	JL copy_tail_f16
	VMOVDQU (SI), Y0
	VMOVDQU Y0, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $32, CX
	JMP copy_loop_f16
copy_tail_f16:
	TESTQ CX, CX
	JZ done_f16
	MOVB (SI), AL
	MOVB AL, (DI)
	INCQ SI
	INCQ DI
	DECQ CX
	JNZ copy_tail_f16
	JMP done_f16

start_f16:
	MOVQ R9, R10
	SHLQ $1, R10             // stride = B * 2
	MOVQ R8, R11
	DECQ R11
	MOVQ R9, R12

col8_loop_f16:
	CMPQ R12, $8
	JL col_scalar_f16

	MOVQ SI, R13
	VCVTPH2PS (R13), Y0      // load 8 half-floats -> 8 float32 in Y0
	MOVQ R11, CX
	ADDQ R10, R13

row8_loop_f16:
	VCVTPH2PS (R13), Y1
	VADDPS Y1, Y0, Y0
	ADDQ R10, R13
	DECQ CX
	JNZ row8_loop_f16

	VCVTPS2PH $0, Y0, X0     // convert 8 float32 -> 8 half-floats in X0
	VMOVDQU X0, (DI)
	ADDQ $16, DI
	ADDQ $16, SI
	SUBQ $8, R12
	JMP col8_loop_f16

col_scalar_f16:
	TESTQ R12, R12
	JZ done_f16

scalar_col_loop_f16:
	MOVQ SI, R13
	MOVWLZX (R13), DX
	VMOVD DX, X0
	VCVTPH2PS X0, X0         // X0 has float32 in lane 0
	MOVQ R11, CX
	ADDQ R10, R13

scalar_row_loop_f16:
	MOVWLZX (R13), DX
	VMOVD DX, X1
	VCVTPH2PS X1, X1
	VADDSS X1, X0, X0
	ADDQ R10, R13
	DECQ CX
	JNZ scalar_row_loop_f16

	VCVTPS2PH $0, X0, X1
	VMOVD X1, DX
	MOVW DX, (DI)
	ADDQ $2, DI
	ADDQ $2, SI
	DECQ R12
	JNZ scalar_col_loop_f16

done_f16:
	VZEROUPPER
	RET

// ----------------------------------------------------------------------------
// BFloat16: reduceLeadingSumBFloat16AVX2(in, out unsafe.Pointer, A, B int)
// ----------------------------------------------------------------------------
TEXT ·reduceLeadingSumBFloat16AVX2(SB), NOSPLIT, $0-32
	MOVQ in+0(FP), SI
	MOVQ out+8(FP), DI
	MOVQ A+16(FP), R8
	MOVQ B+24(FP), R9

	TESTQ R8, R8
	JLE done_bf16
	TESTQ R9, R9
	JLE done_bf16

	CMPQ R8, $1
	JNE start_bf16

	MOVQ R9, CX
	SHLQ $1, CX
copy_loop_bf16:
	CMPQ CX, $32
	JL copy_tail_bf16
	VMOVDQU (SI), Y0
	VMOVDQU Y0, (DI)
	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $32, CX
	JMP copy_loop_bf16
copy_tail_bf16:
	TESTQ CX, CX
	JZ done_bf16
	MOVB (SI), AL
	MOVB AL, (DI)
	INCQ SI
	INCQ DI
	DECQ CX
	JNZ copy_tail_bf16
	JMP done_bf16

start_bf16:
	MOVQ R9, R10
	SHLQ $1, R10             // stride = B * 2
	MOVQ R8, R11
	DECQ R11
	MOVQ R9, R12

col8_loop_bf16:
	CMPQ R12, $8
	JL col_scalar_bf16

	MOVQ SI, R13
	VMOVDQU (R13), X0
	VPMOVZXWD X0, Y0
	VPSLLD $16, Y0, Y0       // Y0 has 8 float32s

	MOVQ R11, CX
	ADDQ R10, R13

row8_loop_bf16:
	VMOVDQU (R13), X1
	VPMOVZXWD X1, Y1
	VPSLLD $16, Y1, Y1
	VADDPS Y1, Y0, Y0
	ADDQ R10, R13
	DECQ CX
	JNZ row8_loop_bf16

	// Store 8 bfloat16s from Y0:
	// Round each lane with round-to-nearest-even
	VEXTRACTF128 $1, Y0, X3  // X3 = lanes 4..7; X0 = lanes 0..3

	// Lane 0:
	VMOVD X0, DX
	MOVL DX, R14
	SHRL $16, R14
	ANDL $1, R14
	ADDL $0x7FFF, R14
	ADDL R14, DX
	SHRL $16, DX
	MOVW DX, 0(DI)

	// Lane 1:
	VPERMILPS $1, X0, X2
	VMOVD X2, DX
	MOVL DX, R14
	SHRL $16, R14
	ANDL $1, R14
	ADDL $0x7FFF, R14
	ADDL R14, DX
	SHRL $16, DX
	MOVW DX, 2(DI)

	// Lane 2:
	VPERMILPS $2, X0, X2
	VMOVD X2, DX
	MOVL DX, R14
	SHRL $16, R14
	ANDL $1, R14
	ADDL $0x7FFF, R14
	ADDL R14, DX
	SHRL $16, DX
	MOVW DX, 4(DI)

	// Lane 3:
	VPERMILPS $3, X0, X2
	VMOVD X2, DX
	MOVL DX, R14
	SHRL $16, R14
	ANDL $1, R14
	ADDL $0x7FFF, R14
	ADDL R14, DX
	SHRL $16, DX
	MOVW DX, 6(DI)

	// Lane 4:
	VMOVD X3, DX
	MOVL DX, R14
	SHRL $16, R14
	ANDL $1, R14
	ADDL $0x7FFF, R14
	ADDL R14, DX
	SHRL $16, DX
	MOVW DX, 8(DI)

	// Lane 5:
	VPERMILPS $1, X3, X2
	VMOVD X2, DX
	MOVL DX, R14
	SHRL $16, R14
	ANDL $1, R14
	ADDL $0x7FFF, R14
	ADDL R14, DX
	SHRL $16, DX
	MOVW DX, 10(DI)

	// Lane 6:
	VPERMILPS $2, X3, X2
	VMOVD X2, DX
	MOVL DX, R14
	SHRL $16, R14
	ANDL $1, R14
	ADDL $0x7FFF, R14
	ADDL R14, DX
	SHRL $16, DX
	MOVW DX, 12(DI)

	// Lane 7:
	VPERMILPS $3, X3, X2
	VMOVD X2, DX
	MOVL DX, R14
	SHRL $16, R14
	ANDL $1, R14
	ADDL $0x7FFF, R14
	ADDL R14, DX
	SHRL $16, DX
	MOVW DX, 14(DI)

	ADDQ $16, DI
	ADDQ $16, SI
	SUBQ $8, R12
	JMP col8_loop_bf16

col_scalar_bf16:
	TESTQ R12, R12
	JZ done_bf16

scalar_col_loop_bf16:
	MOVQ SI, R13
	MOVWLZX (R13), DX
	SHLL $16, DX
	VMOVD DX, X0
	MOVQ R11, CX
	ADDQ R10, R13

scalar_row_loop_bf16:
	MOVWLZX (R13), DX
	SHLL $16, DX
	VMOVD DX, X1
	VADDSS X1, X0, X0
	ADDQ R10, R13
	DECQ CX
	JNZ scalar_row_loop_bf16

	VMOVD X0, DX
	MOVL DX, R14
	SHRL $16, R14
	ANDL $1, R14
	ADDL $0x7FFF, R14
	ADDL R14, DX
	SHRL $16, DX
	MOVW DX, (DI)
	ADDQ $2, DI
	ADDQ $2, SI
	DECQ R12
	JNZ scalar_col_loop_bf16

done_bf16:
	VZEROUPPER
	RET
