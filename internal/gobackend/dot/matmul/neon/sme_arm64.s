// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

#include "textflag.h"

// ===========================================================================
// ARM64 SME (Scalable Matrix Extension) Outer-Product (FMOPA) Microkernels
// ===========================================================================
//
// Target: ARMv9-A with FEAT_SME (512-bit streaming vector length SVL = 64 bytes:
// 16x float32 / uint32 per Z register, or 8x float64 per Z register).
//
// Signal-Safety Canary (Z31):
// On darwin/arm64 (and POSIX OSes whose signal ucontext saves 128-bit NEON Q0-Q31),
// delivering an asynchronous signal (such as Go's SIGURG preemption signal) while in
// SME streaming mode restores Q0-Q31 on sigreturn, which architecturally zeroes the
// upper 384 bits [511:128] of Z0-Z31.
// To make SME microkernels 100% immune to async signal register clobbering:
//   1. Phase 1 (tile_retry -> k_done, read-only on output memory):
//      Sets Z31 lane 15 to 1 (`MOV Z31.S, #1`), initializes ZA (either `ZERO {ZA}` or
//      loads C into ZA via `LD1W` + `MOVA ZA, Z` when accumulate=true), and runs the
//      FMOPA K-loop. At `k_done` (before any memory store), checks lane 15 of Z31 via
//      `LASTB W16, P7, Z31.S`. If 0, a signal occurred during Phase 1; since output
//      memory is untouched, branches back to `tile_retry`.
//   2. Phase 2 (store_retry -> store_done, idempotent write of ZA to output memory):
//      Sets Z31 lane 15 to 1, extracts ZA rows via `MOVA Z, ZA`, and writes to C via
//      `ST1W`. At `store_done`, checks `LASTB W16, P7, Z31.S`; if 0, branches back to
//      `store_retry` (or `tile_retry` when accumulate=false) to re-store ZA to C.

// ---------------------------------------------------------------------------
// 1. smeLargeKernelFloat32Asm_16x16 (1x1 ZA tile: ZA0.S = 16x16 float32)
// ---------------------------------------------------------------------------
// Stack frame (112 bytes):
//   packedLHS_base:    0(FP)
//   packedRHS_base:   24(FP)
//   packedOut_base:   48(FP)
//   lhsPanelRows:     72(FP)
//   rhsPanelCols:     80(FP)
//   contractingLen:   88(FP)
//   lhsActiveRows:    96(FP)
//   rhsActiveCols:   104(FP)
//   accumulate:      112(FP)
TEXT ·smeLargeKernelFloat32Asm_16x16(SB), NOSPLIT, $0-113
	MOVD packedLHS_base+0(FP), R0
	MOVD packedRHS_base+24(FP), R1
	MOVD packedOutput_base+48(FP), R2
	MOVD rhsPanelCols+80(FP), R3
	MOVD contractingLen+88(FP), R4
	MOVD lhsActiveRows+96(FP), R5
	MOVD rhsActiveCols+104(FP), R6
	MOVBU accumulate+112(FP), R7

	LSL $2, R3, R3 // outputRowByteStride = rhsPanelCols * 4
	MOVD $0, R19   // lhsRowIdx = 0

	WORD $0xD503477F // SMSTART (enter streaming mode + enable ZA)

row_strip_loop:
	CMP R5, R19
	BGE sme_f32_16x16_done

	MOVD $0, R20 // rhsColIdx = 0

col_strip_loop:
	CMP R6, R20
	BGE next_row_strip

tile_retry:
	WORD $0x2598E3E7 // PTRUE P7.S
	WORD $0x25B8C03F // MOV Z31.S, #1 (signal canary in lane 15)
	WORD $0x2598E3E0 // PTRUE P0.S
	WORD $0xC00800FF // ZERO {ZA}

	CBZ R7, init_za_done

	// accumulate == true: load existing C tile into ZA0.S
	WORD $0x25A61681 // WHILELT P1.S, X20, X6
	MUL R19, R3, R10
	ADD R2, R10, R10
	ADD R20<<2, R10, R10 // outRowPtr
	MOVD $0, R12
	MOVD R19, R13

init_za_row_loop:
	CMP R5, R13
	BGE init_za_done
	WORD $0xA540A540 // LD1W {Z0.S}, P1/Z, [X10, #0, MUL VL]
	WORD $0xC0800400 // MOVA ZA0H.S[W12, 0], P1/M, Z0.S
	ADD $1, R12, R12
	ADD $1, R13, R13
	ADD R3, R10, R10
	CMP $16, R12
	BLT init_za_row_loop

init_za_done:
	MUL R19, R4, R8
	LSL $2, R8, R8
	ADD R0, R8, R8

	MUL R20, R4, R9
	LSL $2, R9, R9
	ADD R1, R9, R9

	MOVD R4, R11
	CMP $4, R11
	BLT k_rem_check

k_loop_4:
	WORD $0xA540A100 // LD1W {Z0.S}, P0/Z, [X8, #0, MUL VL]
	WORD $0xA540A121 // LD1W {Z1.S}, P0/Z, [X9, #0, MUL VL]
	WORD $0x80810000 // FMOPA ZA0.S, P0/M, P0/M, Z0.S, Z1.S

	WORD $0xA541A102 // LD1W {Z2.S}, P0/Z, [X8, #1, MUL VL]
	WORD $0xA541A123 // LD1W {Z3.S}, P0/Z, [X9, #1, MUL VL]
	WORD $0x80830040 // FMOPA ZA0.S, P0/M, P0/M, Z2.S, Z3.S

	WORD $0xA542A100 // LD1W {Z0.S}, P0/Z, [X8, #2, MUL VL]
	WORD $0xA542A121 // LD1W {Z1.S}, P0/Z, [X9, #2, MUL VL]
	WORD $0x80810000 // FMOPA ZA0.S, P0/M, P0/M, Z0.S, Z1.S

	WORD $0xA543A102 // LD1W {Z2.S}, P0/Z, [X8, #3, MUL VL]
	WORD $0xA543A123 // LD1W {Z3.S}, P0/Z, [X9, #3, MUL VL]
	WORD $0x80830040 // FMOPA ZA0.S, P0/M, P0/M, Z2.S, Z3.S

	ADD $256, R8, R8
	ADD $256, R9, R9
	SUB $4, R11, R11
	CMP $4, R11
	BGE k_loop_4

k_rem_check:
	CBZ R11, k_done

k_loop_1:
	WORD $0xA540A100 // LD1W {Z0.S}, P0/Z, [X8, #0, MUL VL]
	WORD $0xA540A121 // LD1W {Z1.S}, P0/Z, [X9, #0, MUL VL]
	WORD $0x80810000 // FMOPA ZA0.S, P0/M, P0/M, Z0.S, Z1.S
	ADD $64, R8, R8
	ADD $64, R9, R9
	SUB $1, R11, R11
	CBNZ R11, k_loop_1

k_done:
	// Verify no signal zeroed upper bits of Z registers during Phase 1:
	WORD $0x2598E3E7 // PTRUE P7.S
	WORD $0x05A1BFEB // LASTB W11, P7, Z31.S
	CBZ R11, tile_retry

store_retry:
	WORD $0x2598E3E7 // PTRUE P7.S
	WORD $0x25B8C03F // MOV Z31.S, #1 (signal canary for Phase 2)
	WORD $0x25A61680 // WHILELT P0.S, X20, X6

	MUL R19, R3, R10
	ADD R2, R10, R10
	ADD R20<<2, R10, R10

	MOVD $0, R12
	MOVD R19, R13

store_row_loop:
	CMP R5, R13
	BGE store_done

	WORD $0xC0820000 // MOVA Z0.S, P0/M, ZA0H.S[W12, 0]
	WORD $0xE540E140 // ST1W {Z0.S}, P0, [X10, #0, MUL VL]

	ADD $1, R12, R12
	ADD $1, R13, R13
	ADD R3, R10, R10
	CMP $16, R12
	BLT store_row_loop

store_done:
	WORD $0x2598E3E7 // PTRUE P7.S
	WORD $0x05A1BFEB // LASTB W11, P7, Z31.S
	CBZ R11, store_retry

	ADD $16, R20, R20
	B col_strip_loop

next_row_strip:
	ADD $16, R19, R19
	B row_strip_loop

sme_f32_16x16_done:
	WORD $0xD503467F // SMSTOP
	RET

// ---------------------------------------------------------------------------
// 2. smeLargeKernelFloat32Asm_16x64 (1x4 ZA tiles: ZA0..3.S = 16x64 float32)
// ---------------------------------------------------------------------------
TEXT ·smeLargeKernelFloat32Asm_16x64(SB), NOSPLIT, $0-113
	MOVD packedLHS_base+0(FP), R0
	MOVD packedRHS_base+24(FP), R1
	MOVD packedOutput_base+48(FP), R2
	MOVD rhsPanelCols+80(FP), R3
	MOVD contractingLen+88(FP), R4
	MOVD lhsActiveRows+96(FP), R5
	MOVD rhsActiveCols+104(FP), R6
	MOVBU accumulate+112(FP), R7

	LSL $2, R3, R3 // outputRowByteStride = rhsPanelCols * 4
	MOVD $0, R19   // lhsRowIdx = 0

	WORD $0xD503477F // SMSTART

row_strip_loop:
	CMP R5, R19
	BGE sme_f32_16x64_done

	MOVD $0, R20 // rhsColIdx = 0

col_strip_loop:
	CMP R6, R20
	BGE next_row_strip

tile_retry:
	WORD $0x2598E3E7 // PTRUE P7.S
	WORD $0x25B8C03F // MOV Z31.S, #1
	WORD $0x2598E3E0 // PTRUE P0.S
	WORD $0xC00800FF // ZERO {ZA}

	CBZ R7, init_za_done

	// accumulate == true: load existing C tile into ZA0..3.S
	MOVD R20, R14
	WORD $0x25A615C1 // WHILELT P1.S, X14, X6
	ADD $16, R20, R14
	WORD $0x25A615C2 // WHILELT P2.S, X14, X6
	ADD $32, R20, R14
	WORD $0x25A615C3 // WHILELT P3.S, X14, X6
	ADD $48, R20, R14
	WORD $0x25A615C4 // WHILELT P4.S, X14, X6

	MUL R19, R3, R10
	ADD R2, R10, R10
	ADD R20<<2, R10, R10
	MOVD $0, R12
	MOVD R19, R13

init_za_row_loop:
	CMP R5, R13
	BGE init_za_done
	WORD $0xA540A540 // LD1W {Z0.S}, P1/Z, [X10, #0, MUL VL]
	WORD $0xA541A941 // LD1W {Z1.S}, P2/Z, [X10, #1, MUL VL]
	WORD $0xA542AD42 // LD1W {Z2.S}, P3/Z, [X10, #2, MUL VL]
	WORD $0xA543B143 // LD1W {Z3.S}, P4/Z, [X10, #3, MUL VL]
	WORD $0xC0800400 // MOVA ZA0H.S[W12, 0], P1/M, Z0.S
	WORD $0xC0800824 // MOVA ZA1H.S[W12, 0], P2/M, Z1.S
	WORD $0xC0800C48 // MOVA ZA2H.S[W12, 0], P3/M, Z2.S
	WORD $0xC080106C // MOVA ZA3H.S[W12, 0], P4/M, Z3.S
	ADD $1, R12, R12
	ADD $1, R13, R13
	ADD R3, R10, R10
	CMP $16, R12
	BLT init_za_row_loop

init_za_done:
	MUL R19, R4, R8
	LSL $2, R8, R8
	ADD R0, R8, R8

	MUL R20, R4, R9
	LSL $2, R9, R9
	ADD R1, R9, R9

	MOVD R4, R11
	CMP $2, R11
	BLT k_rem_check

k_loop_2:
	// Step 0
	WORD $0xA540A100 // LD1W {Z0.S}, P0/Z, [X8, #0, MUL VL]
	WORD $0xA540A121 // LD1W {Z1.S}, P0/Z, [X9, #0, MUL VL]
	WORD $0xA541A122 // LD1W {Z2.S}, P0/Z, [X9, #1, MUL VL]
	WORD $0xA542A123 // LD1W {Z3.S}, P0/Z, [X9, #2, MUL VL]
	WORD $0xA543A124 // LD1W {Z4.S}, P0/Z, [X9, #3, MUL VL]
	WORD $0x80810000 // FMOPA ZA0.S, P0/M, P0/M, Z0.S, Z1.S
	WORD $0x80820001 // FMOPA ZA1.S, P0/M, P0/M, Z0.S, Z2.S
	WORD $0x80830002 // FMOPA ZA2.S, P0/M, P0/M, Z0.S, Z3.S
	WORD $0x80840003 // FMOPA ZA3.S, P0/M, P0/M, Z0.S, Z4.S

	// Step 1
	WORD $0xA541A100 // LD1W {Z0.S}, P0/Z, [X8, #1, MUL VL]
	WORD $0xA544A121 // LD1W {Z1.S}, P0/Z, [X9, #4, MUL VL]
	WORD $0xA545A122 // LD1W {Z2.S}, P0/Z, [X9, #5, MUL VL]
	WORD $0xA546A123 // LD1W {Z3.S}, P0/Z, [X9, #6, MUL VL]
	WORD $0xA547A124 // LD1W {Z4.S}, P0/Z, [X9, #7, MUL VL]
	WORD $0x80810000 // FMOPA ZA0.S, P0/M, P0/M, Z0.S, Z1.S
	WORD $0x80820001 // FMOPA ZA1.S, P0/M, P0/M, Z0.S, Z2.S
	WORD $0x80830002 // FMOPA ZA2.S, P0/M, P0/M, Z0.S, Z3.S
	WORD $0x80840003 // FMOPA ZA3.S, P0/M, P0/M, Z0.S, Z4.S

	ADD $128, R8, R8
	ADD $512, R9, R9
	SUB $2, R11, R11
	CMP $2, R11
	BGE k_loop_2

k_rem_check:
	CBZ R11, k_done

k_loop_1:
	WORD $0xA540A100 // LD1W {Z0.S}, P0/Z, [X8, #0, MUL VL]
	WORD $0xA540A121 // LD1W {Z1.S}, P0/Z, [X9, #0, MUL VL]
	WORD $0xA541A122 // LD1W {Z2.S}, P0/Z, [X9, #1, MUL VL]
	WORD $0xA542A123 // LD1W {Z3.S}, P0/Z, [X9, #2, MUL VL]
	WORD $0xA543A124 // LD1W {Z4.S}, P0/Z, [X9, #3, MUL VL]
	WORD $0x80810000 // FMOPA ZA0.S, P0/M, P0/M, Z0.S, Z1.S
	WORD $0x80820001 // FMOPA ZA1.S, P0/M, P0/M, Z0.S, Z2.S
	WORD $0x80830002 // FMOPA ZA2.S, P0/M, P0/M, Z0.S, Z3.S
	WORD $0x80840003 // FMOPA ZA3.S, P0/M, P0/M, Z0.S, Z4.S
	ADD $64, R8, R8
	ADD $256, R9, R9
	SUB $1, R11, R11
	CBNZ R11, k_loop_1

k_done:
	WORD $0x2598E3E7 // PTRUE P7.S
	WORD $0x05A1BFEB // LASTB W11, P7, Z31.S
	CBZ R11, tile_retry

store_retry:
	WORD $0x2598E3E7 // PTRUE P7.S
	WORD $0x25B8C03F // MOV Z31.S, #1
	MOVD R20, R14
	WORD $0x25A615C0 // WHILELT P0.S, X14, X6
	ADD $16, R20, R14
	WORD $0x25A615C1 // WHILELT P1.S, X14, X6
	ADD $32, R20, R14
	WORD $0x25A615C2 // WHILELT P2.S, X14, X6
	ADD $48, R20, R14
	WORD $0x25A615C3 // WHILELT P3.S, X14, X6

	MUL R19, R3, R10
	ADD R2, R10, R10
	ADD R20<<2, R10, R10

	MOVD $0, R12
	MOVD R19, R13

store_row_loop:
	CMP R5, R13
	BGE store_done

	WORD $0xC0820000 // MOVA Z0.S, P0/M, ZA0H.S[W12, 0]
	WORD $0xC0820481 // MOVA Z1.S, P1/M, ZA1H.S[W12, 0]
	WORD $0xC0820902 // MOVA Z2.S, P2/M, ZA2H.S[W12, 0]
	WORD $0xC0820D83 // MOVA Z3.S, P3/M, ZA3H.S[W12, 0]
	WORD $0xE540E140 // ST1W {Z0.S}, P0, [X10, #0, MUL VL]
	WORD $0xE541E541 // ST1W {Z1.S}, P1, [X10, #1, MUL VL]
	WORD $0xE542E942 // ST1W {Z2.S}, P2, [X10, #2, MUL VL]
	WORD $0xE543ED43 // ST1W {Z3.S}, P3, [X10, #3, MUL VL]

	ADD $1, R12, R12
	ADD $1, R13, R13
	ADD R3, R10, R10
	CMP $16, R12
	BLT store_row_loop

store_done:
	WORD $0x2598E3E7 // PTRUE P7.S
	WORD $0x05A1BFEB // LASTB W11, P7, Z31.S
	CBZ R11, store_retry

	ADD $64, R20, R20
	B col_strip_loop

next_row_strip:
	ADD $16, R19, R19
	B row_strip_loop

sme_f32_16x64_done:
	WORD $0xD503467F // SMSTOP
	RET

// ---------------------------------------------------------------------------
// 3. smeLargeKernelFloat32Asm_32x32 (2x2 ZA tiles: ZA0..3.S = 32x32 float32)
// ---------------------------------------------------------------------------
// Tile layout:
//   ZA0.S: rows 0..15,  cols 0..15
//   ZA1.S: rows 0..15,  cols 16..31
//   ZA2.S: rows 16..31, cols 0..15
//   ZA3.S: rows 16..31, cols 16..31
TEXT ·smeLargeKernelFloat32Asm_32x32(SB), NOSPLIT, $0-113
	MOVD packedLHS_base+0(FP), R0
	MOVD packedRHS_base+24(FP), R1
	MOVD packedOutput_base+48(FP), R2
	MOVD rhsPanelCols+80(FP), R3
	MOVD contractingLen+88(FP), R4
	MOVD lhsActiveRows+96(FP), R5
	MOVD rhsActiveCols+104(FP), R6
	MOVBU accumulate+112(FP), R7

	LSL $2, R3, R3 // outputRowByteStride = rhsPanelCols * 4
	MOVD $0, R19   // lhsRowIdx = 0

	WORD $0xD503477F // SMSTART

row_strip_loop:
	CMP R5, R19
	BGE sme_f32_32x32_done

	MOVD $0, R20 // rhsColIdx = 0

col_strip_loop:
	CMP R6, R20
	BGE next_row_strip

tile_retry:
	WORD $0x2598E3E7 // PTRUE P7.S
	WORD $0x25B8C03F // MOV Z31.S, #1 (signal canary in lane 15)
	WORD $0x2598E3E0 // PTRUE P0.S
	WORD $0xC00800FF // ZERO {ZA}

	CBZ R7, init_za_done

	// accumulate == true: load existing C tile into ZA0..3.S
	MOVD R20, R14
	WORD $0x25A615C1 // WHILELT P1.S, X14, X6
	ADD $16, R20, R14
	WORD $0x25A615C2 // WHILELT P2.S, X14, X6

	MUL R19, R3, R10
	ADD R2, R10, R10
	ADD R20<<2, R10, R10
	MOVD R10, R16
	LSL $4, R3, R15
	ADD R10, R15, R17

	MOVD $0, R12
	MOVD R19, R13
	ADD $16, R19, R15

init_za_row_loop:
	CMP R5, R13
	BGE init_za_done

	WORD $0xA540A600 // LD1W {Z0.S}, P1/Z, [X16, #0, MUL VL]
	WORD $0xA541AA01 // LD1W {Z1.S}, P2/Z, [X16, #1, MUL VL]
	WORD $0xC0800400 // MOVA ZA0H.S[W12, 0], P1/M, Z0.S
	WORD $0xC0800824 // MOVA ZA1H.S[W12, 0], P2/M, Z1.S

	CMP R5, R15
	BGE init_za_skip_bot
	WORD $0xA540A622 // LD1W {Z2.S}, P1/Z, [X17, #0, MUL VL]
	WORD $0xA541AA23 // LD1W {Z3.S}, P2/Z, [X17, #1, MUL VL]
	WORD $0xC0800448 // MOVA ZA2H.S[W12, 0], P1/M, Z2.S
	WORD $0xC080086C // MOVA ZA3H.S[W12, 0], P2/M, Z3.S

init_za_skip_bot:
	ADD $1, R12, R12
	ADD $1, R13, R13
	ADD $1, R15, R15
	ADD R3, R16, R16
	ADD R3, R17, R17
	CMP $16, R12
	BLT init_za_row_loop

init_za_done:
	MUL R19, R4, R8
	LSL $2, R8, R8
	ADD R0, R8, R8

	MUL R20, R4, R9
	LSL $2, R9, R9
	ADD R1, R9, R9

	MOVD R4, R11
	CMP $4, R11
	BLT k_rem_check

k_loop_4:
	// Step 0
	WORD $0xA540A100 // LD1W {Z0.S}, P0/Z, [X8, #0, MUL VL]
	WORD $0xA541A101 // LD1W {Z1.S}, P0/Z, [X8, #1, MUL VL]
	WORD $0xA540A122 // LD1W {Z2.S}, P0/Z, [X9, #0, MUL VL]
	WORD $0xA541A123 // LD1W {Z3.S}, P0/Z, [X9, #1, MUL VL]
	WORD $0x80820000 // FMOPA ZA0.S, P0/M, P0/M, Z0.S, Z2.S
	WORD $0x80830001 // FMOPA ZA1.S, P0/M, P0/M, Z0.S, Z3.S
	WORD $0x80820022 // FMOPA ZA2.S, P0/M, P0/M, Z1.S, Z2.S
	WORD $0x80830023 // FMOPA ZA3.S, P0/M, P0/M, Z1.S, Z3.S

	// Step 1
	WORD $0xA542A104 // LD1W {Z4.S}, P0/Z, [X8, #2, MUL VL]
	WORD $0xA543A105 // LD1W {Z5.S}, P0/Z, [X8, #3, MUL VL]
	WORD $0xA542A126 // LD1W {Z6.S}, P0/Z, [X9, #2, MUL VL]
	WORD $0xA543A127 // LD1W {Z7.S}, P0/Z, [X9, #3, MUL VL]
	WORD $0x80860080 // FMOPA ZA0.S, P0/M, P0/M, Z4.S, Z6.S
	WORD $0x80870081 // FMOPA ZA1.S, P0/M, P0/M, Z4.S, Z7.S
	WORD $0x808600A2 // FMOPA ZA2.S, P0/M, P0/M, Z5.S, Z6.S
	WORD $0x808700A3 // FMOPA ZA3.S, P0/M, P0/M, Z5.S, Z7.S

	// Step 2
	WORD $0xA544A100 // LD1W {Z0.S}, P0/Z, [X8, #4, MUL VL]
	WORD $0xA545A101 // LD1W {Z1.S}, P0/Z, [X8, #5, MUL VL]
	WORD $0xA544A122 // LD1W {Z2.S}, P0/Z, [X9, #4, MUL VL]
	WORD $0xA545A123 // LD1W {Z3.S}, P0/Z, [X9, #5, MUL VL]
	WORD $0x80820000 // FMOPA ZA0.S, P0/M, P0/M, Z0.S, Z2.S
	WORD $0x80830001 // FMOPA ZA1.S, P0/M, P0/M, Z0.S, Z3.S
	WORD $0x80820022 // FMOPA ZA2.S, P0/M, P0/M, Z1.S, Z2.S
	WORD $0x80830023 // FMOPA ZA3.S, P0/M, P0/M, Z1.S, Z3.S

	// Step 3
	WORD $0xA546A104 // LD1W {Z4.S}, P0/Z, [X8, #6, MUL VL]
	WORD $0xA547A105 // LD1W {Z5.S}, P0/Z, [X8, #7, MUL VL]
	WORD $0xA546A126 // LD1W {Z6.S}, P0/Z, [X9, #6, MUL VL]
	WORD $0xA547A127 // LD1W {Z7.S}, P0/Z, [X9, #7, MUL VL]
	WORD $0x80860080 // FMOPA ZA0.S, P0/M, P0/M, Z4.S, Z6.S
	WORD $0x80870081 // FMOPA ZA1.S, P0/M, P0/M, Z4.S, Z7.S
	WORD $0x808600A2 // FMOPA ZA2.S, P0/M, P0/M, Z5.S, Z6.S
	WORD $0x808700A3 // FMOPA ZA3.S, P0/M, P0/M, Z5.S, Z7.S

	ADD $512, R8, R8
	ADD $512, R9, R9
	SUB $4, R11, R11
	CMP $4, R11
	BGE k_loop_4

k_rem_check:
	CBZ R11, k_done

k_loop_1:
	WORD $0xA540A100 // LD1W {Z0.S}, P0/Z, [X8, #0, MUL VL]
	WORD $0xA541A101 // LD1W {Z1.S}, P0/Z, [X8, #1, MUL VL]
	WORD $0xA540A122 // LD1W {Z2.S}, P0/Z, [X9, #0, MUL VL]
	WORD $0xA541A123 // LD1W {Z3.S}, P0/Z, [X9, #1, MUL VL]
	WORD $0x80820000 // FMOPA ZA0.S, P0/M, P0/M, Z0.S, Z2.S
	WORD $0x80830001 // FMOPA ZA1.S, P0/M, P0/M, Z0.S, Z3.S
	WORD $0x80820022 // FMOPA ZA2.S, P0/M, P0/M, Z1.S, Z2.S
	WORD $0x80830023 // FMOPA ZA3.S, P0/M, P0/M, Z1.S, Z3.S
	ADD $128, R8, R8
	ADD $128, R9, R9
	SUB $1, R11, R11
	CBNZ R11, k_loop_1

k_done:
	WORD $0x2598E3E7 // PTRUE P7.S
	WORD $0x05A1BFEB // LASTB W11, P7, Z31.S
	CBZ R11, tile_retry

store_retry:
	WORD $0x2598E3E7 // PTRUE P7.S
	WORD $0x25B8C03F // MOV Z31.S, #1
	MOVD R20, R14
	WORD $0x25A615C0 // WHILELT P0.S, X14, X6
	ADD $16, R20, R14
	WORD $0x25A615C1 // WHILELT P1.S, X14, X6

	MUL R19, R3, R10
	ADD R2, R10, R10
	ADD R20<<2, R10, R10

	MOVD R10, R16
	LSL $4, R3, R15
	ADD R10, R15, R17

	MOVD $0, R12
	MOVD R19, R13
	ADD $16, R19, R15

store_row_loop:
	CMP R5, R13
	BGE store_done

	WORD $0xC0820000 // MOVA Z0.S, P0/M, ZA0H.S[W12, 0]
	WORD $0xC0820481 // MOVA Z1.S, P1/M, ZA1H.S[W12, 0]
	WORD $0xE540E200 // ST1W {Z0.S}, P0, [X16, #0, MUL VL]
	WORD $0xE541E601 // ST1W {Z1.S}, P1, [X16, #1, MUL VL]

	CMP R5, R15
	BGE store_skip_bot

	WORD $0xC0820102 // MOVA Z2.S, P0/M, ZA2H.S[W12, 0]
	WORD $0xC0820583 // MOVA Z3.S, P1/M, ZA3H.S[W12, 0]
	WORD $0xE540E222 // ST1W {Z2.S}, P0, [X17, #0, MUL VL]
	WORD $0xE541E623 // ST1W {Z3.S}, P1, [X17, #1, MUL VL]

store_skip_bot:
	ADD $1, R12, R12
	ADD $1, R13, R13
	ADD $1, R15, R15
	ADD R3, R16, R16
	ADD R3, R17, R17
	CMP $16, R12
	BLT store_row_loop

store_done:
	WORD $0x2598E3E7 // PTRUE P7.S
	WORD $0x05A1BFEB // LASTB W11, P7, Z31.S
	CBZ R11, store_retry

	ADD $32, R20, R20
	B col_strip_loop

next_row_strip:
	ADD $32, R19, R19
	B row_strip_loop

sme_f32_32x32_done:
	WORD $0xD503467F // SMSTOP
	RET

// ---------------------------------------------------------------------------
// 4. smeLargeKernelFloat16Asm_16x16 (1x1 ZA tile: Float16 -> Float32)
// ---------------------------------------------------------------------------
TEXT ·smeLargeKernelFloat16Asm_16x16(SB), NOSPLIT, $0-113
	MOVD packedLHS_base+0(FP), R0
	MOVD packedRHS_base+24(FP), R1
	MOVD packedOutput_base+48(FP), R2
	MOVD rhsPanelCols+80(FP), R3
	MOVD contractingLen+88(FP), R4
	MOVD lhsActiveRows+96(FP), R5
	MOVD rhsActiveCols+104(FP), R6
	MOVBU accumulate+112(FP), R7

	LSL $2, R3, R3
	MOVD $0, R19

	WORD $0xD503477F // SMSTART

row_strip_loop:
	CMP R5, R19
	BGE sme_f16_16x16_done

	MOVD $0, R20

col_strip_loop:
	CMP R6, R20
	BGE next_row_strip

tile_retry:
	WORD $0x2598E3E7 // PTRUE P7.S
	WORD $0x25B8C03F // MOV Z31.S, #1
	WORD $0x2598E3E0 // PTRUE P0.S
	WORD $0xC00800FF // ZERO {ZA}

	CBZ R7, init_za_done

	WORD $0x25A61681 // WHILELT P1.S, X20, X6
	MUL R19, R3, R10
	ADD R2, R10, R10
	ADD R20<<2, R10, R10
	MOVD $0, R12
	MOVD R19, R13

init_za_row_loop:
	CMP R5, R13
	BGE init_za_done
	WORD $0xA540A540 // LD1W {Z0.S}, P1/Z, [X10, #0, MUL VL]
	WORD $0xC0800400 // MOVA ZA0H.S[W12, 0], P1/M, Z0.S
	ADD $1, R12, R12
	ADD $1, R13, R13
	ADD R3, R10, R10
	CMP $16, R12
	BLT init_za_row_loop

init_za_done:
	MUL R19, R4, R8
	LSL $1, R8, R8
	ADD R0, R8, R8

	MUL R20, R4, R9
	LSL $1, R9, R9
	ADD R1, R9, R9

	MOVD R4, R11
	CBZ R11, k_done

k_loop_1:
	WORD $0xA4C0A100 // LD1H {Z0.S}, P0/Z, [X8, #0, MUL VL]
	WORD $0xA4C0A121 // LD1H {Z1.S}, P0/Z, [X9, #0, MUL VL]
	WORD $0x6589A000 // FCVT Z0.S, P0/M, Z0.H
	WORD $0x6589A021 // FCVT Z1.S, P0/M, Z1.H
	WORD $0x80810000 // FMOPA ZA0.S, P0/M, P0/M, Z0.S, Z1.S
	ADD $32, R8, R8
	ADD $32, R9, R9
	SUB $1, R11, R11
	CBNZ R11, k_loop_1

k_done:
	WORD $0x2598E3E7 // PTRUE P7.S
	WORD $0x05A1BFEB // LASTB W11, P7, Z31.S
	CBZ R11, tile_retry

store_retry:
	WORD $0x2598E3E7 // PTRUE P7.S
	WORD $0x25B8C03F // MOV Z31.S, #1
	WORD $0x25A61680 // WHILELT P0.S, X20, X6

	MUL R19, R3, R10
	ADD R2, R10, R10
	ADD R20<<2, R10, R10

	MOVD $0, R12
	MOVD R19, R13

store_row_loop:
	CMP R5, R13
	BGE store_done

	WORD $0xC0820000 // MOVA Z0.S, P0/M, ZA0H.S[W12, 0]
	WORD $0xE540E140 // ST1W {Z0.S}, P0, [X10, #0, MUL VL]

	ADD $1, R12, R12
	ADD $1, R13, R13
	ADD R3, R10, R10
	CMP $16, R12
	BLT store_row_loop

store_done:
	WORD $0x2598E3E7 // PTRUE P7.S
	WORD $0x05A1BFEB // LASTB W11, P7, Z31.S
	CBZ R11, store_retry

	ADD $16, R20, R20
	B col_strip_loop

next_row_strip:
	ADD $16, R19, R19
	B row_strip_loop

sme_f16_16x16_done:
	WORD $0xD503467F // SMSTOP
	RET

// ---------------------------------------------------------------------------
// 5. smeLargeKernelFloat16Asm_32x32 (2x2 ZA tiles: Float16 -> Float32)
// ---------------------------------------------------------------------------
TEXT ·smeLargeKernelFloat16Asm_32x32(SB), NOSPLIT, $0-113
	MOVD packedLHS_base+0(FP), R0
	MOVD packedRHS_base+24(FP), R1
	MOVD packedOutput_base+48(FP), R2
	MOVD rhsPanelCols+80(FP), R3
	MOVD contractingLen+88(FP), R4
	MOVD lhsActiveRows+96(FP), R5
	MOVD rhsActiveCols+104(FP), R6
	MOVBU accumulate+112(FP), R7

	LSL $2, R3, R3
	MOVD $0, R19

	WORD $0xD503477F // SMSTART

row_strip_loop:
	CMP R5, R19
	BGE sme_f16_32x32_done

	MOVD $0, R20

col_strip_loop:
	CMP R6, R20
	BGE next_row_strip

tile_retry:
	WORD $0x2598E3E7 // PTRUE P7.S
	WORD $0x25B8C03F // MOV Z31.S, #1
	WORD $0x2598E3E0 // PTRUE P0.S
	WORD $0xC00800FF // ZERO {ZA}

	CBZ R7, init_za_done

	MOVD R20, R14
	WORD $0x25A615C1 // WHILELT P1.S, X14, X6
	ADD $16, R20, R14
	WORD $0x25A615C2 // WHILELT P2.S, X14, X6

	MUL R19, R3, R10
	ADD R2, R10, R10
	ADD R20<<2, R10, R10
	MOVD R10, R16
	LSL $4, R3, R15
	ADD R10, R15, R17

	MOVD $0, R12
	MOVD R19, R13
	ADD $16, R19, R15

init_za_row_loop:
	CMP R5, R13
	BGE init_za_done

	WORD $0xA540A600 // LD1W {Z0.S}, P1/Z, [X16, #0, MUL VL]
	WORD $0xA541AA01 // LD1W {Z1.S}, P2/Z, [X16, #1, MUL VL]
	WORD $0xC0800400 // MOVA ZA0H.S[W12, 0], P1/M, Z0.S
	WORD $0xC0800824 // MOVA ZA1H.S[W12, 0], P2/M, Z1.S

	CMP R5, R15
	BGE init_za_skip_bot
	WORD $0xA540A622 // LD1W {Z2.S}, P1/Z, [X17, #0, MUL VL]
	WORD $0xA541AA23 // LD1W {Z3.S}, P2/Z, [X17, #1, MUL VL]
	WORD $0xC0800448 // MOVA ZA2H.S[W12, 0], P1/M, Z2.S
	WORD $0xC080086C // MOVA ZA3H.S[W12, 0], P2/M, Z3.S

init_za_skip_bot:
	ADD $1, R12, R12
	ADD $1, R13, R13
	ADD $1, R15, R15
	ADD R3, R16, R16
	ADD R3, R17, R17
	CMP $16, R12
	BLT init_za_row_loop

init_za_done:
	MUL R19, R4, R8
	LSL $1, R8, R8
	ADD R0, R8, R8

	MUL R20, R4, R9
	LSL $1, R9, R9
	ADD R1, R9, R9

	MOVD R4, R11
	CMP $2, R11
	BLT k_rem_check

k_loop_2:
	// Step 0
	WORD $0xA4C0A100 // LD1H {Z0.S}, P0/Z, [X8, #0, MUL VL]
	WORD $0xA4C1A101 // LD1H {Z1.S}, P0/Z, [X8, #1, MUL VL]
	WORD $0xA4C0A122 // LD1H {Z2.S}, P0/Z, [X9, #0, MUL VL]
	WORD $0xA4C1A123 // LD1H {Z3.S}, P0/Z, [X9, #1, MUL VL]
	WORD $0x6589A000 // FCVT Z0.S, P0/M, Z0.H
	WORD $0x6589A021 // FCVT Z1.S, P0/M, Z1.H
	WORD $0x6589A042 // FCVT Z2.S, P0/M, Z2.H
	WORD $0x6589A063 // FCVT Z3.S, P0/M, Z3.H
	WORD $0x80820000 // FMOPA ZA0.S, P0/M, P0/M, Z0.S, Z2.S
	WORD $0x80830001 // FMOPA ZA1.S, P0/M, P0/M, Z0.S, Z3.S
	WORD $0x80820022 // FMOPA ZA2.S, P0/M, P0/M, Z1.S, Z2.S
	WORD $0x80830023 // FMOPA ZA3.S, P0/M, P0/M, Z1.S, Z3.S

	// Step 1
	WORD $0xA4C2A104 // LD1H {Z4.S}, P0/Z, [X8, #2, MUL VL]
	WORD $0xA4C3A105 // LD1H {Z5.S}, P0/Z, [X8, #3, MUL VL]
	WORD $0xA4C2A126 // LD1H {Z6.S}, P0/Z, [X9, #2, MUL VL]
	WORD $0xA4C3A127 // LD1H {Z7.S}, P0/Z, [X9, #3, MUL VL]
	WORD $0x6589A084 // FCVT Z4.S, P0/M, Z4.H
	WORD $0x6589A0A5 // FCVT Z5.S, P0/M, Z5.H
	WORD $0x6589A0C6 // FCVT Z6.S, P0/M, Z6.H
	WORD $0x6589A0E7 // FCVT Z7.S, P0/M, Z7.H
	WORD $0x80860080 // FMOPA ZA0.S, P0/M, P0/M, Z4.S, Z6.S
	WORD $0x80870081 // FMOPA ZA1.S, P0/M, P0/M, Z4.S, Z7.S
	WORD $0x808600A2 // FMOPA ZA2.S, P0/M, P0/M, Z5.S, Z6.S
	WORD $0x808700A3 // FMOPA ZA3.S, P0/M, P0/M, Z5.S, Z7.S

	ADD $128, R8, R8
	ADD $128, R9, R9
	SUB $2, R11, R11
	CMP $2, R11
	BGE k_loop_2

k_rem_check:
	CBZ R11, k_done

k_loop_1:
	WORD $0xA4C0A100 // LD1H {Z0.S}, P0/Z, [X8, #0, MUL VL]
	WORD $0xA4C1A101 // LD1H {Z1.S}, P0/Z, [X8, #1, MUL VL]
	WORD $0xA4C0A122 // LD1H {Z2.S}, P0/Z, [X9, #0, MUL VL]
	WORD $0xA4C1A123 // LD1H {Z3.S}, P0/Z, [X9, #1, MUL VL]
	WORD $0x6589A000 // FCVT Z0.S, P0/M, Z0.H
	WORD $0x6589A021 // FCVT Z1.S, P0/M, Z1.H
	WORD $0x6589A042 // FCVT Z2.S, P0/M, Z2.H
	WORD $0x6589A063 // FCVT Z3.S, P0/M, Z3.H
	WORD $0x80820000 // FMOPA ZA0.S, P0/M, P0/M, Z0.S, Z2.S
	WORD $0x80830001 // FMOPA ZA1.S, P0/M, P0/M, Z0.S, Z3.S
	WORD $0x80820022 // FMOPA ZA2.S, P0/M, P0/M, Z1.S, Z2.S
	WORD $0x80830023 // FMOPA ZA3.S, P0/M, P0/M, Z1.S, Z3.S
	ADD $64, R8, R8
	ADD $64, R9, R9
	SUB $1, R11, R11
	CBNZ R11, k_loop_1

k_done:
	WORD $0x2598E3E7 // PTRUE P7.S
	WORD $0x05A1BFEB // LASTB W11, P7, Z31.S
	CBZ R11, tile_retry

store_retry:
	WORD $0x2598E3E7 // PTRUE P7.S
	WORD $0x25B8C03F // MOV Z31.S, #1
	MOVD R20, R14
	WORD $0x25A615C0 // WHILELT P0.S, X14, X6
	ADD $16, R20, R14
	WORD $0x25A615C1 // WHILELT P1.S, X14, X6

	MUL R19, R3, R10
	ADD R2, R10, R10
	ADD R20<<2, R10, R10

	MOVD R10, R16
	LSL $4, R3, R15
	ADD R10, R15, R17

	MOVD $0, R12
	MOVD R19, R13
	ADD $16, R19, R15

store_row_loop:
	CMP R5, R13
	BGE store_done

	WORD $0xC0820000 // MOVA Z0.S, P0/M, ZA0H.S[W12, 0]
	WORD $0xC0820481 // MOVA Z1.S, P1/M, ZA1H.S[W12, 0]
	WORD $0xE540E200 // ST1W {Z0.S}, P0, [X16, #0, MUL VL]
	WORD $0xE541E601 // ST1W {Z1.S}, P1, [X16, #1, MUL VL]

	CMP R5, R15
	BGE store_skip_bot

	WORD $0xC0820102 // MOVA Z2.S, P0/M, ZA2H.S[W12, 0]
	WORD $0xC0820583 // MOVA Z3.S, P1/M, ZA3H.S[W12, 0]
	WORD $0xE540E222 // ST1W {Z2.S}, P0, [X17, #0, MUL VL]
	WORD $0xE541E623 // ST1W {Z3.S}, P1, [X17, #1, MUL VL]

store_skip_bot:
	ADD $1, R12, R12
	ADD $1, R13, R13
	ADD $1, R15, R15
	ADD R3, R16, R16
	ADD R3, R17, R17
	CMP $16, R12
	BLT store_row_loop

store_done:
	WORD $0x2598E3E7 // PTRUE P7.S
	WORD $0x05A1BFEB // LASTB W11, P7, Z31.S
	CBZ R11, store_retry

	ADD $32, R20, R20
	B col_strip_loop

next_row_strip:
	ADD $32, R19, R19
	B row_strip_loop

sme_f16_32x32_done:
	WORD $0xD503467F // SMSTOP
	RET

// ---------------------------------------------------------------------------
// 6. smeLargeKernelBFloat16Asm_16x16 (1x1 ZA tile: BFloat16 -> Float32)
// ---------------------------------------------------------------------------
TEXT ·smeLargeKernelBFloat16Asm_16x16(SB), NOSPLIT, $0-113
	MOVD packedLHS_base+0(FP), R0
	MOVD packedRHS_base+24(FP), R1
	MOVD packedOutput_base+48(FP), R2
	MOVD rhsPanelCols+80(FP), R3
	MOVD contractingLen+88(FP), R4
	MOVD lhsActiveRows+96(FP), R5
	MOVD rhsActiveCols+104(FP), R6
	MOVBU accumulate+112(FP), R7

	LSL $2, R3, R3
	MOVD $0, R19

	WORD $0xD503477F // SMSTART

row_strip_loop:
	CMP R5, R19
	BGE sme_bf16_16x16_done

	MOVD $0, R20

col_strip_loop:
	CMP R6, R20
	BGE next_row_strip

tile_retry:
	WORD $0x2598E3E7 // PTRUE P7.S
	WORD $0x25B8C03F // MOV Z31.S, #1
	WORD $0x2598E3E0 // PTRUE P0.S
	WORD $0xC00800FF // ZERO {ZA}

	CBZ R7, init_za_done

	WORD $0x25A61681 // WHILELT P1.S, X20, X6
	MUL R19, R3, R10
	ADD R2, R10, R10
	ADD R20<<2, R10, R10
	MOVD $0, R12
	MOVD R19, R13

init_za_row_loop:
	CMP R5, R13
	BGE init_za_done
	WORD $0xA540A540 // LD1W {Z0.S}, P1/Z, [X10, #0, MUL VL]
	WORD $0xC0800400 // MOVA ZA0H.S[W12, 0], P1/M, Z0.S
	ADD $1, R12, R12
	ADD $1, R13, R13
	ADD R3, R10, R10
	CMP $16, R12
	BLT init_za_row_loop

init_za_done:
	MUL R19, R4, R8
	LSL $1, R8, R8
	ADD R0, R8, R8

	MUL R20, R4, R9
	LSL $1, R9, R9
	ADD R1, R9, R9

	MOVD R4, R11
	CBZ R11, k_done

k_loop_1:
	WORD $0xA4C0A100 // LD1H {Z0.S}, P0/Z, [X8, #0, MUL VL]
	WORD $0xA4C0A121 // LD1H {Z1.S}, P0/Z, [X9, #0, MUL VL]
	WORD $0x04B09C00 // LSL Z0.S, Z0.S, #16
	WORD $0x04B09C21 // LSL Z1.S, Z1.S, #16
	WORD $0x80810000 // FMOPA ZA0.S, P0/M, P0/M, Z0.S, Z1.S
	ADD $32, R8, R8
	ADD $32, R9, R9
	SUB $1, R11, R11
	CBNZ R11, k_loop_1

k_done:
	WORD $0x2598E3E7 // PTRUE P7.S
	WORD $0x05A1BFEB // LASTB W11, P7, Z31.S
	CBZ R11, tile_retry

store_retry:
	WORD $0x2598E3E7 // PTRUE P7.S
	WORD $0x25B8C03F // MOV Z31.S, #1
	WORD $0x25A61680 // WHILELT P0.S, X20, X6

	MUL R19, R3, R10
	ADD R2, R10, R10
	ADD R20<<2, R10, R10

	MOVD $0, R12
	MOVD R19, R13

store_row_loop:
	CMP R5, R13
	BGE store_done

	WORD $0xC0820000 // MOVA Z0.S, P0/M, ZA0H.S[W12, 0]
	WORD $0xE540E140 // ST1W {Z0.S}, P0, [X10, #0, MUL VL]

	ADD $1, R12, R12
	ADD $1, R13, R13
	ADD R3, R10, R10
	CMP $16, R12
	BLT store_row_loop

store_done:
	WORD $0x2598E3E7 // PTRUE P7.S
	WORD $0x05A1BFEB // LASTB W11, P7, Z31.S
	CBZ R11, store_retry

	ADD $16, R20, R20
	B col_strip_loop

next_row_strip:
	ADD $16, R19, R19
	B row_strip_loop

sme_bf16_16x16_done:
	WORD $0xD503467F // SMSTOP
	RET

// ---------------------------------------------------------------------------
// 7. smeLargeKernelBFloat16Asm_32x32 (2x2 ZA tiles: BFloat16 -> Float32)
// ---------------------------------------------------------------------------
TEXT ·smeLargeKernelBFloat16Asm_32x32(SB), NOSPLIT, $0-113
	MOVD packedLHS_base+0(FP), R0
	MOVD packedRHS_base+24(FP), R1
	MOVD packedOutput_base+48(FP), R2
	MOVD rhsPanelCols+80(FP), R3
	MOVD contractingLen+88(FP), R4
	MOVD lhsActiveRows+96(FP), R5
	MOVD rhsActiveCols+104(FP), R6
	MOVBU accumulate+112(FP), R7

	LSL $2, R3, R3
	MOVD $0, R19

	WORD $0xD503477F // SMSTART

row_strip_loop:
	CMP R5, R19
	BGE sme_bf16_32x32_done

	MOVD $0, R20

col_strip_loop:
	CMP R6, R20
	BGE next_row_strip

tile_retry:
	WORD $0x2598E3E7 // PTRUE P7.S
	WORD $0x25B8C03F // MOV Z31.S, #1
	WORD $0x2598E3E0 // PTRUE P0.S
	WORD $0xC00800FF // ZERO {ZA}

	CBZ R7, init_za_done

	MOVD R20, R14
	WORD $0x25A615C1 // WHILELT P1.S, X14, X6
	ADD $16, R20, R14
	WORD $0x25A615C2 // WHILELT P2.S, X14, X6

	MUL R19, R3, R10
	ADD R2, R10, R10
	ADD R20<<2, R10, R10
	MOVD R10, R16
	LSL $4, R3, R15
	ADD R10, R15, R17

	MOVD $0, R12
	MOVD R19, R13
	ADD $16, R19, R15

init_za_row_loop:
	CMP R5, R13
	BGE init_za_done

	WORD $0xA540A600 // LD1W {Z0.S}, P1/Z, [X16, #0, MUL VL]
	WORD $0xA541AA01 // LD1W {Z1.S}, P2/Z, [X16, #1, MUL VL]
	WORD $0xC0800400 // MOVA ZA0H.S[W12, 0], P1/M, Z0.S
	WORD $0xC0800824 // MOVA ZA1H.S[W12, 0], P2/M, Z1.S

	CMP R5, R15
	BGE init_za_skip_bot
	WORD $0xA540A622 // LD1W {Z2.S}, P1/Z, [X17, #0, MUL VL]
	WORD $0xA541AA23 // LD1W {Z3.S}, P2/Z, [X17, #1, MUL VL]
	WORD $0xC0800448 // MOVA ZA2H.S[W12, 0], P1/M, Z2.S
	WORD $0xC080086C // MOVA ZA3H.S[W12, 0], P2/M, Z3.S

init_za_skip_bot:
	ADD $1, R12, R12
	ADD $1, R13, R13
	ADD $1, R15, R15
	ADD R3, R16, R16
	ADD R3, R17, R17
	CMP $16, R12
	BLT init_za_row_loop

init_za_done:
	MUL R19, R4, R8
	LSL $1, R8, R8
	ADD R0, R8, R8

	MUL R20, R4, R9
	LSL $1, R9, R9
	ADD R1, R9, R9

	MOVD R4, R11
	CMP $2, R11
	BLT k_rem_check

k_loop_2:
	// Step 0
	WORD $0xA4C0A100 // LD1H {Z0.S}, P0/Z, [X8, #0, MUL VL]
	WORD $0xA4C1A101 // LD1H {Z1.S}, P0/Z, [X8, #1, MUL VL]
	WORD $0xA4C0A122 // LD1H {Z2.S}, P0/Z, [X9, #0, MUL VL]
	WORD $0xA4C1A123 // LD1H {Z3.S}, P0/Z, [X9, #1, MUL VL]
	WORD $0x04B09C00 // LSL Z0.S, Z0.S, #16
	WORD $0x04B09C21 // LSL Z1.S, Z1.S, #16
	WORD $0x04B09C42 // LSL Z2.S, Z2.S, #16
	WORD $0x04B09C63 // LSL Z3.S, Z3.S, #16
	WORD $0x80820000 // FMOPA ZA0.S, P0/M, P0/M, Z0.S, Z2.S
	WORD $0x80830001 // FMOPA ZA1.S, P0/M, P0/M, Z0.S, Z3.S
	WORD $0x80820022 // FMOPA ZA2.S, P0/M, P0/M, Z1.S, Z2.S
	WORD $0x80830023 // FMOPA ZA3.S, P0/M, P0/M, Z1.S, Z3.S

	// Step 1
	WORD $0xA4C2A104 // LD1H {Z4.S}, P0/Z, [X8, #2, MUL VL]
	WORD $0xA4C3A105 // LD1H {Z5.S}, P0/Z, [X8, #3, MUL VL]
	WORD $0xA4C2A126 // LD1H {Z6.S}, P0/Z, [X9, #2, MUL VL]
	WORD $0xA4C3A127 // LD1H {Z7.S}, P0/Z, [X9, #3, MUL VL]
	WORD $0x04B09C84 // LSL Z4.S, Z4.S, #16
	WORD $0x04B09CA5 // LSL Z5.S, Z5.S, #16
	WORD $0x04B09CC6 // LSL Z6.S, Z6.S, #16
	WORD $0x04B09CE7 // LSL Z7.S, Z7.S, #16
	WORD $0x80860080 // FMOPA ZA0.S, P0/M, P0/M, Z4.S, Z6.S
	WORD $0x80870081 // FMOPA ZA1.S, P0/M, P0/M, Z4.S, Z7.S
	WORD $0x808600A2 // FMOPA ZA2.S, P0/M, P0/M, Z5.S, Z6.S
	WORD $0x808700A3 // FMOPA ZA3.S, P0/M, P0/M, Z5.S, Z7.S

	ADD $128, R8, R8
	ADD $128, R9, R9
	SUB $2, R11, R11
	CMP $2, R11
	BGE k_loop_2

k_rem_check:
	CBZ R11, k_done

k_loop_1:
	WORD $0xA4C0A100 // LD1H {Z0.S}, P0/Z, [X8, #0, MUL VL]
	WORD $0xA4C1A101 // LD1H {Z1.S}, P0/Z, [X8, #1, MUL VL]
	WORD $0xA4C0A122 // LD1H {Z2.S}, P0/Z, [X9, #0, MUL VL]
	WORD $0xA4C1A123 // LD1H {Z3.S}, P0/Z, [X9, #1, MUL VL]
	WORD $0x04B09C00 // LSL Z0.S, Z0.S, #16
	WORD $0x04B09C21 // LSL Z1.S, Z1.S, #16
	WORD $0x04B09C42 // LSL Z2.S, Z2.S, #16
	WORD $0x04B09C63 // LSL Z3.S, Z3.S, #16
	WORD $0x80820000 // FMOPA ZA0.S, P0/M, P0/M, Z0.S, Z2.S
	WORD $0x80830001 // FMOPA ZA1.S, P0/M, P0/M, Z0.S, Z3.S
	WORD $0x80820022 // FMOPA ZA2.S, P0/M, P0/M, Z1.S, Z2.S
	WORD $0x80830023 // FMOPA ZA3.S, P0/M, P0/M, Z1.S, Z3.S
	ADD $64, R8, R8
	ADD $64, R9, R9
	SUB $1, R11, R11
	CBNZ R11, k_loop_1

k_done:
	WORD $0x2598E3E7 // PTRUE P7.S
	WORD $0x05A1BFEB // LASTB W11, P7, Z31.S
	CBZ R11, tile_retry

store_retry:
	WORD $0x2598E3E7 // PTRUE P7.S
	WORD $0x25B8C03F // MOV Z31.S, #1
	MOVD R20, R14
	WORD $0x25A615C0 // WHILELT P0.S, X14, X6
	ADD $16, R20, R14
	WORD $0x25A615C1 // WHILELT P1.S, X14, X6

	MUL R19, R3, R10
	ADD R2, R10, R10
	ADD R20<<2, R10, R10

	MOVD R10, R16
	LSL $4, R3, R15
	ADD R10, R15, R17

	MOVD $0, R12
	MOVD R19, R13
	ADD $16, R19, R15

store_row_loop:
	CMP R5, R13
	BGE store_done

	WORD $0xC0820000 // MOVA Z0.S, P0/M, ZA0H.S[W12, 0]
	WORD $0xC0820481 // MOVA Z1.S, P1/M, ZA1H.S[W12, 0]
	WORD $0xE540E200 // ST1W {Z0.S}, P0, [X16, #0, MUL VL]
	WORD $0xE541E601 // ST1W {Z1.S}, P1, [X16, #1, MUL VL]

	CMP R5, R15
	BGE store_skip_bot

	WORD $0xC0820102 // MOVA Z2.S, P0/M, ZA2H.S[W12, 0]
	WORD $0xC0820583 // MOVA Z3.S, P1/M, ZA3H.S[W12, 0]
	WORD $0xE540E222 // ST1W {Z2.S}, P0, [X17, #0, MUL VL]
	WORD $0xE541E623 // ST1W {Z3.S}, P1, [X17, #1, MUL VL]

store_skip_bot:
	ADD $1, R12, R12
	ADD $1, R13, R13
	ADD $1, R15, R15
	ADD R3, R16, R16
	ADD R3, R17, R17
	CMP $16, R12
	BLT store_row_loop

store_done:
	WORD $0x2598E3E7 // PTRUE P7.S
	WORD $0x05A1BFEB // LASTB W11, P7, Z31.S
	CBZ R11, store_retry

	ADD $32, R20, R20
	B col_strip_loop

next_row_strip:
	ADD $32, R19, R19
	B row_strip_loop

sme_bf16_32x32_done:
	WORD $0xD503467F // SMSTOP
	RET

// ---------------------------------------------------------------------------
// 8. smeLargeKernelFloat64Asm_8x8 (1x1 ZA.D tile: ZA0.D = 8x8 float64)
// ---------------------------------------------------------------------------
TEXT ·smeLargeKernelFloat64Asm_8x8(SB), NOSPLIT, $0-113
	MOVD packedLHS_base+0(FP), R0
	MOVD packedRHS_base+24(FP), R1
	MOVD packedOutput_base+48(FP), R2
	MOVD rhsPanelCols+80(FP), R3
	MOVD contractingLen+88(FP), R4
	MOVD lhsActiveRows+96(FP), R5
	MOVD rhsActiveCols+104(FP), R6
	MOVBU accumulate+112(FP), R7

	LSL $3, R3, R3 // outputRowByteStride = rhsPanelCols * 8
	MOVD $0, R19

	WORD $0xD503477F // SMSTART

row_strip_loop:
	CMP R5, R19
	BGE sme_f64_8x8_done

	MOVD $0, R20

col_strip_loop:
	CMP R6, R20
	BGE next_row_strip

tile_retry:
	WORD $0x25D8E3E7 // PTRUE P7.D
	WORD $0x25F8C03F // MOV Z31.D, #1
	WORD $0x25D8E3E0 // PTRUE P0.D
	WORD $0xC00800FF // ZERO {ZA}

	CBZ R7, init_za_done

	WORD $0x25E61681 // WHILELT P1.D, X20, X6
	MUL R19, R3, R10
	ADD R2, R10, R10
	ADD R20<<3, R10, R10
	MOVD $0, R12
	MOVD R19, R13

init_za_row_loop:
	CMP R5, R13
	BGE init_za_done
	WORD $0xA5E0A540 // LD1D {Z0.D}, P1/Z, [X10, #0, MUL VL]
	WORD $0xC0C00400 // MOVA ZA0H.D[W12, 0], P1/M, Z0.D
	ADD $1, R12, R12
	ADD $1, R13, R13
	ADD R3, R10, R10
	CMP $8, R12
	BLT init_za_row_loop

init_za_done:
	MUL R19, R4, R8
	LSL $3, R8, R8
	ADD R0, R8, R8

	MUL R20, R4, R9
	LSL $3, R9, R9
	ADD R1, R9, R9

	MOVD R4, R11
	CBZ R11, k_done

k_loop_1:
	WORD $0xA5E0A100 // LD1D {Z0.D}, P0/Z, [X8, #0, MUL VL]
	WORD $0xA5E0A121 // LD1D {Z1.D}, P0/Z, [X9, #0, MUL VL]
	WORD $0x80C10000 // FMOPA ZA0.D, P0/M, P0/M, Z0.D, Z1.D
	ADD $64, R8, R8
	ADD $64, R9, R9
	SUB $1, R11, R11
	CBNZ R11, k_loop_1

k_done:
	WORD $0x25D8E3E7 // PTRUE P7.D
	WORD $0x05E1BFEB // LASTB X11, P7, Z31.D
	CBZ R11, tile_retry

store_retry:
	WORD $0x25D8E3E7 // PTRUE P7.D
	WORD $0x25F8C03F // MOV Z31.D, #1
	WORD $0x25E61680 // WHILELT P0.D, X20, X6

	MUL R19, R3, R10
	ADD R2, R10, R10
	ADD R20<<3, R10, R10

	MOVD $0, R12
	MOVD R19, R13

store_row_loop:
	CMP R5, R13
	BGE store_done

	WORD $0xC0C20000 // MOVA Z0.D, P0/M, ZA0H.D[W12, 0]
	WORD $0xE5E0E140 // ST1D {Z0.D}, P0, [X10, #0, MUL VL]

	ADD $1, R12, R12
	ADD $1, R13, R13
	ADD R3, R10, R10
	CMP $8, R12
	BLT store_row_loop

store_done:
	WORD $0x25D8E3E7 // PTRUE P7.D
	WORD $0x05E1BFEB // LASTB X11, P7, Z31.D
	CBZ R11, store_retry

	ADD $8, R20, R20
	B col_strip_loop

next_row_strip:
	ADD $8, R19, R19
	B row_strip_loop

sme_f64_8x8_done:
	WORD $0xD503467F // SMSTOP
	RET

// ---------------------------------------------------------------------------
// 9. smeLargeKernelFloat64Asm_16x16 (2x2 ZA.D tiles: ZA0..3.D = 16x16 float64)
// ---------------------------------------------------------------------------
TEXT ·smeLargeKernelFloat64Asm_16x16(SB), NOSPLIT, $0-113
	MOVD packedLHS_base+0(FP), R0
	MOVD packedRHS_base+24(FP), R1
	MOVD packedOutput_base+48(FP), R2
	MOVD rhsPanelCols+80(FP), R3
	MOVD contractingLen+88(FP), R4
	MOVD lhsActiveRows+96(FP), R5
	MOVD rhsActiveCols+104(FP), R6
	MOVBU accumulate+112(FP), R7

	LSL $3, R3, R3 // outputRowByteStride = rhsPanelCols * 8
	MOVD $0, R19

	WORD $0xD503477F // SMSTART

row_strip_loop:
	CMP R5, R19
	BGE sme_f64_16x16_done

	MOVD $0, R20

col_strip_loop:
	CMP R6, R20
	BGE next_row_strip

tile_retry:
	WORD $0x25D8E3E7 // PTRUE P7.D
	WORD $0x25F8C03F // MOV Z31.D, #1
	WORD $0x25D8E3E0 // PTRUE P0.D
	WORD $0xC00800FF // ZERO {ZA}

	CBZ R7, init_za_done

	MOVD R20, R14
	WORD $0x25E615C1 // WHILELT P1.D, X14, X6
	ADD $8, R20, R14
	WORD $0x25E615C2 // WHILELT P2.D, X14, X6

	MUL R19, R3, R10
	ADD R2, R10, R10
	ADD R20<<3, R10, R10
	MOVD R10, R16
	LSL $3, R3, R15
	ADD R10, R15, R17

	MOVD $0, R12
	MOVD R19, R13
	ADD $8, R19, R15

init_za_row_loop:
	CMP R5, R13
	BGE init_za_done

	WORD $0xA5E0A600 // LD1D {Z0.D}, P1/Z, [X16, #0, MUL VL]
	WORD $0xA5E1AA01 // LD1D {Z1.D}, P2/Z, [X16, #1, MUL VL]
	WORD $0xC0C00400 // MOVA ZA0H.D[W12, 0], P1/M, Z0.D
	WORD $0xC0C00822 // MOVA ZA1H.D[W12, 0], P2/M, Z1.D

	CMP R5, R15
	BGE init_za_skip_bot
	WORD $0xA5E0A622 // LD1D {Z2.D}, P1/Z, [X17, #0, MUL VL]
	WORD $0xA5E1AA23 // LD1D {Z3.D}, P2/Z, [X17, #1, MUL VL]
	WORD $0xC0C00444 // MOVA ZA2H.D[W12, 0], P1/M, Z2.D
	WORD $0xC0C00866 // MOVA ZA3H.D[W12, 0], P2/M, Z3.D

init_za_skip_bot:
	ADD $1, R12, R12
	ADD $1, R13, R13
	ADD $1, R15, R15
	ADD R3, R16, R16
	ADD R3, R17, R17
	CMP $8, R12
	BLT init_za_row_loop

init_za_done:
	MUL R19, R4, R8
	LSL $3, R8, R8
	ADD R0, R8, R8

	MUL R20, R4, R9
	LSL $3, R9, R9
	ADD R1, R9, R9

	MOVD R4, R11
	CMP $2, R11
	BLT k_rem_check

k_loop_2:
	// Step 0
	WORD $0xA5E0A100 // LD1D {Z0.D}, P0/Z, [X8, #0, MUL VL]
	WORD $0xA5E1A101 // LD1D {Z1.D}, P0/Z, [X8, #1, MUL VL]
	WORD $0xA5E0A122 // LD1D {Z2.D}, P0/Z, [X9, #0, MUL VL]
	WORD $0xA5E1A123 // LD1D {Z3.D}, P0/Z, [X9, #1, MUL VL]
	WORD $0x80C20000 // FMOPA ZA0.D, P0/M, P0/M, Z0.D, Z2.D
	WORD $0x80C30001 // FMOPA ZA1.D, P0/M, P0/M, Z0.D, Z3.D
	WORD $0x80C20022 // FMOPA ZA2.D, P0/M, P0/M, Z1.D, Z2.D
	WORD $0x80C30023 // FMOPA ZA3.D, P0/M, P0/M, Z1.D, Z3.D

	// Step 1
	WORD $0xA5E2A104 // LD1D {Z4.D}, P0/Z, [X8, #2, MUL VL]
	WORD $0xA5E3A105 // LD1D {Z5.D}, P0/Z, [X8, #3, MUL VL]
	WORD $0xA5E2A126 // LD1D {Z6.D}, P0/Z, [X9, #2, MUL VL]
	WORD $0xA5E3A127 // LD1D {Z7.D}, P0/Z, [X9, #3, MUL VL]
	WORD $0x80C60080 // FMOPA ZA0.D, P0/M, P0/M, Z4.D, Z6.D
	WORD $0x80C70081 // FMOPA ZA1.D, P0/M, P0/M, Z4.D, Z7.D
	WORD $0x80C600A2 // FMOPA ZA2.D, P0/M, P0/M, Z5.D, Z6.D
	WORD $0x80C700A3 // FMOPA ZA3.D, P0/M, P0/M, Z5.D, Z7.D

	ADD $256, R8, R8
	ADD $256, R9, R9
	SUB $2, R11, R11
	CMP $2, R11
	BGE k_loop_2

k_rem_check:
	CBZ R11, k_done

k_loop_1:
	WORD $0xA5E0A100 // LD1D {Z0.D}, P0/Z, [X8, #0, MUL VL]
	WORD $0xA5E1A101 // LD1D {Z1.D}, P0/Z, [X8, #1, MUL VL]
	WORD $0xA5E0A122 // LD1D {Z2.D}, P0/Z, [X9, #0, MUL VL]
	WORD $0xA5E1A123 // LD1D {Z3.D}, P0/Z, [X9, #1, MUL VL]
	WORD $0x80C20000 // FMOPA ZA0.D, P0/M, P0/M, Z0.D, Z2.D
	WORD $0x80C30001 // FMOPA ZA1.D, P0/M, P0/M, Z0.D, Z3.D
	WORD $0x80C20022 // FMOPA ZA2.D, P0/M, P0/M, Z1.D, Z2.D
	WORD $0x80C30023 // FMOPA ZA3.D, P0/M, P0/M, Z1.D, Z3.D
	ADD $128, R8, R8
	ADD $128, R9, R9
	SUB $1, R11, R11
	CBNZ R11, k_loop_1

k_done:
	WORD $0x25D8E3E7 // PTRUE P7.D
	WORD $0x05E1BFEB // LASTB X11, P7, Z31.D
	CBZ R11, tile_retry

store_retry:
	WORD $0x25D8E3E7 // PTRUE P7.D
	WORD $0x25F8C03F // MOV Z31.D, #1
	MOVD R20, R14
	WORD $0x25E615C0 // WHILELT P0.D, X14, X6
	ADD $8, R20, R14
	WORD $0x25E615C1 // WHILELT P1.D, X14, X6

	MUL R19, R3, R10
	ADD R2, R10, R10
	ADD R20<<3, R10, R10

	MOVD R10, R16
	LSL $3, R3, R15
	ADD R10, R15, R17

	MOVD $0, R12
	MOVD R19, R13
	ADD $8, R19, R15

store_row_loop:
	CMP R5, R13
	BGE store_done

	WORD $0xC0C20000 // MOVA Z0.D, P0/M, ZA0H.D[W12, 0]
	WORD $0xC0C20441 // MOVA Z1.D, P1/M, ZA1H.D[W12, 0]
	WORD $0xE5E0E200 // ST1D {Z0.D}, P0, [X16, #0, MUL VL]
	WORD $0xE5E1E601 // ST1D {Z1.D}, P1, [X16, #1, MUL VL]

	CMP R5, R15
	BGE store_skip_bot

	WORD $0xC0C20082 // MOVA Z2.D, P0/M, ZA2H.D[W12, 0]
	WORD $0xC0C204C3 // MOVA Z3.D, P1/M, ZA3H.D[W12, 0]
	WORD $0xE5E0E222 // ST1D {Z2.D}, P0, [X17, #0, MUL VL]
	WORD $0xE5E1E623 // ST1D {Z3.D}, P1, [X17, #1, MUL VL]

store_skip_bot:
	ADD $1, R12, R12
	ADD $1, R13, R13
	ADD $1, R15, R15
	ADD R3, R16, R16
	ADD R3, R17, R17
	CMP $8, R12
	BLT store_row_loop

store_done:
	WORD $0x25D8E3E7 // PTRUE P7.D
	WORD $0x05E1BFEB // LASTB X11, P7, Z31.D
	CBZ R11, store_retry

	ADD $16, R20, R20
	B col_strip_loop

next_row_strip:
	ADD $16, R19, R19
	B row_strip_loop

sme_f64_16x16_done:
	WORD $0xD503467F // SMSTOP
	RET

// ===========================================================================
// Hybrid SME + NEON Cluster Probing & Strided Sub-Tile NEON Microkernels
// ===========================================================================

#define FMLA_4S_LANE(Vd, Vn, Vm, lane) \
	WORD $(0x4F801000 | (((lane)&1)<<21) | (((lane)&2)<<10) | ((Vm)<<16) | ((Vn)<<5) | (Vd))
#define FMLA_2D_LANE(Vd, Vn, Vm, lane) \
	WORD $(0x4FC01000 | (((lane)&1)<<11) | ((Vm)<<16) | ((Vn)<<5) | (Vd))
#define FCVTL_4S_4H(Vd, Vn)   WORD $(0x0E217800 | ((Vn)<<5) | (Vd))
#define FCVTL2_4S_8H(Vd, Vn)  WORD $(0x4E217800 | ((Vn)<<5) | (Vd))
#define SHLL_4S_4H_16(Vd, Vn)  WORD $(0x2E613800 | ((Vn)<<5) | (Vd))
#define SHLL2_4S_8H_16(Vd, Vn) WORD $(0x6E613800 | ((Vn)<<5) | (Vd))

// func smeProbeCoreClockTicks() uint64
// Executes 256 iterations of 16 independent per-core NEON VFMLA instructions
// timed against the hardware virtual counter CNTVCT_EL0 (1 GHz = 1 ns/tick on Apple Silicon).
// Because NEON execution pipelines are private to each core (4x 128-bit FMA pipes on
// Super/Performance cores vs. 2x on Efficiency cores), this ~250-460ns probe cleanly
// identifies which physical core cluster (Super, Performance, or Efficiency) the thread is on.
TEXT ·smeProbeCoreClockTicks(SB), NOSPLIT, $0-8
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16
	VEOR V4.B16, V4.B16, V4.B16
	VEOR V5.B16, V5.B16, V5.B16
	VEOR V6.B16, V6.B16, V6.B16
	VEOR V7.B16, V7.B16, V7.B16
	VEOR V8.B16, V8.B16, V8.B16
	VEOR V9.B16, V9.B16, V9.B16
	VEOR V10.B16, V10.B16, V10.B16
	VEOR V11.B16, V11.B16, V11.B16
	VEOR V12.B16, V12.B16, V12.B16
	VEOR V13.B16, V13.B16, V13.B16
	VEOR V14.B16, V14.B16, V14.B16
	VEOR V15.B16, V15.B16, V15.B16
	WORD $0xD5033FDF // ISB
	WORD $0xD53BE041 // MRS X1, CNTVCT_EL0
	MOVD $256, R0
sme_probe_loop:
	VFMLA V16.S4, V17.S4, V0.S4
	VFMLA V16.S4, V18.S4, V1.S4
	VFMLA V16.S4, V19.S4, V2.S4
	VFMLA V16.S4, V20.S4, V3.S4
	VFMLA V21.S4, V17.S4, V4.S4
	VFMLA V21.S4, V18.S4, V5.S4
	VFMLA V21.S4, V19.S4, V6.S4
	VFMLA V21.S4, V20.S4, V7.S4
	VFMLA V22.S4, V17.S4, V8.S4
	VFMLA V22.S4, V18.S4, V9.S4
	VFMLA V22.S4, V19.S4, V10.S4
	VFMLA V22.S4, V20.S4, V11.S4
	VFMLA V23.S4, V17.S4, V12.S4
	VFMLA V23.S4, V18.S4, V13.S4
	VFMLA V23.S4, V19.S4, V14.S4
	VFMLA V23.S4, V20.S4, V15.S4
	SUB $1, R0, R0
	CBNZ R0, sme_probe_loop
	WORD $0xD5033FDF // ISB
	WORD $0xD53BE042 // MRS X2, CNTVCT_EL0
	SUB R1, R2, R0
	MOVD R0, ret+0(FP)
	RET

// func neonSubTile4x16StridedFloat32Asm(lhsPtr, rhsPtr, outPtr unsafe.Pointer, contractingLen, lhsStepBytes, rhsStepBytes, outRowStrideBytes int, accumulate bool)
TEXT ·neonSubTile4x16StridedFloat32Asm(SB), NOSPLIT, $0-57
	MOVD lhsPtr+0(FP), R0
	MOVD rhsPtr+8(FP), R1
	MOVD outPtr+16(FP), R2
	MOVD contractingLen+24(FP), R3
	MOVD lhsStepBytes+32(FP), R4
	MOVD rhsStepBytes+40(FP), R5
	MOVD outRowStrideBytes+48(FP), R6
	MOVBU accumulate+56(FP), R7

	CBZ R7, zero_f32
	MOVD R2, R8
	VLD1 (R8), [V0.S4, V1.S4, V2.S4, V3.S4]
	ADD R6, R8, R8
	VLD1 (R8), [V4.S4, V5.S4, V6.S4, V7.S4]
	ADD R6, R8, R8
	VLD1 (R8), [V8.S4, V9.S4, V10.S4, V11.S4]
	ADD R6, R8, R8
	VLD1 (R8), [V12.S4, V13.S4, V14.S4, V15.S4]
	B k_check_f32

zero_f32:
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16
	VEOR V4.B16, V4.B16, V4.B16
	VEOR V5.B16, V5.B16, V5.B16
	VEOR V6.B16, V6.B16, V6.B16
	VEOR V7.B16, V7.B16, V7.B16
	VEOR V8.B16, V8.B16, V8.B16
	VEOR V9.B16, V9.B16, V9.B16
	VEOR V10.B16, V10.B16, V10.B16
	VEOR V11.B16, V11.B16, V11.B16
	VEOR V12.B16, V12.B16, V12.B16
	VEOR V13.B16, V13.B16, V13.B16
	VEOR V14.B16, V14.B16, V14.B16
	VEOR V15.B16, V15.B16, V15.B16

k_check_f32:
	CBZ R3, store_f32

k_loop_f32:
	VLD1 (R0), [V16.S4]
	ADD R4, R0, R0
	VLD1 (R1), [V17.S4, V18.S4, V19.S4, V20.S4]
	ADD R5, R1, R1

	FMLA_4S_LANE(0, 17, 16, 0)
	FMLA_4S_LANE(1, 18, 16, 0)
	FMLA_4S_LANE(2, 19, 16, 0)
	FMLA_4S_LANE(3, 20, 16, 0)

	FMLA_4S_LANE(4, 17, 16, 1)
	FMLA_4S_LANE(5, 18, 16, 1)
	FMLA_4S_LANE(6, 19, 16, 1)
	FMLA_4S_LANE(7, 20, 16, 1)

	FMLA_4S_LANE(8, 17, 16, 2)
	FMLA_4S_LANE(9, 18, 16, 2)
	FMLA_4S_LANE(10, 19, 16, 2)
	FMLA_4S_LANE(11, 20, 16, 2)

	FMLA_4S_LANE(12, 17, 16, 3)
	FMLA_4S_LANE(13, 18, 16, 3)
	FMLA_4S_LANE(14, 19, 16, 3)
	FMLA_4S_LANE(15, 20, 16, 3)

	SUB $1, R3, R3
	CBNZ R3, k_loop_f32

store_f32:
	VST1 [V0.S4, V1.S4, V2.S4, V3.S4], (R2)
	ADD R6, R2, R2
	VST1 [V4.S4, V5.S4, V6.S4, V7.S4], (R2)
	ADD R6, R2, R2
	VST1 [V8.S4, V9.S4, V10.S4, V11.S4], (R2)
	ADD R6, R2, R2
	VST1 [V12.S4, V13.S4, V14.S4, V15.S4], (R2)
	RET

// func neonSubTile8x8StridedFloat16Asm(lhsPtr, rhsPtr, outPtr unsafe.Pointer, contractingLen, lhsStepBytes, rhsStepBytes, outRowStrideBytes int, accumulate bool)
TEXT ·neonSubTile8x8StridedFloat16Asm(SB), NOSPLIT, $0-57
	MOVD lhsPtr+0(FP), R0
	MOVD rhsPtr+8(FP), R1
	MOVD outPtr+16(FP), R2
	MOVD contractingLen+24(FP), R3
	MOVD lhsStepBytes+32(FP), R4
	MOVD rhsStepBytes+40(FP), R5
	MOVD outRowStrideBytes+48(FP), R6
	MOVBU accumulate+56(FP), R7

	CBZ R7, zero_f16
	MOVD R2, R8
	VLD1 (R8), [V0.S4, V1.S4]
	ADD R6, R8, R8
	VLD1 (R8), [V2.S4, V3.S4]
	ADD R6, R8, R8
	VLD1 (R8), [V4.S4, V5.S4]
	ADD R6, R8, R8
	VLD1 (R8), [V6.S4, V7.S4]
	ADD R6, R8, R8
	VLD1 (R8), [V8.S4, V9.S4]
	ADD R6, R8, R8
	VLD1 (R8), [V10.S4, V11.S4]
	ADD R6, R8, R8
	VLD1 (R8), [V12.S4, V13.S4]
	ADD R6, R8, R8
	VLD1 (R8), [V14.S4, V15.S4]
	B k_check_f16

zero_f16:
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16
	VEOR V4.B16, V4.B16, V4.B16
	VEOR V5.B16, V5.B16, V5.B16
	VEOR V6.B16, V6.B16, V6.B16
	VEOR V7.B16, V7.B16, V7.B16
	VEOR V8.B16, V8.B16, V8.B16
	VEOR V9.B16, V9.B16, V9.B16
	VEOR V10.B16, V10.B16, V10.B16
	VEOR V11.B16, V11.B16, V11.B16
	VEOR V12.B16, V12.B16, V12.B16
	VEOR V13.B16, V13.B16, V13.B16
	VEOR V14.B16, V14.B16, V14.B16
	VEOR V15.B16, V15.B16, V15.B16

k_check_f16:
	CBZ R3, store_f16

k_loop_f16:
	VLD1 (R0), [V20.H8]
	ADD R4, R0, R0
	VLD1 (R1), [V21.H8]
	ADD R5, R1, R1

	FCVTL_4S_4H(16, 20)
	FCVTL2_4S_8H(17, 20)
	FCVTL_4S_4H(18, 21)
	FCVTL2_4S_8H(19, 21)

	FMLA_4S_LANE(0, 18, 16, 0)
	FMLA_4S_LANE(1, 19, 16, 0)
	FMLA_4S_LANE(2, 18, 16, 1)
	FMLA_4S_LANE(3, 19, 16, 1)
	FMLA_4S_LANE(4, 18, 16, 2)
	FMLA_4S_LANE(5, 19, 16, 2)
	FMLA_4S_LANE(6, 18, 16, 3)
	FMLA_4S_LANE(7, 19, 16, 3)

	FMLA_4S_LANE(8, 18, 17, 0)
	FMLA_4S_LANE(9, 19, 17, 0)
	FMLA_4S_LANE(10, 18, 17, 1)
	FMLA_4S_LANE(11, 19, 17, 1)
	FMLA_4S_LANE(12, 18, 17, 2)
	FMLA_4S_LANE(13, 19, 17, 2)
	FMLA_4S_LANE(14, 18, 17, 3)
	FMLA_4S_LANE(15, 19, 17, 3)

	SUB $1, R3, R3
	CBNZ R3, k_loop_f16

store_f16:
	VST1 [V0.S4, V1.S4], (R2)
	ADD R6, R2, R2
	VST1 [V2.S4, V3.S4], (R2)
	ADD R6, R2, R2
	VST1 [V4.S4, V5.S4], (R2)
	ADD R6, R2, R2
	VST1 [V6.S4, V7.S4], (R2)
	ADD R6, R2, R2
	VST1 [V8.S4, V9.S4], (R2)
	ADD R6, R2, R2
	VST1 [V10.S4, V11.S4], (R2)
	ADD R6, R2, R2
	VST1 [V12.S4, V13.S4], (R2)
	ADD R6, R2, R2
	VST1 [V14.S4, V15.S4], (R2)
	RET

// func neonSubTile8x8StridedBFloat16Asm(lhsPtr, rhsPtr, outPtr unsafe.Pointer, contractingLen, lhsStepBytes, rhsStepBytes, outRowStrideBytes int, accumulate bool)
TEXT ·neonSubTile8x8StridedBFloat16Asm(SB), NOSPLIT, $0-57
	MOVD lhsPtr+0(FP), R0
	MOVD rhsPtr+8(FP), R1
	MOVD outPtr+16(FP), R2
	MOVD contractingLen+24(FP), R3
	MOVD lhsStepBytes+32(FP), R4
	MOVD rhsStepBytes+40(FP), R5
	MOVD outRowStrideBytes+48(FP), R6
	MOVBU accumulate+56(FP), R7

	CBZ R7, zero_bf16
	MOVD R2, R8
	VLD1 (R8), [V0.S4, V1.S4]
	ADD R6, R8, R8
	VLD1 (R8), [V2.S4, V3.S4]
	ADD R6, R8, R8
	VLD1 (R8), [V4.S4, V5.S4]
	ADD R6, R8, R8
	VLD1 (R8), [V6.S4, V7.S4]
	ADD R6, R8, R8
	VLD1 (R8), [V8.S4, V9.S4]
	ADD R6, R8, R8
	VLD1 (R8), [V10.S4, V11.S4]
	ADD R6, R8, R8
	VLD1 (R8), [V12.S4, V13.S4]
	ADD R6, R8, R8
	VLD1 (R8), [V14.S4, V15.S4]
	B k_check_bf16

zero_bf16:
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16
	VEOR V4.B16, V4.B16, V4.B16
	VEOR V5.B16, V5.B16, V5.B16
	VEOR V6.B16, V6.B16, V6.B16
	VEOR V7.B16, V7.B16, V7.B16
	VEOR V8.B16, V8.B16, V8.B16
	VEOR V9.B16, V9.B16, V9.B16
	VEOR V10.B16, V10.B16, V10.B16
	VEOR V11.B16, V11.B16, V11.B16
	VEOR V12.B16, V12.B16, V12.B16
	VEOR V13.B16, V13.B16, V13.B16
	VEOR V14.B16, V14.B16, V14.B16
	VEOR V15.B16, V15.B16, V15.B16

k_check_bf16:
	CBZ R3, store_bf16

k_loop_bf16:
	VLD1 (R0), [V20.H8]
	ADD R4, R0, R0
	VLD1 (R1), [V21.H8]
	ADD R5, R1, R1

	SHLL_4S_4H_16(16, 20)
	SHLL2_4S_8H_16(17, 20)
	SHLL_4S_4H_16(18, 21)
	SHLL2_4S_8H_16(19, 21)

	FMLA_4S_LANE(0, 18, 16, 0)
	FMLA_4S_LANE(1, 19, 16, 0)
	FMLA_4S_LANE(2, 18, 16, 1)
	FMLA_4S_LANE(3, 19, 16, 1)
	FMLA_4S_LANE(4, 18, 16, 2)
	FMLA_4S_LANE(5, 19, 16, 2)
	FMLA_4S_LANE(6, 18, 16, 3)
	FMLA_4S_LANE(7, 19, 16, 3)

	FMLA_4S_LANE(8, 18, 17, 0)
	FMLA_4S_LANE(9, 19, 17, 0)
	FMLA_4S_LANE(10, 18, 17, 1)
	FMLA_4S_LANE(11, 19, 17, 1)
	FMLA_4S_LANE(12, 18, 17, 2)
	FMLA_4S_LANE(13, 19, 17, 2)
	FMLA_4S_LANE(14, 18, 17, 3)
	FMLA_4S_LANE(15, 19, 17, 3)

	SUB $1, R3, R3
	CBNZ R3, k_loop_bf16

store_bf16:
	VST1 [V0.S4, V1.S4], (R2)
	ADD R6, R2, R2
	VST1 [V2.S4, V3.S4], (R2)
	ADD R6, R2, R2
	VST1 [V4.S4, V5.S4], (R2)
	ADD R6, R2, R2
	VST1 [V6.S4, V7.S4], (R2)
	ADD R6, R2, R2
	VST1 [V8.S4, V9.S4], (R2)
	ADD R6, R2, R2
	VST1 [V10.S4, V11.S4], (R2)
	ADD R6, R2, R2
	VST1 [V12.S4, V13.S4], (R2)
	ADD R6, R2, R2
	VST1 [V14.S4, V15.S4], (R2)
	RET

// func neonSubTile8x4StridedFloat64Asm(lhsPtr, rhsPtr, outPtr unsafe.Pointer, contractingLen, lhsStepBytes, rhsStepBytes, outRowStrideBytes int, accumulate bool)
TEXT ·neonSubTile8x4StridedFloat64Asm(SB), NOSPLIT, $0-57
	MOVD lhsPtr+0(FP), R0
	MOVD rhsPtr+8(FP), R1
	MOVD outPtr+16(FP), R2
	MOVD contractingLen+24(FP), R3
	MOVD lhsStepBytes+32(FP), R4
	MOVD rhsStepBytes+40(FP), R5
	MOVD outRowStrideBytes+48(FP), R6
	MOVBU accumulate+56(FP), R7

	CBZ R7, zero_f64
	MOVD R2, R8
	VLD1 (R8), [V0.D2, V1.D2]
	ADD R6, R8, R8
	VLD1 (R8), [V2.D2, V3.D2]
	ADD R6, R8, R8
	VLD1 (R8), [V4.D2, V5.D2]
	ADD R6, R8, R8
	VLD1 (R8), [V6.D2, V7.D2]
	ADD R6, R8, R8
	VLD1 (R8), [V8.D2, V9.D2]
	ADD R6, R8, R8
	VLD1 (R8), [V10.D2, V11.D2]
	ADD R6, R8, R8
	VLD1 (R8), [V12.D2, V13.D2]
	ADD R6, R8, R8
	VLD1 (R8), [V14.D2, V15.D2]
	B k_check_f64

zero_f64:
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16
	VEOR V4.B16, V4.B16, V4.B16
	VEOR V5.B16, V5.B16, V5.B16
	VEOR V6.B16, V6.B16, V6.B16
	VEOR V7.B16, V7.B16, V7.B16
	VEOR V8.B16, V8.B16, V8.B16
	VEOR V9.B16, V9.B16, V9.B16
	VEOR V10.B16, V10.B16, V10.B16
	VEOR V11.B16, V11.B16, V11.B16
	VEOR V12.B16, V12.B16, V12.B16
	VEOR V13.B16, V13.B16, V13.B16
	VEOR V14.B16, V14.B16, V14.B16
	VEOR V15.B16, V15.B16, V15.B16

k_check_f64:
	CBZ R3, store_f64

k_loop_f64:
	VLD1 (R0), [V16.D2, V17.D2, V18.D2, V19.D2]
	ADD R4, R0, R0
	VLD1 (R1), [V20.D2, V21.D2]
	ADD R5, R1, R1

	FMLA_2D_LANE(0, 20, 16, 0)
	FMLA_2D_LANE(1, 21, 16, 0)
	FMLA_2D_LANE(2, 20, 16, 1)
	FMLA_2D_LANE(3, 21, 16, 1)

	FMLA_2D_LANE(4, 20, 17, 0)
	FMLA_2D_LANE(5, 21, 17, 0)
	FMLA_2D_LANE(6, 20, 17, 1)
	FMLA_2D_LANE(7, 21, 17, 1)

	FMLA_2D_LANE(8, 20, 18, 0)
	FMLA_2D_LANE(9, 21, 18, 0)
	FMLA_2D_LANE(10, 20, 18, 1)
	FMLA_2D_LANE(11, 21, 18, 1)

	FMLA_2D_LANE(12, 20, 19, 0)
	FMLA_2D_LANE(13, 21, 19, 0)
	FMLA_2D_LANE(14, 20, 19, 1)
	FMLA_2D_LANE(15, 21, 19, 1)

	SUB $1, R3, R3
	CBNZ R3, k_loop_f64

store_f64:
	VST1 [V0.D2, V1.D2], (R2)
	ADD R6, R2, R2
	VST1 [V2.D2, V3.D2], (R2)
	ADD R6, R2, R2
	VST1 [V4.D2, V5.D2], (R2)
	ADD R6, R2, R2
	VST1 [V6.D2, V7.D2], (R2)
	ADD R6, R2, R2
	VST1 [V8.D2, V9.D2], (R2)
	ADD R6, R2, R2
	VST1 [V10.D2, V11.D2], (R2)
	ADD R6, R2, R2
	VST1 [V12.D2, V13.D2], (R2)
	ADD R6, R2, R2
	VST1 [V14.D2, V15.D2], (R2)
	RET

