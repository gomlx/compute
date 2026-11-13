// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build arm64 && goexperiment.simd

#include "textflag.h"

// ===========================================================================
// ARM64 SME2 (Scalable Matrix Extension 2) Quantized Panel Microkernel
// ===========================================================================
//
// Target: ARMv9.2-A with FEAT_SME2 (512-bit streaming vector length SVL = 64 bytes:
// 64x int8/uint8 or 16x int32/float32 per Z register, and four 16x16 32-bit ZA0.S..ZA3.S tiles).
//
// Algorithm (16 rows x 16 cols per strip, inside a single SMSTART..SMSTOP session):
//   Phase 0 (In-SME Hardware 16x16 32-bit Matrix Transpose):
//     Uses LD1W {ZA0H.S[W12, 0]} + ST1W {ZA0V.S[W12, 0]} to transpose 16 rows of:
//       - Qx:      [activeRows, totalQuads]uint32       -> packedQx      [totalQuads, 16]uint32
//       - DxPairs: [activeRows, numSubBlocks*2]float32 -> packedDxPairs [numSubBlocks*2, 16]float32
//   Phase 1 (Compute 16x16 Tile Across Sub-Blocks, Read-Only on Output Memory):
//     - Z12..Z27 (four 4-vector aligned groups) hold the 16x16 float32 accumulators for
//       sum_{sb} float32(ZA0.S[r, c]) * (dx[r, sb] * d'_w[sb, c]).
//     - ZA1.S holds the 16x16 float32 accumulator for
//       Bias + sum_{sb} (dx*sumQx)[r, sb] * m'_w[sb, c] (accumulated via 1 FMOPA per sub-block).
//     - For each sub-block sb:
//       1. ZERO {ZA0.S} (preserves ZA1.S).
//       2. Execute quadsPerSB (8 or 4) SUMOPA ZA0.S, P0/M, P0/M, Z_qx.B, Z_uw.B instructions
//          (each SUMOPA computes 16x16x4 = 1,024 signed int8 x unsigned uint8 -> int32 MACs).
//       3. Execute 1 FMOPA ZA1.S, P1/M, P1/M, Z_dxSum.S, Z_mw.S instruction (256 FP32 FMAs).
//       4. Extract 4 rows at a time from ZA0H.S via SME2 4-vector MOVA {Z0.S-Z3.S}, ZA0H.S[W12, 0:3],
//          convert 64 int32s to float32 in 1 instruction via SCVTF {Z0.S-Z3.S}, {Z0.S-Z3.S},
//          and accumulate into Z12..Z27 with dx[r, sb] * d'_w[sb, c].
//     - At sb_done, add ZA1H.S into Z12..Z27 and save the final 16x16 float32 result into ZA0H.S.
//     - Verify the Z31 signal-safety canary (LASTB W11, P7, Z31.S) before touching Out memory.
//   Phase 2 (Idempotent Store from ZA0H.S to Out):
//     - Stores ZA0H.S[W12, 0] to Out with column predicate P2 (WHILELT P2.S, X_col, X_chunkCols)
//       and verifies Z31 canary.

#define SME_TRANSPOSE_16_ROWS(retryLabel, loadLoopLabel, storeLoopLabel, doneLabel) \
	LSL $2, R4, R3; \
	MOVD $0, R9; \
retryLabel: \
	CMP R4, R9; \
	BGE doneLabel; \
	WORD $0x2598E3E7; /* PTRUE P7.S */ \
	WORD $0x25B8C03F; /* MOV Z31.S, #1 */ \
	WORD $0x2598E3E0; /* PTRUE P0.S */ \
	WORD $0xC0080011; /* ZERO {ZA0.S} */ \
	WORD $0x25A41521; /* WHILELT P1.S, X9, X4 */ \
	ADD R9<<2, R1, R16; \
	MOVD $0, R12; \
loadLoopLabel: \
	WORD $0xE09F0600; /* LD1W {ZA0H.S[W12, 0]}, P1/Z, [X16] */ \
	ADD R3, R16, R16; \
	ADD $1, R12, R12; \
	CMP R5, R12; \
	BLT loadLoopLabel; \
	SUB R9, R4, R13; \
	CMP $16, R13; \
	MOVD $16, R14; \
	CSEL LT, R13, R14, R13; \
	ADD R9<<6, R2, R10; \
	MOVD $0, R12; \
storeLoopLabel: \
	WORD $0xE0BF8140; /* ST1W {ZA0V.S[W12, 0]}, P0, [X10] */ \
	ADD $64, R10, R10; \
	ADD $1, R12, R12; \
	CMP R13, R12; \
	BLT storeLoopLabel; \
	WORD $0x2598E3E7; /* PTRUE P7.S */ \
	WORD $0x05A1BFEB; /* LASTB W11, P7, Z31.S */ \
	CBZ R11, retryLabel; \
	ADD $16, R9, R9; \
	B retryLabel; \
doneLabel:

// func smeQuantizedPanelRowsAsm(task *smeQuantizedTask)
TEXT ·smeQuantizedPanelRowsAsm(SB), NOSPLIT, $0-8
	MOVD task+0(FP), R0
	MOVD 56(R0), R20 // totalRows
	CMP $0, R20
	BLE sme_quant_done

	MOVD $0, R19 // rowOffset = 0

	WORD $0xD503477F // SMSTART (enter streaming mode + enable ZA)

row_block_loop:
	SUB R19, R20, R5 // remRows = totalRows - rowOffset
	CMP $16, R5
	MOVD $16, R14
	CSEL LT, R5, R14, R5 // R5 = activeRows = min(16, remRows)

	// -------------------------------------------------------------------
	// Phase 0: Transpose Qx and DxPairs for this 16-row block
	// -------------------------------------------------------------------
	// 0a. Transpose Qx: [activeRows, totalQuads]uint32 -> packedQx [totalQuads, 16]uint32
	MOVD 0(R0), R1  // qxPtr
	MOVD 16(R0), R2 // packedQxPtr
	MOVD 64(R0), R4 // totalQuads
	SME_TRANSPOSE_16_ROWS(pack_qx_retry, pack_qx_load, pack_qx_store, pack_qx_done)

	// 0b. Transpose DxPairs: [activeRows, numSubBlocks*2]float32 -> packedDxPairs [numSubBlocks*2, 16]float32
	MOVD 8(R0), R1  // dxPairsPtr
	MOVD 24(R0), R2 // packedDxPairsPtr
	MOVD 72(R0), R4 // numSubBlocks
	LSL $1, R4, R4  // numSubBlocks * 2
	SME_TRANSPOSE_16_ROWS(pack_dx_retry, pack_dx_load, pack_dx_store, pack_dx_done)

	// -------------------------------------------------------------------
	// Phase 1 & 2: Loop over column strips [0, numStrips)
	// -------------------------------------------------------------------
	MOVD 32(R0), R21  // wStripPtr
	MOVD 40(R0), R22  // outStripPtr
	MOVD 48(R0), R23  // biasStripPtr
	MOVD 104(R0), R24 // curCol = startCol
	MOVD 96(R0), R25  // remStrips = numStrips
	MOVD 112(R0), R6  // chunkCols
	MOVD 120(R0), R7  // outRowBytes

strip_loop:
	CMP $0, R25
	BLE strips_done

tile_retry:
	WORD $0x2598E3E7 // PTRUE P7.S
	WORD $0x25B8C03F // MOV Z31.S, #1 (signal-safety canary)
	WORD $0x2518E3E0 // PTRUE P0.B
	WORD $0x2598E3E1 // PTRUE P1.S
	WORD $0x25A61702 // WHILELT P2.S, X24, X6 (curCol < chunkCols)
	WORD $0xC00800FF // ZERO {ZA}

	// Zero FP32 accumulators Z12..Z27
	WORD $0x25B8C00C // MOV Z12.S, #0
	WORD $0x25B8C00D // MOV Z13.S, #0
	WORD $0x25B8C00E // MOV Z14.S, #0
	WORD $0x25B8C00F // MOV Z15.S, #0
	WORD $0x25B8C010 // MOV Z16.S, #0
	WORD $0x25B8C011 // MOV Z17.S, #0
	WORD $0x25B8C012 // MOV Z18.S, #0
	WORD $0x25B8C013 // MOV Z19.S, #0
	WORD $0x25B8C014 // MOV Z20.S, #0
	WORD $0x25B8C015 // MOV Z21.S, #0
	WORD $0x25B8C016 // MOV Z22.S, #0
	WORD $0x25B8C017 // MOV Z23.S, #0
	WORD $0x25B8C018 // MOV Z24.S, #0
	WORD $0x25B8C019 // MOV Z25.S, #0
	WORD $0x25B8C01A // MOV Z26.S, #0
	WORD $0x25B8C01B // MOV Z27.S, #0

	// Initialize ZA1.S from Bias when biasStripPtr != nil
	CBZ R23, init_za1_done
	WORD $0xA540AAE0 // LD1W {Z0.S}, P2/Z, [X23]
	WORD $0x04603001 // MOV Z1.D, Z0.D
	WORD $0x04603002 // MOV Z2.D, Z0.D
	WORD $0x04603003 // MOV Z3.D, Z0.D
	MOVD $0, R12
	WORD $0xC0840401 // MOVA ZA1H.S[W12, 0:3], {Z0.S-Z3.S}
	MOVD $4, R12
	WORD $0xC0840401 // MOVA ZA1H.S[W12, 0:3], {Z0.S-Z3.S}
	MOVD $8, R12
	WORD $0xC0840401 // MOVA ZA1H.S[W12, 0:3], {Z0.S-Z3.S}
	MOVD $12, R12
	WORD $0xC0840401 // MOVA ZA1H.S[W12, 0:3], {Z0.S-Z3.S}

init_za1_done:
	MOVD 16(R0), R8  // packedQxPtr
	MOVD R21, R9     // wStripPtr
	MOVD 24(R0), R10 // packedDxPairsPtr
	MOVD 72(R0), R15 // numSubBlocks
	MOVD 80(R0), R16 // quadsPerSB

sb_loop:
	WORD $0xC0080011 // ZERO {ZA0.S} (preserves ZA1.S)

	CMP $8, R16
	BEQ sb_quads_8
	CMP $4, R16
	BEQ sb_quads_4

	MOVD R16, R17
sb_quads_gen:
	WORD $0xA400A100 // LD1B {Z0.B}, P0/Z, [X8]
	WORD $0xA400A124 // LD1B {Z4.B}, P0/Z, [X9]
	WORD $0xA0A40000 // SUMOPA ZA0.S, P0/M, P0/M, Z0.B, Z4.B
	ADD $64, R8, R8
	ADD $64, R9, R9
	SUB $1, R17, R17
	CBNZ R17, sb_quads_gen
	B sb_scales

sb_quads_8:
	WORD $0xA400A100 // LD1B {Z0.B}, P0/Z, [X8, #0, MUL VL]
	WORD $0xA401A101 // LD1B {Z1.B}, P0/Z, [X8, #1, MUL VL]
	WORD $0xA402A102 // LD1B {Z2.B}, P0/Z, [X8, #2, MUL VL]
	WORD $0xA403A103 // LD1B {Z3.B}, P0/Z, [X8, #3, MUL VL]
	WORD $0xA400A124 // LD1B {Z4.B}, P0/Z, [X9, #0, MUL VL]
	WORD $0xA401A125 // LD1B {Z5.B}, P0/Z, [X9, #1, MUL VL]
	WORD $0xA402A126 // LD1B {Z6.B}, P0/Z, [X9, #2, MUL VL]
	WORD $0xA403A127 // LD1B {Z7.B}, P0/Z, [X9, #3, MUL VL]
	WORD $0xA0A40000 // SUMOPA ZA0.S, P0/M, P0/M, Z0.B, Z4.B
	WORD $0xA0A50020 // SUMOPA ZA0.S, P0/M, P0/M, Z1.B, Z5.B
	WORD $0xA0A60040 // SUMOPA ZA0.S, P0/M, P0/M, Z2.B, Z6.B
	WORD $0xA0A70060 // SUMOPA ZA0.S, P0/M, P0/M, Z3.B, Z7.B

	WORD $0xA404A100 // LD1B {Z0.B}, P0/Z, [X8, #4, MUL VL]
	WORD $0xA405A101 // LD1B {Z1.B}, P0/Z, [X8, #5, MUL VL]
	WORD $0xA406A102 // LD1B {Z2.B}, P0/Z, [X8, #6, MUL VL]
	WORD $0xA407A103 // LD1B {Z3.B}, P0/Z, [X8, #7, MUL VL]
	WORD $0xA404A124 // LD1B {Z4.B}, P0/Z, [X9, #4, MUL VL]
	WORD $0xA405A125 // LD1B {Z5.B}, P0/Z, [X9, #5, MUL VL]
	WORD $0xA406A126 // LD1B {Z6.B}, P0/Z, [X9, #6, MUL VL]
	WORD $0xA407A127 // LD1B {Z7.B}, P0/Z, [X9, #7, MUL VL]
	WORD $0xA0A40000 // SUMOPA ZA0.S, P0/M, P0/M, Z0.B, Z4.B
	WORD $0xA0A50020 // SUMOPA ZA0.S, P0/M, P0/M, Z1.B, Z5.B
	WORD $0xA0A60040 // SUMOPA ZA0.S, P0/M, P0/M, Z2.B, Z6.B
	WORD $0xA0A70060 // SUMOPA ZA0.S, P0/M, P0/M, Z3.B, Z7.B

	ADD $512, R8, R8
	ADD $512, R9, R9
	B sb_scales

sb_quads_4:
	WORD $0xA400A100 // LD1B {Z0.B}, P0/Z, [X8, #0, MUL VL]
	WORD $0xA401A101 // LD1B {Z1.B}, P0/Z, [X8, #1, MUL VL]
	WORD $0xA402A102 // LD1B {Z2.B}, P0/Z, [X8, #2, MUL VL]
	WORD $0xA403A103 // LD1B {Z3.B}, P0/Z, [X8, #3, MUL VL]
	WORD $0xA400A124 // LD1B {Z4.B}, P0/Z, [X9, #0, MUL VL]
	WORD $0xA401A125 // LD1B {Z5.B}, P0/Z, [X9, #1, MUL VL]
	WORD $0xA402A126 // LD1B {Z6.B}, P0/Z, [X9, #2, MUL VL]
	WORD $0xA403A127 // LD1B {Z7.B}, P0/Z, [X9, #3, MUL VL]
	WORD $0xA0A40000 // SUMOPA ZA0.S, P0/M, P0/M, Z0.B, Z4.B
	WORD $0xA0A50020 // SUMOPA ZA0.S, P0/M, P0/M, Z1.B, Z5.B
	WORD $0xA0A60040 // SUMOPA ZA0.S, P0/M, P0/M, Z2.B, Z6.B
	WORD $0xA0A70060 // SUMOPA ZA0.S, P0/M, P0/M, Z3.B, Z7.B

	ADD $256, R8, R8
	ADD $256, R9, R9

sb_scales:
	// In packedDxPairs, [X10, #0..60] holds dx[0..15] and [X10, #1, MUL VL] holds (dx*sumQx)[0..15]
	WORD $0xA541A548 // LD1W {Z8.S}, P1/Z, [X10, #1, MUL VL]
	WORD $0xA540A529 // LD1W {Z9.S}, P1/Z, [X9, #0, MUL VL]  (d'_w[0..15])
	WORD $0xA541A52A // LD1W {Z10.S}, P1/Z, [X9, #1, MUL VL] (m'_w[0..15])
	WORD $0x808A2501 // FMOPA ZA1.S, P1/M, P1/M, Z8.S, Z10.S

	// Rows 0..3 -> Z12..Z15
	MOVD $0, R12
	WORD $0xC0860400 // MOVA {Z0.S-Z3.S}, ZA0H.S[W12, 0:3]
	WORD $0xC132E000 // SCVTF {Z0.S-Z3.S}, {Z0.S-Z3.S}
	WORD $0x8540C544 // LD1RW {Z4.S}, P1/Z, [X10, #0]
	WORD $0x8541C545 // LD1RW {Z5.S}, P1/Z, [X10, #4]
	WORD $0x8542C546 // LD1RW {Z6.S}, P1/Z, [X10, #8]
	WORD $0x8543C547 // LD1RW {Z7.S}, P1/Z, [X10, #12]
	WORD $0x65890884 // FMUL Z4.S, Z4.S, Z9.S
	WORD $0x658908A5 // FMUL Z5.S, Z5.S, Z9.S
	WORD $0x658908C6 // FMUL Z6.S, Z6.S, Z9.S
	WORD $0x658908E7 // FMUL Z7.S, Z7.S, Z9.S
	WORD $0x65A4040C // FMLA Z12.S, P1/M, Z0.S, Z4.S
	WORD $0x65A5042D // FMLA Z13.S, P1/M, Z1.S, Z5.S
	WORD $0x65A6044E // FMLA Z14.S, P1/M, Z2.S, Z6.S
	WORD $0x65A7046F // FMLA Z15.S, P1/M, Z3.S, Z7.S
	CMP $4, R5
	BLE sb_advance

	// Rows 4..7 -> Z16..Z19
	MOVD $4, R12
	WORD $0xC0860400 // MOVA {Z0.S-Z3.S}, ZA0H.S[W12, 0:3]
	WORD $0xC132E000 // SCVTF {Z0.S-Z3.S}, {Z0.S-Z3.S}
	WORD $0x8544C544 // LD1RW {Z4.S}, P1/Z, [X10, #16]
	WORD $0x8545C545 // LD1RW {Z5.S}, P1/Z, [X10, #20]
	WORD $0x8546C546 // LD1RW {Z6.S}, P1/Z, [X10, #24]
	WORD $0x8547C547 // LD1RW {Z7.S}, P1/Z, [X10, #28]
	WORD $0x65890884 // FMUL Z4.S, Z4.S, Z9.S
	WORD $0x658908A5 // FMUL Z5.S, Z5.S, Z9.S
	WORD $0x658908C6 // FMUL Z6.S, Z6.S, Z9.S
	WORD $0x658908E7 // FMUL Z7.S, Z7.S, Z9.S
	WORD $0x65A40410 // FMLA Z16.S, P1/M, Z0.S, Z4.S
	WORD $0x65A50431 // FMLA Z17.S, P1/M, Z1.S, Z5.S
	WORD $0x65A60452 // FMLA Z18.S, P1/M, Z2.S, Z6.S
	WORD $0x65A70473 // FMLA Z19.S, P1/M, Z3.S, Z7.S
	CMP $8, R5
	BLE sb_advance

	// Rows 8..11 -> Z20..Z23
	MOVD $8, R12
	WORD $0xC0860400 // MOVA {Z0.S-Z3.S}, ZA0H.S[W12, 0:3]
	WORD $0xC132E000 // SCVTF {Z0.S-Z3.S}, {Z0.S-Z3.S}
	WORD $0x8548C544 // LD1RW {Z4.S}, P1/Z, [X10, #32]
	WORD $0x8549C545 // LD1RW {Z5.S}, P1/Z, [X10, #36]
	WORD $0x854AC546 // LD1RW {Z6.S}, P1/Z, [X10, #40]
	WORD $0x854BC547 // LD1RW {Z7.S}, P1/Z, [X10, #44]
	WORD $0x65890884 // FMUL Z4.S, Z4.S, Z9.S
	WORD $0x658908A5 // FMUL Z5.S, Z5.S, Z9.S
	WORD $0x658908C6 // FMUL Z6.S, Z6.S, Z9.S
	WORD $0x658908E7 // FMUL Z7.S, Z7.S, Z9.S
	WORD $0x65A40414 // FMLA Z20.S, P1/M, Z0.S, Z4.S
	WORD $0x65A50435 // FMLA Z21.S, P1/M, Z1.S, Z5.S
	WORD $0x65A60456 // FMLA Z22.S, P1/M, Z2.S, Z6.S
	WORD $0x65A70477 // FMLA Z23.S, P1/M, Z3.S, Z7.S
	CMP $12, R5
	BLE sb_advance

	// Rows 12..15 -> Z24..Z27
	MOVD $12, R12
	WORD $0xC0860400 // MOVA {Z0.S-Z3.S}, ZA0H.S[W12, 0:3]
	WORD $0xC132E000 // SCVTF {Z0.S-Z3.S}, {Z0.S-Z3.S}
	WORD $0x854CC544 // LD1RW {Z4.S}, P1/Z, [X10, #48]
	WORD $0x854DC545 // LD1RW {Z5.S}, P1/Z, [X10, #52]
	WORD $0x854EC546 // LD1RW {Z6.S}, P1/Z, [X10, #56]
	WORD $0x854FC547 // LD1RW {Z7.S}, P1/Z, [X10, #60]
	WORD $0x65890884 // FMUL Z4.S, Z4.S, Z9.S
	WORD $0x658908A5 // FMUL Z5.S, Z5.S, Z9.S
	WORD $0x658908C6 // FMUL Z6.S, Z6.S, Z9.S
	WORD $0x658908E7 // FMUL Z7.S, Z7.S, Z9.S
	WORD $0x65A40418 // FMLA Z24.S, P1/M, Z0.S, Z4.S
	WORD $0x65A50439 // FMLA Z25.S, P1/M, Z1.S, Z5.S
	WORD $0x65A6045A // FMLA Z26.S, P1/M, Z2.S, Z6.S
	WORD $0x65A7047B // FMLA Z27.S, P1/M, Z3.S, Z7.S

sb_advance:
	ADD $128, R9, R9
	ADD $128, R10, R10
	SUB $1, R15, R15
	CBNZ R15, sb_loop

	// Combine Z12..Z27 with ZA1.S and write final float32 result into ZA0.S
	MOVD $0, R12
	WORD $0xC0860420 // MOVA {Z0.S-Z3.S}, ZA1H.S[W12, 0:3]
	WORD $0x6580018C // FADD Z12.S, Z12.S, Z0.S
	WORD $0x658101AD // FADD Z13.S, Z13.S, Z1.S
	WORD $0x658201CE // FADD Z14.S, Z14.S, Z2.S
	WORD $0x658301EF // FADD Z15.S, Z15.S, Z3.S
	WORD $0xC0840580 // MOVA ZA0H.S[W12, 0:3], {Z12.S-Z15.S}

	MOVD $4, R12
	WORD $0xC0860420 // MOVA {Z0.S-Z3.S}, ZA1H.S[W12, 0:3]
	WORD $0x65800210 // FADD Z16.S, Z16.S, Z0.S
	WORD $0x65810231 // FADD Z17.S, Z17.S, Z1.S
	WORD $0x65820252 // FADD Z18.S, Z18.S, Z2.S
	WORD $0x65830273 // FADD Z19.S, Z19.S, Z3.S
	WORD $0xC0840600 // MOVA ZA0H.S[W12, 0:3], {Z16.S-Z19.S}

	MOVD $8, R12
	WORD $0xC0860420 // MOVA {Z0.S-Z3.S}, ZA1H.S[W12, 0:3]
	WORD $0x65800294 // FADD Z20.S, Z20.S, Z0.S
	WORD $0x658102B5 // FADD Z21.S, Z21.S, Z1.S
	WORD $0x658202D6 // FADD Z22.S, Z22.S, Z2.S
	WORD $0x658302F7 // FADD Z23.S, Z23.S, Z3.S
	WORD $0xC0840680 // MOVA ZA0H.S[W12, 0:3], {Z20.S-Z23.S}

	MOVD $12, R12
	WORD $0xC0860420 // MOVA {Z0.S-Z3.S}, ZA1H.S[W12, 0:3]
	WORD $0x65800318 // FADD Z24.S, Z24.S, Z0.S
	WORD $0x65810339 // FADD Z25.S, Z25.S, Z1.S
	WORD $0x6582035A // FADD Z26.S, Z26.S, Z2.S
	WORD $0x6583037B // FADD Z27.S, Z27.S, Z3.S
	WORD $0xC0840700 // MOVA ZA0H.S[W12, 0:3], {Z24.S-Z27.S}

	// Verify Z31 signal-safety canary before writing to Out memory
	WORD $0x2598E3E7 // PTRUE P7.S
	WORD $0x05A1BFEB // LASTB W11, P7, Z31.S
	CBZ R11, tile_retry

store_retry:
	WORD $0x2598E3E7 // PTRUE P7.S
	WORD $0x25B8C03F // MOV Z31.S, #1
	WORD $0x25A61702 // WHILELT P2.S, X24, X6

	MOVD R22, R10
	MOVD $0, R12
store_row_loop:
	WORD $0xE0BF0940 // ST1W {ZA0H.S[W12, 0]}, P2, [X10]
	ADD R7, R10, R10
	ADD $1, R12, R12
	CMP R5, R12
	BLT store_row_loop

	WORD $0x2598E3E7 // PTRUE P7.S
	WORD $0x05A1BFEB // LASTB W11, P7, Z31.S
	CBZ R11, store_retry

	// Advance to next strip
	MOVD 88(R0), R14 // stripBytes
	ADD R14, R21, R21
	ADD $64, R22, R22
	CBZ R23, skip_bias_adv
	ADD $64, R23, R23
skip_bias_adv:
	ADD $16, R24, R24
	SUB $1, R25, R25
	B strip_loop

strips_done:
	ADD $16, R19, R19
	CMP R20, R19
	BGE sme_quant_stop

	// Advance qxPtr, dxPairsPtr, outPtr by 16 rows
	MOVD 0(R0), R1
	MOVD 64(R0), R4
	ADD R4<<6, R1, R1
	MOVD R1, 0(R0)

	MOVD 8(R0), R1
	MOVD 72(R0), R4
	ADD R4<<7, R1, R1
	MOVD R1, 8(R0)

	MOVD 40(R0), R1
	MOVD 120(R0), R7
	ADD R7<<4, R1, R1
	MOVD R1, 40(R0)

	B row_block_loop

sme_quant_stop:
	WORD $0xD503467F // SMSTOP

sme_quant_done:
	RET
