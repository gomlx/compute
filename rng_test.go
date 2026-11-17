// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package compute_test

import (
	"reflect"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
)

func TestRNGAlgorithmAndCapabilities(t *testing.T) {
	for _, algo := range compute.RNGAlgorithmValues() {
		if algo == compute.RNGDefault {
			if algo.StateShape().Ok() {
				t.Errorf("RNGDefault.StateShape() = %s, want invalid", algo.StateShape())
			}
			if _, _, err := algo.StateFromSeed(42); err == nil {
				t.Errorf("RNGDefault.StateFromSeed(42) = nil error, want error")
			}
			continue
		}
		shape := algo.StateShape()
		if !shape.Ok() || shape.DType != dtypes.Uint64 || shape.Rank() != 1 {
			t.Errorf("%s.StateShape() = %s, want 1D Uint64", algo, shape)
		}
		flat1, s1, err := algo.StateFromSeed(42)
		if err != nil {
			t.Fatalf("%s.StateFromSeed(42) failed: %v", algo, err)
		}
		if !s1.Equal(shape) {
			t.Errorf("%s.StateFromSeed shape = %s, want %s", algo, s1, shape)
		}
		flat2, _, err := algo.StateFromSeed(42)
		if err != nil {
			t.Fatalf("%s.StateFromSeed(42) second call failed: %v", algo, err)
		}
		if !reflect.DeepEqual(flat1, flat2) {
			t.Errorf("%s.StateFromSeed(42) not deterministic", algo)
		}
		flat3, _, err := algo.StateFromSeed(43)
		if err != nil {
			t.Fatalf("%s.StateFromSeed(43) failed: %v", algo, err)
		}
		if reflect.DeepEqual(flat1, flat3) {
			t.Errorf("%s.StateFromSeed(42) == StateFromSeed(43)", algo)
		}
	}

	// Backend with explicit DefaultRNGAlgorithm.
	capsExplicit := compute.Capabilities{
		Operations: map[compute.OpType]bool{compute.OpTypeRNGBitGenerator: true},
		RNGAlgorithms: map[compute.RNGAlgorithm]bool{
			compute.RNGDefault:  true,
			compute.RNGThreeFry: true,
		},
		DefaultRNGAlgorithm: compute.RNGThreeFry,
	}
	if !capsExplicit.SupportsRNG(compute.RNGDefault) || !capsExplicit.SupportsRNG(compute.RNGThreeFry) {
		t.Errorf("expected RNGDefault and RNGThreeFry to be supported")
	}
	if capsExplicit.SupportsRNG(compute.RNGPhilox) {
		t.Errorf("expected RNGPhilox to be unsupported")
	}
	if got := capsExplicit.ResolveRNGAlgorithm(compute.RNGDefault); got != compute.RNGThreeFry {
		t.Errorf("ResolveRNGAlgorithm(RNGDefault) = %s, want RNGThreeFry", got)
	}
	sh, err := capsExplicit.RNGStateShape(compute.RNGDefault)
	if err != nil || !sh.Equal(compute.RNGThreeFry.StateShape()) {
		t.Errorf("RNGStateShape(RNGDefault) = (%s, %v), want %s", sh, err, compute.RNGThreeFry.StateShape())
	}
	if _, err := capsExplicit.RNGStateShape(compute.RNGPhilox); !compute.IsNotImplemented(err) {
		t.Errorf("RNGStateShape(RNGPhilox) err = %v, want ErrNotImplemented", err)
	}

	// Backend with private DefaultRNGStateShape (e.g. [2]uint32).
	privShape := shapes.Make(dtypes.Uint32, 2)
	capsPrivate := compute.Capabilities{
		Operations: map[compute.OpType]bool{compute.OpTypeRNGBitGenerator: true},
		RNGAlgorithms: map[compute.RNGAlgorithm]bool{
			compute.RNGDefault: true,
		},
		DefaultRNGAlgorithm:  compute.RNGDefault,
		DefaultRNGStateShape: privShape,
	}
	if got := capsPrivate.ResolveRNGAlgorithm(compute.RNGDefault); got != compute.RNGDefault {
		t.Errorf("ResolveRNGAlgorithm(RNGDefault) = %s, want RNGDefault", got)
	}
	flatPriv, gotPrivShape, err := capsPrivate.RNGStateFromSeed(compute.RNGDefault, 7)
	if err != nil || !gotPrivShape.Equal(privShape) {
		t.Fatalf("capsPrivate.RNGStateFromSeed failed: shape=%s err=%v", gotPrivShape, err)
	}
	if s, ok := flatPriv.([]uint32); !ok || len(s) != 2 {
		t.Errorf("capsPrivate.RNGStateFromSeed flat = %T %v, want []uint32 of len 2", flatPriv, flatPriv)
	}

	cloned := capsPrivate.Clone()
	cloned.RNGAlgorithms[compute.RNGPhilox] = true
	if capsPrivate.SupportsRNG(compute.RNGPhilox) {
		t.Errorf("Capabilities.Clone did not deep-copy RNGAlgorithms")
	}
}
