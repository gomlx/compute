// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package dot_test

import (
	"fmt"
	"os"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/internal/gobackend"
	_ "github.com/gomlx/compute/internal/gobackend/defaultpkgs"
	"github.com/gomlx/compute/internal/must"
	"github.com/gomlx/compute/support/testutil"
	"k8s.io/klog/v2"
)

var backend compute.Backend

func init() {
	klog.InitFlags(nil)
}

func setup() {
	fmt.Printf("Available backends: %q\n", compute.List())
	// Perform your setup logic here
	if os.Getenv(compute.ConfigEnvVar) == "" {
		must.M(os.Setenv(compute.ConfigEnvVar, "go"))
	} else {
		fmt.Printf("\t$%s=%q\n", compute.ConfigEnvVar, os.Getenv(compute.ConfigEnvVar))
	}
	backend = compute.MustNew()
	fmt.Printf("Backend: %s, %s\n", backend.Name(), backend.Description())
}

func teardown() {
	backend.Finalize()
}

func TestMain(m *testing.M) {
	setup()
	code := m.Run() // Run all tests in the file
	teardown()
	os.Exit(code)
}

func TestDotGeneral(t *testing.T) {
	if _, ok := backend.(*gobackend.Backend); !ok {
		t.Skip("Skipping test because backend is not a SimpleGo Backend")
	}

	lhs := [][][]float32{{{1, 2, 3}}, {{4, 5, 6}}}
	rhs := [][][]float32{{{1, 1}, {1, 1}, {1, 1}}, {{1, 1}, {1, 1}, {1, 1}}}
	want := [][][]float32{{{6, 6}}, {{15, 15}}}

	y1, err := testutil.Exec1(backend, []any{lhs, rhs}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
		return f.DotGeneral(params[0], []int{2}, []int{0}, params[1], []int{1}, []int{0}, compute.DotGeneralConfig{})
	})
	if err != nil {
		t.Fatalf("testutil.Exec1 failed: %v", err)
	}
	if ok, diff := testutil.IsEqual(want, y1); !ok {
		t.Fatalf("Unexpected result (-want +got):\n%s", diff)
	}
}

func TestDotGeneralTransposedLarge(t *testing.T) {
	if _, ok := backend.(*gobackend.Backend); !ok {
		t.Skip("Skipping test because backend is not a SimpleGo Backend")
	}

	// Shapes:
	// lhs: [4, 32, 8, 2, 256] -> batch=[0, 2], contracting=[4], cross=[1, 3]
	// rhs: [4, 32, 8, 256] -> batch=[0, 2], contracting=[3], cross=[1]
	
	lhsFlat := make([]float32, 4*32*8*2*256)
	for b := range 4 {
		for q := range 32 {
			for h := range 8 {
				for g := range 2 {
					for d := range 256 {
						idx := (((b*32 + q)*8 + h)*2 + g)*256 + d
						lhsFlat[idx] = float32(((b*32+q)*8+h)*2+g) * 0.0001
					}
				}
			}
		}
	}

	rhsFlat := make([]float32, 4*32*8*256)
	for b := range 4 {
		for k := range 32 {
			for h := range 8 {
				for d := range 256 {
					idx := ((b*32 + k)*8 + h)*256 + d
					rhsFlat[idx] = float32(((b*32+k)*8+h)*256+d) * 0.0001
				}
			}
		}
	}

	// Output size: 4 * 32 * 8 * 2 * 32 = 65536
	wantFlat := make([]float32, 65536)
	for b := range 4 {
		for q := range 32 {
			for h := range 8 {
				for g := range 2 {
					for k := range 32 {
						var sum float64
						lhsIdxBase := (((b*32 + q)*8 + h)*2 + g)*256
						rhsIdxBase := ((b*32 + k)*8 + h)*256
						for d := range 256 {
							sum += float64(lhsFlat[lhsIdxBase+d]) * float64(rhsFlat[rhsIdxBase+d])
						}
						outIdx := (((b*8 + h)*32 + q)*2 + g)*32 + k
						wantFlat[outIdx] = float32(sum)
					}
				}
			}
		}
	}

	gotFlat, err := testutil.Exec1(backend, []any{lhsFlat, rhsFlat}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
		lhs, err := f.Reshape(params[0], 4, 32, 8, 2, 256)
		if err != nil {
			return nil, err
		}
		rhs, err := f.Reshape(params[1], 4, 32, 8, 256)
		if err != nil {
			return nil, err
		}
		res, err := f.DotGeneral(lhs, []int{4}, []int{0, 2}, rhs, []int{3}, []int{0, 2}, compute.DotGeneralConfig{})
		if err != nil {
			return nil, err
		}
		out, err := f.Reshape(res, 65536)
		if err != nil {
			return nil, err
		}
		return out, nil
	})
	if err != nil {
		t.Fatalf("testutil.Exec1 failed: %v", err)
	}

	gotFlatSlice := gotFlat.([]float32)
	
	wantStr := ""
	for i := 0; i < 100; i++ {
		wantStr += fmt.Sprintf("%f ", wantFlat[i])
		if (i+1)%8 == 0 {
			wantStr += "\n"
		}
	}
	t.Logf("WANT (first 100):\n%s", wantStr)

	gotStr := ""
	for i := 0; i < 100; i++ {
		gotStr += fmt.Sprintf("%f ", gotFlatSlice[i])
		if (i+1)%8 == 0 {
			gotStr += "\n"
		}
	}
	t.Logf("GOT (first 100):\n%s", gotStr)

	if ok, diff := testutil.IsInDelta(wantFlat, gotFlat, 5e-3); !ok {
		t.Fatalf("Unexpected result (-want +got):\n%s", diff)
	}
}


