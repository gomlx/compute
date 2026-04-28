// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend_test

/*
func TestBuffers_Bytes(t *testing.T) {
	buf, err := backend.(*gobackend.Backend).getBuffer(dtypes.Int32, 3)
	if err != nil {
		t.Fatalf("Failed to get buffer: %+v", err)
	}
	buf.shape = shapes.Make(dtypes.Int32, 3)
	buf.Zeros()
	if len(buf.flat.([]int32)) != 3 {
		t.Fatalf("Expected length 3, got %d", len(buf.flat.([]int32)))
	}
	flatBytes, err := buf.mutableBytes()
	if err != nil {
		t.Fatalf("Failed to get mutable bytes: %+v", err)
	}
	if len(flatBytes) != 3*int(dtypes.Int32.Size()) {
		t.Fatalf("Expected length %d, got %d", 3*int(dtypes.Int32.Size()), len(flatBytes))
	}
	flatBytes[0] = 1
	flatBytes[4] = 7
	flatBytes[8] = 3
	if ok, diff := testutil.IsEqual([]int32{1, 7, 3}, buf.flat.([]int32)); !ok {
		t.Fatalf("Unexpected result (-want +got):\n%s", diff)
	}
	runtime.KeepAlive(buf)
}

func TestBuffers_Fill(t *testing.T) {
	buf, err := backend.(*Backend).getBuffer(dtypes.Int32, 3)
	if err != nil {
		t.Fatalf("Failed to get buffer: %+v", err)
	}
	buf.shape = shapes.Make(dtypes.Int32, 3)
	if len(buf.flat.([]int32)) != 3 {
		t.Fatalf("Expected length 3, got %d", len(buf.flat.([]int32)))
	}
	if err := buf.Fill(int32(3)); err != nil {
		t.Fatalf("Failed to fill buffer: %+v", err)
	}
	if ok, diff := testutil.IsEqual([]int32{3, 3, 3}, buf.flat.([]int32)); !ok {
		t.Fatalf("Unexpected result (-want +got):\n%s", diff)
	}

	buf.Zeros()
	if ok, diff := testutil.IsEqual([]int32{0, 0, 0}, buf.flat.([]int32)); !ok {
		t.Fatalf("Unexpected result (-want +got):\n%s", diff)
	}
}

*/
