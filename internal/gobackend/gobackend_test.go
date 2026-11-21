// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend_test

import (
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/internal/must"
	"github.com/gomlx/compute/shapes"
	"k8s.io/klog/v2"

	// Registers all the ops.
	_ "github.com/gomlx/compute/internal/gobackend/ops"
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

func makeBuffer(t *testing.T, shape shapes.Shape, flat any) *gobackend.Buffer {
	t.Helper()
	computeBuf, err := backend.BufferFromFlatData(0, flat, shape)
	if err != nil {
		t.Fatalf("BufferFromFlatData failed: %+v", err)
	}
	return computeBuf.(*gobackend.Buffer)
}

func TestMain(m *testing.M) {
	setup()
	code := m.Run() // Run all tests in the file
	teardown()
	os.Exit(code)
}

func TestUnknownBackendOptionError(t *testing.T) {
	_, err := compute.NewWithConfig("go:help")
	if err == nil {
		t.Fatalf("expected error from compute.NewWithConfig(\"go:help\"), got nil")
	}
	for _, want := range []string{
		"quantized_cache=<always|balanced|never>",
		"see documentation in https://gomlx.github.io/docs/backends/#2-the-go-backend-go",
	} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("expected error %q to contain %q", err.Error(), want)
		}
	}
}

func TestQuantizedCacheConfigParsing(t *testing.T) {
	cases := []struct {
		cfg      string
		want     gobackend.QuantizedCachePolicy
		wantDesc string
	}{
		{"", gobackend.QuantizedCacheBalanced, ""},
		{"quantized_cache=balanced", gobackend.QuantizedCacheBalanced, ""},
		{"quantized_cache=BALANCED", gobackend.QuantizedCacheBalanced, ""},
		{"quantized_cache", gobackend.QuantizedCacheAlways, "quantized_cache=always"},
		{"quantized_cache=always", gobackend.QuantizedCacheAlways, "quantized_cache=always"},
		{"quantized_cache=true", gobackend.QuantizedCacheAlways, "quantized_cache=always"},
		{"quantized_cache=1", gobackend.QuantizedCacheAlways, "quantized_cache=always"},
		{"quantized_cache=never", gobackend.QuantizedCacheNever, "quantized_cache=never"},
		{"quantized_cache=none", gobackend.QuantizedCacheNever, "quantized_cache=never"},
		{"quantized_cache=false", gobackend.QuantizedCacheNever, "quantized_cache=never"},
		{"quantized_cache=0", gobackend.QuantizedCacheNever, "quantized_cache=never"},
	}
	for _, tc := range cases {
		bRaw, err := gobackend.New(tc.cfg)
		if err != nil {
			t.Fatalf("gobackend.New(%q): %+v", tc.cfg, err)
		}
		be := bRaw.(*gobackend.Backend)
		if be.QuantizedCache != tc.want {
			t.Errorf("gobackend.New(%q): got QuantizedCache=%v, want %v", tc.cfg, be.QuantizedCache, tc.want)
		}
		desc := be.Description()
		if tc.wantDesc != "" && !strings.Contains(desc, tc.wantDesc) {
			t.Errorf("gobackend.New(%q).Description() = %q, want substring %q", tc.cfg, desc, tc.wantDesc)
		}
		if tc.wantDesc == "" && strings.Contains(desc, "quantized_cache=") {
			t.Errorf("gobackend.New(%q).Description() = %q, did not expect quantized_cache=", tc.cfg, desc)
		}
		bRaw.Finalize()
	}

	if _, err := gobackend.New("quantized_cache=invalid"); err == nil {
		t.Fatalf("expected error for quantized_cache=invalid, got nil")
	}
}
