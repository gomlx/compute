package nontransposed

import (
	"github.com/gomlx/compute/support/envutil"
	"k8s.io/klog/v2"
)

func init() {
	avx512Enabled, err := envutil.ReadBool(envutil.SIMD_AVX512_Env, true)
	if err != nil {
		klog.Fatalf("Invalid value for %q: %+v", envutil.SIMD_AVX512_Env, err)
	}
