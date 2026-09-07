// Package envutil provides utility functions for working with environment variables.
package envutil

import (
	"os"
	"strconv"
	"strings"

	"github.com/gomlx/compute/support/sets"
	"github.com/gomlx/compute/support/xslices"
	"github.com/pkg/errors"
)

var (
	FalseValues = sets.MakeWith("0", "false", "f", "no", "off", "disabled")
	TrueValues  = sets.MakeWith("1", "true", "t", "yes", "on", "enabled")
)

// Environment variables used by the Go backend: mostly SIMD instructions and optimizations.
// By default it uses whatever the CPU supports, but this allows one to disable them in case of issues.
const (
	GoBackendSIMD_AVX512 = "GOMLX_GO_SIMD_AVX512"
	GoBackendSIMD_AVX2   = "GOMLX_GO_SIMD_AVX2"
	GoBackendFusion      = "GOMLX_GO_FUSION"
	GoBackendDotMatmul   = "GOMLX_GO_DOT_MATMUL"
	GoBackendAVX512_ASM  = "GOMLX_GO_AVX512_ASM"
	GoBackendAVX512_KC   = "GOMLX_GO_AVX512_KC"
	GoBackendAVX512_MC   = "GOMLX_GO_AVX512_MC"
	GoBackendAVX512_NC   = "GOMLX_GO_AVX512_NC"
	GoBackendAVX2_ASM    = "GOMLX_GO_AVX2_ASM"
	GoBackendAVX2_KC     = "GOMLX_GO_AVX2_KC"
	GoBackendAVX2_MC     = "GOMLX_GO_AVX2_MC"
	GoBackendAVX2_NC     = "GOMLX_GO_AVX2_NC"
)

// ReadBool reads the boolean value of the environment variable with the given name.
// If not set, it returns the defaultValue.
// It returns an error if the value is set but not a valid boolean value.
func ReadBool(name string, defaultValue bool) (bool, error) {
	val := os.Getenv(name)
	if val == "" {
		return defaultValue, nil
	}
	val = strings.ToLower(val)
	if TrueValues.Has(val) {
		return true, nil
	}
	if FalseValues.Has(val) {
		return false, nil
	}
	return defaultValue, errors.Errorf("invalid value %q for environment variable %q; valid values are %q",
		val, name, append(xslices.Keys(TrueValues), xslices.Keys(FalseValues)...))
}

// MustReadBool reads the boolean value of the environment variable with the given name.
// If not set, it returns the defaultValue.
//
// It panics with an informative error if the value is set but it is not able to parse a valid boolean value.
func MustReadBool(name string, defaultValue bool) bool {
	result, err := ReadBool(name, defaultValue)
	if err != nil {
		panic(errors.Errorf("Invalid value for %q: %+v", name, err))
	}
	return result
}

// ReadInt reads the integer value of the environment variable with the given name.
// If not set, it returns the defaultValue.
// It returns an error if the value is set but not a valid integer value.
func ReadInt(name string, defaultValue int) (int, error) {
	val := os.Getenv(name)
	if val == "" {
		return defaultValue, nil
	}
	valInt, err := strconv.Atoi(val)
	if err != nil {
		return defaultValue, errors.Wrapf(err, "invalid value %q for environment variable %q; must be an integer", val, name)
	}
	return valInt, nil
}

// MustReadInt reads the integer value of the environment variable with the given name.
// If not set, it returns the defaultValue.
//
// It panics with an informative error if the value is set but it is not able to parse a valid integer value.
func MustReadInt(name string, defaultValue int) int {
	result, err := ReadInt(name, defaultValue)
	if err != nil {
		panic(errors.Errorf("Invalid value for %q: %+v", name, err))
	}
	return result
}
