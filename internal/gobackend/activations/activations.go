// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package activations

import (
	"sync"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/dtypes/gotype"
	"github.com/gomlx/compute/internal/gobackend"
	"k8s.io/klog/v2"
)

// InPlaceFn applies an activation function in-place to a contiguous slice of elements.
type InPlaceFn[T any] func(data []T)

// registrationKey uniquely identifies an implementation for an activation type and data type.
type registrationKey struct {
	act   compute.ActivationType
	dtype dtypes.DType
}

type registeredImpl struct {
	name     string
	fn       any
	priority gobackend.RegisterPriority
}

var (
	registryMu sync.RWMutex
	registry   = make(map[registrationKey]*registeredImpl)
)

// Register registers an in-place activation function for a specific activation type and data type.
// If an implementation is already registered with a higher priority, this call is ignored.
func Register[T gotype.Supported](name string, act compute.ActivationType, fn InPlaceFn[T], priority gobackend.RegisterPriority) {
	dtype := dtypes.FromGenericsType[T]()
	key := registrationKey{act: act, dtype: dtype}

	registryMu.Lock()
	defer registryMu.Unlock()

	current, exists := registry[key]
	if exists && priority < current.priority {
		klog.V(2).Infof("Activation %s/%s (%q) ignored: priority %d < existing %d",
			act, dtype, name, priority, current.priority)
		return
	}
	registry[key] = &registeredImpl{
		name:     name,
		fn:       fn,
		priority: priority,
	}
	klog.V(2).Infof("Registered activation %s/%s: %q (priority %d)", act, dtype, name, priority)
}

// Get returns the registered in-place activation function for the given activation type and data type T.
// Returns nil if activation is ActivationNone or if no implementation is registered.
func Get[T gotype.Supported](act compute.ActivationType) InPlaceFn[T] {
	if act == compute.ActivationNone {
		return nil
	}
	dtype := dtypes.FromGenericsType[T]()
	key := registrationKey{act: act, dtype: dtype}

	registryMu.RLock()
	impl, exists := registry[key]
	registryMu.RUnlock()

	if !exists || impl == nil {
		return nil
	}
	return impl.fn.(InPlaceFn[T])
}

const minParallelizeChunk = 4096

// Apply applies the given activation function in-place across the slice.
// If the slice is large and workers are available, work is parallelized in chunks.
func Apply[T gotype.Supported](backend *gobackend.Backend, act compute.ActivationType, data []T) {
	if act == compute.ActivationNone || len(data) == 0 {
		return
	}
	fn := Get[T](act)
	if fn == nil {
		klog.Warningf("No registered activation implementation for act=%s dtype=%s",
			act, dtypes.FromGenericsType[T]())
		return
	}

	n := len(data)
	if backend != nil && backend.Workers != nil && backend.Workers.IsEnabled() && n > minParallelizeChunk {
		var wg sync.WaitGroup
		for i := 0; i < n; i += minParallelizeChunk {
			end := min(i+minParallelizeChunk, n)
			chunk := data[i:end]
			wg.Add(1)
			backend.Workers.WaitToStart(func() {
				fn(chunk)
				wg.Done()
			})
		}
		wg.Wait()
	} else {
		fn(data)
	}
}

// Ensure standard dtypes are referenced to satisfy compiler.
var (
	_ = float16.FromFloat32
	_ = bfloat16.FromFloat32
)
