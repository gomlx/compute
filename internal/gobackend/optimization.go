package gobackend

import (
	"slices"

	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

// This file handle registration of graph transformations (optimizations) that are
// executed during compilation.
//
// Currently, it's trivial, its simply applied in priority order, no conditional execution.
// But this can be changed in the future.

// GraphTransformation is an optimizer of the graph, to be executed during compilation.
type GraphTransformation interface {
	// Apply takes the current graph state and returns an optimized version.
	Apply(builder *Builder) error
	Name() string
}

type priorityTransformationPair struct {
	Priority  int
	Transform GraphTransformation
}

// graphTransformations is a list of graph transformations to be executed during compilation,
// sorted by priority: larger priorities come first.
var graphTransformations []priorityTransformationPair

// RegisterTransformation registers a graph transformation to be executed during compilation.
//
// This should be called in init() functions only, it must not be called after the program starts.
func RegisterTransformation(priority int, transformation GraphTransformation) {
	graphTransformations = append(graphTransformations, priorityTransformationPair{priority, transformation})

	// Keep the list sorted by priority: larger priorities come first.
	slices.SortFunc(graphTransformations, func(a, b priorityTransformationPair) int {
		if a.Priority < b.Priority {
			return 1
		}
		if a.Priority > b.Priority {
			return -1
		}
		return 0
	})
}

func (b *Builder) Optimize() error {
	// Apply all the registered transformations in order.
	for _, p := range graphTransformations {
		if klog.V(1).Enabled() {
			klog.Infof("Applying transformation %q\n", p.Transform.Name())
		}
		if err := p.Transform.Apply(b); err != nil {
			return errors.WithMessagef(err, "transformation %q failed", p.Transform.Name())
		}
	}
	return nil
}
