// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend

import (
	"bytes"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

// ConstantData holds the underlying buffer and any derived cached form for an OpTypeConstant node.
// A ConstantData instance can be shared across multiple Functions and Builders on the same Backend
// when Capabilities.SharedConstants is true.
//
// Separating ConstantData from Buffer keeps Buffer a pure storage container without any knowledge
// of constant semantics or graph-level optimizations.
type ConstantData struct {
	backend *Backend
	shape   shapes.Shape

	// rawBuffer holds the original constant tensor data (exact size, not pooled).
	// Once the constant is transformed into cachedForm (and all consumers use that cachedForm),
	// rawBuffer is atomically set to nil so the original backing slice can be garbage-collected.
	rawBuffer atomic.Pointer[Buffer]

	// isRawBufferPinned is set to true if any compiled graph uses this constant in an uncached
	// operation (e.g., Gather, LayerNorm, or multiple incompatible layouts), preventing rawBuffer release.
	isRawBufferPinned bool

	// mu protects slow-path initialization of cachedForm and compile-time updates to
	// cachedFormSignature and isRawBufferPinned. Fast-path reads of cachedForm during
	// execution use atomic loads without locking mu.
	mu sync.Mutex

	// cachedForm holds a shared derived representation (opaque any; e.g., pre-packed GEMM panels [][]T
	// in DotGeneral) for this constant. Read lock-free via CachedForm() and populated once via GetOrSetCachedForm().
	cachedForm atomic.Value

	// cachedFormSignature records the canonical signature of the constant sub-graph and consumer parameters
	// that produced cachedForm.
	cachedFormSignature string

	// forceCache indicates that a consumer op should populate cachedForm on the first execution
	// (e.g., even for small batch sizes in DotGeneral, unless ForceSmallVariant is active in tests).
	forceCache atomic.Bool
}

// NewConstantData creates a new ConstantData with an exact-sized, non-pooled buffer copied from flat.
func NewConstantData(backend *Backend, shape shapes.Shape, flat any) (*ConstantData, error) {
	if backend.isFinalized {
		return nil, errors.WithStack(ErrBackendAlreadyFinalized)
	}
	buf := NewBuffer(shape)
	dtypes.CopyAnySlice(buf.Flat, flat)
	cd := &ConstantData{
		backend: backend,
		shape:   shape.Clone(),
	}
	cd.rawBuffer.Store(buf)
	return cd, nil
}

// NewAnonymousConstantData creates a ConstantData container without a single backing rawBuffer
// (isRawBufferPinned = true), used by ops like DotGeneral when an input is a composite constant expression.
func NewAnonymousConstantData() *ConstantData {
	return &ConstantData{
		isRawBufferPinned: true,
	}
}

// Backend returns the Backend that owns this ConstantData.
func (cd *ConstantData) Backend() *Backend {
	if cd == nil {
		return nil
	}
	return cd.backend
}

// Shape returns the shape of this constant.
func (cd *ConstantData) Shape() shapes.Shape {
	if cd == nil {
		return shapes.Invalid()
	}
	return cd.shape
}

// Buffer returns the raw Buffer, or nil if it has been released after caching a derived form.
func (cd *ConstantData) Buffer() *Buffer {
	if cd == nil {
		return nil
	}
	return cd.rawBuffer.Load()
}

// CachedForm returns the shared derived cached representation, or nil if none has been set.
// It performs a lock-free atomic load.
func (cd *ConstantData) CachedForm() any {
	if cd == nil {
		return nil
	}
	return cd.cachedForm.Load()
}

// GetOrSetCachedForm returns the cached representation cast to T if already present (via a lock-free atomic load).
// Otherwise, it locks cd.mu, re-checks that cachedForm is still nil, calls computeFn() to build the
// cached representation, stores it in cachedForm, releases rawBuffer (if !isRawBufferPinned), and unlocks cd.mu.
func GetOrSetCachedForm[T any](cd *ConstantData, computeFn func() T) T {
	var zero T
	if cd == nil {
		return zero
	}
	if v := cd.cachedForm.Load(); v != nil {
		if typed, ok := v.(T); ok {
			return typed
		}
		return zero
	}
	cd.mu.Lock()
	defer cd.mu.Unlock()
	if v := cd.cachedForm.Load(); v != nil {
		if typed, ok := v.(T); ok {
			return typed
		}
		return zero
	}
	typed := computeFn()
	if !reflect.ValueOf(&typed).Elem().IsZero() {
		cd.cachedForm.Store(typed)
		if !cd.isRawBufferPinned {
			cd.rawBuffer.Store(nil)
		}
	}
	return typed
}

// ForceCache reports whether populating cachedForm is forced on the first execution.
func (cd *ConstantData) ForceCache() bool {
	return cd != nil && cd.forceCache.Load()
}

// EqualNodeData implements NodeDataComparable for ConstantData.
func (cd *ConstantData) EqualNodeData(other NodeDataComparable) bool {
	if cd == other {
		return true
	}
	o, ok := other.(*ConstantData)
	if !ok || o == nil || cd == nil {
		return false
	}
	if !cd.shape.Equal(o.shape) {
		return false
	}
	b1 := cd.rawBuffer.Load()
	b2 := o.rawBuffer.Load()
	if b1 == nil || b2 == nil {
		return false
	}
	byteSize := int(cd.shape.ByteSize())
	if byteSize > 0 && len(b1.RawBytes) >= byteSize && len(b2.RawBytes) >= byteSize {
		return bytes.Equal(b1.RawBytes[:byteSize], b2.RawBytes[:byteSize])
	}
	return compareFlatData(b1.Flat, b2.Flat)
}

// ConstantCacheNodeData is implemented by op Node.Data types in sub-packages
// (for example, DotGeneral and FusedDense caching pre-packed weight panels)
// that transform constant inputs into ConstantData.cachedForm on first execution
// and never read the raw constant input buffer again once cached.
type ConstantCacheNodeData interface {
	// CacheSignature returns a non-empty canonical string describing how inputIdx is transformed/cached,
	// or "" if inputIdx is not cached by this node.
	CacheSignature(inputIdx int) string

	// BindConstantData binds the shared ConstantData to inputIdx so the op reads/writes its cachedForm.
	BindConstantData(inputIdx int, cd *ConstantData)

	// IsInputCached reports whether inputIdx uses a cached form that has already been populated.
	IsInputCached(inputIdx int) bool
}

// constantCacheTerminal represents a consumer node input slot where a constant transformation
// sub-graph terminates in a ConstantCacheNodeData op (such as DotGeneral or FusedDense).
// For example, in Const(W) -> Transpose -> DotGeneral(lhs, rhs), the DotGeneral node at
// inputSlot=1 (rhs) is the terminal consumer of the constant sub-graph rooted at Const(W).
type constantCacheTerminal struct {
	// node is the consumer node whose Data implements ConstantCacheNodeData.
	node *Node

	// inputSlot is the index in node.Inputs that receives the (possibly transformed) constant.
	inputSlot int

	// sig is the full canonical signature combining the constant sub-graph expression from
	// the source OpTypeConstant to inputSlot with node's CacheSignature(inputSlot).
	sig string
}

// setupConstantSubgraphCaching is called once during compilation (from newFunctionExecutable).
// It inspects all reachable OpTypeConstant nodes in fe.Function and analyzes how each constant is consumed:
//   - It verifies whether all uses of the constant flow exclusively through a deterministic constant
//     transformation sub-graph (e.g., Transpose, Reshape, ConvertDType) into ConstantCacheNodeData consumers
//     (e.g., DotGeneral or FusedDense) with a single canonical signature.
//   - If a shared ConstantData was already compiled and optimized in an earlier graph with a different
//     signature, it returns a compile-time error immediately.
//   - It binds the ConstantData to the terminal consumer nodes (via BindConstantData) and records any
//     intermediate constant-subgraph nodes in fe.skipIfCachedConstant.
//
// The actual computation of cachedForm (and release of rawBuffer) happens lazily on the first execution
// when the consumer op calls ConstantData.GetOrSetCachedForm.
func (fe *FunctionExecutable) setupConstantSubgraphCaching() error {
	f := fe.Function
	for nodeIdx := range fe.NumNodesToProcess {
		constNode := f.Nodes[nodeIdx]
		if constNode == nil || constNode.OpType != compute.OpTypeConstant || fe.NumUses[nodeIdx] == 0 {
			continue
		}
		cd, ok := constNode.Data.(*ConstantData)
		if !ok || cd == nil {
			continue
		}

		canReleaseRaw, sig, terminals, intermediateIdxs := fe.analyzeConstantCaching(constNode)

		cd.mu.Lock()
		if !canReleaseRaw {
			if cd.cachedFormSignature != "" || cd.Buffer() == nil {
				optSig := cd.cachedFormSignature
				cd.mu.Unlock()
				return errors.Errorf(
					"the variable/constant (shape %s) was optimized in a first compilation (%s), and now is being compiled in a different way but can't be recovered; if you plan to have different graphs, please use different Exec objects and don't donate the model.Store",
					cd.shape, optSig)
			}
			cd.isRawBufferPinned = true
			// Even when rawBuffer must be kept for an uncached consumer (e.g. Gather or Add),
			// if all cached consumers in this graph share a single signature sig, we can still
			// cache and share the derived representation alongside rawBuffer (Option B).
			if sig != "" && len(terminals) > 0 {
				for _, term := range terminals {
					cacher := term.node.Data.(ConstantCacheNodeData)
					cacher.BindConstantData(term.inputSlot, cd)
				}
				cd.forceCache.Store(true)
			}
			cd.mu.Unlock()
			continue
		}

		// canReleaseRaw == true with canonical signature sig.
		if cd.cachedFormSignature != "" && cd.cachedFormSignature != sig {
			optSig := cd.cachedFormSignature
			cd.mu.Unlock()
			return errors.Errorf(
				"the variable/constant (shape %s) was optimized in a first compilation (%s), and now is being compiled in a different way (%s) but can't be recovered; if you plan to have different graphs, please use different Exec objects and don't donate the model.Store",
				cd.shape, optSig, sig)
		}

		if cd.isRawBufferPinned {
			// A previous compilation required rawBuffer to remain uncached (e.g., for Gather),
			// so we cannot release rawBuffer, though we can still bind/share cachedForm if compatible.
			for _, term := range terminals {
				cacher := term.node.Data.(ConstantCacheNodeData)
				cacher.BindConstantData(term.inputSlot, cd)
			}
			cd.forceCache.Store(true)
			cd.mu.Unlock()
			continue
		}

		cd.cachedFormSignature = sig
		for _, term := range terminals {
			cacher := term.node.Data.(ConstantCacheNodeData)
			cacher.BindConstantData(term.inputSlot, cd)
		}
		cd.forceCache.Store(true)
		cd.mu.Unlock()

		if len(intermediateIdxs) > 0 {
			if fe.skipIfCachedConstant == nil {
				fe.skipIfCachedConstant = make([]*ConstantData, fe.NumNodesToProcess)
			}
			for _, interIdx := range intermediateIdxs {
				fe.skipIfCachedConstant[interIdx] = cd
			}
		}
	}
	return nil
}

// analyzeConstantCaching checks if all uses of constNode in fe.Function flow exclusively through
// deterministic constant-only operations into cached slots of ConstantCacheNodeData nodes with
// an identical sub-graph + cache signature.
// If constNode has both cached uses (sharing a single signature sig) and uncached uses,
// canReleaseRaw is false while sig and terminals are still returned so Option B (keep raw + cache derived form) works.
func (fe *FunctionExecutable) analyzeConstantCaching(constNode *Node) (
	canReleaseRaw bool, sig string, terminals []constantCacheTerminal, intermediateIdxs []int,
) {
	f := fe.Function
	visited := make(map[int]bool)
	queue := []*Node{constNode}
	visited[constNode.Index] = true
	hasUncachedUse := false

	for len(queue) > 0 {
		curr := queue[0]
		queue = queue[1:]

		// If curr is also a direct output of the function, its raw/uncached value is needed.
		if fe.NumUses[curr.Index] != len(fe.Dependents[curr.Index]) {
			hasUncachedUse = true
		}
		if len(fe.Dependents[curr.Index]) == 0 {
			hasUncachedUse = true
			continue
		}

		seenDep := make(map[int]bool, len(fe.Dependents[curr.Index]))
		for _, depIdx := range fe.Dependents[curr.Index] {
			if seenDep[depIdx] {
				continue
			}
			seenDep[depIdx] = true
			depNode := f.Nodes[depIdx]
			if depNode == nil {
				hasUncachedUse = true
				continue
			}

			// Captured uses in closures cannot be skipped via cached constant form.
			for _, captures := range depNode.CapturedInputs {
				for _, capNode := range captures {
					if capNode == curr {
						hasUncachedUse = true
					}
				}
			}

			cacher, isCacher := depNode.Data.(ConstantCacheNodeData)
			for slot, inNode := range depNode.Inputs {
				if inNode != curr {
					continue
				}
				if isCacher {
					cacheSig := cacher.CacheSignature(slot)
					if cacheSig == "" {
						hasUncachedUse = true
						continue
					}
					exprSig, ok := constantSubgraphExprSignature(curr, constNode)
					if !ok {
						hasUncachedUse = true
						continue
					}
					fullSig := exprSig + "->" + cacheSig
					terminals = append(terminals, constantCacheTerminal{
						node:      depNode,
						inputSlot: slot,
						sig:       fullSig,
					})
				} else {
					// Intermediate node must be a single-output, static-shape, constant-only transform.
					if !depNode.IsConstant() || depNode.Shape.IsDynamic() || depNode.IsMultiOutputs() || len(depNode.CapturedInputs) > 0 {
						hasUncachedUse = true
						continue
					}
					if !visited[depNode.Index] {
						visited[depNode.Index] = true
						intermediateIdxs = append(intermediateIdxs, depNode.Index)
						queue = append(queue, depNode)
					}
				}
			}
		}
	}

	if len(terminals) == 0 {
		return false, "", nil, nil
	}
	sig = terminals[0].sig
	for _, t := range terminals[1:] {
		if t.sig != sig {
			return false, "", nil, nil
		}
	}
	if hasUncachedUse {
		return false, sig, terminals, nil
	}

	// Also ensure any other non-constant ancestors of intermediateIdxs (e.g., in binary constant ops)
	// are exclusively used within this constant sub-graph and are included in intermediateIdxs.
	for i := 0; i < len(intermediateIdxs); i++ {
		interNode := f.Nodes[intermediateIdxs[i]]
		for _, inNode := range interNode.Inputs {
			if inNode.OpType == compute.OpTypeConstant || visited[inNode.Index] {
				continue
			}
			if !inNode.IsConstant() || inNode.Shape.IsDynamic() || inNode.IsMultiOutputs() || len(inNode.CapturedInputs) > 0 {
				return false, sig, terminals, nil
			}
			if fe.NumUses[inNode.Index] != len(fe.Dependents[inNode.Index]) {
				return false, sig, terminals, nil
			}
			visited[inNode.Index] = true
			intermediateIdxs = append(intermediateIdxs, inNode.Index)
		}
	}
	// Verify that every node in visited (except constNode) only feeds nodes in visited or terminals.
	terminalNodeSet := make(map[int]bool, len(terminals))
	for _, t := range terminals {
		terminalNodeSet[t.node.Index] = true
	}
	for _, interIdx := range intermediateIdxs {
		for _, depIdx := range fe.Dependents[interIdx] {
			if !visited[depIdx] && !terminalNodeSet[depIdx] {
				return false, sig, terminals, nil
			}
		}
	}

	return true, sig, terminals, intermediateIdxs
}

// constantSubgraphExprSignature builds a canonical string representation of the constant expression
// tree rooted at node with respect to targetConst.
func constantSubgraphExprSignature(node, targetConst *Node) (string, bool) {
	if node == nil {
		return "", false
	}
	if node == targetConst {
		return fmt.Sprintf("SELF(%s)", node.Shape), true
	}
	if node.OpType == compute.OpTypeConstant {
		cd, ok := node.Data.(*ConstantData)
		if !ok || cd == nil {
			return "", false
		}
		// Include small scalar/vector constant values in signature if available.
		if buf := cd.Buffer(); buf != nil && cd.shape.Size() <= 16 {
			return fmt.Sprintf("Const(%s=%v)", cd.shape, buf.Flat), true
		}
		return fmt.Sprintf("Const(%s,%p)", cd.shape, cd), true
	}
	if !node.IsConstant() || node.Shape.IsDynamic() || node.IsMultiOutputs() || len(node.CapturedInputs) > 0 {
		return "", false
	}

	var sb strings.Builder
	fmt.Fprintf(&sb, "%s(%s", node.OpType, node.Shape)
	if dataSig := nodeDataSignature(node.Data); dataSig != "" {
		fmt.Fprintf(&sb, ",data=%s", dataSig)
	}
	sb.WriteString(",in=[")
	for i, in := range node.Inputs {
		if i > 0 {
			sb.WriteString(",")
		}
		inSig, ok := constantSubgraphExprSignature(in, targetConst)
		if !ok {
			return "", false
		}
		sb.WriteString(inSig)
	}
	sb.WriteString("])")
	return sb.String(), true
}

func nodeDataSignature(data any) string {
	if data == nil {
		return ""
	}
	switch v := data.(type) {
	case []int:
		return fmt.Sprintf("%v", v)
	case int:
		return fmt.Sprintf("%d", v)
	case fmt.Stringer:
		return v.String()
	default:
		return fmt.Sprintf("%+v", v)
	}
}
