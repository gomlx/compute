package ops

import (
	"encoding/binary"
	"math/rand/v2"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

func init() {
	gobackend.RegisterRNGBitGenerator.Register(RNGBitGenerator, gobackend.PriorityGeneric)
	gobackend.MultiOutputsNodeExecutors[compute.OpTypeRNGBitGenerator] = execRNGBitGenerator
}

// RNGBitGenerator generates the given shape filled with random bits.
// It takes as input the current random number generator (RNG) state, see RNGState or RNGStateFromSeed.
// The algorithm is hard-coded to use Philox algorithm for now.
//
// It returns the new state of the RNG and the generated values (with random bits) with the given shape.
func RNGBitGenerator(f *gobackend.Function, stateOp compute.Value, shape shapes.Shape) (newState, values compute.Value, err error) {
	opType := compute.OpTypeRNGBitGenerator
	inputs, err := f.VerifyAndCastValues(opType.String(), stateOp)
	if err != nil {
		return nil, nil, err
	}
	state := inputs[0]
	if !state.Shape.Equal(compute.RNGStateShape) {
		err := errors.Errorf(
			"expected random state to be shaped %s, got state.shape=%s instead for RNGBitGenerator",
			compute.RNGStateShape,
			state.Shape,
		)
		return nil, nil, err
	}

	allInputs := []*gobackend.Node{state}
	var axisInputIndices []int
	if shape.IsDynamic() {
		axisInputIndices = make([]int, shape.Rank())
		var dynValues []compute.Value
		knownAxes := f.KnownDynamicAxisNames()
		for i, dim := range shape.Dimensions {
			axisInputIndices[i] = -1
			if dim != shapes.DynamicDim {
				continue
			}
			axisName := shape.AxisName(i)
			if axisName == "" {
				return nil, nil, errors.Errorf("RNGBitGenerator: dynamic axis %d of shape %s must have an axis name", i, shape)
			}
			dimVal, ok, err := f.DynamicAxisValue(axisName)
			if err != nil {
				return nil, nil, err
			}
			if ok {
				axisInputIndices[i] = 1 + len(dynValues)
				dynValues = append(dynValues, dimVal)
			} else if !knownAxes.Has(axisName) {
				return nil, nil, errors.Errorf("RNGBitGenerator: unknown dynamic axis %q at axis %d of shape %s", axisName, i, shape)
			}
		}
		if len(dynValues) > 0 {
			dynNodes, err := f.VerifyAndCastValues(opType.String(), dynValues...)
			if err != nil {
				return nil, nil, err
			}
			allInputs = append(allInputs, dynNodes...)
		}
	}

	outputShapes := []shapes.Shape{
		state.Shape.Clone(),
		shape.Clone(),
	}
	node := f.NewMultiOutputsNode(opType, outputShapes, allInputs...)
	if axisInputIndices != nil {
		node.Data = axisInputIndices
	}
	newState = node.MultiOutputsNodes[0]
	values = node.MultiOutputsNodes[1]
	return
}

// execRNGBitGenerator is the executor function registered for compute.OpTypeRngBitGenerator.
func execRNGBitGenerator(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, inputsOwned []bool) ([]*gobackend.Buffer, error) {
	state := inputs[0]
	stateFlat := state.Flat.([]uint64)

	targetShape := node.MultiOutputsShapes[1]
	if targetShape.IsDynamic() {
		axisInputIndices, _ := node.Data.([]int)
		concreteDims := make([]int, targetShape.Rank())
		for i, dim := range targetShape.Dimensions {
			if dim != shapes.DynamicDim {
				concreteDims[i] = dim
				continue
			}
			if i >= len(axisInputIndices) || axisInputIndices[i] < 0 || axisInputIndices[i] >= len(inputs) {
				return nil, errors.Errorf("execRNGBitGenerator: unresolved dynamic axis %d (%q) in shape %s", i, targetShape.AxisName(i), targetShape)
			}
			valSize, err := readScalarInt(inputs[axisInputIndices[i]])
			if err != nil {
				return nil, errors.Wrapf(err, "execRNGBitGenerator: reading dynamic dimension for axis %d (%q)", i, targetShape.AxisName(i))
			}
			if valSize < 0 {
				return nil, errors.Errorf("execRNGBitGenerator: dynamic dimension for axis %d (%q) must be nonnegative, got %d", i, targetShape.AxisName(i), valSize)
			}
			concreteDims[i] = valSize
		}
		targetShape = shapes.Make(targetShape.DType, concreteDims...)
	}

	// Reserved outputs:
	rngData, err := backend.GetBuffer(targetShape)
	if err != nil {
		return nil, err
	}
	rngDataBytes, err := rngData.MutableBytes()
	if err != nil {
		return nil, err
	}

	// Generate random using rand/v2:
	rng := rand.NewPCG(stateFlat[0], stateFlat[1]) // Use state and increment as seed
	var randomBits uint64
	for idx := range rngDataBytes {
		if idx%8 == 0 {
			randomBits = rng.Uint64()
		}
		// Take one byte from the randomBits.
		rngDataBytes[idx] = byte(randomBits & 0xFF)
		randomBits >>= 8
	}

	// Update state output - PCG internal state after generating random bytes
	if inputsOwned[0] {
		// We re-use the current state.
		inputs[0] = nil
	} else {
		state, err = backend.GetBuffer(node.MultiOutputsShapes[0])
		if err != nil {
			return nil, err
		}
	}
	stateFlat = state.Flat.([]uint64)

	// See details on Go source code src/math/rand/v2/pcg.go:
	rngState, err := rng.MarshalBinary()
	if err != nil {
		panic(errors.Wrapf(err, "cannot update RNGBitGenerator state"))
	}
	if len(rngState) != 20 && string(rngState[:4]) != "pcg:" {
		return nil, errors.Errorf("format of PCG random number generator changed (we got %d bytes starting with %q, "+
			"we wanted 20 and starting with the string 'pcg:'), pls open an issue in GoMLX",
			len(rngState), rngState[:4])
	}
	stateFlat[0] = binary.LittleEndian.Uint64(rngState[4 : 4+8])
	stateFlat[1] = binary.LittleEndian.Uint64(rngState[4+8 : 4+16])
	return []*gobackend.Buffer{state, rngData}, nil
}
