package ops

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapes"
	"github.com/pkg/errors"
)

func init() {
	gobackend.RegisterIota.Register(Iota, gobackend.PriorityGeneric)
	gobackend.SetNodeExecutor(compute.OpTypeIota, gobackend.PriorityGeneric, execIota)

	// iotaDTypeMap registration:
	iotaDTypeMap.Register(dtypes.Int8, gobackend.PriorityGeneric, execIotaGeneric[int8])
	iotaDTypeMap.Register(dtypes.Int16, gobackend.PriorityGeneric, execIotaGeneric[int16])
	iotaDTypeMap.Register(dtypes.Int32, gobackend.PriorityGeneric, execIotaGeneric[int32])
	iotaDTypeMap.Register(dtypes.Int64, gobackend.PriorityGeneric, execIotaGeneric[int64])
	iotaDTypeMap.Register(dtypes.Uint8, gobackend.PriorityGeneric, execIotaGeneric[uint8])
	iotaDTypeMap.Register(dtypes.Uint16, gobackend.PriorityGeneric, execIotaGeneric[uint16])
	iotaDTypeMap.Register(dtypes.Uint32, gobackend.PriorityGeneric, execIotaGeneric[uint32])
	iotaDTypeMap.Register(dtypes.Uint64, gobackend.PriorityGeneric, execIotaGeneric[uint64])
	iotaDTypeMap.Register(dtypes.Float32, gobackend.PriorityGeneric, execIotaGeneric[float32])
	iotaDTypeMap.Register(dtypes.Float64, gobackend.PriorityGeneric, execIotaGeneric[float64])

	// Manual registration for bfloat16 and float16.
	iotaDTypeMap.Register(dtypes.BFloat16, gobackend.PriorityGeneric, execIotaBFloat16)
	iotaDTypeMap.Register(dtypes.Float16, gobackend.PriorityGeneric, execIotaFloat16)
}

// Iota creates a constant of the given shape with increasing numbers (starting from 0)
// on the given axis. So Iota([2,2], 1) returns [[0 1][0 1]], while Iota([2,2], 0)
// returns [[0 0][1 1]].
func Iota(f *gobackend.Function, shape shapes.Shape, iotaAxis int) (*gobackend.Node, error) {
	if shape.Rank() == 0 {
		return nil, errors.Errorf("Iota: shape must have at least one dimension")
	}
	if iotaAxis < 0 || iotaAxis >= shape.Rank() {
		return nil, errors.Errorf("Iota: iotaAxis (%d) must be in the range [0,%d)", iotaAxis, shape.Rank()-1)
	}
	node, _ := f.GetOrCreateNode(compute.OpTypeIota, shape, nil, iotaAxis)
	return node, nil
}

func execIota(backend *gobackend.Backend, node *gobackend.Node, inputs []*gobackend.Buffer, inputsOwned []bool) (*gobackend.Buffer, error) {
	_, _ = inputs, inputsOwned // There are no inputs.
	output, err := backend.GetBuffer(node.Shape.DType, node.Shape.Size())
	if err != nil {
		return nil, err
	}
	output.RawShape = node.Shape
	iotaAxis := node.Data.(int)
	iotaSize := node.Shape.Dimensions[iotaAxis]
	batchSize := 1
	repeatsSize := 1
	for axis, dim := range node.Shape.Dimensions {
		if axis > iotaAxis {
			repeatsSize *= dim
		} else if axis < iotaAxis {
			batchSize *= dim
		}
	}
	fnAny, err := iotaDTypeMap.Get(node.Shape.DType)
	if err != nil {
		return nil, err
	}
	fn := fnAny.(func(output *gobackend.Buffer, batchSize, iotaSize, repeatsSize int))
	fn(output, batchSize, iotaSize, repeatsSize)
	return output, nil
}

//gobackend:dtypemap execIotaGeneric ints,uints,floats
var iotaDTypeMap = gobackend.NewDTypeMap("Iota")

func execIotaGeneric[T gobackend.PODNumericConstraints](
	output *gobackend.Buffer, batchSize, iotaSize, repeatsSize int) {
	outputFlat := output.Flat.([]T)
	flatIdx := 0
	var value T
	for range batchSize {
		// Repeat starting from 0 for each "batch dimension".
		value = T(0)
		for range iotaSize {
			for range repeatsSize {
				outputFlat[flatIdx] = value
				flatIdx++
			}
			value++
		}
	}
}

func execIotaBFloat16(output *gobackend.Buffer, batchSize, iotaSize, repeatsSize int) {
	outputFlat := output.Flat.([]bfloat16.BFloat16)
	flatIdx := 0
	var value float32
	for range batchSize {
		// Repeat starting from 0 for each "batch dimension".
		value = 0
		for range iotaSize {
			for range repeatsSize {
				outputFlat[flatIdx] = bfloat16.FromFloat32(value)
				flatIdx++
			}
			value++
		}
	}
}

func execIotaFloat16(output *gobackend.Buffer, batchSize, iotaSize, repeatsSize int) {
	outputFlat := output.Flat.([]float16.Float16)
	flatIdx := 0
	var value float32
	for range batchSize {
		// Repeat starting from 0 for each "batch dimension".
		value = 0
		for range iotaSize {
			for range repeatsSize {
				outputFlat[flatIdx] = float16.FromFloat32(value)
				flatIdx++
			}
			value++
		}
	}
}
