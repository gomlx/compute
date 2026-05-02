package ops

// Clamp returns the element-wise clamping operation.
//
// The values max and min can either be a scalar or have the same shape as x.

func init() {
	gobackend.RegisterClamp.Register(Clamp, gobackend.PriorityGeneric)
}

func Clamp(f *gobackend.Function, minV, x, maxV compute.Value) (compute.Value, error) {
	clamped, err := f.Max(minV, x)
	if err != nil {
		return nil, errors.WithMessagef(err, "gobackend.Backend %q: failed Clamp", BackendName)
	}
	clamped, err = f.Min(clamped, maxV)
	if err != nil {
		return nil, errors.WithMessagef(err, "gobackend.Backend %q: failed Clamp", BackendName)
	}
	return clamped, nil
}
