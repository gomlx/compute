package ops

// IsNaN implements compute.Builder interface.
func (f *Function) IsNaN(x compute.Value) (compute.Value, error) {
	result, err := f.NotEqual(x, x)
	if err != nil {
		return nil, errors.WithMessage(err, "while building op IsNaN")
	}
	return result, nil
}
