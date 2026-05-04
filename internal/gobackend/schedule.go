package gobackend

import (
	"github.com/gomlx/compute/support/xslices"
	"github.com/pkg/errors"
)

// This file implements different scheduling strategies.

// ScheduleStrategy defines the scheduling strategy to execute functions.
type ScheduleStrategy int

//go:generate go tool enumer -type ScheduleStrategy -trimprefix=Schedule -output=gen_schedulestrategy_enumer.go schedule.go

const (
	CreationOrderSchedule ScheduleStrategy = iota
	DependencyOrderSchedule
)

// InitSchedule initializes the execution schedule.
func (fe *FunctionExecutable) InitSchedule() {
	switch fe.Backend.SchedulingStrategy {
	case CreationOrderSchedule:
		fe.CreationOrderSchedule()
	case DependencyOrderSchedule:
		fe.DependencyOrderSchedule()
	default:
		panic(errors.Errorf("unknown schedule strategy %s", fe.Backend.SchedulingStrategy))
	}
}

func (fe *FunctionExecutable) CreationOrderSchedule() {
	fe.Schedule = xslices.Iota(0, fe.NumNodesToProcess)
}

func (fe *FunctionExecutable) DependencyOrderSchedule() {
	// TODO: Implement dependency order schedule.
	fe.CreationOrderSchedule()
}
