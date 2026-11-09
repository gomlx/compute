// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gobackend

import (
	"fmt"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/support"
	"github.com/gomlx/compute/support/humanize"
)

// isOpProfileEnabled is true when the GOMLX_OP_PROFILE environment variable is set
// to a non-empty value other than "0" or "false".
//
// Usage:
//   - GOMLX_OP_PROFILE=1 (or "true"): skips 1 warm-up run and prints a per-op profile
//     table every 5 timed runs.
//   - GOMLX_OP_PROFILE=<N> (where N > 1): skips 1 warm-up run and prints a per-op
//     profile table every N timed runs.
//
// Tip: combine with -backend="go:ops_sequential" for isolated wall and process CPU times
// per op without concurrent inter-op overlap.
var (
	isOpProfileEnabled bool
	opProfileInterval  int64 = 5
)

func init() {
	val := strings.TrimSpace(os.Getenv("GOMLX_OP_PROFILE"))
	if val == "" || val == "0" || strings.EqualFold(val, "false") {
		return
	}
	isOpProfileEnabled = true
	if n, err := strconv.ParseInt(val, 10, 64); err == nil && n > 1 {
		opProfileInterval = n
	}
}

// opProfileTimer holds the start timestamps for a single node execution when profiling is enabled.
type opProfileTimer struct {
	startWall time.Time
	startCPU  time.Duration
}

type opProfileEntry struct {
	calls  atomic.Int64
	wallNs atomic.Int64
	cpuNs  atomic.Int64
}

var (
	opProfileByType [compute.OpTypeLast]opProfileEntry
	opProfileBySig  sync.Map // map[string]*opProfileEntry
	opProfileRuns   atomic.Int64
)

// startOpProfile captures the starting wall and process CPU time before executing a node.
func startOpProfile(execMode OpsExecutionType) opProfileTimer {
	t := opProfileTimer{startWall: time.Now()}
	if execMode == OpsExecutionSequential {
		t.startCPU = support.CPUTime()
	}
	return t
}

// recordOpProfileNode records the elapsed wall and CPU time for a completed node execution.
func recordOpProfileNode(node *Node, execMode OpsExecutionType, timer opProfileTimer) {
	// Skip recording during the first (warm-up) run.
	if opProfileRuns.Load() < 1 {
		return
	}
	wall := time.Since(timer.startWall)
	var cpu time.Duration
	if execMode == OpsExecutionSequential {
		cpu = support.CPUTime() - timer.startCPU
	}
	if int(node.OpType) < len(opProfileByType) {
		opProfileByType[node.OpType].calls.Add(1)
		opProfileByType[node.OpType].wallNs.Add(int64(wall))
		opProfileByType[node.OpType].cpuNs.Add(int64(cpu))
	}

	var sb strings.Builder
	sb.WriteString(node.OpType.String())
	sb.WriteByte('(')
	for i, in := range node.Inputs {
		if i > 0 {
			sb.WriteString(", ")
		}
		sb.WriteString(in.Shape.String())
	}
	sb.WriteString(") -> ")
	sb.WriteString(node.Shape.String())
	sig := sb.String()

	val, ok := opProfileBySig.Load(sig)
	if !ok {
		val, _ = opProfileBySig.LoadOrStore(sig, &opProfileEntry{})
	}
	entry := val.(*opProfileEntry)
	entry.calls.Add(1)
	entry.wallNs.Add(int64(wall))
	entry.cpuNs.Add(int64(cpu))
}

// recordOpProfileRun tracks top-level function executions and prints + resets the profile
// every opProfileInterval timed runs (after skipping the initial warm-up run).
func (fe *FunctionExecutable) recordOpProfileRun(execMode OpsExecutionType) {
	if fe.Function.RawParent != nil {
		// Only count top-level function executions, not nested closures.
		return
	}
	r := opProfileRuns.Add(1)
	if r == 1 {
		// Warm-up run completed; subsequent runs will be recorded.
		return
	}
	timedRuns := r - 1
	if timedRuns%opProfileInterval != 0 {
		return
	}
	printAndResetOpProfile(timedRuns, execMode)
}

func printAndResetOpProfile(runs int64, execMode OpsExecutionType) {
	type row struct {
		name  string
		calls int64
		wall  time.Duration
		cpu   time.Duration
	}

	var rows []row
	var totalWall, totalCPU time.Duration
	for opIdx := range opProfileByType {
		c := opProfileByType[opIdx].calls.Swap(0)
		w := time.Duration(opProfileByType[opIdx].wallNs.Swap(0))
		cpu := time.Duration(opProfileByType[opIdx].cpuNs.Swap(0))
		if c > 0 {
			rows = append(rows, row{
				name:  compute.OpType(opIdx).String(),
				calls: c,
				wall:  w,
				cpu:   cpu,
			})
			totalWall += w
			totalCPU += cpu
		}
	}
	slices.SortFunc(rows, func(a, b row) int {
		return int(b.wall - a.wall)
	})

	modeStr := "parallel"
	if execMode == OpsExecutionSequential {
		modeStr = "sequential"
	}

	fmt.Printf("\n=== GOMLX_OP_PROFILE Summary (averaged over %d timed runs, mode=%s) ===\n\n", runs, modeStr)
	fmt.Printf("| OpType | Calls/run | Wall/run | Wall %% | CPU/run | Avg Cores |\n")
	fmt.Printf("| :--- | ---: | ---: | ---: | ---: | ---: |\n")
	for _, r := range rows {
		wallPerRun := r.wall / time.Duration(runs)
		cpuPerRun := r.cpu / time.Duration(runs)
		pct := 0.0
		if totalWall > 0 {
			pct = 100.0 * float64(r.wall) / float64(totalWall)
		}
		cores := 0.0
		if r.wall > 0 {
			cores = float64(r.cpu) / float64(r.wall)
		}
		fmt.Printf("| `%s` | %d | %s | %.1f%% | %s | %.1f |\n",
			r.name, r.calls/runs, humanize.Duration(wallPerRun), pct, humanize.Duration(cpuPerRun), cores)
	}
	totalCores := 0.0
	if totalWall > 0 {
		totalCores = float64(totalCPU) / float64(totalWall)
	}
	fmt.Printf("| **Total (Sum of Ops)** | — | **%s** | **100.0%%** | **%s** | **%.1f** |\n\n",
		humanize.Duration(totalWall/time.Duration(runs)),
		humanize.Duration(totalCPU/time.Duration(runs)),
		totalCores)

	var sigRows []row
	opProfileBySig.Range(func(key, value any) bool {
		e := value.(*opProfileEntry)
		c := e.calls.Swap(0)
		w := time.Duration(e.wallNs.Swap(0))
		cpu := time.Duration(e.cpuNs.Swap(0))
		if c > 0 {
			sigRows = append(sigRows, row{
				name:  key.(string),
				calls: c,
				wall:  w,
				cpu:   cpu,
			})
		}
		return true
	})
	slices.SortFunc(sigRows, func(a, b row) int {
		return int(b.wall - a.wall)
	})

	fmt.Printf("| Top Individual Op Signature | Calls/run | Wall/run | Wall/call | CPU/run | Avg Cores |\n")
	fmt.Printf("| :--- | ---: | ---: | ---: | ---: | ---: |\n")
	limit := min(len(sigRows), 35)
	for _, r := range sigRows[:limit] {
		wallPerRun := r.wall / time.Duration(runs)
		wallPerCall := r.wall / time.Duration(r.calls)
		cpuPerRun := r.cpu / time.Duration(runs)
		cores := 0.0
		if r.wall > 0 {
			cores = float64(r.cpu) / float64(r.wall)
		}
		fmt.Printf("| `%s` | %d | %s | %s | %s | %.1f |\n",
			r.name, r.calls/runs, humanize.Duration(wallPerRun), humanize.Duration(wallPerCall), humanize.Duration(cpuPerRun), cores)
	}
	fmt.Println()

	// Reset timed run counter to 1 so the next interval of `runs` starts fresh without repeating warm-up.
	opProfileRuns.Store(1)
}
