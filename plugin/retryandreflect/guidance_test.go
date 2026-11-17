// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package retryandreflect

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// update rewrites the files of TestDefaultGuidanceUnchanged from what the
// plugin answers now: go test ./plugin/retryandreflect -run
// TestDefaultGuidanceUnchanged -update. Use it only for a deliberate change of
// the built-in text.
var update = flag.Bool("update", false, "rewrite the golden files in testdata")

const guidanceTestTool = "lookup_order"

var (
	errOrderNotFound = errors.New("order not found")
	errToolPanicked  = toolinternal.NewPanicError(guidanceTestTool, "secret-user-data-123", []byte("goroutine 28 [running]:\n"))
	// panicLeaks are the parts of errToolPanicked's text that must not reach
	// the model.
	panicLeaks = []string{"secret-user-data-123", "goroutine", "PanicError"}
)

func guidanceTestArgs() map[string]any { return map[string]any{"order_id": "A-1"} }

// failTool reports toolErr to the plugin failures times in the invocation
// inv1, as the flow does for a failing tool, and returns what the plugin
// answered the last time.
func failTool(t *testing.T, p *plugin.Plugin, failures int, toolErr error) (map[string]any, error) {
	t.Helper()
	var res map[string]any
	var err error
	for range failures {
		res, err = p.OnToolErrorCallback()(&mockContext{invocationID: "inv1"}, &mockTool{name: guidanceTestTool}, guidanceTestArgs(), toolErr)
	}
	return res, err
}

func newPlugin(t *testing.T, opts ...PluginOption) *plugin.Plugin {
	t.Helper()
	p, err := New(opts...)
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	return p
}

// fixedGuidance returns a GuidanceFunc that always answers text.
func fixedGuidance(text string) GuidanceFunc {
	return func(agent.Context, ToolFailure) (string, error) { return text, nil }
}

// failingGuidance returns a GuidanceFunc that fails the test when called.
func failingGuidance(t *testing.T, name string) GuidanceFunc {
	return func(agent.Context, ToolFailure) (string, error) {
		t.Errorf("%s was called, want it not called", name)
		return "", nil
	}
}

// TestDefaultGuidanceUnchanged compares the plugin's whole response with files
// recorded from the plugin before guidance became configurable, so the
// built-in text cannot drift. A nil GuidanceFunc must behave like no option.
func TestDefaultGuidanceUnchanged(t *testing.T) {
	cases := []struct {
		name       string
		golden     string
		maxRetries int
		failures   int
		err        error
	}{
		{name: "reflection", golden: "default_reflection.json", maxRetries: 2, failures: 1, err: errOrderNotFound},
		{name: "retry limit exceeded", golden: "default_retry_exceeded.json", maxRetries: 2, failures: 3, err: errOrderNotFound},
		{name: "max retries zero", golden: "default_max_retries_zero.json", maxRetries: 0, failures: 1, err: errOrderNotFound},
		{name: "reflection after a panic", golden: "default_reflection_panic.json", maxRetries: 2, failures: 1, err: errToolPanicked},
	}
	options := []struct {
		name string
		opts []PluginOption
	}{
		{name: "no guidance options"},
		{name: "nil guidance funcs", opts: []PluginOption{WithReflectionGuidance(nil), WithRetryExceededGuidance(nil)}},
	}

	for _, c := range cases {
		for _, o := range options {
			t.Run(c.name+"/"+o.name, func(t *testing.T) {
				p := newPlugin(t, append([]PluginOption{WithMaxRetries(c.maxRetries)}, o.opts...)...)

				res, err := failTool(t, p, c.failures, c.err)
				if err != nil {
					t.Fatalf("OnToolErrorCallback() failed: %v", err)
				}

				got, err := json.MarshalIndent(res, "", "  ")
				if err != nil {
					t.Fatalf("Failed to marshal the response: %v", err)
				}
				got = append(got, '\n')
				golden := filepath.Join("testdata", c.golden)
				if *update {
					if err := os.WriteFile(golden, got, 0o644); err != nil {
						t.Fatalf("Failed to write the golden file: %v", err)
					}
				}
				want, err := os.ReadFile(golden)
				if err != nil {
					t.Fatalf("Failed to read the golden file: %v", err)
				}
				// The response of the panic case would carry the panic text
				// if the plugin leaked it, so a mismatch names the file only.
				if string(got) != string(want) {
					t.Errorf("response differs from %s; rerun with -update to see the difference in git diff", golden)
				}
			})
		}
	}
}

// TestCustomGuidanceReplacesOnlyGuidance checks each option against a plugin
// without it: the reflection_guidance field carries the application's text and
// every other field of the response is the same.
func TestCustomGuidanceReplacesOnlyGuidance(t *testing.T) {
	for _, tc := range []struct {
		name       string
		maxRetries int
		failures   int
		opt        PluginOption
		want       string
	}{
		{
			name:       "reflection",
			maxRetries: 2,
			failures:   2,
			opt:        WithReflectionGuidance(fixedGuidance("Check the order ID.")),
			want:       "Check the order ID.",
		},
		{
			name:       "retry limit exceeded",
			maxRetries: 2,
			failures:   3,
			opt:        WithRetryExceededGuidance(fixedGuidance("Tell the user the lookup is down.")),
			want:       "Tell the user the lookup is down.",
		},
		{
			name:       "max retries zero",
			maxRetries: 0,
			failures:   1,
			opt:        WithRetryExceededGuidance(fixedGuidance("Tell the user the lookup is down.")),
			want:       "Tell the user the lookup is down.",
		},
		{
			name:       "empty text",
			maxRetries: 2,
			failures:   1,
			opt:        WithReflectionGuidance(fixedGuidance("")),
			want:       "",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			want, err := failTool(t, newPlugin(t, WithMaxRetries(tc.maxRetries)), tc.failures, errOrderNotFound)
			if err != nil {
				t.Fatalf("OnToolErrorCallback() without the option failed: %v", err)
			}
			want["reflection_guidance"] = tc.want

			got, err := failTool(t, newPlugin(t, WithMaxRetries(tc.maxRetries), tc.opt), tc.failures, errOrderNotFound)
			if err != nil {
				t.Fatalf("OnToolErrorCallback() failed: %v", err)
			}

			if diff := cmp.Diff(want, got); diff != "" {
				t.Errorf("response mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestOtherGuidanceFuncNotCalled checks that each option covers its own kind
// of failure only.
func TestOtherGuidanceFuncNotCalled(t *testing.T) {
	t.Run("retries remain", func(t *testing.T) {
		p := newPlugin(t, WithMaxRetries(2), WithRetryExceededGuidance(failingGuidance(t, "the retry-exceeded GuidanceFunc")))
		if _, err := failTool(t, p, 2, errOrderNotFound); err != nil {
			t.Fatalf("OnToolErrorCallback() failed: %v", err)
		}
	})
	t.Run("retry limit exceeded", func(t *testing.T) {
		calls := 0
		p := newPlugin(t, WithMaxRetries(2), WithReflectionGuidance(func(agent.Context, ToolFailure) (string, error) {
			calls++
			return "", nil
		}))
		if _, err := failTool(t, p, 3, errOrderNotFound); err != nil {
			t.Fatalf("OnToolErrorCallback() failed: %v", err)
		}
		if calls != 2 {
			t.Errorf("the reflection GuidanceFunc was called %d times over 3 failures with 2 retries, want 2", calls)
		}
	})
}

// TestGuidanceFuncReceivesFailure checks every field a GuidanceFunc is given.
// DefaultGuidance is compared with the response of a plugin that has no
// guidance option and saw the same failures.
func TestGuidanceFuncReceivesFailure(t *testing.T) {
	const errorDetails = "*errors.errorString: order not found"

	for _, tc := range []struct {
		name       string
		maxRetries int
		failures   int
		// wantReflection and wantExceeded are the failures each GuidanceFunc
		// receives, in order, without DefaultGuidance.
		wantReflection []ToolFailure
		wantExceeded   []ToolFailure
	}{
		{
			name:       "retry count grows and stops at the limit",
			maxRetries: 2,
			failures:   4,
			wantReflection: []ToolFailure{
				{ToolName: guidanceTestTool, ToolArgs: guidanceTestArgs(), Err: errOrderNotFound, ErrorDetails: errorDetails, RetryCount: 1, MaxRetries: 2},
				{ToolName: guidanceTestTool, ToolArgs: guidanceTestArgs(), Err: errOrderNotFound, ErrorDetails: errorDetails, RetryCount: 2, MaxRetries: 2},
			},
			wantExceeded: []ToolFailure{
				{ToolName: guidanceTestTool, ToolArgs: guidanceTestArgs(), Err: errOrderNotFound, ErrorDetails: errorDetails, RetryCount: 2, MaxRetries: 2},
				{ToolName: guidanceTestTool, ToolArgs: guidanceTestArgs(), Err: errOrderNotFound, ErrorDetails: errorDetails, RetryCount: 2, MaxRetries: 2},
			},
		},
		{
			name:       "max retries zero",
			maxRetries: 0,
			failures:   2,
			wantExceeded: []ToolFailure{
				{ToolName: guidanceTestTool, ToolArgs: guidanceTestArgs(), Err: errOrderNotFound, ErrorDetails: errorDetails, RetryCount: 0, MaxRetries: 0},
				{ToolName: guidanceTestTool, ToolArgs: guidanceTestArgs(), Err: errOrderNotFound, ErrorDetails: errorDetails, RetryCount: 0, MaxRetries: 0},
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var gotReflection, gotExceeded []ToolFailure
			var gotCtx []agent.Context
			record := func(got *[]ToolFailure) GuidanceFunc {
				return func(ctx agent.Context, f ToolFailure) (string, error) {
					*got = append(*got, f)
					gotCtx = append(gotCtx, ctx)
					return "custom", nil
				}
			}
			p := newPlugin(t, WithMaxRetries(tc.maxRetries), WithReflectionGuidance(record(&gotReflection)), WithRetryExceededGuidance(record(&gotExceeded)))
			plain := newPlugin(t, WithMaxRetries(tc.maxRetries))
			ctx := &mockContext{invocationID: "inv1"}
			tl := &mockTool{name: guidanceTestTool}

			// The built-in text of each failure, in the order the two
			// GuidanceFuncs together are called.
			var wantDefaults []string
			for range tc.failures {
				if _, err := p.OnToolErrorCallback()(ctx, tl, guidanceTestArgs(), errOrderNotFound); err != nil {
					t.Fatalf("OnToolErrorCallback() failed: %v", err)
				}
				res, err := plain.OnToolErrorCallback()(ctx, tl, guidanceTestArgs(), errOrderNotFound)
				if err != nil {
					t.Fatalf("OnToolErrorCallback() without the options failed: %v", err)
				}
				text, ok := res["reflection_guidance"].(string)
				if !ok {
					t.Fatalf("reflection_guidance without the options is a %T, want a string", res["reflection_guidance"])
				}
				wantDefaults = append(wantDefaults, text)
			}

			var gotDefaults []string
			for _, f := range slices.Concat(gotReflection, gotExceeded) {
				gotDefaults = append(gotDefaults, f.DefaultGuidance)
			}
			if diff := cmp.Diff(wantDefaults, gotDefaults); diff != "" {
				t.Errorf("DefaultGuidance differs from the guidance of a plugin without the options (-want +got):\n%s", diff)
			}
			opts := []cmp.Option{cmpopts.EquateErrors(), cmpopts.IgnoreFields(ToolFailure{}, "DefaultGuidance")}
			if diff := cmp.Diff(tc.wantReflection, gotReflection, opts...); diff != "" {
				t.Errorf("failures given to the reflection GuidanceFunc mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tc.wantExceeded, gotExceeded, opts...); diff != "" {
				t.Errorf("failures given to the retry-exceeded GuidanceFunc mismatch (-want +got):\n%s", diff)
			}
			for i, got := range gotCtx {
				if got != agent.Context(ctx) {
					t.Errorf("call %d: the GuidanceFunc did not receive the context of the failed call", i)
				}
			}
		})
	}
}

func TestGuidanceFuncErrorIsReturned(t *testing.T) {
	errGuidance := errors.New("guidance unavailable")
	failing := func(agent.Context, ToolFailure) (string, error) { return "ignored", errGuidance }

	for _, tc := range []struct {
		name       string
		maxRetries int
		failures   int
		opt        PluginOption
	}{
		{name: "reflection", maxRetries: 2, failures: 1, opt: WithReflectionGuidance(failing)},
		{name: "retry limit exceeded", maxRetries: 1, failures: 2, opt: WithRetryExceededGuidance(failing)},
		{name: "max retries zero", maxRetries: 0, failures: 1, opt: WithRetryExceededGuidance(failing)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := newPlugin(t, WithMaxRetries(tc.maxRetries), tc.opt)

			res, err := failTool(t, p, tc.failures, errOrderNotFound)

			// Identity, not errors.Is: the plugin returns the error unchanged.
			if err != errGuidance {
				t.Errorf("OnToolErrorCallback() error = %v, want %v", err, errGuidance)
			}
			if res != nil {
				t.Errorf("OnToolErrorCallback() response = %v, want nil", res)
			}
		})
	}
}

// TestGuidanceFuncErrorStillCountsFailure checks that a failure whose
// GuidanceFunc failed still uses up a retry.
func TestGuidanceFuncErrorStillCountsFailure(t *testing.T) {
	fail := true
	p := newPlugin(t, WithMaxRetries(3), WithReflectionGuidance(func(agent.Context, ToolFailure) (string, error) {
		if fail {
			return "", errors.New("guidance unavailable")
		}
		return "custom", nil
	}))

	if _, err := failTool(t, p, 1, errOrderNotFound); err == nil {
		t.Fatal("OnToolErrorCallback() succeeded, want the GuidanceFunc's error")
	}
	fail = false
	res, err := failTool(t, p, 1, errOrderNotFound)
	if err != nil {
		t.Fatalf("OnToolErrorCallback() failed: %v", err)
	}

	if got := res["retry_count"]; got != 2 {
		t.Errorf("retry_count after one failed and one answered failure = %v, want 2", got)
	}
}

// TestGuidanceFuncNotCalledWhenToolErrorIsReturned covers the failures the
// plugin gives no guidance for: it hands the tool's error back, or leaves a
// confirmation request alone.
func TestGuidanceFuncNotCalledWhenToolErrorIsReturned(t *testing.T) {
	for _, tc := range []struct {
		name                 string
		maxRetries           int
		errorIfRetryExceeded bool
		failures             int
		toolErr              error
		wantErr              error
	}{
		{name: "error if retry exceeded", maxRetries: 1, errorIfRetryExceeded: true, failures: 2, toolErr: errOrderNotFound, wantErr: errOrderNotFound},
		{name: "error if retry exceeded with max retries zero", maxRetries: 0, errorIfRetryExceeded: true, failures: 1, toolErr: errOrderNotFound, wantErr: errOrderNotFound},
		{name: "confirmation required", maxRetries: 1, failures: 1, toolErr: fmt.Errorf("%w: lookup", tool.ErrConfirmationRequired)},
		{name: "confirmation rejected", maxRetries: 1, failures: 1, toolErr: fmt.Errorf("%w: lookup", tool.ErrConfirmationRejected)},
		{name: "confirmation required with max retries zero", maxRetries: 0, failures: 1, toolErr: tool.ErrConfirmationRequired},
	} {
		t.Run(tc.name, func(t *testing.T) {
			opts := []PluginOption{
				WithMaxRetries(tc.maxRetries),
				WithErrorIfRetryExceeded(tc.errorIfRetryExceeded),
				WithRetryExceededGuidance(failingGuidance(t, "the retry-exceeded GuidanceFunc")),
			}
			// With more than one failure the earlier ones have retries left
			// and call the reflection GuidanceFunc legitimately.
			if tc.failures == 1 {
				opts = append(opts, WithReflectionGuidance(failingGuidance(t, "the reflection GuidanceFunc")))
			}
			p := newPlugin(t, opts...)

			res, err := failTool(t, p, tc.failures, tc.toolErr)

			if err != tc.wantErr {
				t.Errorf("OnToolErrorCallback() error = %v, want %v", err, tc.wantErr)
			}
			if res != nil {
				t.Errorf("OnToolErrorCallback() response = %v, want nil", res)
			}
		})
	}
}

// TestGuidanceFuncNeverSeesPanicText gives the GuidanceFunc a recovered panic
// and has it copy every text it received, the error's included, into the
// guidance. None of it may carry the panic value or the stack, because the
// guidance goes to the model.
func TestGuidanceFuncNeverSeesPanicText(t *testing.T) {
	const wantErrText = "Invoking `lookup_order()` failed with an internal error."

	for _, tc := range []struct {
		name     string
		err      error
		failures int
	}{
		{name: "reflection", err: errToolPanicked, failures: 1},
		{name: "retry limit exceeded", err: errToolPanicked, failures: 2},
		{name: "wrapped panic", err: fmt.Errorf("lookup failed: %w", errToolPanicked), failures: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var got []ToolFailure
			echo := func(_ agent.Context, f ToolFailure) (string, error) {
				got = append(got, f)
				return fmt.Sprintf("%v | %s | %s", f.Err, f.ErrorDetails, f.DefaultGuidance), nil
			}
			p := newPlugin(t, WithMaxRetries(1), WithReflectionGuidance(echo), WithRetryExceededGuidance(echo))

			res, err := failTool(t, p, tc.failures, tc.err)
			if err != nil {
				t.Fatalf("OnToolErrorCallback() failed: %v", err)
			}

			if len(got) != tc.failures {
				t.Fatalf("GuidanceFunc called %d times, want %d", len(got), tc.failures)
			}
			last := got[len(got)-1]
			// The messages below leave out what was received: on a failure it
			// would be the panic text.
			if last.Err == nil || last.Err.Error() != wantErrText {
				t.Errorf("ToolFailure.Err does not read %q", wantErrText)
			}
			if want := "panic: " + wantErrText; last.ErrorDetails != want {
				t.Errorf("ToolFailure.ErrorDetails is not %q", want)
			}
			text, _ := res["reflection_guidance"].(string)
			if !strings.HasPrefix(text, wantErrText+" | ") {
				t.Error("reflection_guidance does not carry what the GuidanceFunc returned")
			}
			for key, value := range res {
				text, ok := value.(string)
				if !ok {
					continue
				}
				for _, leaked := range panicLeaks {
					if strings.Contains(text, leaked) {
						t.Errorf("response[%q] leaks %q", key, leaked)
					}
				}
			}
		})
	}
}

// TestGuidanceFuncRunsWithoutPluginLock calls the GuidanceFunc while checking
// that the plugin's mutex is free: application code that runs under it would
// stall every other tool failure for as long as it takes, and deadlock if it
// reached the plugin again.
func TestGuidanceFuncRunsWithoutPluginLock(t *testing.T) {
	for _, tc := range []struct {
		name       string
		maxRetries int
		failures   int
	}{
		{name: "reflection", maxRetries: 1, failures: 1},
		{name: "retry limit exceeded", maxRetries: 1, failures: 2},
		{name: "max retries zero", maxRetries: 0, failures: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := &retryAndReflect{
				maxRetries:            tc.maxRetries,
				scope:                 Invocation,
				scopedFailureCounters: make(map[string]map[string]int),
			}
			calls := 0
			fn := func(agent.Context, ToolFailure) (string, error) {
				calls++
				if !r.mu.TryLock() {
					t.Error("GuidanceFunc called with the plugin's mutex held")
					return "", nil
				}
				r.mu.Unlock()
				return "custom", nil
			}
			r.reflectionGuidance, r.retryExceededGuidance = fn, fn

			for range tc.failures {
				if _, err := r.onToolError(&mockContext{invocationID: "inv1"}, &mockTool{name: guidanceTestTool}, nil, errOrderNotFound); err != nil {
					t.Fatalf("onToolError() failed: %v", err)
				}
			}

			if calls != tc.failures {
				t.Errorf("GuidanceFunc called %d times, want %d", calls, tc.failures)
			}
		})
	}
}

// TestGuidanceFuncConcurrentFailures fails tools from many goroutines at once.
// Run with -race. Under the Global scope all of them share one counter, so the
// retry counts the GuidanceFunc sees must be exactly 1..n, in any order.
func TestGuidanceFuncConcurrentFailures(t *testing.T) {
	const n = 64

	for _, scope := range []TrackingScope{Invocation, Global} {
		t.Run(string(scope), func(t *testing.T) {
			var mu sync.Mutex
			var retryCounts []int
			record := func(_ agent.Context, f ToolFailure) (string, error) {
				mu.Lock()
				defer mu.Unlock()
				retryCounts = append(retryCounts, f.RetryCount)
				return fmt.Sprintf("attempt %d", f.RetryCount), nil
			}
			p := newPlugin(t, WithMaxRetries(n), WithTrackingScope(scope), WithReflectionGuidance(record), WithRetryExceededGuidance(record))

			var wg sync.WaitGroup
			for i := range n {
				wg.Go(func() {
					ctx := &mockContext{invocationID: fmt.Sprintf("inv%d", i)}
					res, err := p.OnToolErrorCallback()(ctx, &mockTool{name: guidanceTestTool}, guidanceTestArgs(), errOrderNotFound)
					if err != nil {
						t.Errorf("OnToolErrorCallback() failed: %v", err)
						return
					}
					if want := fmt.Sprintf("attempt %d", res["retry_count"]); res["reflection_guidance"] != want {
						t.Errorf("reflection_guidance = %v, want %q", res["reflection_guidance"], want)
					}
				})
			}
			wg.Wait()

			want := slices.Repeat([]int{1}, n)
			if scope == Global {
				for i := range want {
					want[i] = i + 1
				}
			}
			slices.Sort(retryCounts)
			if diff := cmp.Diff(want, retryCounts); diff != "" {
				t.Errorf("retry counts given to the GuidanceFunc mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestCustomGuidanceThroughRunner drives a failing tool through a full run, to
// pin what the model receives: the application's guidance, and, when the
// GuidanceFunc fails, its error as the outcome of the tool call.
func TestCustomGuidanceThroughRunner(t *testing.T) {
	for _, tc := range []struct {
		name string
		fn   GuidanceFunc
		want map[string]any
	}{
		{
			name: "guidance",
			fn:   fixedGuidance("Check the order ID."),
			want: map[string]any{
				"response_type":       reflectAndRetryResponseType,
				"error_type":          "*errors.errorString",
				"error_details":       "order not found",
				"retry_count":         1,
				"reflection_guidance": "Check the order ID.",
			},
		},
		{
			name: "guidance func error",
			fn: func(agent.Context, ToolFailure) (string, error) {
				return "", errors.New("guidance unavailable")
			},
			want: map[string]any{"error": "guidance unavailable"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			failing, err := functiontool.New(functiontool.Config{Name: guidanceTestTool, Description: "fails"},
				func(agent.Context, noArgs) (map[string]any, error) {
					return nil, errOrderNotFound
				})
			if err != nil {
				t.Fatalf("Failed to create the tool: %v", err)
			}
			a, err := llmagent.New(llmagent.Config{
				Name: "agent",
				Model: &testutil.MockModel{Responses: []*genai.Content{
					genai.NewContentFromFunctionCall(guidanceTestTool, map[string]any{}, genai.RoleModel),
					genai.NewContentFromText("done", genai.RoleModel),
				}},
				Tools: []tool.Tool{failing},
			})
			if err != nil {
				t.Fatalf("Failed to create the agent: %v", err)
			}
			p := newPlugin(t, WithReflectionGuidance(tc.fn))
			testRunner := testutil.NewTestAgentRunnerWithPluginManager(t, a, runner.PluginConfig{Plugins: []*plugin.Plugin{p}})

			events, err := testutil.CollectEvents(testRunner.Run(t, "ses1", "start"))
			if err != nil {
				t.Fatalf("Run failed: %v", err)
			}

			var got []map[string]any
			for _, ev := range events {
				if ev.Content == nil {
					continue
				}
				for _, part := range ev.Content.Parts {
					if part.FunctionResponse != nil {
						got = append(got, part.FunctionResponse.Response)
					}
				}
			}
			if diff := cmp.Diff([]map[string]any{tc.want}, got); diff != "" {
				t.Errorf("function responses mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
