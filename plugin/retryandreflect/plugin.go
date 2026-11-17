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

// Package retryandreflect provides a plugin that provides self-healing,
// concurrent-safe error recovery for tool failures.
//
// This is the Go version of the Python plugin.
// See https://github.com/google/adk-py/blob/main/google/adk/plugins/retry_and_reflect_plugin.py
//
// # Guidance
//
// When a tool call fails, the plugin answers it with a response whose
// reflection_guidance field tells the model what to do next: how to correct
// the call while retries remain, and to stop using the tool once the limit set
// by [WithMaxRetries] is exceeded. Both texts are built in. An application
// replaces either with [WithReflectionGuidance] and
// [WithRetryExceededGuidance], which take a [GuidanceFunc]:
//
//	p, err := retryandreflect.New(
//		retryandreflect.WithReflectionGuidance(func(ctx agent.Context, f retryandreflect.ToolFailure) (string, error) {
//			return f.DefaultGuidance + "\nCheck the order ID format before retrying.", nil
//		}),
//	)
//
// The function replaces the reflection_guidance field only. The other fields
// of the response are the same with and without it.
package retryandreflect

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"text/template"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"

	_ "embed"
)

//go:embed reflection.md
var reflection string
var reflectionTemplate = template.Must(template.New("ReflectionTemplate").Parse(reflection))

//go:embed exceeded.md
var exceeded string
var exceededTemplate = template.Must(template.New("ExceededTemplate").Parse(exceeded))

const (
	reflectAndRetryResponseType = "ERROR_HANDLED_BY_REFLECT_AND_RETRY_PLUGIN"
	globalScopeKey              = "__global_reflect_and_retry_scope__"
	// panicErrorType stands in for the Go type of a recovered tool panic, which
	// names an internal package the model has no use for.
	panicErrorType = "panic"
)

// TrackingScope defines the lifecycle scope for tracking tool failure counts.
type TrackingScope string

const (
	// Invocation tracks failures per-invocation.
	Invocation TrackingScope = "invocation"
	// Global tracks failures globally across all turns and users.
	Global TrackingScope = "global"
)

// ToolFailure describes a failed tool call to a [GuidanceFunc].
//
// Every field is safe to show the model: a panic recovered from the tool is
// described by a generic message, never by the panic value or the stack.
type ToolFailure struct {
	// ToolName is the name of the tool that failed.
	ToolName string
	// ToolArgs are the arguments of the failed call. The map is the one the
	// tool and the other callbacks received, so a GuidanceFunc must not modify
	// it.
	ToolArgs map[string]any
	// Err is the error the call failed with, for use with [errors.Is] and
	// [errors.As]. The one exception is a panic recovered from the tool, at
	// any wrapping depth: its error carries the panic value and the goroutine
	// stack, so Err is a new error holding only the generic message. This
	// differs from adk-python, which passes the raised exception as it is,
	// because here the text of whatever the function returns reaches the
	// model.
	Err error
	// ErrorDetails is the error as the built-in guidance shows it, in the form
	// "<error type>: <message>".
	ErrorDetails string
	// RetryCount is the number of consecutive failures of the tool so far,
	// this one included. It never exceeds MaxRetries: once the limit is
	// exceeded it stays at MaxRetries, and it is 0 when MaxRetries is 0.
	RetryCount int
	// MaxRetries is the limit set by [WithMaxRetries].
	MaxRetries int
	// DefaultGuidance is the built-in text for this failure, for a
	// GuidanceFunc to return, extend or ignore.
	DefaultGuidance string
}

// GuidanceFunc returns the guidance the plugin gives the model for a failed
// tool call. The string it returns, an empty one included, becomes the
// reflection_guidance field of the plugin's response in place of the built-in
// text.
//
// ctx is the context of the failed call. The function runs once per failure
// the plugin answers with guidance, and may run concurrently for tool calls
// made in parallel, so it must be safe for concurrent use.
//
// If the function returns an error, the plugin gives no guidance and returns
// that error from its OnToolErrorCallback unchanged. No later on-tool-error
// callback runs. The error replaces the tool's own as the outcome of the call,
// and unless an after-tool callback replaces it, its text is reported to the
// model. The failure still counts towards the retry limit. In adk-python an
// error of the provider ends the run instead.
type GuidanceFunc func(ctx agent.Context, failure ToolFailure) (string, error)

type retryAndReflect struct {
	mu                    sync.Mutex
	maxRetries            int
	errorIfRetryExceeded  bool
	scope                 TrackingScope
	scopedFailureCounters map[string]map[string]int
	reflectionGuidance    GuidanceFunc
	retryExceededGuidance GuidanceFunc
}

// PluginOption is an option for configuring the ReflectAndRetryToolPlugin.
type PluginOption func(*retryAndReflect)

// WithMaxRetries sets the maximum number of retries for a tool.
func WithMaxRetries(maxRetries int) PluginOption {
	return func(r *retryAndReflect) {
		r.maxRetries = maxRetries
	}
}

// WithErrorIfRetryExceeded sets whether to return an error if the retry limit is exceeded.
// If set to true, then the original error is returned, otherwise instead of the original error,
// the plugin will return a new instruction "createToolRetryExceedMsg" telling LLM to stop using this tool for the current task.
func WithErrorIfRetryExceeded(errorIfRetryExceeded bool) PluginOption {
	return func(r *retryAndReflect) {
		r.errorIfRetryExceeded = errorIfRetryExceeded
	}
}

// WithTrackingScope sets the tracking scope for tool failures.
func WithTrackingScope(scope TrackingScope) PluginOption {
	return func(r *retryAndReflect) {
		r.scope = scope
	}
}

// WithReflectionGuidance sets the function that writes the guidance for a
// failed tool call while retries remain. A nil fn keeps the built-in text.
//
// The function is never called with [WithMaxRetries] of 0, which leaves no
// retries.
func WithReflectionGuidance(fn GuidanceFunc) PluginOption {
	return func(r *retryAndReflect) {
		r.reflectionGuidance = fn
	}
}

// WithRetryExceededGuidance sets the function that writes the guidance for a
// failed tool call once the retry limit is exceeded, which with
// [WithMaxRetries] of 0 is every failure. A nil fn keeps the built-in text.
//
// The function is never called when [WithErrorIfRetryExceeded] is true,
// because the plugin then returns the tool's error instead of guidance.
func WithRetryExceededGuidance(fn GuidanceFunc) PluginOption {
	return func(r *retryAndReflect) {
		r.retryExceededGuidance = fn
	}
}

// New creates a new reflect and retry tool plugin.
func New(opts ...PluginOption) (*plugin.Plugin, error) {
	r := &retryAndReflect{
		maxRetries:            3, // A sensible default
		errorIfRetryExceeded:  false,
		scope:                 Invocation,
		scopedFailureCounters: make(map[string]map[string]int),
	}

	for _, opt := range opts {
		opt(r)
	}

	if r.maxRetries < 0 {
		return nil, fmt.Errorf("maxRetries must be a non-negative integer")
	}

	return plugin.New(plugin.Config{
		Name:                "RetryAndReflectPlugin",
		AfterToolCallback:   r.afterTool,
		OnToolErrorCallback: r.onToolError,
		AfterRunCallback:    r.afterRun,
	})
}

// MustNew creates a new reflect and retry tool plugin and panics if it fails.
func MustNew(opts ...PluginOption) *plugin.Plugin {
	p, err := New(opts...)
	if err != nil {
		panic(err)
	}
	return p
}

func (r *retryAndReflect) afterTool(ctx agent.Context, tool tool.Tool, args, result map[string]any, err error) (map[string]any, error) {
	if err == nil {
		isReflectResponse := false
		if rt, ok := result["response_type"].(string); ok && rt == reflectAndRetryResponseType {
			isReflectResponse = true
		}
		// On success, reset the failure count for this specific tool within its scope.
		// But do not reset if OnToolErrorCallback just produced a reflection response.
		if !isReflectResponse {
			r.resetFailuresForTool(ctx, tool.Name())
		}
	}
	return nil, nil
}

// afterRun drops the failure counts of an invocation once its run has ended.
// Nothing reads them afterwards, so keeping them would grow
// scopedFailureCounters with every invocation that saw a failure. Every scope
// but Global keys its counts by invocation ID, as scopeKey does.
// The counts of a run that paused on an unanswered long-running call are kept:
// the run that answers the call reuses the invocation ID and continues the
// same retry budget, as in adk-python.
func (r *retryAndReflect) afterRun(ictx agent.InvocationContext) {
	if r.scope == Global {
		return
	}
	invocationID := ictx.InvocationID()
	r.mu.Lock()
	_, tracked := r.scopedFailureCounters[invocationID]
	r.mu.Unlock()
	// Most runs record no failure; skip the scan of the session events for them.
	if !tracked || hasOpenLongRunningCall(ictx.Session(), invocationID) {
		return
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.scopedFailureCounters, invocationID)
}

// hasOpenLongRunningCall reports whether the invocation has a long-running call
// that no user function response has answered yet. A failure response from
// this plugin also settles the call, since no answer will follow.
func hasOpenLongRunningCall(sess session.Session, invocationID string) bool {
	if sess == nil {
		return false
	}
	open := make(map[string]struct{})
	for ev := range sess.Events().All() {
		if ev == nil || ev.InvocationID != invocationID {
			continue
		}
		for _, id := range ev.LongRunningToolIDs {
			open[id] = struct{}{}
		}
		if ev.Content == nil {
			continue
		}
		for _, part := range ev.Content.Parts {
			if part == nil || part.FunctionResponse == nil || part.FunctionResponse.ID == "" {
				continue
			}
			fr := part.FunctionResponse
			if ev.Author == "user" || fr.Response["response_type"] == reflectAndRetryResponseType {
				delete(open, fr.ID)
			}
		}
	}
	return len(open) > 0
}

func (r *retryAndReflect) onToolError(ctx agent.Context, tool tool.Tool, args map[string]any, err error) (map[string]any, error) {
	return r.handleToolError(ctx, tool, args, err)
}

func (r *retryAndReflect) handleToolError(ctx agent.Context, failedTool tool.Tool, args map[string]any, err error) (map[string]any, error) {
	// skip if the error is tool.ErrConfirmationRequired.
	if errors.Is(err, tool.ErrConfirmationRequired) || errors.Is(err, tool.ErrConfirmationRejected) {
		return nil, nil
	}

	if r.maxRetries == 0 {
		if r.errorIfRetryExceeded {
			return nil, err
		}
		return r.createToolRetryExceedMsg(ctx, failedTool, args, err)
	}

	// The count is taken in a call of its own so that the mutex is released
	// before the response is built: that may call a GuidanceFunc, which is
	// application code.
	currentRetries := r.recordFailure(r.scopeKey(ctx), failedTool.Name())

	if currentRetries <= r.maxRetries {
		return r.createToolReflectionResponse(ctx, failedTool, args, err, currentRetries)
	}

	// Max Retry exceeded
	if r.errorIfRetryExceeded {
		return nil, err
	}
	return r.createToolRetryExceedMsg(ctx, failedTool, args, err)
}

// recordFailure counts one more consecutive failure of the tool named toolName
// in the scope scopeKey and returns the new count.
func (r *retryAndReflect) recordFailure(scopeKey, toolName string) int {
	r.mu.Lock()
	defer r.mu.Unlock()

	toolFailureCounter, ok := r.scopedFailureCounters[scopeKey]
	if !ok {
		toolFailureCounter = make(map[string]int)
		r.scopedFailureCounters[scopeKey] = toolFailureCounter
	}
	toolFailureCounter[toolName]++
	return toolFailureCounter[toolName]
}

func (r *retryAndReflect) scopeKey(ctx agent.Context) string {
	if r.scope == Global {
		return globalScopeKey
	}
	return ctx.InvocationID()
}

func (r *retryAndReflect) resetFailuresForTool(ctx agent.Context, toolName string) {
	scopeKey := r.scopeKey(ctx)

	r.mu.Lock()
	defer r.mu.Unlock()
	if scope, ok := r.scopedFailureCounters[scopeKey]; ok {
		delete(scope, toolName)
		if len(scope) == 0 {
			delete(r.scopedFailureCounters, scopeKey)
		}
	}
}

// modelError is the type and text of a failed call of the tool named toolName
// that the plugin may put in the response it answers with, which the model,
// the session and REST clients all see. A panic recovered from the tool's
// handler, at any wrapping depth, becomes an opaque type and a generic
// message, because the panic error's own text carries the panic value and the
// goroutine stack.
func modelError(toolName string, err error) (errType, details string) {
	if text, panicked := toolinternal.ModelErrorText(toolName, err); panicked {
		return panicErrorType, text
	}
	return fmt.Sprintf("%T", err), err.Error()
}

func (r *retryAndReflect) formatToolArgs(toolArgs map[string]any) string {
	argsBytes, err := json.MarshalIndent(toolArgs, "", "  ")
	if err != nil {
		return fmt.Sprintf("%+v", toolArgs)
	}
	return string(argsBytes)
}

// templateData represents the variables in the templates.
type templateData struct {
	ToolName     string
	ErrorDetails string
	ArgsSummary  string
	RetryCount   int
	MaxRetries   int
}

// resolveGuidance returns the reflection_guidance text of the response to the failure
// d describes: defaultGuidance, or what fn makes of the failure when the
// application set one.
func resolveGuidance(ctx agent.Context, fn GuidanceFunc, d templateData, toolArgs map[string]any, toolErr error, defaultGuidance string) (string, error) {
	if fn == nil {
		return defaultGuidance, nil
	}
	if text, panicked := toolinternal.ModelErrorText(d.ToolName, toolErr); panicked {
		toolErr = errors.New(text)
	}
	return fn(ctx, ToolFailure{
		ToolName:        d.ToolName,
		ToolArgs:        toolArgs,
		Err:             toolErr,
		ErrorDetails:    d.ErrorDetails,
		RetryCount:      d.RetryCount,
		MaxRetries:      d.MaxRetries,
		DefaultGuidance: defaultGuidance,
	})
}

func (r *retryAndReflect) createToolReflectionResponse(ctx agent.Context, tool tool.Tool, toolArgs map[string]any, toolErr error, retryCount int) (map[string]any, error) {
	argsSummary := r.formatToolArgs(toolArgs)
	errType, errDetails := modelError(tool.Name(), toolErr)

	d := templateData{
		ToolName:     tool.Name(),
		ErrorDetails: errType + ": " + errDetails,
		ArgsSummary:  argsSummary,
		RetryCount:   retryCount,
		MaxRetries:   r.maxRetries,
	}

	var buf bytes.Buffer
	err := reflectionTemplate.Execute(&buf, d)
	if err != nil {
		return nil, nil
	}
	text, err := resolveGuidance(ctx, r.reflectionGuidance, d, toolArgs, toolErr, strings.TrimSpace(buf.String()))
	if err != nil {
		return nil, err
	}

	return map[string]any{
		"response_type":       reflectAndRetryResponseType,
		"error_type":          errType,
		"error_details":       errDetails,
		"retry_count":         retryCount,
		"reflection_guidance": text,
	}, nil
}

func (r *retryAndReflect) createToolRetryExceedMsg(ctx agent.Context, tool tool.Tool, toolArgs map[string]any, toolErr error) (map[string]any, error) {
	argsSummary := r.formatToolArgs(toolArgs)
	errType, errDetails := modelError(tool.Name(), toolErr)

	d := templateData{
		ToolName:     tool.Name(),
		ErrorDetails: errType + ": " + errDetails,
		ArgsSummary:  argsSummary,
		RetryCount:   r.maxRetries,
		MaxRetries:   r.maxRetries,
	}

	var buf bytes.Buffer
	err := exceededTemplate.Execute(&buf, d)
	if err != nil {
		return nil, nil
	}
	text, err := resolveGuidance(ctx, r.retryExceededGuidance, d, toolArgs, toolErr, strings.TrimSpace(buf.String()))
	if err != nil {
		return nil, err
	}

	return map[string]any{
		"response_type":       reflectAndRetryResponseType,
		"error_type":          errType,
		"error_details":       errDetails,
		"retry_count":         r.maxRetries,
		"reflection_guidance": text,
	}, nil
}
