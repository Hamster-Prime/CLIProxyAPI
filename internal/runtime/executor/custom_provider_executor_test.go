package executor

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v7/sdk/translator"
	"github.com/tidwall/gjson"
)

type customProviderRoundTripper func(*http.Request) (*http.Response, error)

func (f customProviderRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func customProviderJSONResponse(body string) *http.Response {
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}

func TestCustomProviderExecutorRequestToFormatAndEndpoint(t *testing.T) {
	tests := []struct {
		protocol string
		format   sdktranslator.Format
		path     string
	}{
		{protocol: config.CustomProviderProtocolCompletions, format: sdktranslator.FormatOpenAI, path: "/chat/completions"},
		{protocol: config.CustomProviderProtocolResponses, format: sdktranslator.FormatOpenAIResponse, path: "/responses"},
		{protocol: config.CustomProviderProtocolMessages, format: sdktranslator.FormatClaude, path: "/messages"},
	}
	for _, tt := range tests {
		t.Run(tt.protocol, func(t *testing.T) {
			executor := NewCustomProviderExecutor("custom-provider:test", tt.protocol, &config.Config{})
			got := executor.RequestToFormat(cliproxyexecutor.Request{}, cliproxyexecutor.Options{SourceFormat: sdktranslator.FormatOpenAI})
			if got != tt.format {
				t.Fatalf("RequestToFormat() = %q, want %q", got, tt.format)
			}
			gotFormat, gotPath := executor.upstreamTarget(nil, cliproxyexecutor.Options{})
			if gotFormat != tt.format || gotPath != tt.path {
				t.Fatalf("upstreamTarget() = (%q, %q), want (%q, %q)", gotFormat, gotPath, tt.format, tt.path)
			}
		})
	}

	compact := NewCustomProviderExecutor("custom-provider:test", config.CustomProviderProtocolResponses, &config.Config{})
	format, path := compact.upstreamTarget(nil, cliproxyexecutor.Options{Alt: "responses/compact"})
	if format != sdktranslator.FormatOpenAIResponse || path != "/responses/compact" {
		t.Fatalf("compact upstreamTarget() = (%q, %q)", format, path)
	}
}

func TestCustomProviderExecutorProtocolIsVisibleThroughEmbeddedExecution(t *testing.T) {
	executor := NewCustomProviderExecutor("custom-provider:anthropic", config.CustomProviderProtocolMessages, &config.Config{})
	var usageExecutor interface {
		Protocol() string
	}
	usageExecutor = executor.OpenAICompatExecutor
	if got := usageExecutor.Protocol(); got != config.CustomProviderProtocolMessages {
		t.Fatalf("embedded executor protocol = %q, want %q", got, config.CustomProviderProtocolMessages)
	}
}

func TestCustomProviderExecutorPrepareRequestProtectsProtocolHeaders(t *testing.T) {
	req, err := http.NewRequest(http.MethodPost, "https://example.test/v1/messages", nil)
	if err != nil {
		t.Fatal(err)
	}
	executor := NewCustomProviderExecutor("custom-provider:anthropic", config.CustomProviderProtocolMessages, &config.Config{})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{
		"api_key":              "secret",
		"header:x-api-key":     "overridden",
		"header:authorization": "overridden",
	}}
	if errPrepare := executor.PrepareRequest(req, auth); errPrepare != nil {
		t.Fatalf("PrepareRequest() error = %v", errPrepare)
	}
	if got := req.Header.Get("x-api-key"); got != "secret" {
		t.Fatalf("x-api-key = %q, want secret", got)
	}
	if got := req.Header.Get("Authorization"); got != "" {
		t.Fatalf("Authorization = %q, want empty", got)
	}
	if got := req.Header.Get("anthropic-version"); got != "2023-06-01" {
		t.Fatalf("anthropic-version = %q, want default version", got)
	}
}

func TestCustomProviderExecutorResolvesPromptCacheConfig(t *testing.T) {
	cfg := &config.Config{CustomProvider: []config.CustomProvider{{
		Name:                  "Anthropic Gateway",
		SupportPromptCacheKey: true,
		Models:                []config.CustomProviderModel{{Name: "upstream", Alias: "alias"}},
	}}}
	executor := NewCustomProviderExecutor("custom-provider:anthropic gateway", config.CustomProviderProtocolMessages, cfg)
	auth := &cliproxyauth.Auth{Provider: "custom-provider:anthropic gateway", Attributes: map[string]string{
		"custom_provider": "true",
		"custom_name":     "Anthropic Gateway",
		"provider_key":    "custom-provider:anthropic gateway",
	}}
	compat := executor.resolveEffectiveCompatConfig(auth)
	if compat == nil || !compat.SupportPromptCacheKey || compat.Name != "Anthropic Gateway" {
		t.Fatalf("effective config = %#v", compat)
	}
}

func TestCustomProviderExecutorRecognizesMessageErrors(t *testing.T) {
	if !openAICompatErrorEvent("message_error") {
		t.Fatal("message_error event should be treated as an upstream error")
	}
	streamErr, ok := openAICompatStreamDataError([]byte(`{"type":"message_error","error":{"message":"bad"}}`), "message_error")
	if !ok || streamErr.code != http.StatusBadGateway {
		t.Fatalf("message_error payload = (%+v, %v), want bad gateway error", streamErr, ok)
	}
}

func TestCustomProviderExecutorResponsesTranslatesChatRequest(t *testing.T) {
	var gotPath string
	var gotAuth string
	var gotBody []byte
	roundTripper := customProviderRoundTripper(func(req *http.Request) (*http.Response, error) {
		gotPath = req.URL.Path
		gotAuth = req.Header.Get("Authorization")
		gotBody, _ = io.ReadAll(req.Body)
		return customProviderJSONResponse(`{"id":"resp_1","object":"response","created_at":1700000000,"model":"upstream","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]}`), nil
	})
	ctx := context.WithValue(context.Background(), "cliproxy.roundtripper", http.RoundTripper(roundTripper))
	executor := NewCustomProviderExecutor("custom-provider:responses", config.CustomProviderProtocolResponses, &config.Config{})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{
		"base_url": "https://gateway.example/v1",
		"api_key":  "secret",
	}}
	payload := []byte(`{"model":"client-model","messages":[{"role":"user","content":"hello"}]}`)
	resp, errExecute := executor.Execute(ctx, auth, cliproxyexecutor.Request{Model: "upstream", Payload: payload}, cliproxyexecutor.Options{
		SourceFormat:    sdktranslator.FormatOpenAI,
		ResponseFormat:  sdktranslator.FormatOpenAI,
		OriginalRequest: payload,
	})
	if errExecute != nil {
		t.Fatalf("Execute() error = %v", errExecute)
	}
	if gotPath != "/v1/responses" {
		t.Fatalf("path = %q, want /v1/responses", gotPath)
	}
	if gotAuth != "Bearer secret" {
		t.Fatalf("Authorization = %q, want Bearer secret", gotAuth)
	}
	if got := gjson.GetBytes(gotBody, "input.0.role").String(); got != "user" {
		t.Fatalf("input role = %q, want user; body=%s", got, gotBody)
	}
	if got := gjson.GetBytes(gotBody, "input.0.content.0.text").String(); got != "hello" {
		t.Fatalf("input text = %q, want hello; body=%s", got, gotBody)
	}
	if got := gjson.GetBytes(resp.Payload, "choices.0.message.content").String(); got != "ok" {
		t.Fatalf("response content = %q, want ok; body=%s", got, resp.Payload)
	}
}

func TestCustomProviderExecutorResponsesPreservesNativeInput(t *testing.T) {
	var gotBody []byte
	roundTripper := customProviderRoundTripper(func(req *http.Request) (*http.Response, error) {
		gotBody, _ = io.ReadAll(req.Body)
		return customProviderJSONResponse(`{"id":"resp_2","object":"response","created_at":1700000001,"model":"upstream","status":"completed","output":[]}`), nil
	})
	ctx := context.WithValue(context.Background(), "cliproxy.roundtripper", http.RoundTripper(roundTripper))
	executor := NewCustomProviderExecutor("custom-provider:responses", config.CustomProviderProtocolResponses, &config.Config{})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{
		"base_url": "https://gateway.example/v1",
		"api_key":  "secret",
	}}
	payload := []byte(`{"model":"client-model","input":"hello"}`)
	if _, errExecute := executor.Execute(ctx, auth, cliproxyexecutor.Request{Model: "upstream", Payload: payload}, cliproxyexecutor.Options{
		SourceFormat:    sdktranslator.FormatOpenAIResponse,
		ResponseFormat:  sdktranslator.FormatOpenAIResponse,
		OriginalRequest: payload,
	}); errExecute != nil {
		t.Fatalf("Execute() error = %v", errExecute)
	}
	if got := gjson.GetBytes(gotBody, "input").String(); got != "hello" {
		t.Fatalf("native input = %q, want hello; body=%s", got, gotBody)
	}
	if gjson.GetBytes(gotBody, "messages").Exists() {
		t.Fatalf("native Responses body unexpectedly contains messages: %s", gotBody)
	}
}

func TestCustomProviderExecutorResponsesNormalizesMessagesAtResponsesEndpoint(t *testing.T) {
	var gotBody []byte
	roundTripper := customProviderRoundTripper(func(req *http.Request) (*http.Response, error) {
		gotBody, _ = io.ReadAll(req.Body)
		return customProviderJSONResponse(`{"id":"resp_messages","object":"response","created_at":1700000001,"model":"upstream","status":"completed","output":[]}`), nil
	})
	ctx := context.WithValue(context.Background(), "cliproxy.roundtripper", http.RoundTripper(roundTripper))
	executor := NewCustomProviderExecutor("custom-provider:responses", config.CustomProviderProtocolResponses, &config.Config{})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{
		"base_url": "https://gateway.example/v1",
		"api_key":  "secret",
	}}
	payload := []byte(`{"model":"client-model","messages":[{"role":"user","content":"hello"}]}`)
	if _, errExecute := executor.Execute(ctx, auth, cliproxyexecutor.Request{Model: "upstream", Payload: payload}, cliproxyexecutor.Options{
		SourceFormat:    sdktranslator.FormatOpenAIResponse,
		ResponseFormat:  sdktranslator.FormatOpenAIResponse,
		OriginalRequest: payload,
	}); errExecute != nil {
		t.Fatalf("Execute() error = %v", errExecute)
	}
	if got := gjson.GetBytes(gotBody, "input.0.content.0.text").String(); got != "hello" {
		t.Fatalf("normalized input text = %q; body=%s", got, gotBody)
	}
	if gjson.GetBytes(gotBody, "messages").Exists() {
		t.Fatalf("normalized Responses body unexpectedly contains messages: %s", gotBody)
	}
}

func TestCustomProviderExecutorResponsesRejectsEmptyTranslatedInput(t *testing.T) {
	called := false
	roundTripper := customProviderRoundTripper(func(req *http.Request) (*http.Response, error) {
		called = true
		return customProviderJSONResponse(`{}`), nil
	})
	ctx := context.WithValue(context.Background(), "cliproxy.roundtripper", http.RoundTripper(roundTripper))
	executor := NewCustomProviderExecutor("custom-provider:responses", config.CustomProviderProtocolResponses, &config.Config{})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{
		"base_url": "https://gateway.example/v1",
		"api_key":  "secret",
	}}
	if _, errExecute := executor.Execute(ctx, auth, cliproxyexecutor.Request{Model: "upstream", Payload: []byte(`{"model":"client-model","messages":[]}`)}, cliproxyexecutor.Options{
		SourceFormat:   sdktranslator.FormatOpenAI,
		ResponseFormat: sdktranslator.FormatOpenAI,
	}); errExecute == nil {
		t.Fatal("Execute() should reject an empty Responses input")
	}
	if called {
		t.Fatal("upstream should not be called for an empty Responses input")
	}
}

func TestCustomProviderExecutorResponsesBridgesMessagesAlongsideInstructions(t *testing.T) {
	root := gjson.ParseBytes(normalizeCustomResponsesRequest(
		sdktranslator.FormatOpenAIResponse,
		"model",
		[]byte(`{"instructions":"keep this instruction","previous_response_id":"resp_prev","messages":[{"role":"user","content":"hello"}]}`),
		false,
	))
	if got := root.Get("input.0.content.0.text").String(); got != "hello" {
		t.Fatalf("bridged input text = %q, want hello; output=%s", got, root.Raw)
	}
	if got := root.Get("instructions").String(); got != "keep this instruction" {
		t.Fatalf("instructions = %q, want preserved instruction; output=%s", got, root.Raw)
	}
	if got := root.Get("previous_response_id").String(); got != "resp_prev" {
		t.Fatalf("previous_response_id = %q, want resp_prev; output=%s", got, root.Raw)
	}
}

func TestCustomProviderExecutorResponsesRejectsSemanticEmptyInput(t *testing.T) {
	if openAICompatResponsesRequestHasInput([]byte(`{"input":[{"type":"message","role":"user","content":[]}]}`)) {
		t.Fatal("empty message should not count as usable Responses input")
	}
	if openAICompatResponsesRequestHasInput([]byte(`{"input":[{"type":"message","content":[{"type":"unsupported","value":""}]}]}`)) {
		t.Fatal("unsupported empty content should not count as usable Responses input")
	}
}

func TestCustomProviderExecutorCompletionsRejectsUnmappedResponsesInput(t *testing.T) {
	called := false
	roundTripper := customProviderRoundTripper(func(req *http.Request) (*http.Response, error) {
		called = true
		return customProviderJSONResponse(`{}`), nil
	})
	ctx := context.WithValue(context.Background(), "cliproxy.roundtripper", http.RoundTripper(roundTripper))
	executor := NewCustomProviderExecutor("custom-provider:completions", config.CustomProviderProtocolCompletions, &config.Config{})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{
		"base_url": "https://gateway.example/v1",
		"api_key":  "secret",
	}}
	payload := []byte(`{"model":"client-model","input":[{"type":"input_file","file_id":"file_1"}]}`)
	if _, errExecute := executor.Execute(ctx, auth, cliproxyexecutor.Request{Model: "upstream", Payload: payload}, cliproxyexecutor.Options{
		SourceFormat:   sdktranslator.FormatOpenAIResponse,
		ResponseFormat: sdktranslator.FormatOpenAIResponse,
	}); errExecute == nil {
		t.Fatal("Execute() should reject an unmapped Responses input")
	}
	if called {
		t.Fatal("upstream should not be called for an unmapped Responses input")
	}
}

func TestCustomProviderExecutorResponsesTranslatesEventOnlyStream(t *testing.T) {
	roundTripper := customProviderRoundTripper(func(_ *http.Request) (*http.Response, error) {
		body := "event: response.created\ndata: {\"response\":{\"id\":\"resp_stream\",\"model\":\"upstream\",\"created_at\":1700000000}}\n\n" +
			"event: response.output_text.delta\ndata: {\"delta\":\"hello\"}\n\n" +
			"event: response.completed\ndata: {\"response\":{\"id\":\"resp_stream\",\"model\":\"upstream\",\"created_at\":1700000000,\"usage\":{\"input_tokens\":2,\"output_tokens\":1,\"total_tokens\":3}}}\n\n"
		response := customProviderJSONResponse(body)
		response.Header.Set("Content-Type", "text/event-stream")
		return response, nil
	})
	ctx := context.WithValue(context.Background(), "cliproxy.roundtripper", http.RoundTripper(roundTripper))
	executor := NewCustomProviderExecutor("custom-provider:responses", config.CustomProviderProtocolResponses, &config.Config{})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{
		"base_url": "https://gateway.example/v1",
		"api_key":  "secret",
	}}
	payload := []byte(`{"model":"client-model","messages":[{"role":"user","content":"hello"}],"stream":true}`)
	result, errExecute := executor.ExecuteStream(ctx, auth, cliproxyexecutor.Request{Model: "upstream", Payload: payload}, cliproxyexecutor.Options{
		Stream:          true,
		SourceFormat:    sdktranslator.FormatOpenAI,
		ResponseFormat:  sdktranslator.FormatOpenAI,
		OriginalRequest: payload,
	})
	if errExecute != nil {
		t.Fatalf("ExecuteStream() error = %v", errExecute)
	}
	var chunks [][]byte
	for chunk := range result.Chunks {
		if chunk.Err != nil {
			t.Fatalf("stream chunk error = %v", chunk.Err)
		}
		chunks = append(chunks, chunk.Payload)
	}
	if len(chunks) != 3 {
		t.Fatalf("stream chunks = %d, want 3", len(chunks))
	}
	if got := gjson.GetBytes(chunks[1], "choices.0.delta.content").String(); got != "hello" {
		t.Fatalf("content delta = %q, want hello; chunk=%s", got, chunks[1])
	}
	if got := gjson.GetBytes(chunks[2], "choices.0.finish_reason").String(); got != "stop" {
		t.Fatalf("finish reason = %q, want stop; chunk=%s", got, chunks[2])
	}
	if got := gjson.GetBytes(chunks[2], "usage.total_tokens").Int(); got != 3 {
		t.Fatalf("total tokens = %d, want 3; chunk=%s", got, chunks[2])
	}
}
