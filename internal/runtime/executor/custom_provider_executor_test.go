package executor

import (
	"net/http"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v7/sdk/translator"
)

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
