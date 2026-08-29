package config

import (
	"strings"
	"testing"
)

func TestParseConfigBytesCustomProviderNormalizesProtocol(t *testing.T) {
	cfg, errParse := ParseConfigBytes([]byte(`
custom-provider:
  - name: "  DeepSeek  "
    protocol: " chat-completions "
    base-url: " https://deepseek.example/v1 "
    headers:
      X-Test: " value "
    api-key-entries:
      - api-key: " deepseek-key "
  - name: OpenRouter
    base-url: https://openrouter.example/v1
  - name: MissingURL
    protocol: messages
`))
	if errParse != nil {
		t.Fatalf("ParseConfigBytes() error = %v", errParse)
	}
	if len(cfg.CustomProvider) != 2 {
		t.Fatalf("custom-provider count = %d, want 2", len(cfg.CustomProvider))
	}
	if got := cfg.CustomProvider[0].Name; got != "DeepSeek" {
		t.Fatalf("name = %q, want DeepSeek", got)
	}
	if got := cfg.CustomProvider[0].Protocol; got != CustomProviderProtocolCompletions {
		t.Fatalf("protocol = %q, want %q", got, CustomProviderProtocolCompletions)
	}
	if got := cfg.CustomProvider[0].BaseURL; got != "https://deepseek.example/v1" {
		t.Fatalf("base-url = %q, want trimmed URL", got)
	}
	if got := cfg.CustomProvider[0].APIKeyEntries[0].APIKey; got != " deepseek-key " {
		// The config sanitizer intentionally mirrors OpenAI compatibility and does
		// not mutate credential material; runtime synthesis trims it before use.
		t.Fatalf("api-key = %q, want original value", got)
	}
	if got := cfg.CustomProvider[0].Headers["X-Test"]; got != "value" {
		t.Fatalf("header = %q, want trimmed value", got)
	}
	if got := cfg.CustomProvider[1].Protocol; got != CustomProviderProtocolCompletions {
		t.Fatalf("empty protocol = %q, want %q", got, CustomProviderProtocolCompletions)
	}
}

func TestParseConfigBytesCustomProviderRejectsInvalidProtocol(t *testing.T) {
	_, errParse := ParseConfigBytes([]byte(`
custom-provider:
  - name: invalid
    protocol: xml
    base-url: https://example.test
`))
	if errParse == nil || !strings.Contains(errParse.Error(), "custom-provider[0].protocol") {
		t.Fatalf("ParseConfigBytes() error = %v, want custom provider protocol validation", errParse)
	}
}

func TestParseConfigBytesCustomProviderRejectsInvalidWeight(t *testing.T) {
	_, errParse := ParseConfigBytes([]byte(`
custom-provider:
  - name: invalid
    base-url: https://example.test
    api-key-entries:
      - api-key: key
        weight: nope
`))
	if errParse == nil || !strings.Contains(errParse.Error(), "custom-provider[0].api-key-entries[0].weight") {
		t.Fatalf("ParseConfigBytes() error = %v, want custom provider weight validation", errParse)
	}
}

func TestCloneForRuntimeCustomProviderDoesNotShareReferences(t *testing.T) {
	cfg := &Config{CustomProvider: []CustomProvider{{
		Name:    "provider",
		BaseURL: "https://example.test",
		Headers: map[string]string{"X-Test": "original"},
		APIKeyEntries: []CustomProviderAPIKey{{
			APIKey: "key",
		}},
		Models: []CustomProviderModel{{Name: "model", Alias: "alias"}},
	}}}
	clone := cfg.CloneForRuntime()
	clone.CustomProvider[0].Headers["X-Test"] = "clone"
	clone.CustomProvider[0].APIKeyEntries[0].APIKey = "clone-key"
	clone.CustomProvider[0].Models[0].Alias = "clone-alias"
	if cfg.CustomProvider[0].Headers["X-Test"] != "original" {
		t.Fatalf("original headers were mutated through clone")
	}
	if cfg.CustomProvider[0].APIKeyEntries[0].APIKey != "key" {
		t.Fatalf("original API key was mutated through clone")
	}
	if cfg.CustomProvider[0].Models[0].Alias != "alias" {
		t.Fatalf("original model was mutated through clone")
	}
}
