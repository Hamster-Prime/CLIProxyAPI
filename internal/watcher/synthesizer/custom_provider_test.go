package synthesizer

import (
	"testing"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	coreauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
)

func TestConfigSynthesizerCustomProvider(t *testing.T) {
	synth := NewConfigSynthesizer()
	auths, errSynthesize := synth.Synthesize(&SynthesisContext{
		Config: &config.Config{CustomProvider: []config.CustomProvider{{
			Name:     "OpenRouter",
			Protocol: " anthropic-messages ",
			Prefix:   " team-a ",
			BaseURL:  " https://openrouter.example/v1 ",
			Headers:  map[string]string{"X-Test": "value"},
			Models:   []config.CustomProviderModel{{Name: "upstream", Alias: "client"}},
			APIKeyEntries: []config.CustomProviderAPIKey{{
				APIKey:   " secret ",
				ProxyURL: " http://proxy.example ",
			}},
		}}},
		Now:         time.Now(),
		IDGenerator: NewStableIDGenerator(),
	})
	if errSynthesize != nil {
		t.Fatalf("Synthesize() error = %v", errSynthesize)
	}
	if len(auths) != 1 {
		t.Fatalf("auth count = %d, want 1", len(auths))
	}
	auth := auths[0]
	if auth.Provider != "custom-provider:openrouter" {
		t.Fatalf("provider = %q, want custom-provider:openrouter", auth.Provider)
	}
	if auth.Prefix != "team-a" {
		t.Fatalf("prefix = %q, want team-a", auth.Prefix)
	}
	if auth.ProxyURL != "http://proxy.example" {
		t.Fatalf("proxy URL = %q, want trimmed URL", auth.ProxyURL)
	}
	attrs := auth.Attributes
	if attrs["custom_provider"] != "true" || attrs["custom_name"] != "OpenRouter" || attrs["custom_service"] != "openrouter" {
		t.Fatalf("custom attrs = %#v", attrs)
	}
	if attrs["provider_key"] != "custom-provider:openrouter" {
		t.Fatalf("provider_key = %q", attrs["provider_key"])
	}
	if attrs["protocol"] != config.CustomProviderProtocolMessages {
		t.Fatalf("protocol = %q, want %q", attrs["protocol"], config.CustomProviderProtocolMessages)
	}
	if attrs["base_url"] != "https://openrouter.example/v1" {
		t.Fatalf("base_url = %q, want trimmed URL", attrs["base_url"])
	}
	if attrs["api_key"] != "secret" {
		t.Fatalf("api_key = %q, want trimmed key", attrs["api_key"])
	}
	if attrs["header:X-Test"] != "value" {
		t.Fatalf("header = %q, want value", attrs["header:X-Test"])
	}
	if auth.Status != coreauth.StatusActive {
		t.Fatalf("status = %q, want active", auth.Status)
	}
}

func TestConfigSynthesizerCustomProviderSkipsDisabledAndFallsBackWithoutKeys(t *testing.T) {
	auths, errSynthesize := NewConfigSynthesizer().Synthesize(&SynthesisContext{
		Config: &config.Config{CustomProvider: []config.CustomProvider{
			{Name: "disabled", Disabled: true, BaseURL: "https://disabled.example"},
			{Name: "no-key", BaseURL: "https://no-key.example"},
		}},
		Now:         time.Now(),
		IDGenerator: NewStableIDGenerator(),
	})
	if errSynthesize != nil {
		t.Fatalf("Synthesize() error = %v", errSynthesize)
	}
	if len(auths) != 1 {
		t.Fatalf("auth count = %d, want 1", len(auths))
	}
	if _, ok := auths[0].Attributes["api_key"]; ok {
		t.Fatalf("fallback auth unexpectedly contains api_key")
	}
	if auths[0].Attributes["protocol"] != config.CustomProviderProtocolCompletions {
		t.Fatalf("fallback protocol = %q, want %q", auths[0].Attributes["protocol"], config.CustomProviderProtocolCompletions)
	}
}
