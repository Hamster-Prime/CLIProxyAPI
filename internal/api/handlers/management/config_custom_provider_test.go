package management

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	coreauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
)

func TestGetCustomProvidersIncludesProtocol(t *testing.T) {
	h := NewHandlerWithoutConfigFilePath(&config.Config{
		CustomProvider: []config.CustomProvider{{
			Name:     "OpenRouter",
			Protocol: config.CustomProviderProtocolMessages,
			BaseURL:  "https://openrouter.example/v1",
			APIKeyEntries: []config.CustomProviderAPIKey{{
				APIKey:   "secret",
				ProxyURL: "direct",
			}},
		}},
	}, nil)

	rec := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(rec)
	ctx.Request = httptest.NewRequest(http.MethodGet, "/v0/management/custom-provider", nil)
	h.GetCustomProviders(ctx)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	var body struct {
		Providers []struct {
			Name     string `json:"name"`
			Protocol string `json:"protocol"`
			Keys     []struct {
				APIKey string `json:"api-key"`
			} `json:"api-key-entries"`
		} `json:"custom-provider"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(body.Providers) != 1 || body.Providers[0].Name != "OpenRouter" {
		t.Fatalf("providers = %+v, want one OpenRouter provider", body.Providers)
	}
	if body.Providers[0].Protocol != config.CustomProviderProtocolMessages {
		t.Fatalf("protocol = %q, want %q", body.Providers[0].Protocol, config.CustomProviderProtocolMessages)
	}
	if len(body.Providers[0].Keys) != 1 || body.Providers[0].Keys[0].APIKey != "secret" {
		t.Fatalf("api-key entries = %+v, want secret", body.Providers[0].Keys)
	}
}

func TestPutCustomProvidersRejectsInvalidProtocolAndDuplicateName(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{
			name: "invalid protocol",
			body: `[{"name":"provider","protocol":"xml","base-url":"https://example.test"}]`,
		},
		{
			name: "duplicate names",
			body: `[{"name":"Provider","base-url":"https://one.test"},{"name":" provider ","base-url":"https://two.test"}]`,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &config.Config{}
			h := NewHandlerWithoutConfigFilePath(cfg, nil)
			rec := httptest.NewRecorder()
			ctx, _ := gin.CreateTestContext(rec)
			ctx.Request = httptest.NewRequest(http.MethodPut, "/v0/management/custom-provider", strings.NewReader(tc.body))
			ctx.Request.Header.Set("Content-Type", "application/json")
			h.PutCustomProviders(ctx)
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400; body=%s", rec.Code, rec.Body.String())
			}
			if len(cfg.CustomProvider) != 0 {
				t.Fatalf("config changed after rejected request: %+v", cfg.CustomProvider)
			}
		})
	}
}

func TestPatchCustomProviderProtocol(t *testing.T) {
	cfg := &config.Config{CustomProvider: []config.CustomProvider{{
		Name:     "DeepSeek",
		Protocol: config.CustomProviderProtocolCompletions,
		BaseURL:  "https://api.deepseek.example/v1",
	}}}
	h := &Handler{cfg: cfg, configFilePath: writeTestConfigFile(t)}
	rec := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(rec)
	ctx.Request = httptest.NewRequest(http.MethodPatch, "/v0/management/custom-provider", strings.NewReader(`{"name":"deepseek","value":{"protocol":"responses"}}`))
	ctx.Request.Header.Set("Content-Type", "application/json")
	h.PatchCustomProvider(ctx)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	if got := cfg.CustomProvider[0].Protocol; got != config.CustomProviderProtocolResponses {
		t.Fatalf("protocol = %q, want %q", got, config.CustomProviderProtocolResponses)
	}
}

func TestAPICallTransportUsesCustomProviderProxyURL(t *testing.T) {
	h := &Handler{cfg: &config.Config{CustomProvider: []config.CustomProvider{{
		Name:    "OpenRouter",
		BaseURL: "https://openrouter.example/v1",
		APIKeyEntries: []config.CustomProviderAPIKey{{
			APIKey:   "secret",
			ProxyURL: "http://proxy.example:8080",
		}},
	}}}}
	auth := &coreauth.Auth{
		Provider: "custom-provider:openrouter",
		Attributes: map[string]string{
			"api_key":         "secret",
			"custom_provider": "true",
			"custom_name":     "OpenRouter",
			"provider_key":    "custom-provider:openrouter",
		},
	}
	transport, ok := h.apiCallTransport(auth, "").(*http.Transport)
	if !ok {
		t.Fatalf("transport type = %T, want *http.Transport", h.apiCallTransport(auth, ""))
	}
	req, err := http.NewRequest(http.MethodGet, "https://example.com", nil)
	if err != nil {
		t.Fatalf("http.NewRequest returned error: %v", err)
	}
	proxyURL, err := transport.Proxy(req)
	if err != nil {
		t.Fatalf("transport.Proxy returned error: %v", err)
	}
	if proxyURL == nil || proxyURL.String() != "http://proxy.example:8080" {
		t.Fatalf("proxy URL = %v, want http://proxy.example:8080", proxyURL)
	}
}
