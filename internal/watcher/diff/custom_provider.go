package diff

import (
	"fmt"
	"sort"
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
)

// DiffCustomProviders produces human-readable changes for custom-provider entries.
// Credential values are deliberately represented only by counts.
func DiffCustomProviders(oldList, newList []config.CustomProvider) []string {
	oldMap := make(map[string]config.CustomProvider, len(oldList))
	newMap := make(map[string]config.CustomProvider, len(newList))
	for i, entry := range oldList {
		oldMap[customProviderDiffKey(entry, i)] = entry
	}
	for i, entry := range newList {
		newMap[customProviderDiffKey(entry, i)] = entry
	}
	keys := make(map[string]struct{}, len(oldMap)+len(newMap))
	for key := range oldMap {
		keys[key] = struct{}{}
	}
	for key := range newMap {
		keys[key] = struct{}{}
	}
	ordered := make([]string, 0, len(keys))
	for key := range keys {
		ordered = append(ordered, key)
	}
	sort.Strings(ordered)
	changes := make([]string, 0, len(ordered))
	for _, key := range ordered {
		oldEntry, oldOK := oldMap[key]
		newEntry, newOK := newMap[key]
		label := customProviderDiffLabel(oldEntry, newEntry, key)
		detail := describeCustomProviderUpdate(oldEntry, newEntry)
		switch {
		case !oldOK:
			changes = append(changes, fmt.Sprintf("provider added: %s (protocol=%s, api-keys=%d, models=%d)", label, customProviderProtocol(newEntry), customProviderKeyCount(newEntry), customProviderModelCount(newEntry)))
		case !newOK:
			changes = append(changes, fmt.Sprintf("provider removed: %s (protocol=%s, api-keys=%d, models=%d)", label, customProviderProtocol(oldEntry), customProviderKeyCount(oldEntry), customProviderModelCount(oldEntry)))
		case detail != "":
			changes = append(changes, fmt.Sprintf("provider updated: %s %s", label, detail))
		}
	}
	return changes
}

func customProviderDiffKey(entry config.CustomProvider, index int) string {
	name := strings.ToLower(strings.TrimSpace(entry.Name))
	if name != "" {
		return "name:" + name
	}
	base := strings.TrimSpace(entry.BaseURL)
	if base != "" {
		return "base:" + base
	}
	return fmt.Sprintf("index:%d", index)
}

func customProviderDiffLabel(oldEntry, newEntry config.CustomProvider, key string) string {
	for _, entry := range []config.CustomProvider{newEntry, oldEntry} {
		if name := strings.TrimSpace(entry.Name); name != "" {
			return name
		}
	}
	return key
}

func customProviderProtocol(entry config.CustomProvider) string {
	return config.NormalizeCustomProviderProtocol(entry.Protocol)
}

func customProviderKeyCount(entry config.CustomProvider) int {
	count := 0
	for _, key := range entry.APIKeyEntries {
		if strings.TrimSpace(key.APIKey) != "" {
			count++
		}
	}
	return count
}

func customProviderModelCount(entry config.CustomProvider) int {
	count := 0
	for _, model := range entry.Models {
		if strings.TrimSpace(model.Name) != "" || strings.TrimSpace(model.Alias) != "" {
			count++
		}
	}
	return count
}

func describeCustomProviderUpdate(oldEntry, newEntry config.CustomProvider) string {
	details := make([]string, 0, 11)
	if customProviderProtocol(oldEntry) != customProviderProtocol(newEntry) {
		details = append(details, fmt.Sprintf("protocol %s -> %s", customProviderProtocol(oldEntry), customProviderProtocol(newEntry)))
	}
	if oldEntry.Disabled != newEntry.Disabled {
		details = append(details, fmt.Sprintf("disabled %t -> %t", oldEntry.Disabled, newEntry.Disabled))
	}
	if oldEntry.SupportPromptCacheKey != newEntry.SupportPromptCacheKey {
		details = append(details, fmt.Sprintf("support-prompt-cache-key %t -> %t", oldEntry.SupportPromptCacheKey, newEntry.SupportPromptCacheKey))
	}
	if !optionalBoolEqual(oldEntry.DisableCooling, newEntry.DisableCooling) {
		details = append(details, fmt.Sprintf("disable-cooling %s -> %s", formatOptionalBool(oldEntry.DisableCooling), formatOptionalBool(newEntry.DisableCooling)))
	}
	if !optionalIntEqual(oldEntry.RequestRetry, newEntry.RequestRetry) {
		details = append(details, fmt.Sprintf("request-retry %s -> %s", formatOptionalInt(oldEntry.RequestRetry), formatOptionalInt(newEntry.RequestRetry)))
	}
	if strings.TrimSpace(oldEntry.BaseURL) != strings.TrimSpace(newEntry.BaseURL) {
		details = append(details, "base-url updated")
	}
	if customProviderKeyCount(oldEntry) != customProviderKeyCount(newEntry) {
		details = append(details, fmt.Sprintf("api-keys %d -> %d", customProviderKeyCount(oldEntry), customProviderKeyCount(newEntry)))
	}
	if customProviderModelCount(oldEntry) != customProviderModelCount(newEntry) {
		details = append(details, fmt.Sprintf("models %d -> %d", customProviderModelCount(oldEntry), customProviderModelCount(newEntry)))
	}
	if !equalStringMap(oldEntry.Headers, newEntry.Headers) {
		details = append(details, "headers updated")
	}
	if len(details) == 0 {
		return ""
	}
	return "(" + strings.Join(details, ", ") + ")"
}
