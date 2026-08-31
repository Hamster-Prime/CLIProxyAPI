package responses

import (
	"bytes"
	"context"
	"strings"

	openaichat "github.com/router-for-me/CLIProxyAPI/v7/internal/translator/openai/claude"
	openairesponses "github.com/router-for-me/CLIProxyAPI/v7/internal/translator/openai/openai/responses"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// ConvertClaudeRequestToOpenAIResponses uses the existing Claude-to-Chat
// translator and then emits standard Responses input items. Keeping the two
// conversions composed avoids a second, subtly different tool/image mapper.
func ConvertClaudeRequestToOpenAIResponses(modelName string, inputRawJSON []byte, stream bool) []byte {
	chatRequest := openaichat.ConvertClaudeRequestToOpenAI(modelName, inputRawJSON, stream)
	return openairesponses.ConvertOpenAIChatCompletionsRequestToOpenAIResponses(modelName, chatRequest, stream)
}

type responsesToClaudeState struct {
	responsesParam any
	claudeParam    any
	finished       bool
}

// ConvertOpenAIResponsesResponseToClaude converts Responses SSE events into
// Anthropic Messages SSE events by first normalizing them to Chat chunks and
// reusing the established Chat-to-Claude response converter.
func ConvertOpenAIResponsesResponseToClaude(ctx context.Context, modelName string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, param *any) [][]byte {
	state := responsesToClaudeStateFromParam(param)
	chatChunks := openairesponses.ConvertOpenAIResponsesResponseToOpenAIChatCompletions(ctx, modelName, originalRequestRawJSON, requestRawJSON, rawJSON, &state.responsesParam)
	var out [][]byte
	for _, chunk := range chatChunks {
		if bytes.Equal(bytes.TrimSpace(chunk), []byte("[DONE]")) {
			out = append(out, openaichat.ConvertOpenAIResponseToClaude(ctx, modelName, originalRequestRawJSON, requestRawJSON, []byte("data: [DONE]"), &state.claudeParam)...)
			state.finished = true
			continue
		}
		line := append([]byte("data: "), chunk...)
		out = append(out, openaichat.ConvertOpenAIResponseToClaude(ctx, modelName, originalRequestRawJSON, requestRawJSON, line, &state.claudeParam)...)
	}
	// Responses streams terminate with response.completed rather than [DONE].
	// Feed a synthetic [DONE] to Anthropic's converter so it emits message_stop.
	if !state.finished && responsesPayloadIsTerminal(rawJSON) {
		out = append(out, openaichat.ConvertOpenAIResponseToClaude(ctx, modelName, originalRequestRawJSON, requestRawJSON, []byte("data: [DONE]"), &state.claudeParam)...)
		state.finished = true
	}
	if param != nil {
		*param = state
	}
	return out
}

func ConvertOpenAIResponsesResponseToClaudeNonStream(ctx context.Context, modelName string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, param *any) []byte {
	if normalized, ok := openairesponses.NormalizeOpenAIResponsesError(rawJSON); ok {
		return responsesErrorToClaude(normalized)
	}
	chat := openairesponses.ConvertOpenAIResponsesResponseToOpenAIChatCompletionsNonStream(ctx, modelName, originalRequestRawJSON, requestRawJSON, rawJSON, param)
	return openaichat.ConvertOpenAIResponseToClaudeNonStream(ctx, modelName, originalRequestRawJSON, requestRawJSON, chat, param)
}

func responsesToClaudeStateFromParam(param *any) responsesToClaudeState {
	if param != nil {
		if state, ok := (*param).(responsesToClaudeState); ok {
			return state
		}
		if state, ok := (*param).(*responsesToClaudeState); ok && state != nil {
			return *state
		}
	}
	return responsesToClaudeState{}
}

func responsesPayloadIsTerminal(rawJSON []byte) bool {
	typ, payload := openairesponses.ParseOpenAIResponsesStreamEvent(rawJSON)
	if bytes.Equal(bytes.TrimSpace(payload), []byte("[DONE]")) {
		return true
	}
	switch strings.ToLower(strings.TrimSpace(typ)) {
	case "response.completed", "response.incomplete", "response.done", "response.failed", "response.error", "error":
		return true
	}
	root := gjson.ParseBytes(payload)
	for _, path := range []string{"response.error", "error"} {
		errorNode := root.Get(path)
		if errorNode.Exists() && errorNode.Type != gjson.Null {
			return true
		}
	}
	status := strings.ToLower(strings.TrimSpace(root.Get("response.status").String()))
	return status == "completed" || status == "incomplete" || status == "failed"
}

func responsesErrorToClaude(openAIError []byte) []byte {
	root := gjson.ParseBytes(openAIError)
	errorNode := root.Get("error")
	errorType := strings.TrimSpace(errorNode.Get("type").String())
	if errorType == "" {
		errorType = "api_error"
	}
	message := errorNode.Get("message").String()
	if strings.TrimSpace(message) == "" {
		message = "upstream Responses request failed"
	}
	out := []byte(`{"type":"error","error":{"type":"api_error","message":""}}`)
	// Use the raw JSON setter so provider messages remain correctly escaped.
	var errSet error
	out, errSet = sjson.SetBytes(out, "error.type", errorType)
	if errSet != nil {
		return openAIError
	}
	out, errSet = sjson.SetBytes(out, "error.message", message)
	if errSet != nil {
		return openAIError
	}
	return out
}
