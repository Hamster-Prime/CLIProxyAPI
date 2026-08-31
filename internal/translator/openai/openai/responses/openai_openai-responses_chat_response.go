package responses

import (
	"bytes"
	"context"
	"fmt"
	"strings"
	"time"

	translatorcommon "github.com/router-for-me/CLIProxyAPI/v7/internal/translator/common"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// ConvertOpenAIResponsesResponseToOpenAIChatCompletions converts Responses SSE
// events into OpenAI Chat Completions chunks. The executor/HTTP handler adds
// the final SSE framing and [DONE] marker, so each returned chunk is JSON.
func ConvertOpenAIResponsesResponseToOpenAIChatCompletions(_ context.Context, _ string, _, _, rawJSON []byte, param *any) [][]byte {
	if param == nil {
		var local any
		param = &local
	}
	if *param == nil {
		*param = newResponsesToChatStreamState()
	}
	state, ok := (*param).(*responsesToChatStreamState)
	if !ok {
		state = newResponsesToChatStreamState()
		*param = state
	}
	state.ensureMaps()
	typ, payload := ParseOpenAIResponsesStreamEvent(rawJSON)
	if len(payload) == 0 || bytes.Equal(bytes.TrimSpace(payload), []byte("[DONE]")) || state.done {
		return nil
	}
	root := gjson.ParseBytes(payload)
	if !root.Exists() {
		return nil
	}
	if root.Get("choices").Exists() {
		// Be tolerant of an upstream that already returned Chat Completions.
		return [][]byte{payload}
	}

	state.ensureIdentity(root)
	if typ == "" {
		typ = root.Get("type").String()
	}
	typ = strings.ToLower(strings.TrimSpace(typ))
	var out [][]byte
	switch typ {
	case "response.created", "response.in_progress":
		if !state.started {
			state.started = true
			out = append(out, chatChunk(state.id, state.model, state.created, []byte(`{"role":"assistant"}`), nil, gjson.Result{}))
		}
	case "response.output_item.added":
		item := root.Get("item")
		out = append(out, responsesOutputItemToChatChunksWithFallback(state, root, item, false, "added")...)
	case "response.output_item.done":
		item := root.Get("item")
		out = append(out, responsesOutputItemToChatChunks(state, root, item, true)...)
	case "response.output_text.delta":
		text := firstNonEmptyString(root.Get("delta").String(), root.Get("text").String())
		if textDelta := state.textChunk(root, text, false); textDelta != "" {
			state.started = true
			chunkDelta := []byte(`{"role":"assistant","content":""}`)
			chunkDelta, _ = sjson.SetBytes(chunkDelta, "content", textDelta)
			out = append(out, chatChunk(state.id, state.model, state.created, chunkDelta, nil, gjson.Result{}))
		}
	case "response.output_text.done":
		text := firstNonEmptyString(root.Get("text").String(), root.Get("delta").String())
		if delta := state.textChunk(root, text, true); delta != "" {
			state.started = true
			contentDelta := []byte(`{"role":"assistant","content":""}`)
			contentDelta, _ = sjson.SetBytes(contentDelta, "content", delta)
			out = append(out, chatChunk(state.id, state.model, state.created, contentDelta, nil, gjson.Result{}))
		}
	case "response.reasoning_summary_text.delta", "response.reasoning_summary_text.done":
		text := firstNonEmptyString(root.Get("delta").String(), root.Get("text").String())
		if delta := state.reasoningChunk(root, text, typ == "response.reasoning_summary_text.done"); delta != "" {
			state.started = true
			reasoningDelta := []byte(`{"role":"assistant","reasoning_content":""}`)
			reasoningDelta, _ = sjson.SetBytes(reasoningDelta, "reasoning_content", delta)
			out = append(out, chatChunk(state.id, state.model, state.created, reasoningDelta, nil, gjson.Result{}))
		}
	case "response.function_call_arguments.delta", "response.custom_tool_call_input.delta":
		state.hasToolCall = true
		index, intro := state.prepareToolCall(root, gjson.Result{})
		if len(intro) > 0 {
			state.started = true
			out = append(out, chatChunk(state.id, state.model, state.created, intro, nil, gjson.Result{}))
		}
		out = append(out, state.toolArgumentChunks(index, root.Get("delta").String())...)
	case "response.function_call_arguments.done", "response.custom_tool_call_input.done":
		state.hasToolCall = true
		index, intro := state.prepareToolCall(root, gjson.Result{})
		if len(intro) > 0 {
			state.started = true
			out = append(out, chatChunk(state.id, state.model, state.created, intro, nil, gjson.Result{}))
		}
		fullArgs := root.Get("arguments").String()
		if typ == "response.custom_tool_call_input.done" || fullArgs == "" {
			fullArgs = firstNonEmptyString(fullArgs, root.Get("input").String())
		}
		out = append(out, state.toolArgumentChunksComplete(index, fullArgs)...)
	case "response.completed", "response.incomplete", "response.done":
		response := root.Get("response")
		if !response.Exists() {
			response = root
		} else {
			state.ensureIdentity(response)
		}
		if response.Get("output").IsArray() {
			response.Get("output").ForEach(func(index gjson.Result, item gjson.Result) bool {
				fallback := fmt.Sprintf("output-%d", index.Int())
				out = append(out, responsesOutputItemToChatChunksWithFallback(state, response, item, true, fallback)...)
				return true
			})
		}
		finish := "stop"
		status := strings.ToLower(strings.TrimSpace(response.Get("status").String()))
		if status == "" {
			status = strings.ToLower(strings.TrimSpace(root.Get("status").String()))
		}
		if typ == "response.incomplete" || status == "incomplete" {
			finish = "length"
			switch strings.ToLower(strings.TrimSpace(response.Get("incomplete_details.reason").String())) {
			case "content_filter":
				finish = "content_filter"
			}
		}
		if response.Get("output").IsArray() {
			response.Get("output").ForEach(func(_, item gjson.Result) bool {
				if item.Get("type").String() == "function_call" || item.Get("type").String() == "custom_tool_call" {
					state.hasToolCall = true
					finish = "tool_calls"
				}
				return true
			})
		}
		if state.hasToolCall {
			finish = "tool_calls"
		}
		usage := response.Get("usage")
		if !usage.Exists() {
			usage = root.Get("usage")
		}
		out = append(out, chatChunk(state.id, state.model, state.created, []byte(`{"role":"assistant"}`), &finish, usage))
		state.done = true
	case "response.failed", "response.error", "error":
		// The executor turns these payloads into a request error. Do not emit a
		// successful Chat completion when a translator is used directly.
		state.done = true
	}
	return out
}

// ConvertOpenAIResponsesResponseToOpenAIChatCompletionsNonStream converts a
// completed Responses object into a Chat Completions response object.
func ConvertOpenAIResponsesResponseToOpenAIChatCompletionsNonStream(_ context.Context, modelName string, _, _, rawJSON []byte, _ *any) []byte {
	root := gjson.ParseBytes(rawJSON)
	if normalized, ok := NormalizeOpenAIResponsesError(rawJSON); ok {
		return normalized
	}
	if root.Get("choices").Exists() {
		return rawJSON
	}
	if root.Get("type").String() == "response.completed" || root.Get("type").String() == "response.incomplete" || root.Get("type").String() == "response.done" {
		if response := root.Get("response"); response.Exists() {
			root = response
		}
	}
	if !root.Exists() {
		return rawJSON
	}
	if modelName == "" {
		modelName = root.Get("model").String()
	}
	id := root.Get("id").String()
	if id == "" {
		id = fmt.Sprintf("chatcmpl_%d", time.Now().UnixNano())
	}
	created := root.Get("created_at").Int()
	if created == 0 {
		created = time.Now().Unix()
	}
	out := []byte(`{"id":"","object":"chat.completion","created":0,"model":"","choices":[]}`)
	out, _ = sjson.SetBytes(out, "id", id)
	out, _ = sjson.SetBytes(out, "created", created)
	out, _ = sjson.SetBytes(out, "model", modelName)

	var textParts []string
	var contentParts [][]byte
	var toolCalls [][]byte
	appendOutputContentPart := func(part gjson.Result) {
		if !part.Exists() {
			return
		}
		if part.Type == gjson.String {
			if part.String() != "" {
				textParts = append(textParts, part.String())
			}
			return
		}
		typ := strings.ToLower(strings.TrimSpace(part.Get("type").String()))
		if typ == "" {
			switch {
			case part.Get("text").Exists():
				typ = "text"
			case part.Get("refusal").Exists():
				typ = "refusal"
			case part.Get("image_url").Exists() || part.Get("url").Exists():
				typ = "image_url"
			}
		}
		switch typ {
		case "output_text", "input_text", "text":
			if text := part.Get("text"); text.Type == gjson.String {
				textParts = append(textParts, text.String())
			}
		case "refusal":
			textParts = append(textParts, firstNonEmptyString(part.Get("refusal").String(), part.Get("text").String()))
		case "image_url", "input_image", "image":
			if contentPart := responsesContentPartToChat(part); len(contentPart) > 0 {
				contentParts = append(contentParts, contentPart)
			}
		}
	}
	root.Get("output").ForEach(func(_, item gjson.Result) bool {
		switch item.Get("type").String() {
		case "message":
			content := item.Get("content")
			if content.IsArray() {
				content.ForEach(func(_, part gjson.Result) bool {
					appendOutputContentPart(part)
					return true
				})
			} else if content.IsObject() || content.Type == gjson.String {
				appendOutputContentPart(content)
			} else if text := item.Get("text"); text.Type == gjson.String {
				appendOutputContentPart(text)
			}
		case "output_text", "input_text", "text", "refusal", "image_url", "input_image", "image":
			appendOutputContentPart(item)
		case "function_call", "custom_tool_call":
			call := []byte(`{"id":"","type":"function","function":{"name":"","arguments":""}}`)
			call, _ = sjson.SetBytes(call, "id", firstNonEmptyString(item.Get("call_id").String(), item.Get("id").String()))
			call, _ = sjson.SetBytes(call, "function.name", item.Get("name").String())
			args := item.Get("arguments")
			if !args.Exists() {
				args = item.Get("input")
			}
			if args.Type == gjson.String {
				call, _ = sjson.SetBytes(call, "function.arguments", args.String())
			} else if args.Exists() {
				call, _ = sjson.SetBytes(call, "function.arguments", args.Raw)
			}
			toolCalls = append(toolCalls, call)
		}
		return true
	})

	message := []byte(`{"role":"assistant","content":""}`)
	text := strings.Join(textParts, "")
	if len(contentParts) > 0 {
		if text != "" {
			textPart := []byte(`{"type":"text","text":""}`)
			textPart, _ = sjson.SetBytes(textPart, "text", text)
			contentParts = append([][]byte{textPart}, contentParts...)
		}
		message = translatorcommon.SetRawArrayItems(message, "content", contentParts)
	} else {
		message, _ = sjson.SetBytes(message, "content", text)
	}
	if len(toolCalls) > 0 {
		message = translatorcommon.SetRawArrayItems(message, "tool_calls", toolCalls)
	}
	choice := []byte(`{"index":0,"message":{},"finish_reason":"stop"}`)
	choice, _ = sjson.SetRawBytes(choice, "message", message)
	if len(toolCalls) > 0 {
		choice, _ = sjson.SetBytes(choice, "finish_reason", "tool_calls")
	}
	out = translatorcommon.SetRawArrayItems(out, "choices", [][]byte{choice})
	if usage := root.Get("usage"); usage.Exists() {
		chatUsage := []byte(`{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}`)
		promptTokens := usage.Get("input_tokens").Int()
		completionTokens := usage.Get("output_tokens").Int()
		totalTokens := usage.Get("total_tokens").Int()
		if totalTokens == 0 {
			totalTokens = promptTokens + completionTokens
		}
		chatUsage, _ = sjson.SetBytes(chatUsage, "prompt_tokens", promptTokens)
		chatUsage, _ = sjson.SetBytes(chatUsage, "completion_tokens", completionTokens)
		chatUsage, _ = sjson.SetBytes(chatUsage, "total_tokens", totalTokens)
		if cached := usage.Get("input_tokens_details.cached_tokens"); cached.Exists() {
			chatUsage, _ = sjson.SetBytes(chatUsage, "prompt_tokens_details.cached_tokens", cached.Int())
		}
		out, _ = sjson.SetRawBytes(out, "usage", chatUsage)
	}
	return out
}

// NormalizeOpenAIResponsesError converts a Responses failure envelope into an
// OpenAI-compatible error object. It returns the original payload unchanged
// when it is not an error envelope.
func NormalizeOpenAIResponsesError(rawJSON []byte) ([]byte, bool) {
	root := gjson.ParseBytes(rawJSON)
	if !root.Exists() {
		return rawJSON, false
	}
	if root.Get("choices").Exists() {
		return rawJSON, false
	}
	errorNode := root.Get("error")
	if !errorNode.Exists() || errorNode.Type == gjson.Null {
		errorNode = root.Get("response.error")
	}
	typ := strings.ToLower(strings.TrimSpace(root.Get("type").String()))
	isFailure := typ == "response.failed" || typ == "response.error" || typ == "error" || (errorNode.Exists() && errorNode.Type != gjson.Null)
	if !isFailure {
		return rawJSON, false
	}
	errorType := strings.TrimSpace(errorNode.Get("type").String())
	if errorType == "" {
		errorType = "upstream_error"
	}
	message := strings.TrimSpace(errorNode.Get("message").String())
	if message == "" {
		message = strings.TrimSpace(root.Get("message").String())
	}
	if message == "" {
		message = "upstream Responses request failed"
	}
	out := []byte(`{"error":{"message":"","type":""}}`)
	out, _ = sjson.SetBytes(out, "error.message", message)
	out, _ = sjson.SetBytes(out, "error.type", errorType)
	if code := firstNonEmptyString(errorNode.Get("code").String(), root.Get("code").String()); code != "" {
		out, _ = sjson.SetBytes(out, "error.code", code)
	}
	return out, true
}

type responsesToChatStreamState struct {
	id                string
	model             string
	created           int64
	started           bool
	done              bool
	hasToolCall       bool
	nextTool          int
	toolIndexes       map[string]int
	toolIDs           map[int]string
	toolNames         map[int]string
	toolArguments     map[int]*strings.Builder
	toolArgumentsSent map[int]int
	toolIntroduced    map[int]bool
	textBuffers       map[string]*strings.Builder
	textSent          map[string]int
	reasoningBuffers  map[string]*strings.Builder
	reasoningSent     map[string]int
}

func newResponsesToChatStreamState() *responsesToChatStreamState {
	return &responsesToChatStreamState{
		toolIndexes:       make(map[string]int),
		toolIDs:           make(map[int]string),
		toolNames:         make(map[int]string),
		toolArguments:     make(map[int]*strings.Builder),
		toolArgumentsSent: make(map[int]int),
		toolIntroduced:    make(map[int]bool),
		textBuffers:       make(map[string]*strings.Builder),
		textSent:          make(map[string]int),
		reasoningBuffers:  make(map[string]*strings.Builder),
		reasoningSent:     make(map[string]int),
	}
}

func (s *responsesToChatStreamState) ensureMaps() {
	if s.toolIndexes == nil {
		s.toolIndexes = make(map[string]int)
	}
	if s.toolIDs == nil {
		s.toolIDs = make(map[int]string)
	}
	if s.toolNames == nil {
		s.toolNames = make(map[int]string)
	}
	if s.toolArguments == nil {
		s.toolArguments = make(map[int]*strings.Builder)
	}
	if s.toolArgumentsSent == nil {
		s.toolArgumentsSent = make(map[int]int)
	}
	if s.toolIntroduced == nil {
		s.toolIntroduced = make(map[int]bool)
	}
	if s.textBuffers == nil {
		s.textBuffers = make(map[string]*strings.Builder)
	}
	if s.textSent == nil {
		s.textSent = make(map[string]int)
	}
	if s.reasoningBuffers == nil {
		s.reasoningBuffers = make(map[string]*strings.Builder)
	}
	if s.reasoningSent == nil {
		s.reasoningSent = make(map[string]int)
	}
}

func (s *responsesToChatStreamState) ensureIdentity(root gjson.Result) {
	if s.id == "" {
		s.id = firstNonEmptyString(root.Get("id").String(), root.Get("response.id").String())
	}
	if s.model == "" {
		s.model = firstNonEmptyString(root.Get("model").String(), root.Get("response.model").String())
	}
	if s.created == 0 {
		s.created = root.Get("created_at").Int()
		if s.created == 0 {
			s.created = root.Get("response.created_at").Int()
		}
		if s.created == 0 {
			s.created = time.Now().Unix()
		}
	}
	if s.id == "" {
		s.id = fmt.Sprintf("resp_%d", time.Now().UnixNano())
	}
}

func (s *responsesToChatStreamState) toolIndex(key string) int {
	s.ensureMaps()
	if key == "" {
		key = fmt.Sprintf("tool-%d", s.nextTool)
	}
	if index, ok := s.toolIndexes[key]; ok {
		return index
	}
	index := s.nextTool
	s.nextTool++
	s.toolIndexes[key] = index
	return index
}

func responsesItemKey(event, item gjson.Result, fallback string) string {
	for _, value := range []string{
		item.Get("id").String(),
		event.Get("item_id").String(),
		item.Get("call_id").String(),
		event.Get("call_id").String(),
		item.Get("output_index").String(),
		event.Get("output_index").String(),
	} {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	if strings.TrimSpace(fallback) != "" {
		return strings.TrimSpace(fallback)
	}
	return "default"
}

func appendResponsesStreamValue(buffers map[string]*strings.Builder, sent map[string]int, key, value string, complete bool) string {
	if value == "" {
		return ""
	}
	builder := buffers[key]
	if builder == nil {
		builder = &strings.Builder{}
		buffers[key] = builder
	}
	current := builder.String()
	if complete {
		switch {
		case current == "":
			builder.WriteString(value)
		case strings.HasPrefix(value, current):
			builder.WriteString(value[len(current):])
		case strings.HasPrefix(current, value):
			// The complete event repeated content already received as deltas.
		case sent[key] == 0:
			builder.Reset()
			builder.WriteString(value)
		}
	} else {
		builder.WriteString(value)
	}
	all := builder.String()
	start := sent[key]
	if start >= len(all) {
		return ""
	}
	delta := all[start:]
	sent[key] = len(all)
	return delta
}

func (s *responsesToChatStreamState) textChunk(event gjson.Result, value string, complete bool) string {
	s.ensureMaps()
	return appendResponsesStreamValue(s.textBuffers, s.textSent, responsesItemKey(event, gjson.Result{}, "text"), value, complete)
}

func (s *responsesToChatStreamState) textChunkForKey(key, value string, complete bool) string {
	s.ensureMaps()
	return appendResponsesStreamValue(s.textBuffers, s.textSent, key, value, complete)
}

func (s *responsesToChatStreamState) reasoningChunk(event gjson.Result, value string, complete bool) string {
	s.ensureMaps()
	return appendResponsesStreamValue(s.reasoningBuffers, s.reasoningSent, responsesItemKey(event, gjson.Result{}, "reasoning"), value, complete)
}

func (s *responsesToChatStreamState) reasoningChunkForKey(key, value string, complete bool) string {
	s.ensureMaps()
	return appendResponsesStreamValue(s.reasoningBuffers, s.reasoningSent, key, value, complete)
}

func (s *responsesToChatStreamState) prepareToolCall(event, item gjson.Result) (int, []byte) {
	return s.prepareToolCallWithFallback(event, item, "tool")
}

func (s *responsesToChatStreamState) prepareToolCallWithFallback(event, item gjson.Result, fallback string) (int, []byte) {
	s.ensureMaps()
	key := responsesItemKey(event, item, fallback)
	index := s.toolIndex(key)
	if callID := firstNonEmptyString(item.Get("call_id").String(), event.Get("call_id").String(), item.Get("id").String(), event.Get("item_id").String()); callID != "" {
		s.toolIDs[index] = callID
	}
	if name := firstNonEmptyString(item.Get("name").String(), event.Get("name").String()); name != "" {
		s.toolNames[index] = name
	}
	callID := s.toolIDs[index]
	if callID == "" {
		callID = fmt.Sprintf("call_%s_%d", strings.TrimPrefix(s.id, "resp_"), index)
		s.toolIDs[index] = callID
	}
	if s.toolIntroduced[index] || s.toolNames[index] == "" {
		return index, nil
	}
	s.toolIntroduced[index] = true
	delta := []byte(`{"role":"assistant","tool_calls":[{"index":0,"id":"","type":"function","function":{"name":"","arguments":""}}]}`)
	delta, _ = sjson.SetBytes(delta, "tool_calls.0.index", index)
	delta, _ = sjson.SetBytes(delta, "tool_calls.0.id", callID)
	delta, _ = sjson.SetBytes(delta, "tool_calls.0.function.name", s.toolNames[index])
	return index, delta
}

func (s *responsesToChatStreamState) toolArgumentChunks(index int, value string) [][]byte {
	s.ensureMaps()
	if value == "" {
		return nil
	}
	builder := s.toolArguments[index]
	if builder == nil {
		builder = &strings.Builder{}
		s.toolArguments[index] = builder
	}
	builder.WriteString(value)
	return s.pendingToolArgumentChunks(index)
}

func (s *responsesToChatStreamState) toolArgumentChunksComplete(index int, value string) [][]byte {
	s.ensureMaps()
	if value != "" {
		builder := s.toolArguments[index]
		if builder == nil {
			builder = &strings.Builder{}
			s.toolArguments[index] = builder
		}
		current := builder.String()
		switch {
		case current == "":
			builder.WriteString(value)
		case strings.HasPrefix(value, current):
			builder.WriteString(value[len(current):])
		case strings.HasPrefix(current, value):
		case s.toolArgumentsSent[index] == 0:
			builder.Reset()
			builder.WriteString(value)
		}
	}
	return s.pendingToolArgumentChunks(index)
}

func (s *responsesToChatStreamState) pendingToolArgumentChunks(index int) [][]byte {
	if !s.toolIntroduced[index] {
		return nil
	}
	builder := s.toolArguments[index]
	if builder == nil {
		return nil
	}
	all := builder.String()
	start := s.toolArgumentsSent[index]
	if start >= len(all) {
		return nil
	}
	args := all[start:]
	s.toolArgumentsSent[index] = len(all)
	delta := []byte(`{"role":"assistant","tool_calls":[{"index":0,"type":"function","function":{"arguments":""}}]}`)
	delta, _ = sjson.SetBytes(delta, "tool_calls.0.index", index)
	if callID := s.toolIDs[index]; callID != "" {
		delta, _ = sjson.SetBytes(delta, "tool_calls.0.id", callID)
	}
	if name := s.toolNames[index]; name != "" {
		delta, _ = sjson.SetBytes(delta, "tool_calls.0.function.name", name)
	}
	delta, _ = sjson.SetBytes(delta, "tool_calls.0.function.arguments", args)
	s.started = true
	return [][]byte{chatChunk(s.id, s.model, s.created, delta, nil, gjson.Result{})}
}

func responsesOutputItemToChatChunks(state *responsesToChatStreamState, event, item gjson.Result, complete bool) [][]byte {
	return responsesOutputItemToChatChunksWithFallback(state, event, item, complete, "output-item")
}

func responsesOutputItemToChatChunksWithFallback(state *responsesToChatStreamState, event, item gjson.Result, complete bool, fallback string) [][]byte {
	if !item.Exists() {
		return nil
	}
	var out [][]byte
	itemType := strings.ToLower(strings.TrimSpace(item.Get("type").String()))
	key := responsesItemKey(event, item, fallback)
	switch itemType {
	case "message":
		content := item.Get("content")
		emitTextPart := func(part gjson.Result) {
			if !part.Exists() {
				return
			}
			partType := strings.ToLower(strings.TrimSpace(part.Get("type").String()))
			if partType == "" {
				switch {
				case part.Get("text").Exists():
					partType = "output_text"
				case part.Get("refusal").Exists():
					partType = "refusal"
				}
			}
			if partType != "output_text" && partType != "input_text" && partType != "text" && partType != "refusal" {
				return
			}
			text := part.Get("text").String()
			if partType == "refusal" {
				text = firstNonEmptyString(text, part.Get("refusal").String())
			}
			if delta := state.textChunkForKey(key, text, complete); delta != "" {
				state.started = true
				chunkDelta := []byte(`{"role":"assistant","content":""}`)
				chunkDelta, _ = sjson.SetBytes(chunkDelta, "content", delta)
				out = append(out, chatChunk(state.id, state.model, state.created, chunkDelta, nil, gjson.Result{}))
			}
		}
		if content.Type == gjson.String {
			if delta := state.textChunkForKey(key, content.String(), complete); delta != "" {
				state.started = true
				chunkDelta := []byte(`{"role":"assistant","content":""}`)
				chunkDelta, _ = sjson.SetBytes(chunkDelta, "content", delta)
				out = append(out, chatChunk(state.id, state.model, state.created, chunkDelta, nil, gjson.Result{}))
			}
		} else if content.IsArray() {
			content.ForEach(func(_, part gjson.Result) bool {
				emitTextPart(part)
				return true
			})
		} else if content.IsObject() {
			emitTextPart(content)
		} else if text := item.Get("text"); text.Type == gjson.String {
			emitTextPart(text)
		}
		if !state.started && complete {
			state.started = true
			out = append(out, chatChunk(state.id, state.model, state.created, []byte(`{"role":"assistant"}`), nil, gjson.Result{}))
		}
	case "reasoning":
		state.started = true
		if summary := item.Get("summary"); summary.IsArray() {
			summary.ForEach(func(_, part gjson.Result) bool {
				text := firstNonEmptyString(part.Get("text").String(), part.Get("summary_text").String())
				if delta := state.reasoningChunkForKey(key, text, complete); delta != "" {
					chunkDelta := []byte(`{"role":"assistant","reasoning_content":""}`)
					chunkDelta, _ = sjson.SetBytes(chunkDelta, "reasoning_content", delta)
					out = append(out, chatChunk(state.id, state.model, state.created, chunkDelta, nil, gjson.Result{}))
				}
				return true
			})
		}
	case "function_call", "custom_tool_call":
		state.hasToolCall = true
		index, intro := state.prepareToolCallWithFallback(event, item, fallback)
		if len(intro) > 0 {
			state.started = true
			out = append(out, chatChunk(state.id, state.model, state.created, intro, nil, gjson.Result{}))
		}
		args := firstNonEmptyString(item.Get("arguments").String(), item.Get("input").String())
		if complete {
			out = append(out, state.toolArgumentChunksComplete(index, args)...)
		} else {
			out = append(out, state.toolArgumentChunks(index, args)...)
		}
	}
	return out
}

func chatChunk(id, model string, created int64, delta []byte, finish *string, usage gjson.Result) []byte {
	chunk := []byte(`{"id":"","object":"chat.completion.chunk","created":0,"model":"","choices":[{"index":0,"delta":{},"finish_reason":null}]}`)
	chunk, _ = sjson.SetBytes(chunk, "id", id)
	chunk, _ = sjson.SetBytes(chunk, "created", created)
	chunk, _ = sjson.SetBytes(chunk, "model", model)
	chunk, _ = sjson.SetRawBytes(chunk, "choices.0.delta", delta)
	if finish != nil {
		chunk, _ = sjson.SetBytes(chunk, "choices.0.finish_reason", *finish)
	}
	if usage.Exists() {
		mapped := []byte(`{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}`)
		promptTokens := usage.Get("input_tokens").Int()
		completionTokens := usage.Get("output_tokens").Int()
		totalTokens := usage.Get("total_tokens").Int()
		if totalTokens == 0 {
			totalTokens = promptTokens + completionTokens
		}
		mapped, _ = sjson.SetBytes(mapped, "prompt_tokens", promptTokens)
		mapped, _ = sjson.SetBytes(mapped, "completion_tokens", completionTokens)
		mapped, _ = sjson.SetBytes(mapped, "total_tokens", totalTokens)
		chunk, _ = sjson.SetRawBytes(chunk, "usage", mapped)
	}
	return chunk
}

// ParseOpenAIResponsesStreamEvent extracts the event name and JSON payload from
// either a raw Responses data object or an SSE frame. Some gateways put the
// event name only in the SSE event field, so callers should use the returned
// name when the payload has no type field.
func ParseOpenAIResponsesStreamEvent(raw []byte) (string, []byte) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 {
		return "", nil
	}
	if gjson.ValidBytes(trimmed) {
		root := gjson.ParseBytes(trimmed)
		return strings.ToLower(strings.TrimSpace(root.Get("type").String())), trimmed
	}
	var eventName string
	var dataLines [][]byte
	for _, line := range bytes.Split(trimmed, []byte("\n")) {
		line = bytes.TrimSpace(bytes.TrimRight(line, "\r"))
		switch {
		case bytes.HasPrefix(line, []byte("event:")):
			eventName = strings.TrimSpace(string(line[len("event:"):]))
		case bytes.HasPrefix(line, []byte("data:")):
			dataLines = append(dataLines, bytes.TrimSpace(bytes.Clone(line[len("data:"):])))
		case bytes.HasPrefix(line, []byte(":")), bytes.HasPrefix(line, []byte("id:")), bytes.HasPrefix(line, []byte("retry:")):
			// SSE metadata does not belong to the JSON payload.
		default:
			if len(dataLines) == 0 && eventName == "" {
				dataLines = append(dataLines, bytes.Clone(line))
			}
		}
	}
	if len(dataLines) == 0 {
		return strings.ToLower(strings.TrimSpace(eventName)), nil
	}
	payload := bytes.TrimSpace(bytes.Join(dataLines, []byte("\n")))
	root := gjson.ParseBytes(payload)
	typ := ""
	if root.Exists() {
		typ = root.Get("type").String()
	}
	if strings.TrimSpace(typ) == "" {
		typ = eventName
	}
	return strings.ToLower(strings.TrimSpace(typ)), payload
}

func responsesStreamPayload(raw []byte) []byte {
	_, payload := ParseOpenAIResponsesStreamEvent(raw)
	return payload
}

func firstNonEmptyString(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}
