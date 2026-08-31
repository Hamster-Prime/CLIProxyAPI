package responses

import (
	"fmt"
	"strings"

	translatorcommon "github.com/router-for-me/CLIProxyAPI/v7/internal/translator/common"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// ConvertOpenAIChatCompletionsRequestToOpenAIResponses converts an OpenAI Chat
// Completions request into the Responses request shape. This is used when a
// client speaks Chat Completions but the selected upstream exposes /responses.
func ConvertOpenAIChatCompletionsRequestToOpenAIResponses(modelName string, inputRawJSON []byte, stream bool) []byte {
	root := gjson.ParseBytes(inputRawJSON)
	out := []byte(`{"model":"","input":[],"stream":false}`)
	if strings.TrimSpace(modelName) == "" {
		modelName = strings.TrimSpace(root.Get("model").String())
	}
	out, _ = sjson.SetBytes(out, "model", modelName)
	if streamValue := root.Get("stream"); streamValue.Exists() {
		out, _ = sjson.SetBytes(out, "stream", streamValue.Bool())
	} else {
		out, _ = sjson.SetBytes(out, "stream", stream)
	}

	var inputItems [][]byte
	var systemInstructions []string
	if messages := root.Get("messages"); messages.Exists() && messages.IsArray() {
		messages.ForEach(func(_, message gjson.Result) bool {
			role := strings.ToLower(strings.TrimSpace(message.Get("role").String()))
			if role == "" {
				role = "user"
			}
			if role == "system" {
				if text := chatMessageText(message.Get("content")); text != "" {
					systemInstructions = append(systemInstructions, text)
				}
				return true
			}
			if role != "user" && role != "assistant" && role != "developer" && role != "tool" {
				role = "user"
			}
			if role == "tool" {
				if item := chatToolMessageToResponses(message); len(item) > 0 {
					inputItems = append(inputItems, item)
				}
				return true
			}

			messageItem, hasContent := chatMessageToResponses(message, role)
			if hasContent {
				inputItems = append(inputItems, messageItem)
			}
			if role == "assistant" {
				message.Get("tool_calls").ForEach(func(_, call gjson.Result) bool {
					if item := chatToolCallToResponses(call); len(item) > 0 {
						inputItems = append(inputItems, item)
					}
					return true
				})
			}
			return true
		})
	}
	if len(inputItems) > 0 {
		out = translatorcommon.SetRawArrayItems(out, "input", inputItems)
	}
	if len(systemInstructions) > 0 {
		out, _ = sjson.SetBytes(out, "instructions", strings.Join(systemInstructions, "\n\n"))
	}

	// A few clients send a Responses-shaped payload to the Chat endpoint. Keep
	// that input if no messages were present instead of replacing it with an
	// empty request.
	if len(inputItems) == 0 {
		if input := root.Get("input"); input.Exists() {
			out, _ = sjson.SetRawBytes(out, "input", []byte(input.Raw))
		}
	}

	out = copyChatResponsesScalarFields(out, root)
	out = convertChatToolsToResponses(out, root.Get("tools"))
	out = convertChatToolChoiceToResponses(out, root.Get("tool_choice"))
	if format := root.Get("response_format"); format.Exists() {
		if converted := chatResponseFormatToResponses(format); len(converted) > 0 {
			out, _ = sjson.SetRawBytes(out, "text.format", converted)
		}
	}
	return out
}

// NormalizeOpenAIResponsesRequest normalizes image URL shapes accepted by
// OpenAI-compatible clients while preserving the rest of a native Responses
// request. The public Responses schema uses a string image_url, but several
// clients send {"url":"..."} instead.
func NormalizeOpenAIResponsesRequest(inputRawJSON []byte) []byte {
	if !gjson.ValidBytes(inputRawJSON) {
		return inputRawJSON
	}
	root := gjson.ParseBytes(inputRawJSON)
	if !root.Exists() {
		return inputRawJSON
	}
	input := root.Get("input")
	if !input.IsArray() {
		return inputRawJSON
	}
	updated := inputRawJSON
	changed := false
	input.ForEach(func(inputIndex, item gjson.Result) bool {
		// Some clients put an image directly in an input item instead of
		// wrapping it in a message/content array. Normalize that shape too.
		if normalizeResponsesImagePart(&updated, fmt.Sprintf("input.%d", inputIndex.Int()), item) {
			changed = true
		}
		content := item.Get("content")
		if content.IsArray() {
			content.ForEach(func(contentIndex, part gjson.Result) bool {
				if normalizeResponsesImagePart(&updated, fmt.Sprintf("input.%d.content.%d", inputIndex.Int(), contentIndex.Int()), part) {
					changed = true
				}
				return true
			})
		} else if content.IsObject() {
			if normalizeResponsesImagePart(&updated, fmt.Sprintf("input.%d.content", inputIndex.Int()), content) {
				changed = true
			}
		}
		return true
	})
	// Empty message items are commonly produced by compatibility clients when
	// they replay an assistant turn without content. Most Responses gateways
	// reject the entire request for that item, so remove only the explicitly
	// empty, known input shapes while preserving unknown future item types.
	if normalized, removed := removeEmptyResponsesInputItems(updated); removed {
		updated = normalized
		changed = true
	}
	if !changed {
		return inputRawJSON
	}
	return updated
}

// HasUsableOpenAIResponsesInput reports whether a Responses request contains
// content that an upstream can process. It deliberately treats identifiers and
// empty message shells as metadata, not as input text.
func HasUsableOpenAIResponsesInput(inputRawJSON []byte) bool {
	if !gjson.ValidBytes(inputRawJSON) {
		return false
	}
	root := gjson.ParseBytes(inputRawJSON)
	if responsesValueHasText(root.Get("instructions")) || responsesValueHasText(root.Get("previous_response_id")) {
		return true
	}
	return responsesInputValueHasContent(root.Get("input"))
}

func responsesInputValueHasContent(value gjson.Result) bool {
	if !value.Exists() || value.Type == gjson.Null {
		return false
	}
	if value.Type == gjson.String {
		return strings.TrimSpace(value.String()) != ""
	}
	if value.IsArray() {
		for _, item := range value.Array() {
			if responsesInputItemHasContent(item) {
				return true
			}
		}
		return false
	}
	if value.IsObject() {
		return responsesInputItemHasContent(value)
	}
	return false
}

func responsesInputItemHasContent(item gjson.Result) bool {
	if !item.Exists() || item.Type == gjson.Null {
		return false
	}
	if item.Type == gjson.String {
		return strings.TrimSpace(item.String()) != ""
	}
	if item.IsArray() {
		for _, child := range item.Array() {
			if responsesInputItemHasContent(child) {
				return true
			}
		}
		return false
	}
	if !item.IsObject() {
		return false
	}

	typ := strings.ToLower(strings.TrimSpace(item.Get("type").String()))
	switch typ {
	case "message":
		return responsesContentValueHasContent(item.Get("content")) ||
			responsesValueHasText(item.Get("text")) ||
			responsesValueHasText(item.Get("reasoning_content")) ||
			responsesInputValueHasContent(item.Get("tool_calls"))
	case "input_text", "output_text", "text":
		return responsesValueHasText(item.Get("text"))
	case "input_image", "image_url", "image":
		return responsesImageURL(item) != ""
	case "input_file":
		return responsesValueHasText(item.Get("file_data")) ||
			responsesValueHasText(item.Get("file_id")) ||
			responsesValueHasText(item.Get("file_url"))
	case "function_call":
		return responsesValueHasText(item.Get("name")) ||
			responsesValueHasText(item.Get("call_id")) ||
			responsesValueHasContent(item.Get("arguments"))
	case "custom_tool_call":
		return responsesValueHasText(item.Get("name")) ||
			responsesValueHasText(item.Get("call_id")) ||
			responsesValueHasContent(item.Get("input"))
	case "function_call_output", "custom_tool_call_output":
		// A tool result may intentionally be an empty string; its call_id still
		// makes it a meaningful input item.
		return responsesValueHasText(item.Get("call_id")) || responsesValueHasContent(item.Get("output"))
	case "reasoning":
		return responsesValueHasText(item.Get("encrypted_content")) ||
			responsesContentValueHasContent(item.Get("summary")) ||
			responsesContentValueHasContent(item.Get("content"))
	case "compaction_trigger":
		return true
	}

	// For an untyped object, inspect content-bearing fields. Ignore common
	// bookkeeping keys so an object containing only id/type/role is not accepted.
	for key, value := range item.Map() {
		switch strings.ToLower(strings.TrimSpace(key)) {
		case "type", "id", "role", "status", "sequence_number", "output_index", "content_index", "item_id":
			continue
		}
		if responsesValueHasContent(value) {
			return true
		}
	}
	return false
}

func responsesContentValueHasContent(value gjson.Result) bool {
	if !value.Exists() || value.Type == gjson.Null {
		return false
	}
	if value.Type == gjson.String {
		return strings.TrimSpace(value.String()) != ""
	}
	if value.IsArray() {
		for _, part := range value.Array() {
			if responsesContentPartHasContent(part) {
				return true
			}
		}
		return false
	}
	if value.IsObject() {
		return responsesContentPartHasContent(value)
	}
	return false
}

func responsesContentPartHasContent(part gjson.Result) bool {
	if !part.Exists() || part.Type == gjson.Null {
		return false
	}
	if part.Type == gjson.String {
		return strings.TrimSpace(part.String()) != ""
	}
	if !part.IsObject() {
		return false
	}
	typ := strings.ToLower(strings.TrimSpace(part.Get("type").String()))
	if typ == "" {
		switch {
		case part.Get("text").Type == gjson.String:
			typ = "text"
		case part.Get("refusal").Type == gjson.String:
			typ = "refusal"
		case part.Get("image_url").Exists() || part.Get("url").Exists():
			typ = "image_url"
		case part.Get("file_data").Exists() || part.Get("file_id").Exists() || part.Get("file_url").Exists():
			typ = "input_file"
		}
	}
	switch typ {
	case "input_text", "output_text", "text", "summary_text", "reasoning_text":
		return responsesValueHasText(part.Get("text"))
	case "refusal":
		return responsesValueHasText(part.Get("refusal")) || responsesValueHasText(part.Get("text"))
	case "input_image", "image_url", "image":
		return responsesImageURL(part) != ""
	case "input_file":
		return responsesValueHasText(part.Get("file_data")) ||
			responsesValueHasText(part.Get("file_id")) ||
			responsesValueHasText(part.Get("file_url"))
	case "input_audio", "audio":
		return responsesValueHasText(part.Get("audio_url")) || responsesValueHasText(part.Get("data"))
	default:
		return false
	}
}

func responsesValueHasText(value gjson.Result) bool {
	return value.Exists() && value.Type == gjson.String && strings.TrimSpace(value.String()) != ""
}

func responsesValueHasContent(value gjson.Result) bool {
	if !value.Exists() || value.Type == gjson.Null {
		return false
	}
	if value.Type == gjson.String {
		return strings.TrimSpace(value.String()) != ""
	}
	if value.IsArray() {
		for _, child := range value.Array() {
			if responsesValueHasContent(child) {
				return true
			}
		}
		return false
	}
	if value.IsObject() {
		if typ := strings.TrimSpace(value.Get("type").String()); typ != "" {
			return responsesInputItemHasContent(value)
		}
		for key, child := range value.Map() {
			switch strings.ToLower(strings.TrimSpace(key)) {
			case "type", "id", "role", "status":
				continue
			}
			if responsesValueHasContent(child) {
				return true
			}
		}
	}
	return false
}

func removeEmptyResponsesInputItems(payload []byte) ([]byte, bool) {
	root := gjson.ParseBytes(payload)
	input := root.Get("input")
	if !input.IsArray() {
		return payload, false
	}
	kept := make([][]byte, 0, len(input.Array()))
	removed := false
	for _, item := range input.Array() {
		if responsesInputItemIsRemovableEmpty(item) {
			removed = true
			continue
		}
		kept = append(kept, []byte(item.Raw))
	}
	if !removed {
		return payload, false
	}
	updated, errSet := sjson.SetRawBytes(payload, "input", translatorcommon.JoinRawArray(kept))
	if errSet != nil {
		return payload, false
	}
	return updated, true
}

func responsesInputItemIsRemovableEmpty(item gjson.Result) bool {
	if !item.Exists() || item.Type == gjson.Null {
		return true
	}
	if item.Type == gjson.String {
		return strings.TrimSpace(item.String()) == ""
	}
	if item.IsArray() {
		return len(item.Array()) == 0
	}
	if !item.IsObject() {
		return false
	}
	typ := strings.ToLower(strings.TrimSpace(item.Get("type").String()))
	switch typ {
	case "message", "input_text", "output_text", "text", "input_image", "image_url", "image", "input_file", "reasoning":
		return !responsesInputItemHasContent(item)
	default:
		return false
	}
}

func normalizeResponsesImagePart(payload *[]byte, path string, part gjson.Result) bool {
	typ := strings.ToLower(strings.TrimSpace(part.Get("type").String()))
	if typ != "input_image" && typ != "image_url" && typ != "image" {
		return false
	}
	url := responsesImageURL(part)
	if url == "" {
		return false
	}
	changed := false
	if imageURL := part.Get("image_url"); imageURL.Type == gjson.String && strings.TrimSpace(imageURL.String()) != "" {
		// The canonical Responses shape already has a string image_url. Remove
		// any legacy top-level url so strict upstream validators do not reject it.
		if part.Get("url").Exists() {
			if updated, errSet := sjson.DeleteBytes(*payload, path+".url"); errSet == nil {
				*payload = updated
				changed = true
			}
		}
		return changed
	}
	updated, errSet := sjson.SetBytes(*payload, path+".image_url", url)
	if errSet != nil {
		return false
	}
	*payload = updated
	changed = true
	if part.Get("detail").Type != gjson.String && part.Get("image_url.detail").Type == gjson.String {
		if updated, errSet = sjson.SetBytes(*payload, path+".detail", part.Get("image_url.detail").String()); errSet == nil {
			*payload = updated
		}
	}
	if part.Get("url").Exists() {
		if updated, errSet = sjson.DeleteBytes(*payload, path+".url"); errSet == nil {
			*payload = updated
		}
	}
	return changed
}

func chatMessageToResponses(message gjson.Result, role string) ([]byte, bool) {
	parts := make([][]byte, 0, 2)
	content := message.Get("content")
	if content.Type == gjson.String {
		// Preserve an explicitly empty user/assistant message as a real input
		// item; omitting it can make image/tool-only turns look empty upstream.
		if content.Exists() || role == "user" {
			parts = append(parts, chatTextPartToResponses(content.String(), role))
		}
	} else if content.IsArray() {
		content.ForEach(func(_, part gjson.Result) bool {
			if converted := chatContentPartToResponses(part, role); len(converted) > 0 {
				parts = append(parts, converted)
			}
			return true
		})
	} else if content.IsObject() {
		// Some OpenAI-compatible clients send a single content part as an object
		// instead of wrapping it in an array.
		if converted := chatContentPartToResponses(content, role); len(converted) > 0 {
			parts = append(parts, converted)
		}
	}
	if len(parts) == 0 {
		return nil, false
	}
	item := []byte(`{"type":"message","role":"","content":[]}`)
	item, _ = sjson.SetBytes(item, "role", role)
	item = translatorcommon.SetRawArrayItems(item, "content", parts)
	return item, true
}

func chatMessageText(content gjson.Result) string {
	if content.Type == gjson.String {
		return strings.TrimSpace(content.String())
	}
	if !content.IsArray() {
		return ""
	}
	var texts []string
	content.ForEach(func(_, part gjson.Result) bool {
		if text := part.Get("text"); text.Type == gjson.String && text.String() != "" {
			texts = append(texts, text.String())
		}
		return true
	})
	return strings.Join(texts, "\n")
}

func chatTextPartToResponses(text, role string) []byte {
	partType := "input_text"
	if role == "assistant" {
		partType = "output_text"
	}
	part := []byte(`{"type":"","text":""}`)
	part, _ = sjson.SetBytes(part, "type", partType)
	part, _ = sjson.SetBytes(part, "text", text)
	return part
}

func chatContentPartToResponses(part gjson.Result, role string) []byte {
	typ := strings.ToLower(strings.TrimSpace(part.Get("type").String()))
	if typ == "" && part.Get("text").Exists() {
		typ = "text"
	}
	switch typ {
	case "text", "input_text", "output_text":
		return chatTextPartToResponses(part.Get("text").String(), role)
	case "image_url", "input_image", "image":
		imageURL := responsesImageURL(part)
		if imageURL == "" && typ == "image" {
			imageURL = chatClaudeImageURL(part)
		}
		if imageURL == "" {
			return nil
		}
		image := []byte(`{"type":"input_image","image_url":""}`)
		image, _ = sjson.SetBytes(image, "image_url", imageURL)
		if detail := part.Get("detail"); detail.Type == gjson.String {
			image, _ = sjson.SetBytes(image, "detail", detail.String())
		}
		return image
	default:
		return nil
	}
}

func chatClaudeImageURL(part gjson.Result) string {
	source := part.Get("source")
	if source.Exists() {
		if source.Get("type").String() == "base64" {
			media := source.Get("media_type").String()
			if media == "" {
				media = "application/octet-stream"
			}
			if data := source.Get("data").String(); data != "" {
				return "data:" + media + ";base64," + data
			}
		}
		if url := source.Get("url").String(); url != "" {
			return url
		}
	}
	return strings.TrimSpace(part.Get("url").String())
}

func chatToolMessageToResponses(message gjson.Result) []byte {
	callID := strings.TrimSpace(message.Get("tool_call_id").String())
	if callID == "" {
		return nil
	}
	item := []byte(`{"type":"function_call_output","call_id":"","output":""}`)
	item, _ = sjson.SetBytes(item, "call_id", callID)
	if output := message.Get("content"); output.Exists() {
		if output.Type == gjson.String {
			item, _ = sjson.SetBytes(item, "output", output.String())
		} else {
			parts := make([][]byte, 0, 2)
			if output.IsArray() {
				output.ForEach(func(_, part gjson.Result) bool {
					if converted := chatContentPartToResponses(part, "user"); len(converted) > 0 {
						parts = append(parts, converted)
					}
					return true
				})
			}
			if len(parts) > 0 {
				item, _ = sjson.SetRawBytes(item, "output", translatorcommon.JoinRawArray(parts))
			}
		}
	}
	return item
}

func chatToolCallToResponses(call gjson.Result) []byte {
	name := strings.TrimSpace(call.Get("function.name").String())
	if name == "" {
		name = strings.TrimSpace(call.Get("name").String())
	}
	if name == "" {
		return nil
	}
	callID := strings.TrimSpace(call.Get("id").String())
	if callID == "" {
		callID = strings.TrimSpace(call.Get("call_id").String())
	}
	item := []byte(`{"type":"function_call","call_id":"","name":"","arguments":""}`)
	item, _ = sjson.SetBytes(item, "call_id", callID)
	item, _ = sjson.SetBytes(item, "name", name)
	args := call.Get("function.arguments")
	if !args.Exists() {
		args = call.Get("arguments")
	}
	if args.Type == gjson.String {
		item, _ = sjson.SetBytes(item, "arguments", args.String())
	} else if args.Exists() {
		item, _ = sjson.SetBytes(item, "arguments", args.Raw)
	}
	return item
}

func copyChatResponsesScalarFields(out []byte, root gjson.Result) []byte {
	for _, mapping := range []struct{ from, to string }{
		{from: "temperature", to: "temperature"},
		{from: "top_p", to: "top_p"},
		{from: "parallel_tool_calls", to: "parallel_tool_calls"},
		{from: "reasoning_effort", to: "reasoning.effort"},
		{from: "service_tier", to: "service_tier"},
		{from: "user", to: "user"},
		{from: "metadata", to: "metadata"},
		{from: "store", to: "store"},
		{from: "previous_response_id", to: "previous_response_id"},
		{from: "modalities", to: "modalities"},
	} {
		from, to := mapping.from, mapping.to
		value := root.Get(from)
		if value.Exists() {
			out, _ = sjson.SetRawBytes(out, to, []byte(value.Raw))
		}
	}
	// Chat clients may send both spellings; max_tokens has historically taken
	// precedence in this proxy, so resolve it deterministically.
	if maxTokens := root.Get("max_tokens"); maxTokens.Exists() {
		out, _ = sjson.SetRawBytes(out, "max_output_tokens", []byte(maxTokens.Raw))
	} else if maxTokens := root.Get("max_completion_tokens"); maxTokens.Exists() {
		out, _ = sjson.SetRawBytes(out, "max_output_tokens", []byte(maxTokens.Raw))
	}
	return out
}

func convertChatToolsToResponses(out []byte, tools gjson.Result) []byte {
	if !tools.Exists() || !tools.IsArray() {
		return out
	}
	converted := make([][]byte, 0, len(tools.Array()))
	tools.ForEach(func(_, tool gjson.Result) bool {
		typ := tool.Get("type").String()
		if typ == "" {
			typ = "function"
		}
		if typ == "function" {
			fn := tool.Get("function")
			name := strings.TrimSpace(fn.Get("name").String())
			if name == "" {
				name = strings.TrimSpace(tool.Get("name").String())
			}
			if name == "" {
				return true
			}
			item := []byte(`{"type":"function","name":""}`)
			item, _ = sjson.SetBytes(item, "name", name)
			for _, field := range []string{"description", "parameters", "strict"} {
				value := fn.Get(field)
				if !value.Exists() {
					value = tool.Get(field)
				}
				if value.Exists() {
					item, _ = sjson.SetRawBytes(item, field, []byte(value.Raw))
				}
			}
			converted = append(converted, item)
		} else if typ == "custom" {
			item := []byte(tool.Raw)
			if gjson.ValidBytes(item) {
				converted = append(converted, item)
			}
		}
		return true
	})
	if len(converted) > 0 {
		out = translatorcommon.SetRawArrayItems(out, "tools", converted)
	}
	return out
}

func convertChatToolChoiceToResponses(out []byte, choice gjson.Result) []byte {
	if !choice.Exists() {
		return out
	}
	if choice.Type == gjson.String {
		out, _ = sjson.SetBytes(out, "tool_choice", choice.String())
		return out
	}
	if choice.IsObject() {
		name := choice.Get("function.name").String()
		if name == "" {
			name = choice.Get("name").String()
		}
		if name != "" {
			mapped := []byte(`{"type":"function","name":""}`)
			mapped, _ = sjson.SetBytes(mapped, "name", name)
			out, _ = sjson.SetRawBytes(out, "tool_choice", mapped)
			return out
		}
	}
	out, _ = sjson.SetRawBytes(out, "tool_choice", []byte(choice.Raw))
	return out
}

func chatResponseFormatToResponses(format gjson.Result) []byte {
	typ := format.Get("type").String()
	if typ == "" {
		return nil
	}
	if typ == "json_schema" {
		format = format.Get("json_schema")
		out := []byte(`{"type":"json_schema"}`)
		for _, field := range []string{"name", "description", "strict", "schema"} {
			if value := format.Get(field); value.Exists() {
				if field == "schema" {
					out, _ = sjson.SetRawBytes(out, field, []byte(value.Raw))
				} else {
					out, _ = sjson.SetRawBytes(out, field, []byte(value.Raw))
				}
			}
		}
		return out
	}
	return []byte(format.Raw)
}
