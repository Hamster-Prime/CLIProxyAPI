package responses

import (
	"context"
	"testing"

	"github.com/tidwall/gjson"
)

func TestConvertOpenAIChatCompletionsRequestToOpenAIResponsesPreservesInputAndTools(t *testing.T) {
	raw := []byte(`{
		"model":"client-model",
		"messages":[
			{"role":"system","content":"follow the policy"},
			{"role":"user","content":[{"type":"text","text":"look"},{"type":"image_url","image_url":{"url":"https://example.com/a.png"}}]},
			{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"x\"}"}}]},
			{"role":"tool","tool_call_id":"call_1","content":"done"}
		],
		"max_tokens":128,
		"tools":[{"type":"function","function":{"name":"lookup","description":"find","parameters":{"type":"object"}}}],
		"tool_choice":{"type":"function","function":{"name":"lookup"}},
		"response_format":{"type":"json_object"}
	}`)

	out := gjson.ParseBytes(ConvertOpenAIChatCompletionsRequestToOpenAIResponses("resolved-model", raw, false))
	if got := out.Get("model").String(); got != "resolved-model" {
		t.Fatalf("model = %q, want resolved-model", got)
	}
	if got := out.Get("instructions").String(); got != "follow the policy" {
		t.Fatalf("instructions = %q", got)
	}
	if got := out.Get("input.#").Int(); got != 3 {
		t.Fatalf("input count = %d, want 3; output=%s", got, out.Raw)
	}
	if got := out.Get("input.0.content.1.type").String(); got != "input_image" {
		t.Fatalf("image type = %q", got)
	}
	if got := out.Get("input.0.content.1.image_url").String(); got != "https://example.com/a.png" {
		t.Fatalf("image URL = %q", got)
	}
	if got := out.Get("input.1.type").String(); got != "function_call" {
		t.Fatalf("tool call type = %q", got)
	}
	if got := out.Get("input.2.type").String(); got != "function_call_output" {
		t.Fatalf("tool result type = %q", got)
	}
	if got := out.Get("tools.0.name").String(); got != "lookup" {
		t.Fatalf("tool name = %q", got)
	}
	if got := out.Get("tool_choice.name").String(); got != "lookup" {
		t.Fatalf("tool choice = %q", got)
	}
	if got := out.Get("text.format.type").String(); got != "json_object" {
		t.Fatalf("text format = %q", got)
	}
}

func TestConvertOpenAIChatCompletionsRequestToOpenAIResponsesAcceptsObjectContent(t *testing.T) {
	raw := []byte(`{"model":"model","messages":[{"role":"user","content":{"type":"image_url","image_url":{"url":"https://example.com/image.png"}}}]}`)
	out := gjson.ParseBytes(ConvertOpenAIChatCompletionsRequestToOpenAIResponses("model", raw, false))
	if got := out.Get("input.#").Int(); got != 1 {
		t.Fatalf("input count = %d, want 1; output=%s", got, out.Raw)
	}
	if got := out.Get("input.0.content.0.type").String(); got != "input_image" {
		t.Fatalf("content type = %q, want input_image", got)
	}
	if got := out.Get("input.0.content.0.image_url").String(); got != "https://example.com/image.png" {
		t.Fatalf("image URL = %q", got)
	}
}

func TestNormalizeOpenAIResponsesRequestAcceptsObjectImageURL(t *testing.T) {
	raw := []byte(`{"model":"model","input":[{"type":"message","role":"user","content":[{"type":"input_image","image_url":{"url":"https://example.com/image.png","detail":"high"}}]}]}`)
	out := gjson.ParseBytes(NormalizeOpenAIResponsesRequest(raw))
	if got := out.Get("input.0.content.0.image_url").String(); got != "https://example.com/image.png" {
		t.Fatalf("image URL = %q", got)
	}
	if got := out.Get("input.0.content.0.detail").String(); got != "high" {
		t.Fatalf("detail = %q, want high", got)
	}
}

func TestNormalizeOpenAIResponsesRequestAcceptsDirectImageItem(t *testing.T) {
	raw := []byte(`{"model":"model","input":[{"type":"input_image","image_url":{"url":"https://example.com/direct.png","detail":"low"}}]}`)
	out := gjson.ParseBytes(NormalizeOpenAIResponsesRequest(raw))
	if got := out.Get("input.0.image_url").String(); got != "https://example.com/direct.png" {
		t.Fatalf("image URL = %q", got)
	}
	if got := out.Get("input.0.detail").String(); got != "low" {
		t.Fatalf("detail = %q, want low", got)
	}
}

func TestNormalizeOpenAIResponsesRequestUsesFallbackURLWhenImageURLIsEmpty(t *testing.T) {
	raw := []byte(`{"model":"model","input":[{"type":"input_image","image_url":"","url":"https://example.com/fallback.png"}]}`)
	out := gjson.ParseBytes(NormalizeOpenAIResponsesRequest(raw))
	if got := out.Get("input.0.image_url").String(); got != "https://example.com/fallback.png" {
		t.Fatalf("image URL = %q, want fallback URL; output=%s", got, out.Raw)
	}
}

func TestNormalizeOpenAIResponsesRequestRemovesLegacyTopLevelURL(t *testing.T) {
	raw := []byte(`{"model":"model","input":[{"type":"input_image","image_url":"https://example.com/image.png","url":"https://example.com/legacy.png"}]}`)
	out := gjson.ParseBytes(NormalizeOpenAIResponsesRequest(raw))
	if got := out.Get("input.0.image_url").String(); got != "https://example.com/image.png" {
		t.Fatalf("image URL = %q", got)
	}
	if out.Get("input.0.url").Exists() {
		t.Fatalf("legacy top-level url was not removed: %s", out.Raw)
	}
}

func TestNormalizeOpenAIResponsesRequestDropsEmptyInputItems(t *testing.T) {
	raw := []byte(`{"model":"model","input":[{"type":"message","role":"user","content":[]},{"type":"input_text","text":""},{"type":"message","role":"user","content":"hello"}]}`)
	out := gjson.ParseBytes(NormalizeOpenAIResponsesRequest(raw))
	if got := out.Get("input.#").Int(); got != 1 {
		t.Fatalf("input count = %d, want 1; output=%s", got, out.Raw)
	}
	if got := out.Get("input.0.content").String(); got != "hello" {
		t.Fatalf("remaining input = %q, want hello", got)
	}
}

func TestHasUsableOpenAIResponsesInputChecksSemanticContent(t *testing.T) {
	tests := []struct {
		name string
		raw  string
		want bool
	}{
		{name: "empty message", raw: `{"input":[{"type":"message","role":"user","content":[]}]}`, want: false},
		{name: "empty text part", raw: `{"input":[{"type":"input_text","text":""}]}`, want: false},
		{name: "image object", raw: `{"input":[{"type":"input_image","image_url":{"url":"https://example.com/image.png"}}]}`, want: true},
		{name: "tool call", raw: `{"input":[{"type":"function_call","name":"lookup","arguments":"{}"}]}`, want: true},
		{name: "continuation", raw: `{"previous_response_id":"resp_1","input":[]}`, want: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := HasUsableOpenAIResponsesInput([]byte(tt.raw)); got != tt.want {
				t.Fatalf("HasUsableOpenAIResponsesInput() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestConvertOpenAIResponsesResponseToOpenAIChatCompletionsStream(t *testing.T) {
	var param any
	events := [][]byte{
		[]byte(`data: {"type":"response.created","response":{"id":"resp_1","model":"model","created_at":1700000000}}`),
		[]byte(`data: {"type":"response.output_text.delta","delta":"hello"}`),
		[]byte(`data: {"type":"response.completed","response":{"id":"resp_1","model":"model","created_at":1700000000,"status":"completed","usage":{"input_tokens":2,"output_tokens":1,"total_tokens":3}}}`),
	}
	var chunks [][]byte
	for _, event := range events {
		chunks = append(chunks, ConvertOpenAIResponsesResponseToOpenAIChatCompletions(context.Background(), "model", nil, nil, event, &param)...)
	}
	if len(chunks) != 3 {
		t.Fatalf("chunks = %d, want 3", len(chunks))
	}
	if got := gjson.GetBytes(chunks[1], "choices.0.delta.content").String(); got != "hello" {
		t.Fatalf("content delta = %q", got)
	}
	if got := gjson.GetBytes(chunks[2], "choices.0.finish_reason").String(); got != "stop" {
		t.Fatalf("finish reason = %q", got)
	}
	if got := gjson.GetBytes(chunks[2], "usage.prompt_tokens").Int(); got != 2 {
		t.Fatalf("prompt tokens = %d", got)
	}
}

func TestConvertOpenAIResponsesResponseToOpenAIChatCompletionsDoneOnlyEvents(t *testing.T) {
	var param any
	events := [][]byte{
		[]byte("event: response.created\ndata: {\"response\":{\"id\":\"resp_done\",\"model\":\"model\",\"created_at\":1700000000}}"),
		[]byte("event: response.output_text.done\ndata: {\"item_id\":\"msg_1\",\"text\":\"final answer\"}"),
		[]byte("event: response.output_item.done\ndata: {\"output_index\":0,\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"final answer\"}]}}"),
		[]byte("event: response.done\ndata: {\"response\":{\"id\":\"resp_done\",\"model\":\"model\",\"created_at\":1700000000,\"status\":\"completed\",\"output\":[{\"id\":\"msg_1\",\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"final answer\"}]}],\"usage\":{\"input_tokens\":2,\"output_tokens\":2}}}"),
	}
	var chunks [][]byte
	for _, event := range events {
		chunks = append(chunks, ConvertOpenAIResponsesResponseToOpenAIChatCompletions(context.Background(), "model", nil, nil, event, &param)...)
	}
	if len(chunks) < 3 {
		t.Fatalf("chunks = %d, want text and terminal chunks; output=%q", len(chunks), chunks)
	}
	var content string
	for _, chunk := range chunks {
		content += gjson.GetBytes(chunk, "choices.0.delta.content").String()
	}
	if content != "final answer" {
		t.Fatalf("content = %q, want final answer", content)
	}
	if got := gjson.GetBytes(chunks[len(chunks)-1], "choices.0.finish_reason").String(); got != "stop" {
		t.Fatalf("finish reason = %q", got)
	}
}

func TestConvertOpenAIResponsesResponseToOpenAIChatCompletionsReasoningAndToolDone(t *testing.T) {
	var param any
	events := [][]byte{
		[]byte(`data: {"type":"response.created","response":{"id":"resp_tool","model":"model","created_at":1700000000}}`),
		[]byte(`data: {"type":"response.output_item.added","item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"lookup"}}`),
		[]byte(`data: {"type":"response.function_call_arguments.done","item_id":"fc_1","arguments":"{\"q\":1}"}`),
		[]byte(`data: {"type":"response.reasoning_summary_text.delta","item_id":"reason_1","delta":"thinking"}`),
		[]byte(`data: {"type":"response.completed","response":{"id":"resp_tool","model":"model","created_at":1700000000,"output":[{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":"{\"q\":1}"}],"usage":{"input_tokens":1,"output_tokens":1}}}`),
	}
	var chunks [][]byte
	for _, event := range events {
		chunks = append(chunks, ConvertOpenAIResponsesResponseToOpenAIChatCompletions(context.Background(), "model", nil, nil, event, &param)...)
	}
	if len(chunks) < 4 {
		t.Fatalf("chunks = %d, want role/tool/reasoning/terminal; output=%q", len(chunks), chunks)
	}
	foundArgs := false
	foundReasoning := false
	for _, chunk := range chunks {
		if gjson.GetBytes(chunk, "choices.0.delta.tool_calls.0.function.arguments").String() == `{"q":1}` {
			foundArgs = true
		}
		if gjson.GetBytes(chunk, "choices.0.delta.reasoning_content").String() == "thinking" {
			foundReasoning = true
		}
	}
	if !foundArgs || !foundReasoning {
		t.Fatalf("missing tool args or reasoning: %q", chunks)
	}
}

func TestConvertOpenAIResponsesResponseToOpenAIChatCompletionsNonStream(t *testing.T) {
	raw := []byte(`{"id":"resp_2","object":"response","created_at":1700000001,"model":"model","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"answer"}]},{"type":"function_call","call_id":"call_2","name":"lookup","arguments":"{\"q\":1}"}],"usage":{"input_tokens":4,"output_tokens":2,"total_tokens":6}}`)
	out := ConvertOpenAIResponsesResponseToOpenAIChatCompletionsNonStream(context.Background(), "model", nil, nil, raw, nil)
	root := gjson.ParseBytes(out)
	if got := root.Get("choices.0.message.content").String(); got != "answer" {
		t.Fatalf("content = %q", got)
	}
	if got := root.Get("choices.0.message.tool_calls.0.function.name").String(); got != "lookup" {
		t.Fatalf("tool name = %q", got)
	}
	if got := root.Get("choices.0.finish_reason").String(); got != "tool_calls" {
		t.Fatalf("finish reason = %q", got)
	}
}

func TestConvertOpenAIResponsesResponseToOpenAIChatCompletionsNonStreamAcceptsObjectAndStringContent(t *testing.T) {
	tests := []struct {
		name    string
		content string
		want    string
	}{
		{name: "string", content: `"plain answer"`, want: "plain answer"},
		{name: "object", content: `{"type":"output_text","text":"object answer"}`, want: "object answer"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			raw := []byte(`{"id":"resp_content","model":"model","output":[{"type":"message","content":` + tt.content + `}]}`)
			out := gjson.ParseBytes(ConvertOpenAIResponsesResponseToOpenAIChatCompletionsNonStream(context.Background(), "model", nil, nil, raw, nil))
			if got := out.Get("choices.0.message.content").String(); got != tt.want {
				t.Fatalf("content = %q, want %q; output=%s", got, tt.want, out.Raw)
			}
		})
	}
}

func TestConvertOpenAIResponsesResponseToOpenAIChatCompletionsStreamDoneEvents(t *testing.T) {
	t.Run("output text done fills missing delta and deduplicates complete item", func(t *testing.T) {
		var param any
		send := func(event string) [][]byte {
			return ConvertOpenAIResponsesResponseToOpenAIChatCompletions(
				context.Background(), "model", nil, nil, []byte("data: "+event), &param,
			)
		}

		if out := send(`{"type":"response.output_item.added","output_index":0,"item":{"id":"msg_1","type":"message","role":"assistant","content":[]}}`); len(out) != 0 {
			t.Fatalf("empty output item added emitted %d chunks", len(out))
		}
		if out := send(`{"type":"response.output_text.done","item_id":"msg_1","output_index":0,"text":"complete text"}`); len(out) != 1 {
			t.Fatalf("output_text.done chunks = %d, want 1", len(out))
		} else if got := gjson.GetBytes(out[0], "choices.0.delta.content").String(); got != "complete text" {
			t.Fatalf("output_text.done content = %q", got)
		}
		// A complete item carrying the same text must not repeat the text.
		if out := send(`{"type":"response.output_item.done","output_index":0,"item":{"id":"msg_1","type":"message","content":[{"type":"output_text","text":"complete text"}]}}`); len(out) != 0 {
			t.Fatalf("duplicate output_item.done emitted %d chunks", len(out))
		}
	})

	t.Run("function call arguments done fills missing delta", func(t *testing.T) {
		var param any
		send := func(event string) [][]byte {
			return ConvertOpenAIResponsesResponseToOpenAIChatCompletions(
				context.Background(), "model", nil, nil, []byte("data: "+event), &param,
			)
		}

		if out := send(`{"type":"response.output_item.added","output_index":0,"item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"lookup","arguments":""}}`); len(out) != 1 {
			t.Fatalf("function output item added chunks = %d, want 1", len(out))
		}
		if out := send(`{"type":"response.function_call_arguments.done","item_id":"fc_1","output_index":0,"arguments":"{\"q\":1}"}`); len(out) != 1 {
			t.Fatalf("function arguments done chunks = %d, want 1", len(out))
		} else if got := gjson.GetBytes(out[0], "choices.0.delta.tool_calls.0.function.arguments").String(); got != `{"q":1}` {
			t.Fatalf("function arguments = %q", got)
		}
		if out := send(`{"type":"response.output_item.done","output_index":0,"item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"lookup","arguments":"{\"q\":1}"}}`); len(out) != 0 {
			t.Fatalf("duplicate function output item done emitted %d chunks", len(out))
		}
	})

	t.Run("reasoning summary done emits only the unseen suffix", func(t *testing.T) {
		var param any
		send := func(event string) [][]byte {
			return ConvertOpenAIResponsesResponseToOpenAIChatCompletions(
				context.Background(), "model", nil, nil, []byte("data: "+event), &param,
			)
		}

		if out := send(`{"type":"response.reasoning_summary_text.delta","item_id":"rs_1","output_index":0,"delta":"think"}`); len(out) != 1 {
			t.Fatalf("reasoning delta chunks = %d, want 1", len(out))
		}
		if out := send(`{"type":"response.reasoning_summary_text.done","item_id":"rs_1","output_index":0,"text":"think harder"}`); len(out) != 1 {
			t.Fatalf("reasoning done chunks = %d, want 1", len(out))
		} else if got := gjson.GetBytes(out[0], "choices.0.delta.reasoning_content").String(); got != " harder" {
			t.Fatalf("reasoning suffix = %q, want %q", got, " harder")
		}
	})
}

func TestConvertOpenAIResponsesResponseToOpenAIChatCompletionsStreamResponseDone(t *testing.T) {
	var param any
	out := ConvertOpenAIResponsesResponseToOpenAIChatCompletions(
		context.Background(), "model", nil, nil,
		[]byte("event: response.done\ndata: {\"response\":{\"id\":\"resp_done\",\"model\":\"model\",\"created_at\":1700000000,\"status\":\"completed\",\"usage\":{\"input_tokens\":3,\"output_tokens\":2,\"total_tokens\":5},\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"answer\"}]}]}}"),
		&param,
	)
	if len(out) != 2 {
		t.Fatalf("response.done chunks = %d, want text and terminal chunks", len(out))
	}
	if got := gjson.GetBytes(out[0], "choices.0.delta.content").String(); got != "answer" {
		t.Fatalf("response.done text = %q", got)
	}
	terminal := gjson.ParseBytes(out[len(out)-1])
	if got := terminal.Get("choices.0.finish_reason").String(); got != "stop" {
		t.Fatalf("response.done finish reason = %q", got)
	}
	if got := terminal.Get("usage.total_tokens").Int(); got != 5 {
		t.Fatalf("response.done total tokens = %d", got)
	}

	// The newer event can also carry the response fields at the top level.
	param = nil
	out = ConvertOpenAIResponsesResponseToOpenAIChatCompletions(
		context.Background(), "model", nil, nil,
		[]byte(`{"type":"response.done","id":"resp_top","model":"model","created_at":1700000001,"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"usage":{"input_tokens":1,"output_tokens":1}}`),
		&param,
	)
	if len(out) != 1 {
		t.Fatalf("top-level response.done chunks = %d, want 1", len(out))
	}
	if got := gjson.GetBytes(out[0], "choices.0.finish_reason").String(); got != "length" {
		t.Fatalf("top-level response.done finish reason = %q, want length", got)
	}
}
