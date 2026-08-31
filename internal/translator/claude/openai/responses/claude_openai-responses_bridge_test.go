package responses

import (
	"context"
	"strings"
	"testing"

	"github.com/tidwall/gjson"
)

func TestConvertClaudeRequestToOpenAIResponsesProducesInput(t *testing.T) {
	raw := []byte(`{"model":"claude-model","max_tokens":64,"messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}]}`)
	out := gjson.ParseBytes(ConvertClaudeRequestToOpenAIResponses("resolved-model", raw, false))
	if got := out.Get("model").String(); got != "resolved-model" {
		t.Fatalf("model = %q", got)
	}
	if got := out.Get("input.#").Int(); got != 1 {
		t.Fatalf("input count = %d; output=%s", got, out.Raw)
	}
	if got := out.Get("input.0.content.0.text").String(); got != "hello" {
		t.Fatalf("input text = %q", got)
	}
}

func TestConvertOpenAIResponsesRequestToClaudePreservesPrimitiveInput(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{name: "top-level string", input: `"hello"`, want: "hello"},
		{name: "array string", input: `["hello"]`, want: "hello"},
		{name: "message content string item", input: `[{"type":"message","role":"user","content":["hello"]}]`, want: "hello"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			raw := []byte(`{"model":"model","input":` + tt.input + `}`)
			out := gjson.ParseBytes(ConvertOpenAIResponsesRequestToClaude("model", raw, false))
			if got := out.Get("messages.0.content").String(); got != tt.want {
				t.Fatalf("content = %q, want %q; output=%s", got, tt.want, out.Raw)
			}
		})
	}
}

func TestConvertOpenAIResponsesResponseToClaudeEmitsMessageStop(t *testing.T) {
	var param any
	request := []byte(`{"model":"claude-model","stream":true,"messages":[{"role":"user","content":"hello"}]}`)
	events := [][]byte{
		[]byte(`data: {"type":"response.created","response":{"id":"resp_1","model":"model","created_at":1700000000}}`),
		[]byte(`data: {"type":"response.output_text.delta","delta":"answer"}`),
		[]byte(`data: {"type":"response.completed","response":{"id":"resp_1","model":"model","created_at":1700000000,"status":"completed","usage":{"input_tokens":2,"output_tokens":1,"total_tokens":3}}}`),
	}
	var output strings.Builder
	for _, event := range events {
		for _, chunk := range ConvertOpenAIResponsesResponseToClaude(context.Background(), "model", request, request, event, &param) {
			output.Write(chunk)
		}
	}
	body := output.String()
	if !strings.Contains(body, "message_start") {
		t.Fatalf("missing message_start: %s", body)
	}
	if !strings.Contains(body, "text_delta") || !strings.Contains(body, "answer") {
		t.Fatalf("missing text delta: %s", body)
	}
	if !strings.Contains(body, "message_stop") {
		t.Fatalf("missing message_stop: %s", body)
	}
}

func TestConvertOpenAIResponsesResponseToClaudeHandlesEventOnlyDone(t *testing.T) {
	var param any
	request := []byte(`{"model":"claude-model","stream":true,"messages":[{"role":"user","content":"hello"}]}`)
	var output strings.Builder
	for _, event := range [][]byte{
		[]byte("event: response.created\ndata: { \"response\": { \"id\": \"resp_2\", \"model\": \"model\", \"created_at\": 1700000000 } }\n\n"),
		[]byte("event: response.output_text.done\ndata: { \"item_id\": \"msg_2\", \"text\": \"answer\" }\n\n"),
		[]byte("event: response.done\ndata: { \"response\": { \"id\": \"resp_2\", \"model\": \"model\", \"created_at\": 1700000000, \"status\": \"completed\" } }\n\n"),
	} {
		for _, chunk := range ConvertOpenAIResponsesResponseToClaude(context.Background(), "model", request, request, event, &param) {
			output.Write(chunk)
		}
	}
	if !strings.Contains(output.String(), "answer") || !strings.Contains(output.String(), "message_stop") {
		t.Fatalf("event-only output missing answer or message_stop: %s", output.String())
	}
}

func TestConvertOpenAIResponsesResponseToClaudeNonStreamPropagatesFailure(t *testing.T) {
	raw := []byte(`{"type":"response.failed","response":{"error":{"type":"server_error","message":"upstream failed"}}}`)
	out := ConvertOpenAIResponsesResponseToClaudeNonStream(context.Background(), "model", nil, nil, raw, nil)
	root := gjson.ParseBytes(out)
	if got := root.Get("type").String(); got != "error" {
		t.Fatalf("error type = %q, want error; body=%s", got, out)
	}
	if got := root.Get("error.message").String(); got != "upstream failed" {
		t.Fatalf("error message = %q", got)
	}
}
