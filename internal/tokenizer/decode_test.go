package tokenizer

import (
	"os"
	"testing"
)

func TestDecode203(t *testing.T) {
	data, err := os.ReadFile("testdata/tokenizer.json")
	if err != nil {
		t.Fatalf("Failed to read tokenizer: %v", err)
	}

	tok, err := NewFromJSON(data)
	if err != nil {
		t.Fatalf("Failed to create tokenizer: %v", err)
	}

	// Test decode of token 203
	decoded := tok.Decode([]int{203})
	t.Logf("Decode [203]: %q", decoded)
	if decoded != "\n" {
		t.Errorf("Expected newline, got %q", decoded)
	}

	// Test with multiple 203s
	decoded = tok.Decode([]int{203, 203, 65})
	t.Logf("Decode [203,203,65]: %q", decoded)
	if decoded != "\n\nO" {
		t.Errorf("Expected newline newline O, got %q", decoded)
	}
}
