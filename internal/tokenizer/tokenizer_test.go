package tokenizer

import (
	"os"
	"testing"
)

func TestTokenizer(t *testing.T) {
	data, err := os.ReadFile("testdata/tokenizer.json")
	if err != nil {
		t.Fatalf("Failed to read tokenizer: %v", err)
	}

	tok, err := NewFromJSON(data)
	if err != nil {
		t.Fatalf("Failed to create tokenizer: %v", err)
	}

	// Test 1: Raw prompt
	rawPrompt := "Explain a hash table in one sentence:"
	ids := tok.Encode(rawPrompt)
	expected := []int{37394, 312, 3677, 1858, 328, 1591, 15430, 44}
	
	if len(ids) != len(expected) {
		t.Errorf("Encode length mismatch: got %d, expected %d", len(ids), len(expected))
	} else {
		match := true
		for i := range ids {
			if ids[i] != expected[i] {
				match = false
				break
			}
		}
		if !match {
			t.Errorf("Encode mismatch: got %v, expected %v", ids, expected)
		} else {
			t.Logf("Raw prompt test PASSED: %v", ids)
		}
	}

	// Test 2: Chat template
	messages := []Message{{Role: "user", Content: "Explain a hash table in one sentence:"}}
	chatIDs := tok.ApplyChatTemplate(messages, true)
	t.Logf("Chat template IDs (first 20): %v", chatIDs[:min(20, len(chatIDs))])
	
	// Test 3: PT-BR accents
	ptbrPrompt := "O que é uma função em Go?"
	ptbrIDs := tok.Encode(ptbrPrompt)
	t.Logf("PT-BR prompt IDs: %v", ptbrIDs)
	
	// Test 4: Decode
	decoded := tok.Decode(expected)
	t.Logf("Decoded: %s", decoded)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
