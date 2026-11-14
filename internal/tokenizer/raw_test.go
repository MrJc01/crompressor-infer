package tokenizer

import (
	"os"
	"testing"
)

func TestRawPrompt(t *testing.T) {
	data, err := os.ReadFile("testdata/tokenizer.json")
	if err != nil {
		t.Fatalf("Failed to read tokenizer: %v", err)
	}

	tok, err := NewFromJSON(data)
	if err != nil {
		t.Fatalf("Failed to create tokenizer: %v", err)
	}

	rawPrompt := "Explain a hash table in one sentence:"
	ids := tok.Encode(rawPrompt)
	expected := []int{37394, 312, 3677, 1858, 328, 1591, 15430, 44}
	
	if len(ids) != len(expected) {
		t.Errorf("Encode length mismatch: got %d, expected %d", len(ids), len(expected))
	} else {
		match := true
		for i := range ids {
			if ids[i] != expected[i] {
				t.Errorf("Encode mismatch at %d: got %d, expected %d", i, ids[i], expected[i])
				match = false
			}
		}
		if match {
			t.Logf("Raw prompt test PASSED: %v", ids)
		}
	}
	
	decoded := tok.Decode(ids)
	t.Logf("Decoded: %s", decoded)
}
