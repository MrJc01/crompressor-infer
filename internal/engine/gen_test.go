package engine

import (
	"testing"
)

func TestGenerate(t *testing.T) {
	model, err := Open(Config{
		ModelPath:    "../../artifacts/granite-1b.cromllm",
		CacheExperts: 768,
		Threads:      6,
	})
	if err != nil {
		t.Fatalf("Failed to open model: %v", err)
	}
	defer model.Close()

	hidden := make([]float32, 1024)
	logits := make([]float32, 49155)
	inputIDs := []int{37394, 312, 3677, 1858, 328, 1591, 15430, 44}

	// Prefill
	for i := 0; i < len(inputIDs)-1; i++ {
		model.Forward(inputIDs[:i+1], i, hidden, logits)
	}

	// Generate a few tokens
	generated := make([]int, 0, 5)
	pos := len(inputIDs) - 1
	for step := 0; step < 5; step++ {
		model.Forward(inputIDs, pos, hidden, logits)
		
		nextID := 0
		maxLogit := float32(-1e30)
		for i := 0; i < 49155; i++ {
			if logits[i] > maxLogit {
				maxLogit = logits[i]
				nextID = i
			}
		}
		
		t.Logf("Step %d: nextID=%d, maxLogit=%.4f", step, nextID, maxLogit)
		
		if nextID == 0 { // EOS
			break
		}
		
		generated = append(generated, nextID)
		pos++
	}
	
	t.Logf("Generated: %v", generated)
}
