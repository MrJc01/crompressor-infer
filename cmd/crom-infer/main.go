package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"runtime"
	"runtime/debug"
	"strings"
	"time"

	"github.com/MrJc01/crompressor-infer/internal/engine"
	"github.com/MrJc01/crompressor-infer/internal/tokenizer"
)

type Config struct {
	ModelPath     string
	Prompt        string
	Chat          bool
	MaxTokens     int
	Threads       int
	CacheExperts  int
	Ctx           int
	RepeatPenalty float64
	RepeatWindow  int
	Stream        bool
	JSONOutput    bool
	TokenizeOnly  bool
}

func main() {
	cfg := Config{}
	flag.StringVar(&cfg.ModelPath, "model", "artifacts/granite-1b.cromllm", "Model path")
	flag.StringVar(&cfg.Prompt, "prompt", "", "Prompt text")
	flag.BoolVar(&cfg.Chat, "chat", false, "Apply Granite chat template")
	flag.IntVar(&cfg.MaxTokens, "max-tokens", 128, "Max tokens to generate")
	flag.IntVar(&cfg.Threads, "threads", 6, "Number of threads")
	flag.IntVar(&cfg.CacheExperts, "cache-experts", 768, "Number of experts to cache in RAM")
	flag.IntVar(&cfg.Ctx, "ctx", 2048, "Context window size")
	flag.Float64Var(&cfg.RepeatPenalty, "repeat-penalty", 1.10, "Repetition penalty factor (1.0 = disabled)")
	flag.IntVar(&cfg.RepeatWindow, "repeat-window", 64, "Repetition penalty window size")
	flag.BoolVar(&cfg.Stream, "stream", false, "Stream output in real-time")
	flag.BoolVar(&cfg.JSONOutput, "json", false, "Output JSON with metrics")
	flag.BoolVar(&cfg.TokenizeOnly, "tokenize-only", false, "Only tokenize the prompt and print JSON ids")
	flag.Parse()

	if cfg.Prompt == "" {
		fmt.Fprintln(os.Stderr, "error: --prompt is required")
		os.Exit(1)
	}

	if cfg.TokenizeOnly {
		runTokenizeOnly(cfg)
		return
	}

	runtime.GOMAXPROCS(cfg.Threads)
	// LockOSThread keeps every crom_forward call on the same OS thread.
	// libgomp builds its worker team per master OS thread; if the goroutine
	// migrates between OS threads, each forward pass rebuilds the team (or
	// degrades to serial) and throughput collapses ~100x.
	runtime.LockOSThread()
	// The Go heap here is tiny and bounded (~a few hundred KB for the token
	// slices); GC workers would otherwise steal cores from the OpenMP team
	// mid-inference and cut tokens/s by 3-5x on unlucky runs.
	debug.SetGCPercent(-1)

	// Load model
	startLoad := time.Now()
	model, err := engine.Open(engine.Config{
		ModelPath:    cfg.ModelPath,
		CacheExperts: cfg.CacheExperts,
		Threads:      cfg.Threads,
		MaxSeq:       cfg.Ctx,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load model: %v\n", err)
		os.Exit(1)
	}
	defer model.Close()
	loadMs := time.Since(startLoad).Seconds() * 1000

	// Get tokenizer
	tokData, err := model.GetTokenizerData()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to get tokenizer: %v\n", err)
		os.Exit(1)
	}

	tok, err := tokenizer.NewFromJSON(tokData)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create tokenizer: %v\n", err)
		os.Exit(1)
	}

	// Prepare input
	var inputIDs []int
	if cfg.Chat {
		inputIDs = tok.ApplyChatTemplate(
			[]tokenizer.Message{{Role: "user", Content: cfg.Prompt}},
			true,
		)
	} else {
		inputIDs = tok.Encode(cfg.Prompt)
	}

	// Generate
	vocabSize := model.VocabSize()
	hiddenSize := model.HiddenSize()
	maxSeq := model.MaxSeq()

	hidden := make([]float32, hiddenSize)
	logits := make([]float32, vocabSize)
	generated := make([]int, 0, cfg.MaxTokens)
	promptLen := len(inputIDs)

	// Prefill: run each prompt token once through the KV cache.
	prefillStart := time.Now()
	for i := 0; i < len(inputIDs)-1; i++ {
		model.Forward(inputIDs, i, hidden, logits)
	}
	prefillMs := time.Since(prefillStart).Seconds() * 1000

	// Generation: one forward per new token, reusing the KV cache (no reset,
	// no re-sending of the whole history into the model).
	decodeStart := time.Now()
	pos := len(inputIDs) - 1
	for step := 0; step < cfg.MaxTokens && pos < maxSeq-1; step++ {
		model.Forward(inputIDs, pos, hidden, logits)

		// Apply repetition penalty
		if cfg.RepeatPenalty > 0 && cfg.RepeatPenalty != 1.0 && cfg.RepeatWindow > 0 {
			start := len(inputIDs) - cfg.RepeatWindow
			if start < 0 {
				start = 0
			}
			window := inputIDs[start:]
			seen := make(map[int]bool, len(window))
			penalty := float32(cfg.RepeatPenalty)
			for _, id := range window {
				if id < 0 || id >= vocabSize || seen[id] {
					continue
				}
				seen[id] = true
				if logits[id] > 0 {
					logits[id] /= penalty
				} else {
					logits[id] *= penalty
				}
			}
		}

		// Sample next token (greedy)
		nextID := 0
		maxLogit := float32(-1e30)
		for i := 0; i < vocabSize; i++ {
			if logits[i] > maxLogit {
				maxLogit = logits[i]
				nextID = i
			}
		}

		generated = append(generated, nextID)
		inputIDs = append(inputIDs, nextID)
		pos++

		// Check for EOS
		if nextID == tok.EOSID() {
			break
		}

		// Stream output
		if cfg.Stream {
			fmt.Print(tok.Decode([]int{nextID}))
		}
	}
	decodeMs := time.Since(decodeStart).Seconds() * 1000

	// Get stats
	stats := model.GetStats()
	total := stats.Hits + stats.Misses
	hitPct := 100.0
	if total > 0 {
		hitPct = 100.0 * float64(stats.Hits) / float64(total)
	}

	prefillTokS := 0.0
	if prefillMs > 0 && promptLen-1 > 0 {
		prefillTokS = float64(promptLen-1) / (prefillMs / 1000.0)
	}
	decodeTokS := 0.0
	if decodeMs > 0 && len(generated) > 0 {
		decodeTokS = float64(len(generated)) / (decodeMs / 1000.0)
	}

	if cfg.JSONOutput {
		out := map[string]interface{}{
			"text":          tok.Decode(generated),
			"ids":           generated,
			"load_ms":       loadMs,
			"prefill_tok_s": prefillTokS,
			"decode_tok_s":  decodeTokS,
			"rss_mb":        float64(getRSSKB()) / 1024.0,
			"cache_hit_pct": hitPct,
			"ssd_stream_mb": float64(stats.StreamBytes) / (1024.0 * 1024.0),
			"attn_ms":       stats.AttnMs,
			"moe_ms":        stats.MoeMs,
			"lm_head_ms":    stats.LMHeadMs,
		}
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		enc.Encode(out)
		return
	}

	fmt.Printf("%s\n", tok.Decode(generated))
	fmt.Printf("load_ms=%.1f prefill_tok/s=%.2f decode_tok/s=%.2f RSS_MB=%.1f cache_hit%%=%.2f ssd_stream_MB=%.2f attn_ms=%.1f moe_ms=%.1f lm_head_ms=%.1f\n",
		loadMs, prefillTokS, decodeTokS, float64(getRSSKB())/1024.0, hitPct,
		float64(stats.StreamBytes)/(1024.0*1024.0),
		stats.AttnMs, stats.MoeMs, stats.LMHeadMs)
}

func runTokenizeOnly(cfg Config) {
	tokData, err := engine.ReadTokenizerData(cfg.ModelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to read tokenizer: %v\n", err)
		os.Exit(1)
	}
	tok, err := tokenizer.NewFromJSON(tokData)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create tokenizer: %v\n", err)
		os.Exit(1)
	}

	var ids []int
	if cfg.Chat {
		ids = tok.ApplyChatTemplate([]tokenizer.Message{{Role: "user", Content: cfg.Prompt}}, true)
	} else {
		ids = tok.Encode(cfg.Prompt)
	}
	if ids == nil {
		ids = []int{}
	}
	enc := json.NewEncoder(os.Stdout)
	enc.Encode(map[string]interface{}{"ids": ids})
}

func getRSSKB() int64 {
	data, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return 0
	}
	lines := strings.Split(string(data), "\n")
	for _, line := range lines {
		if strings.HasPrefix(line, "VmRSS:") {
			var rss int64
			fmt.Sscanf(line, "VmRSS: %d kB", &rss)
			return rss
		}
	}
	return 0
}
