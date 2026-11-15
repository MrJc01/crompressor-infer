package main

import (
	"crypto/sha256"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
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
	DeltaPath     string
	Prompt        string
	Chat          bool
	MaxTokens     int
	Threads       int
	CacheExperts  int
	Ctx           int
	RepeatPenalty float64
	RepeatWindow  int
	Stream        bool
	SyncIO        bool
	JSONOutput    bool
	TokenizeOnly  bool
}

func main() {
	if len(os.Args) > 1 {
		if os.Args[1] == "serve-delta" {
			runServeDelta(os.Args[2:])
			return
		}
		if os.Args[1] == "pull-delta" {
			runPullDelta(os.Args[2:])
			return
		}
	}

	cfg := Config{}
	flag.StringVar(&cfg.ModelPath, "model", "artifacts/granite-1b.cromllm", "Model path")
	flag.StringVar(&cfg.DeltaPath, "delta", "", "Path to .cromdelta Multi-Fine-Tune patch")
	flag.StringVar(&cfg.Prompt, "prompt", "", "Prompt text")
	flag.BoolVar(&cfg.Chat, "chat", false, "Apply Granite chat template")
	flag.IntVar(&cfg.MaxTokens, "max-tokens", 128, "Max tokens to generate")
	flag.IntVar(&cfg.Threads, "threads", 6, "Number of threads")
	flag.IntVar(&cfg.CacheExperts, "cache-experts", 768, "Number of experts to cache in RAM")
	flag.IntVar(&cfg.Ctx, "ctx", 2048, "Context window size")
	flag.Float64Var(&cfg.RepeatPenalty, "repeat-penalty", 1.10, "Repetition penalty factor (1.0 = disabled)")
	flag.IntVar(&cfg.RepeatWindow, "repeat-window", 64, "Repetition penalty window size")
	flag.BoolVar(&cfg.Stream, "stream", false, "Stream output in real-time")
	flag.BoolVar(&cfg.SyncIO, "sync-io", false, "Force synchronous SSD I/O instead of async pthread prefetch")
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
		DeltaPath:    cfg.DeltaPath,
		CacheExperts: cfg.CacheExperts,
		Threads:      cfg.Threads,
		MaxSeq:       cfg.Ctx,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load model: %v\n", err)
		os.Exit(1)
	}
	defer model.Close()
	model.SetSyncIO(cfg.SyncIO)
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

	// Prefill: run prompt tokens through the model to build the KV cache.
	prefillStart := time.Now()
	if promptLen > 1 {
		// Batch prefill processes promptLen-1 tokens and computes logits only for the last prompt token
		if err := model.ForwardBatch(inputIDs[:promptLen-1], 0, nil); err != nil {
			fmt.Fprintf(os.Stderr, "ForwardBatch failed: %v\n", err)
			os.Exit(1)
		}
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

func runServeDelta(args []string) {
	fs := flag.NewFlagSet("serve-delta", flag.ExitOnError)
	filePath := fs.String("file", "artifacts/granite-1b-instruct.cromdelta", "Path to .cromdelta file to serve")
	addr := fs.String("addr", "127.0.0.1:18421", "HTTP address to bind")
	fs.Parse(args)

	data, err := os.ReadFile(*filePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading delta file %s: %v\n", *filePath, err)
		os.Exit(1)
	}

	hash := sha256.Sum256(data)
	shaHex := fmt.Sprintf("%x", hash)
	fileSize := len(data)

	http.HandleFunc("/meta", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"filename": *filePath,
			"size":     fileSize,
			"sha256":   shaHex,
		})
	})

	http.HandleFunc("/delta", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/octet-stream")
		w.Header().Set("X-SHA256", shaHex)
		w.Header().Set("Content-Length", fmt.Sprintf("%d", fileSize))
		w.WriteHeader(http.StatusOK)

		chunkSize := 1024 * 1024 // 1 MiB chunks
		for i := 0; i < fileSize; i += chunkSize {
			end := i + chunkSize
			if end > fileSize {
				end = fileSize
			}
			w.Write(data[i:end])
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
		}
	})

	fmt.Printf("Serving %s (%d bytes, SHA-256: %s) on http://%s\n", *filePath, fileSize, shaHex, *addr)
	if err := http.ListenAndServe(*addr, nil); err != nil {
		fmt.Fprintf(os.Stderr, "Server failed: %v\n", err)
		os.Exit(1)
	}
}

func runPullDelta(args []string) {
	fs := flag.NewFlagSet("pull-delta", flag.ExitOnError)
	urlStr := fs.String("url", "http://127.0.0.1:18421", "Base URL of server")
	outPath := fs.String("out", "/tmp/pulled.cromdelta", "Destination file path")
	fs.Parse(args)

	baseURL := strings.TrimRight(*urlStr, "/")
	respMeta, err := http.Get(baseURL + "/meta")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to fetch metadata from %s/meta: %v\n", baseURL, err)
		os.Exit(1)
	}
	defer respMeta.Body.Close()

	var meta struct {
		Filename string `json:"filename"`
		Size     int64  `json:"size"`
		SHA256   string `json:"sha256"`
	}
	if err := json.NewDecoder(respMeta.Body).Decode(&meta); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to decode metadata: %v\n", err)
		os.Exit(1)
	}

	start := time.Now()
	respDelta, err := http.Get(baseURL + "/delta")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to fetch delta from %s/delta: %v\n", baseURL, err)
		os.Exit(1)
	}
	defer respDelta.Body.Close()

	outFile, err := os.Create(*outPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create output file %s: %v\n", *outPath, err)
		os.Exit(1)
	}
	defer outFile.Close()

	hasher := sha256.New()
	multiWriter := io.MultiWriter(outFile, hasher)

	buf := make([]byte, 1024*1024) // 1 MiB chunk
	var totalCopied int64
	for {
		nr, rerr := respDelta.Body.Read(buf)
		if nr > 0 {
			nw, werr := multiWriter.Write(buf[:nr])
			if werr != nil {
				fmt.Fprintf(os.Stderr, "Write error: %v\n", werr)
				os.Exit(1)
			}
			totalCopied += int64(nw)
		}
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			fmt.Fprintf(os.Stderr, "Read error: %v\n", rerr)
			os.Exit(1)
		}
	}

	elapsed := time.Since(start)
	pulledHash := fmt.Sprintf("%x", hasher.Sum(nil))

	if pulledHash != meta.SHA256 {
		fmt.Fprintf(os.Stderr, "SHA256 mismatch! Expected %s, got %s\n", meta.SHA256, pulledHash)
		os.Exit(1)
	}

	mbSec := (float64(totalCopied) / (1024 * 1024)) / elapsed.Seconds()
	fmt.Printf("Pull completed successfully!\n")
	fmt.Printf("File: %s\n", *outPath)
	fmt.Printf("Size: %d bytes (%.2f MB)\n", totalCopied, float64(totalCopied)/(1024*1024))
	fmt.Printf("SHA-256: %s (VERIFIED MATCH)\n", pulledHash)
	fmt.Printf("Elapsed: %v (%.2f MB/s)\n", elapsed.Round(time.Millisecond), mbSec)
}
