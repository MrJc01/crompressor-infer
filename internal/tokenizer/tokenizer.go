package tokenizer

import (
	"encoding/json"
	"sort"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"
)

// Tokenizer implements the Granite ByteLevel BPE tokenizer.
// It reproduces HuggingFace's GPT2TokenizerFast behaviour:
//   - added/special tokens are matched on the raw text before BPE
//   - ByteLevel pre-tokenization uses the GPT-2 regex semantics
//   - BPE merges use the model merge ranks over the byte->unicode alphabet
type Tokenizer struct {
	vocab         map[string]int
	idToToken     map[int]string
	merges        map[string]int // key: left + "\x00" + right -> rank
	addedList     []addedToken   // sorted by content length desc
	specialTokens map[int]string
	byteToUni     [256]rune
	uniToByte     map[rune]byte
	eosID         int
}

type addedToken struct {
	content string
	id      int
}

// TokenizerJSON represents the structure of tokenizer.json
type TokenizerJSON struct {
	Model struct {
		Type   string         `json:"type"`
		Vocab  map[string]int `json:"vocab"`
		Merges [][2]string    `json:"merges"`
	} `json:"model"`
	AddedTokens []struct {
		ID      int    `json:"id"`
		Content string `json:"content"`
		Special bool   `json:"special"`
	} `json:"added_tokens"`
}

// bytesToUnicode builds the standard GPT-2 byte <-> unicode tables.
func bytesToUnicode() ([256]rune, map[rune]byte) {
	var bs []int
	for b := int('!'); b <= int('~'); b++ {
		bs = append(bs, b)
	}
	for b := int('¡'); b <= int('¬'); b++ {
		bs = append(bs, b)
	}
	for b := int('®'); b <= int('ÿ'); b++ {
		bs = append(bs, b)
	}
	cs := append([]int{}, bs...)
	n := 0
	seen := make(map[int]bool, 256)
	for _, b := range bs {
		seen[b] = true
	}
	for b := 0; b < 256; b++ {
		if !seen[b] {
			bs = append(bs, b)
			cs = append(cs, 256+n)
			n++
		}
	}
	var b2u [256]rune
	u2b := make(map[rune]byte, 256)
	for i, b := range bs {
		b2u[b] = rune(cs[i])
		u2b[rune(cs[i])] = byte(b)
	}
	return b2u, u2b
}

// NewFromJSON creates a Tokenizer from tokenizer.json bytes
func NewFromJSON(data []byte) (*Tokenizer, error) {
	var tj TokenizerJSON
	if err := json.Unmarshal(data, &tj); err != nil {
		return nil, err
	}

	b2u, u2b := bytesToUnicode()

	t := &Tokenizer{
		vocab:         make(map[string]int, len(tj.Model.Vocab)),
		idToToken:     make(map[int]string, len(tj.Model.Vocab)),
		merges:        make(map[string]int, len(tj.Model.Merges)),
		specialTokens: make(map[int]string),
		byteToUni:     b2u,
		uniToByte:     u2b,
		eosID:         0,
	}

	for token, id := range tj.Model.Vocab {
		t.vocab[token] = id
		t.idToToken[id] = token
	}

	for i, merge := range tj.Model.Merges {
		t.merges[merge[0]+"\x00"+merge[1]] = i
	}

	for _, at := range tj.AddedTokens {
		t.addedList = append(t.addedList, addedToken{content: at.Content, id: at.ID})
		if at.Special {
			t.specialTokens[at.ID] = at.Content
		}
		// Granite: <|end_of_text|> (id 0) is the EOS token.
		if at.Content == "<|end_of_text|>" {
			t.eosID = at.ID
		}
	}

	// Longest-content-first so overlapping added tokens match greedily.
	sort.SliceStable(t.addedList, func(i, j int) bool {
		return len(t.addedList[i].content) > len(t.addedList[j].content)
	})

	return t, nil
}

// EOSID returns the end-of-text token id.
func (t *Tokenizer) EOSID() int { return t.eosID }

// Encode encodes text to token IDs (no chat template).
func (t *Tokenizer) Encode(text string) []int {
	ids := make([]int, 0, len(text))
	var seg strings.Builder

	i := 0
	for i < len(text) {
		if id, n, ok := t.matchAddedAt(text[i:]); ok {
			if seg.Len() > 0 {
				ids = append(ids, t.encodeSegment(seg.String())...)
				seg.Reset()
			}
			ids = append(ids, id)
			i += n
			continue
		}
		r, size := utf8.DecodeRuneInString(text[i:])
		seg.WriteRune(r)
		i += size
	}
	if seg.Len() > 0 {
		ids = append(ids, t.encodeSegment(seg.String())...)
	}
	return ids
}

func (t *Tokenizer) matchAddedAt(s string) (id int, n int, ok bool) {
	for _, at := range t.addedList {
		if strings.HasPrefix(s, at.content) {
			return at.id, len(at.content), true
		}
	}
	return 0, 0, false
}

func (t *Tokenizer) encodeSegment(seg string) []int {
	var ids []int
	for _, chunk := range t.preTokenize(seg) {
		syms := make([]string, 0, len(chunk))
		for _, b := range []byte(chunk) {
			syms = append(syms, string(t.byteToUni[b]))
		}
		ids = append(ids, t.bpe(syms)...)
	}
	return ids
}

// preTokenize reproduces the GPT-2 ByteLevel regex:
//
//	's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
func (t *Tokenizer) preTokenize(text string) []string {
	rs := []rune(text)
	n := len(rs)
	isSpace := unicode.IsSpace
	isLetter := unicode.IsLetter
	isNumber := unicode.IsNumber
	out := make([]string, 0, n)

	for i := 0; i < n; {
		// contractions
		if rs[i] == '\'' && i+1 < n {
			if rs[i+1] == 's' || rs[i+1] == 't' || rs[i+1] == 'm' || rs[i+1] == 'd' {
				out = append(out, string(rs[i:i+2]))
				i += 2
				continue
			}
			if i+2 < n {
				two := string(rs[i+1 : i+3])
				if two == "re" || two == "ve" || two == "ll" {
					out = append(out, string(rs[i:i+3]))
					i += 3
					continue
				}
			}
		}

		// " ?\p{L}+"
		if rs[i] == ' ' && i+1 < n && isLetter(rs[i+1]) {
			j := i + 1
			for j < n && isLetter(rs[j]) {
				j++
			}
			out = append(out, string(rs[i:j]))
			i = j
			continue
		}
		if rs[i] != ' ' && isLetter(rs[i]) {
			j := i
			for j < n && isLetter(rs[j]) {
				j++
			}
			out = append(out, string(rs[i:j]))
			i = j
			continue
		}

		// " ?\p{N}+"
		if rs[i] == ' ' && i+1 < n && isNumber(rs[i+1]) {
			j := i + 1
			for j < n && isNumber(rs[j]) {
				j++
			}
			out = append(out, string(rs[i:j]))
			i = j
			continue
		}
		if rs[i] != ' ' && isNumber(rs[i]) {
			j := i
			for j < n && isNumber(rs[j]) {
				j++
			}
			out = append(out, string(rs[i:j]))
			i = j
			continue
		}

		// " ?[^\s\p{L}\p{N}]+"
		isPunct := func(r rune) bool {
			return !isSpace(r) && !isLetter(r) && !isNumber(r)
		}
		if rs[i] == ' ' && i+1 < n && isPunct(rs[i+1]) {
			j := i + 1
			for j < n && isPunct(rs[j]) {
				j++
			}
			out = append(out, string(rs[i:j]))
			i = j
			continue
		}
		if rs[i] != ' ' && isPunct(rs[i]) {
			j := i
			for j < n && isPunct(rs[j]) {
				j++
			}
			out = append(out, string(rs[i:j]))
			i = j
			continue
		}

		// whitespace: "\s+(?!\S)" then "\s+"
		if isSpace(rs[i]) {
			j := i
			for j < n && isSpace(rs[j]) {
				j++
			}
			run := j - i
			if j == n {
				out = append(out, string(rs[i:j]))
				i = j
				continue
			}
			// followed by a non-space: (?!\S) leaves the last space behind
			if run > 1 {
				out = append(out, string(rs[i:i+run-1]))
				i = i + run - 1
				continue
			}
			out = append(out, string(rs[i:i+1]))
			i++
			continue
		}

		// fallback (should not happen for valid UTF-8)
		out = append(out, string(rs[i:i+1]))
		i++
	}
	return out
}

// bpe applies merge ranks to the byte-level symbols of a pre-token.
func (t *Tokenizer) bpe(syms []string) []int {
	if len(syms) == 0 {
		return nil
	}
	for len(syms) > 1 {
		bestRank := -1
		bestPair := ""
		for i := 0; i+1 < len(syms); i++ {
			key := syms[i] + "\x00" + syms[i+1]
			if r, ok := t.merges[key]; ok {
				if bestRank == -1 || r < bestRank {
					bestRank = r
					bestPair = key
				}
			}
		}
		if bestRank == -1 {
			break
		}
		next := make([]string, 0, len(syms))
		for i := 0; i < len(syms); i++ {
			if i+1 < len(syms) && syms[i]+"\x00"+syms[i+1] == bestPair {
				next = append(next, syms[i]+syms[i+1])
				i++
			} else {
				next = append(next, syms[i])
			}
		}
		syms = next
	}

	ids := make([]int, len(syms))
	for i, s := range syms {
		if id, ok := t.vocab[s]; ok {
			ids[i] = id
		}
	}
	return ids
}

// Decode decodes token IDs back to text.
func (t *Tokenizer) Decode(ids []int) string {
	var out strings.Builder
	var buf []byte
	flush := func() {
		if len(buf) > 0 {
			out.Write(buf)
			buf = buf[:0]
		}
	}
	for _, id := range ids {
		if content, ok := t.specialTokens[id]; ok {
			flush()
			out.WriteString(content)
			continue
		}
		token, ok := t.idToToken[id]
		if !ok {
			continue
		}
		for _, r := range token {
			if b, ok := t.uniToByte[r]; ok {
				buf = append(buf, b)
			} else {
				buf = append(buf, []byte(string(r))...)
			}
		}
	}
	flush()
	return out.String()
}

// ApplyChatTemplate applies the Granite chat template and tokenizes it.
func (t *Tokenizer) ApplyChatTemplate(messages []Message, addGenerationPrompt bool) []int {
	var b strings.Builder

	var system string
	loop := messages
	if len(messages) > 0 && messages[0].Role == "system" {
		system = messages[0].Content
		loop = messages[1:]
	} else {
		system = "Knowledge Cutoff Date: April 2024.\nToday's Date: " +
			time.Now().Format("January 02, 2006") +
			".\nYou are Granite, developed by IBM. You are a helpful AI assistant."
	}

	b.WriteString("<|start_of_role|>system<|end_of_role|>")
	b.WriteString(system)
	b.WriteString("<|end_of_text|>\n")

	for i, msg := range loop {
		b.WriteString("<|start_of_role|>")
		b.WriteString(msg.Role)
		b.WriteString("<|end_of_role|>")
		b.WriteString(msg.Content)
		b.WriteString("<|end_of_text|>\n")
		if i == len(loop)-1 && addGenerationPrompt {
			b.WriteString("<|start_of_role|>assistant<|end_of_role|>")
		}
	}

	return t.Encode(b.String())
}

type Message struct {
	Role    string
	Content string
}

func GetEOSID() int {
	return 0
}

func GetSpecialTokens() map[string]int {
	return map[string]int{
		"<|end_of_text|>":   0,
		"<fim_prefix>":      1,
		"<fim_middle>":      2,
		"<fim_suffix>":      3,
		"<fim_pad>":         4,
		"<filename>":        5,
		"<gh_stars>":        6,
		"<issue_start>":     7,
		"<issue_comment>":   8,
		"<issue_closed>":    9,
		"<jupyter_start>":   10,
		"<jupyter_text>":    11,
		"<jupyter_code>":    12,
		"<jupyter_output>":  13,
		"<empty_output>":    14,
		"<commit_before>":   15,
		"<commit_msg>":      16,
		"<commit_after>":    17,
		"<reponame>":        18,
		"<|start_of_role|>": 49152,
		"<|end_of_role|>":   49153,
		"<|tool_call|>":     49154,
	}
}
