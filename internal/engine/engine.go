package engine

/*
#cgo CFLAGS: -I${SRCDIR}/../../engine
#cgo LDFLAGS: -L${SRCDIR}/../../engine -lcrom_infer -lm -fopenmp

#include "crom_infer.h"
#include <stdlib.h>
*/
import "C"
import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"runtime"
	"unsafe"
)

// Model wraps the C crom_model_t
type Model struct {
	model C.crom_model_t
	closed bool
}

// Config holds model configuration
type Config struct {
	ModelPath     string
	CacheExperts  int
	Threads       int
}

// Open loads the model
func Open(config Config) (*Model, error) {
	if config.Threads <= 0 {
		config.Threads = 6
	}
	if config.CacheExperts <= 0 {
		config.CacheExperts = 768
	}

	cPath := C.CString(config.ModelPath)
	defer C.free(unsafe.Pointer(cPath))

	var model C.crom_model_t
	ret := C.crom_model_open(cPath, &model, C.int(config.CacheExperts), C.int(config.Threads))
	if ret != 0 {
		return nil, errors.New("failed to open model")
	}

	m := &Model{model: model}
	runtime.SetFinalizer(m, (*Model).Close)
	return m, nil
}

// Close releases resources
func (m *Model) Close() error {
	if m.closed {
		return nil
	}
	C.crom_model_close(&m.model)
	m.closed = true
	return nil
}

// Forward runs one forward pass
func (m *Model) Forward(inputIDs []int, position int, hidden, outLogits []float32) error {
	n := len(inputIDs)
	cInputIDs := make([]C.int, n)
	for i, id := range inputIDs {
		cInputIDs[i] = C.int(id)
	}
	cHidden := (*C.float)(unsafe.Pointer(&hidden[0]))
	cOutLogits := (*C.float)(unsafe.Pointer(&outLogits[0]))

	ret := C.crom_forward(&m.model, &cInputIDs[0], C.int(n), C.int(position), cHidden, cOutLogits)
	if ret != 0 {
		return errors.New("forward failed")
	}
	return nil
}

// Embed looks up token embedding
func (m *Model) Embed(tokenID int, hidden []float32) error {
	cHidden := (*C.float)(unsafe.Pointer(&hidden[0]))
	ret := C.crom_embed(&m.model, C.int(tokenID), cHidden)
	if ret != 0 {
		return errors.New("embed failed")
	}
	return nil
}

// ResetKV zeros the KV cache
func (m *Model) ResetKV() {
	C.crom_reset_kv(&m.model)
}

// GetTokenizerData returns the embedded tokenizer.json
func (m *Model) GetTokenizerData() ([]byte, error) {
    var size C.size_t
    ptr := C.crom_tokenizer_data(&m.model, (*C.size_t)(unsafe.Pointer(&size)))
    if ptr == nil {
        return nil, errors.New("tokenizer not found")
    }
    // Don't free - C side caches the buffer
    return C.GoBytes(ptr, C.int(size)), nil
}

// EOSID returns the EOS token ID
func EOSID() int {
	return int(C.crom_eos_id())
}

// Stats holds model statistics
type Stats struct {
	Hits         int64
	Misses       int64
	Bytes        int64
	StreamBytes  int64
	IOMs         float64
	AttnMs       float64
	MoeMs        float64
	LMHeadMs     float64
}

// GetStats returns model statistics
func (m *Model) GetStats() Stats {
	var hits, misses, bytes, streamBytes C.longlong
	var ioMs, attnMs, moeMs, lmHeadMs C.double
	C.crom_stats(&m.model,
		(*C.longlong)(unsafe.Pointer(&hits)),
		(*C.longlong)(unsafe.Pointer(&misses)),
		(*C.longlong)(unsafe.Pointer(&bytes)),
		(*C.longlong)(unsafe.Pointer(&streamBytes)),
		(*C.double)(unsafe.Pointer(&ioMs)),
		(*C.double)(unsafe.Pointer(&attnMs)),
		(*C.double)(unsafe.Pointer(&moeMs)),
		(*C.double)(unsafe.Pointer(&lmHeadMs)),
	)
	return Stats{
		Hits:         int64(hits),
		Misses:       int64(misses),
		Bytes:        int64(bytes),
		StreamBytes:  int64(streamBytes),
		IOMs:         float64(ioMs),
		AttnMs:       float64(attnMs),
		MoeMs:        float64(moeMs),
		LMHeadMs:     float64(lmHeadMs),
	}
}

// ResetKV zeros the KV cache
func (m *Model) ResetKVCache() {
	C.crom_reset_kv(&m.model)
}

// ReadTokenizerData reads the embedded tokenizer.json straight from the
// .cromllm header without loading any model weights. The tokenizer blob lives
// at offset tokenizer_offset (header byte 128) with tokenizer_size (byte 136).
func ReadTokenizerData(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	hdr := make([]byte, 144)
	if _, err := io.ReadFull(f, hdr); err != nil {
		return nil, err
	}
	if string(hdr[:8]) != "CROMLM01" {
		return nil, fmt.Errorf("bad magic: %q", string(hdr[:8]))
	}
	off := binary.LittleEndian.Uint64(hdr[128:136])
	size := binary.LittleEndian.Uint64(hdr[136:144])
	if size == 0 {
		return nil, errors.New("tokenizer not present")
	}

	buf := make([]byte, size)
	if _, err := f.ReadAt(buf, int64(off)); err != nil {
		return nil, err
	}
	return buf, nil
}
