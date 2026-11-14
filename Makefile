.PHONY: all clean test

GO_SRCS := $(wildcard cmd/crom-infer/*.go internal/engine/*.go internal/tokenizer/*.go)

# Build the C static library
engine/libcrom_infer.a: engine/crom_infer.c engine/crom_infer.h
	gcc -O3 -march=native -fopenmp -Wall -Wextra -c engine/crom_infer.c -o engine/crom_infer.o
	ar rcs engine/libcrom_infer.a engine/crom_infer.o

# Build the C shared library (para ctypes em Python)
engine/libcrom_infer.so: engine/crom_infer.c engine/crom_infer.h
	gcc -shared -fPIC -O3 -march=native -fopenmp -Wall -Wextra engine/crom_infer.c -o engine/libcrom_infer.so -lm -lpthread

# Build the C CLI for validation
engine/crom_cli: engine/crom_cli.c engine/crom_infer.c engine/crom_infer.h
	gcc -O3 -march=native -fopenmp -Wall -Wextra engine/crom_cli.c engine/crom_infer.c -o engine/crom_cli -lm -lpthread

# Build the Go CLI
bin/crom-infer: engine/libcrom_infer.a $(GO_SRCS)
	go build -o bin/crom-infer ./cmd/crom-infer

all: bin/crom-infer engine/libcrom_infer.so engine/crom_cli

# Run tokenizer validation
test-tokenizer:
	python3 tools/check_tokenizer.py

# Run the official duel
duel: bin/crom-infer
	python3 tools/duelo_ollama.py

clean:
	rm -f engine/crom_infer.o engine/libcrom_infer.a engine/libcrom_infer.so engine/crom_cli
	rm -rf bin/

test: all test-tokenizer
