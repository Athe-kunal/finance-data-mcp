MODEL := allenai/olmOCR-2-7B-1025-FP8

GPU_MEMORY_UTILIZATION ?= 1.00
MAX_MODEL_LEN          ?= 16384
TENSOR_PARALLEL_SIZE   ?= 1
DATA_PARALLEL_SIZE     ?= 1
PORT                   ?= 8000
SERVER                 ?= localhost

.PHONY: vllm-olmocr-serve
vllm-olmocr-serve:
	uv run vllm serve $(MODEL) \
		--gpu-memory-utilization $(GPU_MEMORY_UTILIZATION) \
		--max-model-len $(MAX_MODEL_LEN) \
		--tensor-parallel-size $(TENSOR_PARALLEL_SIZE) \
		--data-parallel-size $(DATA_PARALLEL_SIZE) \
		--max-num-batched-tokens 32768 \
		--mm-encoder-tp-mode "data" \
		--max-num-seqs 8192 \
		--port $(PORT) \
		--host $(SERVER)

.PHONY: guidellm-benchmark
guidellm-benchmark:
	uv run guidellm benchmark \
		--target "http://localhost:$(PORT)" \
		--profile throughput \
		--max-seconds 300 \
		--rate 20 \
		--data "prompt_tokens=300,output_tokens=2048" \
		--output-path benchmark.yaml