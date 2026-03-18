MODEL := allenai/olmOCR-2-7B-1025-FP8
EMBD_MODEL := Qwen/Qwen3-Embedding-0.6B

GPU_MEMORY_UTILIZATION ?= 0.98
EMBD_GPU_MEMORY_UTILIZATION ?= 0.5
MAX_MODEL_LEN          ?= 16384
TENSOR_PARALLEL_SIZE   ?= 1
DATA_PARALLEL_SIZE     ?= 1
PORT                   ?= 8000
EMBD_PORT			   ?= 8888
API_PORT               ?= 8081
SERVER                 ?= localhost
IMAGE_NAME             ?= sec-filings-md
GPU_DEVICE             ?= 3
SEC_API_ORGANIZATION   ?= Your-Organization
SEC_API_EMAIL          ?= your-email@example.com

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

.PHONY: vllm-embd-serve
vllm-embd-serve:
	uv run vllm serve $(EMBD_MODEL) \
		--gpu-memory-utilization $(EMBD_GPU_MEMORY_UTILIZATION) \
		--runner pooling \
		--max-model-len 4096 \
		--port $(EMBD_PORT) \
		--host $(SERVER)

.PHONY: start-server
start-server:
	uv run uvicorn server:app --host 0.0.0.0 --reload --port $(API_PORT)

.PHONY: guidellm-benchmark
guidellm-benchmark:
	uv run guidellm benchmark \
		--target "http://localhost:$(PORT)" \
		--profile throughput \
		--max-seconds 300 \
		--rate 20 \
		--data "prompt_tokens=300,output_tokens=2048" \
		--output-path benchmark.yaml

.PHONY: docker-build
docker-build:
	docker build -t $(IMAGE_NAME) .

.PHONY: docker-run
docker-run:
	docker run --gpus device=$(GPU_DEVICE) \
		-e SEC_API_ORGANIZATION="$(SEC_API_ORGANIZATION)" \
		-e SEC_API_EMAIL="$(SEC_API_EMAIL)" \
		-v ./sec_data:/app/sec_data \
		-v ./localworkspace:/app/localworkspace \
		-p $(API_PORT):8081 \
		$(IMAGE_NAME)

.PHONY: docker-start
docker-start: docker-build docker-run

.PHONY: docker-stop
docker-stop:
	docker ps -q --filter "publish=$(API_PORT)" | xargs -r docker stop

.PHONY: docker-remove
docker-remove:
	docker ps -aq --filter "publish=$(API_PORT)" | xargs -r docker rm