MODEL := allenai/olmOCR-2-7B-1025-FP8
EMBD_MODEL := Qwen/Qwen3-Embedding-0.6B
RERANKER_MODEL := Qwen/Qwen3-Reranker-0.6B

GPU_MEMORY_UTILIZATION      ?= 0.5
EMBD_GPU_MEMORY_UTILIZATION ?= 0.1
RERANKER_GPU_MEMORY_UTILIZATION ?= 0.3
MAX_MODEL_LEN               ?= 8192
TENSOR_PARALLEL_SIZE        ?= 1
DATA_PARALLEL_SIZE          ?= 1
PORT                        ?= 8000
EMBD_PORT                   ?= 8002
RERANKER_PORT               ?= 8003
API_PORT                    ?= 8081
SERVER                      ?= localhost
IMAGE_NAME                  ?= sec-filings-md
GPU_DEVICE                  ?= 3
SEC_API_ORGANIZATION        ?= Your-Organization
SEC_API_EMAIL               ?= your-email@example.com

.PHONY: vllm-olmocr-serve
vllm-olmocr-serve:
	CUDA_VISIBLE_DEVICES=$(GPU_DEVICE) uv run vllm serve $(MODEL) \
		--gpu-memory-utilization $(GPU_MEMORY_UTILIZATION) \
		--max-model-len $(MAX_MODEL_LEN) \
		--tensor-parallel-size $(TENSOR_PARALLEL_SIZE) \
		--data-parallel-size $(DATA_PARALLEL_SIZE) \
		--max-num-batched-tokens 65536 \
		--mm-encoder-tp-mode "data" \
		--limit-mm-per-prompt '{"video": 0}' \
		--max-num-seqs 8192 \
		--port $(PORT) \
		--host $(SERVER)

.PHONY: vllm-embd-serve
vllm-embd-serve:
	CUDA_VISIBLE_DEVICES=$(GPU_DEVICE) uv run vllm serve $(EMBD_MODEL) \
		--gpu-memory-utilization $(EMBD_GPU_MEMORY_UTILIZATION) \
		--runner pooling \
		--max-model-len 8192 \
		--port $(EMBD_PORT) \
		--host $(SERVER)

# --task score \

.PHONY: vllm-reranker-serve
vllm-reranker-serve:
	CUDA_VISIBLE_DEVICES=$(GPU_DEVICE) uv run vllm serve $(RERANKER_MODEL) \
		--gpu-memory-utilization $(RERANKER_GPU_MEMORY_UTILIZATION) \
		--hf-overrides '{"architectures": ["Qwen3ForSequenceClassification"], "classifier_from_token": ["no", "yes"], "is_original_qwen3_reranker": true}' \
		--port $(RERANKER_PORT) \
		--host $(SERVER)


.PHONY: start-server
start-server:
	PROCESSED_INDEX_START_WATCHER=true uv run uvicorn server:app --host 0.0.0.0 --reload --port $(API_PORT)

.PHONY: run-ocr
run-ocr:
	nohup $(MAKE) vllm-olmocr-serve >> olmocr.log 2>&1 &

.PHONY: run-embd
run-embd:
	nohup $(MAKE) vllm-embd-serve >> embd.log 2>&1 &

.PHONY: run-server
run-server:
	nohup $(MAKE) start-server >> server.log 2>&1 &

.PHONY: run-reranker
run-reranker:
	nohup $(MAKE) vllm-reranker-serve >> reranker.log 2>&1 &

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
	docker run --gpus all \
		-e GPU_DEVICE="$(GPU_DEVICE)" \
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

# Dev: Black reformats sources; ty needs ocr-md + mcp so third-party imports resolve.
.PHONY: lint
lint:
	uv run --group dev black .
	uv run --group dev --group ocr-md --group mcp ty check .

.PHONY: lint-check
lint-check:
	uv run --group dev black --check .
	uv run --group dev --group ocr-md --group mcp ty check .

.PHONY: build-cache
build-cache:
	uv run python -c "from finance_data.common.processed_data_index import rebuild_processed_data_cache; rebuild_processed_data_cache()"
