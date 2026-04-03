import ray
from ray.data.llm import vLLMEngineProcessorConfig, build_processor

if __name__ == "__main__":
    # Initialize Ray
    ray.init()

    # simple dataset
    ds = ray.data.from_items(
        [
            {"prompt": "What is machine learning?"},
            {"prompt": "Explain neural networks in one sentence."},
        ]
    )

    # Minimal vLLM configuration
    config = vLLMEngineProcessorConfig(
        model_source="Qwen/Qwen2.5-1.5B-Instruct",
        concurrency=1,  # 1 vLLM engine replica
        batch_size=32,  # 32 samples per batch
        engine_kwargs={
            "max_model_len": 4096,  # Fit into test GPU memory
        },
    )

    # Build processor
    # preprocess: converts input row to format expected by vLLM (OpenAI chat format)
    # postprocess: extracts generated text from vLLM output
    processor = build_processor(
        config,
        preprocess=lambda row: {
            "messages": [{"role": "user", "content": row["prompt"]}],
            "sampling_params": {"temperature": 0.7, "max_tokens": 100},
        },
        postprocess=lambda row: {
            "prompt": row["prompt"],
            "response": row["generated_text"],
        },
    )

    # inference
    ds = processor(ds)

    # iterate through the results
    for result in ds.iter_rows():
        print(f"Q: {result['prompt']}")
        print(f"A: {result['response']}\n")

    # Alternative ways to get results:
    # results = ds.take(10)  # Get first 10 results
    # ds.show(limit=5)       # Print first 5 results
    # ds.write_parquet("output.parquet")  # Save to file
