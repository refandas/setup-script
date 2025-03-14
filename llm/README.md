# Infinity Embedding and VLLM

This guide provides setup and execution instructions for **[Infinity Embedding](https://github.com/michaelfeil/infinity)** and **[VLLM](https://github.com/vllm-project/vllm)**, two tools used for efficient model serving.

---

## [Infinity Embedding](https://github.com/michaelfeil/infinity)

Infinity Embedding is a tool designed for serving embeddings effiiciently using CUDA-enables GPUs.

### Setup Environment Variables

Before starting the Infinity Embedding server, set the required environment variables:

```bash
export INFINITY_API_TOKEN="<API TOKEN>"
export INFINITY_MODEL_PATH="<MODEL PATH>"
```

- INFINITY_API_TOKEN: The API token required for authentication.
- INFINITY_MODEL_PATH: The full file path where the embedding model is stored.

### Start the Infinity Embedding Server

Use the following command to serve the model:

```bash
nohup infinity_emb v2 \
  --model-id $INFINITY_MODEL_PATH \
  --api-key $INFINITY_API_TOKEN \
  --device cuda \
  --batch-size 32 \
  --port 7997 \
  > infinity.log 2>&1 &
```

---

## [VLLM](https://github.com/vllm-project/vllm)

VLLM is an efficient model inference server optimized for large language models.

### Setup Environment Variables

Before running the VLLM server, define the required environment variables:

```bash
export VLLM_API_TOKEN="<API TOKEN>"
export VLLM_MODEL_PATH="<MODEL PATH>"
```

- VLLM_API_TOKEN: The API token for authentication
- VLLM_MODEL_PATH: The full file path where the model is stored

### Start the VLLM Server

VLLM can be deployed on single-GPU or multi-GPU setups.

#### Single GPU Mode

To serve the model using a single GPU, run:

```bash
nohup vllm serve $VLLM_MODEL_PATH \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key $VLLM_API_TOKEN \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.95 \
    --disable-log-requests \
    > vllm.log 2>&1 &
```

#### Multi GPU Mode

For multi-GPU support, use the following command:

```bash
nohup vllm serve $VLLM_MODEL_PATH \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key $VLLM_API_TOKEN \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json \
    --tensor-parallel-size 2 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.95 \
    --disable-log-requests \
    > vllm.log 2>&1 &
```

#### Additional Argument for Multi-GPU Mode

- `--tensor-parallel-size 2`: Distributes computation across 2 GPUs (modify as needed based on hardware availability).

---

## Summary

- **Infinity Embedding** is used for efficient embedding serving.
- **VLLM** is a high-performance LLM inference server, supporting both single-GPU and multi-GPU execution.
- Logs are stored in `infinity.log` and `vllm.log` for debugging.
