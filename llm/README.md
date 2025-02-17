# Infinity Embedding and VLLM

## Infinity Embedding

### Setup env

```bash
export INFINITY_API_TOKEN="<API TOKEN>"
export INFINITY_MODEL_PATH="<MODEL PATH>"
```

### Serve

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

## VLLM

### Setup env

```bash
export VLLM_API_TOKEN="<API TOKEN>"
export VLLM_MODEL_PATH="<MODEL PATH>"
```

### Serve

#### For Single GPU

```bash
nohup vllm serve $VLLM_MODEL_PATH \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key $VLLM_API_TOKEN \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.95 \
    > vllm.log 2>&1 &
```

### For Multi GPU

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
    > vllm.log 2>&1 &
```
