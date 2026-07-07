
curl -X POST "http://localhost:8080/v1/chat/completions" -H "Content-Type: application/json" --data '{
  "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
  "messages": [
  {
  "role": "user",
  "content": "Alice and Bob play the following game. A stack of n tokens lies before them. The players take turns with Alice going first. On each turn, the player removes either 1 token or 4 tokens from the stack. Whoever removes the last token wins. Find the number of positive integers n less than or equal to 2024 for which there exists a strategy for Bob that guarantees that Bob will win the game regardless of Alice play. \n The correct answer is 1688, could you explain why?"
  }
  ]
  }'



genai-perf profile -m deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --url deepseek-svc-qwen-32b \
  --service-kind openai \
  --endpoint-type chat \
  --num-prompts 100 \
  --synthetic-input-tokens-mean 200 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 100 \
  --output-tokens-stddev 0 \
  --concurrency 20 \
  --streaming \
  --tokenizer hf-internal-testing/llama-tokenizer




genai-perf profile -m zai-org/GLM-5.2-FP8 \
  --url glm-5-2 \
  --service-kind openai \
  --endpoint-type chat \
  --num-prompts 100 \
  --synthetic-input-tokens-mean 200 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 100 \
  --output-tokens-stddev 0 \
  --concurrency 20 \
  --streaming \
  --tokenizer zai-org/GLM-5.2-FP8


====

  vllm bench serve \
  --model zai-org/GLM-5.2-FP8 \
  --dataset-name random \
  --random-input 8000 \
  --random-output 1024 \
  --request-rate 10 \
  --num-prompts 32 \
  --ignore-eos


genai-perf profile -m zai-org/GLM-5.2-FP8 \
  --url glm-5-2 \
  --service-kind openai \
  --endpoint-type chat \
  --num-prompts 100 \
  --synthetic-input-tokens-mean 8000 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 1024 \
  --output-tokens-stddev 0 \
  --concurrency 20 \
  --streaming \
  --tokenizer zai-org/GLM-5.2-FP8


  genai-perf profile -m zai-org/GLM-5.2-FP8 \
    --url glm-5-2.default.svc.cluster.local:80 \
    --service-kind openai \
    --endpoint-type chat \
    --num-prompts 100 \
    --synthetic-input-tokens-mean 8000 \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean 1024 \
    --output-tokens-stddev 0 \
    --concurrency 20 \
    --streaming \
    --tokenizer zai-org/GLM-5.2-FP8 \
    --extra-inputs max_tokens:1024 \
    --extra-inputs ignore_eos:true \
    --extra-inputs 'chat_template_kwargs:{"enable_thinking":false}'


genai-perf profile -m zai-org/GLM-5.2-FP8 \
  --url glm-5-2 \
  --endpoint-type chat \
  --num-prompts 100 \
  --synthetic-input-tokens-mean 8000 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 1024 \
  --output-tokens-stddev 0 \
  --concurrency 20 \
  --streaming \
  --tokenizer zai-org/GLM-5.2-FP8