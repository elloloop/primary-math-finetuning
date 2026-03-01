"""FastAPI inference server wrapping vLLM for fine-tuned model serving."""

import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-7B")
LORA_PATH = os.environ.get("LORA_PATH", "")
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "4096"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.9"))

llm: Optional[LLM] = None
lora_request: Optional[LoRARequest] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm, lora_request
    logger.info("Loading model from %s", MODEL_PATH)
    enable_lora = bool(LORA_PATH)
    llm = LLM(
        model=MODEL_PATH,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        enable_lora=enable_lora,
        trust_remote_code=True,
    )
    if LORA_PATH:
        logger.info("Loading LoRA adapter from %s", LORA_PATH)
        lora_request = LoRARequest("finetuned", 1, LORA_PATH)
    logger.info("Model loaded successfully")
    yield
    logger.info("Shutting down")


app = FastAPI(title="Primary Math Inference Server", lifespan=lifespan)


# --- Request / Response schemas ---


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    stop: Optional[list[str]] = None


class CompletionResponse(BaseModel):
    text: str
    finish_reason: str
    usage: dict


class BatchCompletionRequest(BaseModel):
    prompts: list[str]
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    stop: Optional[list[str]] = None


class BatchCompletionResponse(BaseModel):
    completions: list[CompletionResponse]


# --- Endpoints ---


@app.get("/health")
async def health():
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model": MODEL_PATH, "lora": LORA_PATH or None}


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(request: CompletionRequest):
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    params = SamplingParams(
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=request.stop,
    )
    outputs = llm.generate([request.prompt], params, lora_request=lora_request)
    output = outputs[0].outputs[0]

    return CompletionResponse(
        text=output.text,
        finish_reason=output.finish_reason,
        usage={
            "prompt_tokens": len(outputs[0].prompt_token_ids),
            "completion_tokens": len(output.token_ids),
        },
    )


@app.post("/v1/batch", response_model=BatchCompletionResponse)
async def batch_completions(request: BatchCompletionRequest):
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.prompts:
        raise HTTPException(status_code=400, detail="No prompts provided")

    params = SamplingParams(
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=request.stop,
    )
    outputs = llm.generate(request.prompts, params, lora_request=lora_request)

    completions = []
    for out in outputs:
        result = out.outputs[0]
        completions.append(
            CompletionResponse(
                text=result.text,
                finish_reason=result.finish_reason,
                usage={
                    "prompt_tokens": len(out.prompt_token_ids),
                    "completion_tokens": len(result.token_ids),
                },
            )
        )

    return BatchCompletionResponse(completions=completions)


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
