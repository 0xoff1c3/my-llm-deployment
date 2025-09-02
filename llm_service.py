import os
from openai import OpenAI
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import re
import logging
import gc

# Setup logging
logging.basicConfig(level=logging.INFO, filename='llm_service.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not set. Please set your Hugging Face token.")
    
def call_hf_api(prompt: str, max_tokens: int = 100) -> str:
    """Call Hugging Face Inference API for Llama-3.1-8B-Instruct"""
    try:
        client = InferenceClient(
            model="meta-llama/Llama-3.1-8B-Instruct",
            token=HF_TOKEN
        )
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        content = completion.choices[0].message["content"] if completion.choices else ""
        return content.strip()
    except Exception as e:
        logging.error(f"HF API error: {repr(e)}")
        return f"[Error in HF API: {repr(e)}]"
    finally:
        gc.collect()

def refine_query(query: str) -> str:
    """Refine query for better document retrieval"""
    prompt = (
        "As an SAP log analyst, rewrite this query to be clear, specific, and optimized for searching log data. "
        "Focus on key terms like product IDs (e.g., AB663), forecast models (e.g., JLR_SF_BESTFIT_CONTINUOUS_0M), "
        "or metrics (e.g., MAPE, outliers). Keep it concise.\n"
        f"Query: {query}"
    )
    refined = call_hf_api(prompt, max_tokens=40)
    logging.info(f"Refined query: {refined}")
    return refined

def get_answer(query: str, docs: list[str]) -> dict:
    """Generate concise answer from retrieved logs"""
    context = "\n".join(docs[:6]) if docs else ""
    prompt = (
        "You are a seasoned SAP log analyst.\n"
        "Objective:\n"
        f"Analyze the provided log lines and answer the query: {query}\n"
        "Response Format:\n"
        "- Line 1: Direct, concise answer to the query.\n"
        "- Next up to 5 bullet points: Evidence-based reasoning using specific log details.\n"
        "Instructions:\n"
        "- Use only the provided log lines—no assumptions or external knowledge.\n"
        "- Each bullet must be distinct, concrete, and insightful.\n"
        "- Keep total response ≤ 6 lines.\n"
    )
    if context:
        prompt += f"Logs:\n{context}\n\n"
    else:
        prompt += "No logs found. Use general knowledge.\n\n"
    prompt += f"Question: {query}\nAnswer:"

    try:
        raw_answer = call_hf_api(prompt, max_tokens=150)

        # Split into lines
        lines = [line.strip("•- ") for line in raw_answer.splitlines() if line.strip()]
        if not lines:
            return {"main": "No answer generated.", "bullets": []}

        main_answer = lines[0]  # first line = concise answer
        bullets = lines[1:] if len(lines) > 1 else []

        logging.info(f"Generated structured answer: {main_answer}, bullets: {bullets}")
        return {"main": main_answer, "bullets": bullets}

    except Exception as e:
        logging.error(f"Error answering: {repr(e)}")
        return {"main": f"[Error answering: {repr(e)}]", "bullets": []}
    finally:
        gc.collect()