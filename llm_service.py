# llm_service.py
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

# def call_hf_api(prompt: str, max_tokens: int = 100) -> str:
#     """Call Hugging Face Inference API for Llama-3.1-8B-Instruct"""
#     try:
#         client = OpenAI(
#             base_url="https://api-inference.huggingface.co/models/meta-llama/Llama-3.1-8B-Instruct",
#             api_key=HF_TOKEN,
#         )
#         completion = client.chat.completions.create(
#             model="meta-llama/Llama-3.1-8B-Instruct",
#             messages=[{"role": "user", "content": prompt}],
#             max_tokens=max_tokens
#         )
#         # Assuming 'completion' is the response object from the OpenAI API
#         content = completion.choices[0].message.content if completion and completion.choices and completion.choices[0].message.content else ""
#         return content.strip() 
#     except Exception as e:
#         logging.error(f"HF API error: {repr(e)}")
#         return f"[Error in HF API: {repr(e)}]"
#     finally:
#         gc.collect()
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

def get_answer(query: str, docs: list[str]) -> str:
    """Generate concise answer from retrieved logs"""
    context = "\n".join(docs[:6]) if docs else ""
    prompt = (
        "You are an expert SAP log analyst.\n"
        "Task: Analyze the provided logs and answer the question in up to 5 unique sentences. "
        "Use bullet points for key insights. Include specific metrics (e.g., MAPE) or IDs (e.g., AB663, 13094) if available. "
        "Avoid repetition and generic statements. Summarize findings in 1-2 sentences.\n"
    )
    if context:
        prompt += f"Logs:\n{context}\n\n"
    else:
        prompt += "No logs found. Use general knowledge.\n\n"
    prompt += f"Question: {query}\nAnswer:"
    try:
        answer = call_hf_api(prompt, max_tokens=100)
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        unique_sentences = []
        for s in sentences:
            if s and s not in unique_sentences and len(unique_sentences) < 5:
                unique_sentences.append(s)
        answer = ' '.join(unique_sentences)
        logging.info(f"Generated answer: {answer}")
        return answer
    except Exception as e:
        logging.error(f"Error answering: {repr(e)}")
        return f"[Error answering: {repr(e)}]"
    finally:
        gc.collect()