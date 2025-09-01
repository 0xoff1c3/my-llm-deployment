# llm_service.py
import os
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv
from transformers import pipeline as text_pipeline
import logging
import gc
from openai import OpenAI

# Setup logging
logging.basicConfig(level=logging.INFO, filename='llm_service.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not set. Please set your Hugging Face token.")

phi_pipeline = None


def load_refiner():
    """Load local Flan-T5 query refiner pipeline"""
    return text_pipeline(
        task="text2text-generation",
        model="google/flan-t5-small",
        device=-1  # CPU
    )

refiner_pipeline = load_refiner()

def refine_query(query: str) -> str:
    """
    Refine user query for better document retrieval.
    """
    result = refiner_pipeline(
        f"Rewrite this query for better document search: {query}",
        max_new_tokens=40
    )
    return result[0]["generated_text"].strip()
def get_answer(query: str, docs: list[str]) -> str:
    """
    Generate an answer using Llama-3.1-8B-Instruct via Hugging Face Inference API.
    """
    try:
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=os.environ["HF_TOKEN"],
        )
        context = "\n".join(docs[:10]) if docs else ""
        prompt = (
            "You are an expert SAP log analyst.\n"
            "Keep the combined total of Answer + Reasoning within 5 lines.\n\n"
            "Provide your response in two parts:\n\n\n"
            "Part 1 – Answer:\n"
            "- Use concise bullet points (max 3).\n"
            "- Each bullet starts on a new line.\n"
            "- Include numbers if available.\n"
            "- Perform math operations if required.\n"
            "- Avoid repetition and summarize unique insights.\n\n"
            "Part 2 – Reasoning:\n"
            "- Match each bullet with a brief explanation.\n"
            "- Use numbered bullets to align with the answer.\n"
            "- Focus on logic, data patterns, or contextual factors."
        )
        if context:
            prompt += "Logs:\n" + context + "\n\n"
        else:
            prompt += "No logs found. Use general knowledge.\n\n"
        prompt += f"Question: {query}\nAnswer:"

        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct:cerebras",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )
        # Assuming 'completion' is the response object from the OpenAI API
        content = completion.choices[0].message.content if completion and completion.choices and completion.choices[0].message.content else ""
        answer = content.strip()
        logging.info(f"Generated answer: {answer}")
        return answer
    except Exception as e:
        logging.error(f"Error answering: {repr(e)}")
        return f"[Error answering: {repr(e)}]"
    finally:
        gc.collect()