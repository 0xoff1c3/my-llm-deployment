# my-llm-deployment
Deploy custom LLM with SAP AI Core and BAS

# Mini LLM Doc Assistant (SAP BAS + Hugging Face API)

A free, simple LLM-powered RAG using Hugging Face’s public inference API.  
Built in SAP Business Application Studio (BAS), no paid SAP or external cloud needed.

## Setup

1. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

2. Run the app:
    ```
    flask run
    ```

3. Paste your Hugging Face API token in the input field to use the app.

## How it Works

- Uses [Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) via Hugging Face’s free API.
- No data is stored; your token is used only for requests.

---

**Built with SAP BAS, open-source, and free tiers only!**
