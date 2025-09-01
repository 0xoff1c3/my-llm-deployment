# app.py
from flask import Flask, render_template, request
from retriever import retrieve_documents
from llm_service import refine_query, get_answer
import gc

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_query = request.form["query"].strip()
        if not user_query or len(user_query) > 500:
            return render_template("index.html", error="Query must be non-empty and under 500 characters.")
        refined_query = refine_query(user_query)
        docs = retrieve_documents(refined_query, top_k=12)
        answer = get_answer(user_query, docs)
        gc.collect()
        return render_template(
            "results.html",
            query=user_query,
            refined_query=refined_query,
            docs=docs,
            answer=answer
        )
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, port=8000)