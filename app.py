from flask import Flask, render_template, request
from retriever import retrieve_documents
from llm_service import refine_query, get_answer
import gc

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_query = request.form["query"].strip()
        from_date = request.form.get("from_date", "").strip()
        to_date = request.form.get("to_date", "").strip()

        if not user_query or len(user_query) > 500:
            return render_template("index.html", error="Query must be non-empty and under 500 characters.")

        refined_query = refine_query(user_query)

        docs = retrieve_documents(
            refined_query,
            from_date=from_date if from_date else "",
            to_date=to_date if to_date else ""
        )

        answer = get_answer(user_query, docs)
        gc.collect()
        display_docs = docs[:12]
        return render_template(
            "results.html",
            query=user_query,
            refined_query=refined_query,
            docs=display_docs,
            answer=answer
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
