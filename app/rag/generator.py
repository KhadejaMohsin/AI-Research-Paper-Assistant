"""Step 3 of RAG: give the retrieved chunks to the LLM and get a cited answer."""
from app import config
from app.rag.retriever import retrieve
from app.rag.vectorstore import get_openai_client

NOT_FOUND = "I couldn't find that in the uploaded documents."


def load_system_prompt() -> str:
    return config.PROMPT_FILE.read_text()


def build_context(chunks: list[dict]) -> str:
    """Label each chunk with its source so the model can cite it."""
    return "\n\n---\n\n".join(
        f"[{c['source']}, p.{c['page']}]\n{c['text']}" for c in chunks
    )


def generate_answer(question: str, chunks: list[dict]) -> str:
    if not chunks:
        return NOT_FOUND
    response = get_openai_client().chat.completions.create(
        model=config.CHAT_MODEL,
        temperature=0,  # keep answers stable, which also makes evaluation fairer
        messages=[
            {"role": "system", "content": load_system_prompt()},
            {
                "role": "user",
                "content": f"Context passages:\n\n{build_context(chunks)}\n\nQuestion: {question}",
            },
        ],
    )
    return response.choices[0].message.content


def answer_question(question: str, k: int = 5) -> dict:
    """The full RAG pipeline: retrieve, then generate."""
    chunks = retrieve(question, k)
    return {"question": question, "answer": generate_answer(question, chunks), "sources": chunks}
