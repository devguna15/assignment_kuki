import os
import json
import numpy as np
import faiss
import unicodedata
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS as LCFAISS
from langchain_core.documents import Document
from llm_utilization import call_mistral_chat
from difflib import get_close_matches

# --- Configuration ---
JSON_PATH = "faqs.json"
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
INDEX_PATH = "faiss_index"

def robust_normalize(category: str) -> str:
    """Lowercase, strip, '&'->'and', remove all non-alphanum."""
    if not category:
        return ""
    category = unicodedata.normalize('NFKD', category.lower()).strip()
    category = category.replace("&", "and")
    return re.sub(r'[^a-z0-9]', '', category)

def load_faqs():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def load_categories():
    faqs = load_faqs()
    return sorted(set(f['Product_Category'].strip() for f in faqs))

def classify_category(user_query, categories):
    norm_cats = [robust_normalize(c) for c in categories]
    category_list = ", ".join(categories)
    prompt = (
        "You are an e-commerce AI assistant. "
        "Classify the user's question into ONE category from the list below. "
        "Return ONLY the category name, nothing else. "
        "If not found, reply with 'General'.\n\n"
        f"Categories: {category_list}\n"
        f"Customer question: {user_query}\nCategory:"
    )
    llm_category = call_mistral_chat(prompt, system_message="You are an e-commerce assistant.", max_tokens=10).strip()
    print(f"[DEBUG] LLM CATEGORY RAW OUTPUT: '{llm_category}'")  # <-- See the LLM's raw category output
    llm_category_norm = robust_normalize(llm_category)
    if llm_category_norm in norm_cats:
        matched = categories[norm_cats.index(llm_category_norm)]
        print(f"[DEBUG] Matched category: '{matched}'")  # <-- See the matched category
        return matched
    match = get_close_matches(llm_category, categories, n=1, cutoff=0.6)
    if match:
        print(f"[DEBUG] Fuzzy-matched category: '{match[0]}'")
        return match[0]
    print("[DEBUG] No good category match, using 'General'")
    return "General"

def build_faiss_index():
    faqs = load_faqs()
    docs = [
        Document(
            page_content=f"Q: {f['Customer_Question']} A: {f['Detailed_Response']}",
            metadata=f
        )
        for f in faqs
    ]
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    vectordb = LCFAISS.from_documents(docs, embedding=embeddings)
    vectordb.save_local(INDEX_PATH)
    print(f"FAISS index built and saved to '{INDEX_PATH}'.")

def get_relevant_faqs(user_query, top_k=5, similarity_threshold=1.2):
    faqs = load_faqs()
    categories = load_categories()
    category = classify_category(user_query, categories)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    questions = [f['Customer_Question'] for f in faqs]
    faqs_emb = np.array(embeddings.embed_documents(questions)).astype("float32")
    user_emb = np.array(embeddings.embed_query(user_query)).reshape(1, -1).astype("float32")
    context_faqs = []
    used_category_filter = False

    # Print which branch is being used for retrieval
    if robust_normalize(category) != robust_normalize("General"):
        print(f"[DEBUG] Retrieval: Category-filtered search for category '{category}'")
        indices = [
            i for i, row in enumerate(faqs)
            if robust_normalize(row["Product_Category"]) == robust_normalize(category)
        ]
        if indices:
            cat_vectors = faqs_emb[indices]
            cat_index = faiss.IndexFlatL2(cat_vectors.shape[1])
            cat_index.add(cat_vectors)
            D, I = cat_index.search(user_emb, min(top_k, len(indices)))
            I = [[indices[j] for j in row] for row in I]
            used_category_filter = True
        else:
            print(f"[DEBUG] No FAQs found for category '{category}', falling back to general search.")
            faiss_index = faiss.IndexFlatL2(faqs_emb.shape[1])
            faiss_index.add(faqs_emb)
            D, I = faiss_index.search(user_emb, top_k)
    else:
        print("[DEBUG] Retrieval: General search (no category filtering)")
        faiss_index = faiss.IndexFlatL2(faqs_emb.shape[1])
        faiss_index.add(faqs_emb)
        D, I = faiss_index.search(user_emb, top_k)

    for rank, idx in enumerate(I[0]):
        faq = dict(faqs[idx])
        faq["rank"] = rank + 1
        faq["used_category_filter"] = used_category_filter
        faq["distance"] = D[0][rank]
        context_faqs.append(faq)

    if all(d > similarity_threshold for d in D[0]) or not context_faqs:
        return []
    return context_faqs

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2 and sys.argv[1] == "build":
        build_faiss_index()
    # ... (rest of your code)