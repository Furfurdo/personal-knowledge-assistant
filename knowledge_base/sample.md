# Sample Note

RAG (Retrieval-Augmented Generation) combines retrieval and generation.

- Step 1: Convert documents into embeddings.
- Step 2: Store vectors in a vector database (FAISS).
- Step 3: Retrieve top-k related chunks for each question.
- Step 4: Ask an LLM to answer using only retrieved context.

Benefits:
- Reduces hallucination
- Uses your private knowledge
- Easy to update by re-indexing
