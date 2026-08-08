"""code-explain: a local, RAG-powered CLI for answering questions about a codebase.

Stage 1 implements RAG question-answering over an indexed repository using
Ollama for embeddings and the LLM, and sqlite-vec for the vector store. Stage 2
adds a code-relationship graph that expands retrieval with caller/callee
context. Stage 3 adds an agentic-edit layer that explores the codebase with
tools and proposes patches. The architecture isolates each stage behind small
protocols (``VectorStore``, ``Reranker``) so each layer slots in without
rewrites; the vector backend (sqlite-vec or LanceDB) is swappable too.
"""

__version__ = "0.2.0"