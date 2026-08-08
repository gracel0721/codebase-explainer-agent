"""code-explain: a local, RAG-powered CLI for answering questions about a codebase.

Stage 1 (this release) implements RAG question-answering over an indexed
repository using Ollama for embeddings and the LLM, and sqlite-vec for the
vector store. The architecture deliberately isolates each stage behind small
protocols so the later code-graph and agentic-edit stages can slot in without
rewrites.
"""

__version__ = "0.1.0"