# Documentation Helper

Documentation Helper is a Python project for building a LangChain agent that can ingest, search, and analyze technical documentation. The goal is to turn documentation from websites and local files into a queryable knowledge base that produces grounded answers with source context.

> [!NOTE]
> The project is currently an early scaffold. Dependencies and development tooling are configured, but the documentation ingestion and agent workflows have not been implemented yet.

## Goals

- Load documentation from websites and local files.
- Clean, split, and enrich documents with source metadata.
- Generate embeddings and store them in Chroma or Pinecone.
- Retrieve relevant documentation for each question.
- Use a LangChain agent and tools to analyze the retrieved context.
- Return answers with enough source information to verify them.
- Expose the agent through a Streamlit interface or FastAPI service.
- Trace and evaluate agent behavior with LangSmith.

## Planned workflow

```text
Documentation sources
	→ Load and normalize content
	→ Split content into searchable chunks
	→ Create embeddings
	→ Store chunks in a vector database
	→ Retrieve relevant context
	→ Analyze context with the LangChain agent
	→ Return a grounded answer with sources
```

## Technology

- **LangChain** for agent, tool, and retrieval workflows.
- **OpenAI** for language-model and embedding integrations.
- **Unstructured**, **Beautiful Soup**, and **Firecrawl** for document ingestion.
- **Chroma** and **Pinecone** for vector storage and semantic search.
- **Tavily** for optional web-search tools.
- **Streamlit** and **FastAPI** for user-facing interfaces.
- **LangSmith** for tracing, debugging, and evaluation.
- **uv** for Python dependency and environment management.

## Requirements

- Python 3.13 or newer
- [uv](https://docs.astral.sh/uv/)
- An OpenAI API key
- Provider credentials for any optional services you enable, such as Pinecone, Tavily, Firecrawl, or LangSmith

## Setup

Clone the repository:

```bash
git clone https://github.com/igor518/document-helper.git
cd document-helper
```

Create the project environment and install all locked dependencies:

```bash
uv sync
```

Create a local `.env` file:

```dotenv
OPENAI_API_KEY=your-api-key

# Optional integrations
LANGSMITH_API_KEY=your-langsmith-api-key
LANGSMITH_TRACING=true
PINECONE_API_KEY=your-pinecone-api-key
TAVILY_API_KEY=your-tavily-api-key
FIRECRAWL_API_KEY=your-firecrawl-api-key
```

The `.env` file is ignored by Git and must never be committed.

## Development

Run Python commands inside the managed environment with `uv run`:

```bash
uv run python main.py
```

Format and sort Python code:

```bash
uv run black .
uv run isort .
```

Inspect the resolved dependency tree:

```bash
uv tree
```

## Roadmap

- [ ] Define application configuration and supported documentation sources.
- [ ] Implement web and file document loaders.
- [ ] Add chunking and metadata normalization.
- [ ] Build Chroma and Pinecone indexing workflows.
- [ ] Create retrieval tools for the LangChain agent.
- [ ] Add grounded answers and source citations.
- [ ] Build a Streamlit chat interface.
- [ ] Add a FastAPI endpoint.
- [ ] Add automated tests and LangSmith evaluations.

## License

No license has been added yet.
