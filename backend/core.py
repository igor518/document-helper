import os
import re
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
vectorstore = PineconeVectorStore(index_name=os.getenv("PINECONE_INDEX_NAME"), embedding=embeddings)
model = ChatOpenAI(
	model=os.getenv("OPENAI_CHAT_MODEL", "gpt-6-luna"),
	reasoning_effort="none",
	use_responses_api=True,
)

CASUAL_RESPONSES = {
	"bye": "Goodbye!",
	"goodbye": "Goodbye!",
	"good afternoon": "Good afternoon! How can I help with LangChain documentation?",
	"good evening": "Good evening! How can I help with LangChain documentation?",
	"good morning": "Good morning! How can I help with LangChain documentation?",
	"hello": "Hello! How can I help with LangChain documentation?",
	"hey": "Hey! How can I help with LangChain documentation?",
	"hi": "Hi! How can I help with LangChain documentation?",
	"how are you": "I'm doing well, thanks! How can I help with LangChain documentation?",
	"how are you doing": "I'm doing well, thanks! How can I help with LangChain documentation?",
	"thank you": "You're welcome!",
	"thanks": "You're welcome!",
}

LANGCHAIN_TERMS = (
	"agent",
	"agents",
	"chain",
	"chatopenai",
	"chroma",
	"create agent",
	"deep agent",
	"deep agents",
	"document loader",
	"embedding",
	"embeddings",
	"langchain",
	"langgraph",
	"langsmith",
	"llm",
	"middleware",
	"pinecone",
	"prompt",
	"rag",
	"retrieval",
	"retriever",
	"text splitter",
	"tool",
	"tools",
	"vector store",
	"vectorstore",
)

OUT_OF_SCOPE_RESPONSE = (
	"I can only answer questions about LangChain and its related documentation. "
	"Ask me about LangChain, LangGraph, LangSmith, agents, tools, retrieval, "
	"embeddings, or vector stores."
)


def _normalize_query(query: str) -> str:
	"""Normalize user input for deterministic routing."""
	return " ".join(query.casefold().replace("_", " ").strip().rstrip("!.,?").split())


def _casual_response(query: str) -> str | None:
	"""Return an immediate response for simple conversation."""
	return CASUAL_RESPONSES.get(_normalize_query(query))


def _is_langchain_question(query: str) -> bool:
	"""Return whether a query contains a supported LangChain concept."""
	normalized = _normalize_query(query)
	return any(
		re.search(rf"\b{re.escape(term)}\b", normalized)
		for term in LANGCHAIN_TERMS
	)


def _message_text(message: Any) -> str:
	"""Extract plain text from LangChain message content blocks."""
	return "\n".join(
		block.get("text", "")
		for block in message.content_blocks
		if block.get("type") == "text"
	).strip()


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
	"""Retrieve context from vectorstore."""
	retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
	retrieved_docs = retriever.invoke(query)
	serialized = "\n\n".join(
		f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}"
		for doc in retrieved_docs
	)
	return serialized, retrieved_docs


def run_llm(query: str) -> Dict[str, Any]:
	"""Run the RAG pipeline to answer the query using retrieved documents."""
	if casual_response := _casual_response(query):
		return {
			"answer": casual_response,
			"context": [],
		}

	if not _is_langchain_question(query):
		return {
			"answer": OUT_OF_SCOPE_RESPONSE,
			"context": [],
		}

	system_prompt = (
		"You are a helpful AI assistant that answers questions about LangChain documentation. "
		"Call retrieve_context only for substantive questions about LangChain documentation. "
		"Do not answer questions outside the LangChain ecosystem. "
		"Cite sources only when retrieved documentation includes source metadata. "
		"If you cannot find the answer in the retrieved documentation, say so."
	)
	agent = create_agent(
		model,
		tools=[retrieve_context],
		system_prompt=system_prompt,
		debug=False,
	)
	messages = [{"role": "user", "content": query}]
	response = agent.invoke({"messages": messages})
	answer = _message_text(response["messages"][-1])
	context_docs = []

	for message in response["messages"]:
		if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
			if isinstance(message.artifact, list):
				context_docs.extend(message.artifact)

	return {
		"answer": answer,
		"context": context_docs,
	}
