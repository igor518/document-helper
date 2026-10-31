import os
import re
import socket
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from ipaddress import ip_address
from typing import Any, Dict, Literal, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

load_dotenv()

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
vectorstore = PineconeVectorStore(index_name=os.getenv("PINECONE_INDEX_NAME"), embedding=embeddings)
model = ChatOpenAI(
	model=os.getenv("OPENAI_CHAT_MODEL", "gpt-6-luna"),
	reasoning_effort="none",
	use_responses_api=True,
)


class QueryClassification(BaseModel):
	"""Structured routing decision for a user question."""

	category: Literal["langchain", "casual", "out_of_scope", "ambiguous"] = Field(
		description="The category that best describes the latest user message."
	)
	confidence: float = Field(
		ge=0.0,
		le=1.0,
		description="Confidence in the category from 0.0 to 1.0.",
	)


query_classifier = model.with_structured_output(QueryClassification)

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

EXPLICIT_LANGCHAIN_TERMS = (
	"chatopenai",
	"create agent",
	"deep agent",
	"deep agents",
	"langchain",
	"langchain community",
	"langchain openai",
	"langchain pinecone",
	"langgraph",
	"langsmith",
	"pineconevectorstore",
	"recursivecharactertextsplitter",
	"recursive character text splitter",
)

OUT_OF_SCOPE_RESPONSE = (
	"I can only answer questions about LangChain and its related documentation. "
	"Ask me about LangChain, LangGraph, LangSmith, agents, tools, retrieval, "
	"embeddings, or vector stores."
)
CLARIFICATION_RESPONSE = (
	"I'm not sure whether that question is about LangChain. "
	"Please mention the LangChain feature or documentation topic you mean."
)
CASUAL_FALLBACK_RESPONSE = "Hello! How can I help with LangChain documentation?"
CLASSIFICATION_CONFIDENCE_THRESHOLD = float(
	os.getenv("QUERY_CLASSIFICATION_CONFIDENCE", "0.80")
)
RETRIEVAL_RELEVANCE_THRESHOLD = float(
	os.getenv("RETRIEVAL_RELEVANCE_THRESHOLD", "0.55")
)
MAX_HISTORY_MESSAGES = 6
SOURCE_VALIDATION_TIMEOUT = float(os.getenv("SOURCE_VALIDATION_TIMEOUT", "3.0"))
SOURCE_VALIDATION_WORKERS = 4
MARKDOWN_LINK_PATTERN = re.compile(
	r"\[([^\]]+)]\(https?://[^)\s]+\)",
	flags=re.IGNORECASE,
)
BARE_URL_PATTERN = re.compile(r"https?://[^\s<>]+", flags=re.IGNORECASE)


def _normalize_query(query: str) -> str:
	"""Normalize user input for deterministic routing."""
	return " ".join(query.casefold().replace("_", " ").strip().rstrip("!.,?").split())


def _casual_response(query: str) -> str | None:
	"""Return an immediate response for simple conversation."""
	return CASUAL_RESPONSES.get(_normalize_query(query))


def _is_explicit_langchain_question(query: str) -> bool:
	"""Return whether a query contains an unambiguous LangChain identifier."""
	normalized = _normalize_query(query)
	return any(
		re.search(rf"\b{re.escape(term)}\b", normalized)
		for term in EXPLICIT_LANGCHAIN_TERMS
	)


def _history_messages(
	chat_history: Sequence[Dict[str, Any]] | None,
) -> list[Dict[str, str]]:
	"""Return a small, sanitized conversation window for routing and answering."""
	messages = []

	for message in (chat_history or [])[-MAX_HISTORY_MESSAGES:]:
		role = str(message.get("role", ""))
		content = str(message.get("content", "")).strip()

		if role in {"user", "assistant"} and content:
			messages.append({"role": role, "content": content})

	return messages


def _history_cache_key(
	chat_history: Sequence[Dict[str, Any]] | None,
) -> tuple[tuple[str, str], ...]:
	"""Build a hashable representation of recent conversation history."""
	return tuple(
		(message["role"], message["content"])
		for message in _history_messages(chat_history)
	)


@lru_cache(maxsize=256)
def _classify_query_cached(
	query: str,
	history: tuple[tuple[str, str], ...],
) -> QueryClassification:
	"""Classify an ambiguous query using a fast structured-output model call."""
	messages = [
		{
			"role": "system",
			"content": (
				"Classify the latest user message for a LangChain documentation assistant. "
				"Use conversation history to resolve follow-ups and pronouns. "
				"Choose langchain only when the user is asking about LangChain, LangGraph, "
				"LangSmith, or their documented integrations and concepts. Choose casual "
				"only for greetings, thanks, and brief social conversation. Choose "
				"out_of_scope for requests about people, current events, general knowledge, "
				"or other technologies without a LangChain connection. Choose ambiguous "
				"when the intended subject cannot be determined. Do not answer the question."
			),
		},
	]
	messages.extend(
		{"role": role, "content": content}
		for role, content in history
	)
	messages.append({"role": "user", "content": query})
	return query_classifier.invoke(messages)


def _classify_query(
	query: str,
	chat_history: Sequence[Dict[str, Any]] | None,
) -> QueryClassification:
	"""Classify a query while keeping repeated requests out of the model API."""
	return _classify_query_cached(query, _history_cache_key(chat_history))


def _message_text(message: Any) -> str:
	"""Extract plain text from LangChain message content blocks."""
	return "\n".join(
		block.get("text", "")
		for block in message.content_blocks
		if block.get("type") == "text"
	).strip()


def _document_source(document: Any) -> str:
	"""Read an HTTP source URL from document metadata."""
	metadata = getattr(document, "metadata", None) or {}
	return str(metadata.get("source") or metadata.get("url") or "").strip()


def _is_public_http_url(url: str) -> bool:
	"""Reject malformed URLs and hosts that resolve to local or private networks."""
	parsed = urlparse(url)

	if (
		parsed.scheme not in {"http", "https"}
		or not parsed.hostname
		or parsed.username
		or parsed.password
	):
		return False

	try:
		default_port = 80 if parsed.scheme == "http" else 443
		addresses = socket.getaddrinfo(parsed.hostname, parsed.port or default_port)
	except OSError:
		return False

	for address in addresses:
		try:
			value = ip_address(address[4][0].split("%", maxsplit=1)[0])
		except ValueError:
			return False

		if not value.is_global:
			return False

	return True


@lru_cache(maxsize=512)
def _reachable_source_url(url: str) -> str | None:
	"""Return the final URL when a public source responds without an HTTP error."""
	if not _is_public_http_url(url):
		return None

	request = Request(
		url,
		headers={
			"Accept": "text/html,application/xhtml+xml",
			"Range": "bytes=0-0",
			"User-Agent": "DocumentationHelper/1.0 source-validator",
		},
		method="GET",
	)

	try:
		with urlopen(request, timeout=SOURCE_VALIDATION_TIMEOUT) as response:
			if 200 <= response.status < 400:
				return response.geturl()
	except (HTTPError, URLError, TimeoutError, ValueError, OSError):
		return None

	return None


def _documents_with_reachable_sources(documents: Sequence[Any]) -> list[Any]:
	"""Keep documents whose source URLs exist and store their final redirect URLs."""
	if not documents:
		return []

	sources = [_document_source(document) for document in documents]
	worker_count = min(SOURCE_VALIDATION_WORKERS, len(sources))

	with ThreadPoolExecutor(max_workers=worker_count) as executor:
		resolved_sources = list(executor.map(_reachable_source_url, sources))

	validated_documents = []

	for document, resolved_source in zip(documents, resolved_sources):
		if not resolved_source:
			continue

		document.metadata = dict(document.metadata or {})
		document.metadata["source"] = resolved_source
		validated_documents.append(document)

	return validated_documents


def _strip_unverified_links(answer: str) -> str:
	"""Remove model-generated links; verified sources are rendered separately by the UI."""
	answer = MARKDOWN_LINK_PATTERN.sub(r"\1", answer)
	answer = BARE_URL_PATTERN.sub("", answer)
	return "\n".join(line.rstrip() for line in answer.splitlines()).strip()


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
	"""Retrieve only sufficiently relevant context from the vector store."""
	retriever = vectorstore.as_retriever(
		search_type="similarity_score_threshold",
		search_kwargs={
			"k": 4,
			"score_threshold": RETRIEVAL_RELEVANCE_THRESHOLD,
		},
	)
	retrieved_docs = _documents_with_reachable_sources(retriever.invoke(query))

	if not retrieved_docs:
		return (
			"No sufficiently relevant LangChain documentation was found for this query.",
			[],
		)

	serialized = "\n\n".join(
		f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}"
		for doc in retrieved_docs
	)
	return serialized, retrieved_docs


def run_llm(
	query: str,
	chat_history: Sequence[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
	"""Run the RAG pipeline to answer the query using retrieved documents."""
	if casual_response := _casual_response(query):
		return {
			"answer": casual_response,
			"context": [],
		}

	if not _is_explicit_langchain_question(query):
		try:
			classification = _classify_query(query, chat_history)
		except Exception:
			return {
				"answer": CLARIFICATION_RESPONSE,
				"context": [],
			}

		if classification.confidence < CLASSIFICATION_CONFIDENCE_THRESHOLD:
			return {
				"answer": CLARIFICATION_RESPONSE,
				"context": [],
			}

		if classification.category == "casual":
			return {
				"answer": CASUAL_FALLBACK_RESPONSE,
				"context": [],
			}

		if classification.category == "out_of_scope":
			return {
				"answer": OUT_OF_SCOPE_RESPONSE,
				"context": [],
			}

		if classification.category != "langchain":
			return {
				"answer": CLARIFICATION_RESPONSE,
				"context": [],
			}

	system_prompt = (
		"You are a helpful AI assistant that answers questions about LangChain documentation. "
		"The query has already been classified as relevant to LangChain. "
		"Call retrieve_context exactly once before answering. "
		"Do not answer questions outside the LangChain ecosystem. "
		"Base the answer on retrieved documentation, not general model knowledge. "
		"Do not create Markdown links, citations, or URLs in the answer because the "
		"application renders verified source links separately. "
		"If no sufficiently relevant documentation is returned, say that the answer "
		"was not found in the indexed documentation."
	)
	agent = create_agent(
		model,
		tools=[retrieve_context],
		system_prompt=system_prompt,
		debug=False,
	)
	messages = _history_messages(chat_history)
	messages.append({"role": "user", "content": query})
	response = agent.invoke({"messages": messages})
	answer = _strip_unverified_links(_message_text(response["messages"][-1]))
	context_docs = []

	for message in response["messages"]:
		if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
			if isinstance(message.artifact, list):
				context_docs.extend(message.artifact)

	return {
		"answer": answer,
		"context": context_docs,
	}
