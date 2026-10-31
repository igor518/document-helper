import asyncio
import hashlib
import os
import ssl
from typing import List

import certifi
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyExtract

from logger import (Colors, log_error, log_header, log_info, log_success, log_warning)

load_dotenv()

ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

PINECONE_INDEX_NAME = os.getenv(
	"PINECONE_INDEX_NAME",
	"langchain-docs-vectors-2026",
)
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "langchain-docs-v2")
DEFAULT_DOCUMENTATION_URLS = (
	"https://docs.langchain.com/oss/python/langchain/overview",
	"https://docs.langchain.com/oss/python/langchain/agents",
	"https://docs.langchain.com/oss/python/langchain/tools",
	"https://docs.langchain.com/oss/python/langchain/retrieval",
	"https://docs.langchain.com/langsmith/home",
	"https://docs.langchain.com/oss/python/langgraph/overview",
	"https://docs.langchain.com/oss/python/deepagents/overview",
)
MIN_DOCUMENT_LENGTH = 200


embeddings = OpenAIEmbeddings(
	model='text-embedding-3-small',
	show_progress_bar=False,
	chunk_size=50,
	retry_min_seconds=10
)

vectorstore = PineconeVectorStore(
	index_name=PINECONE_INDEX_NAME,
	embedding=embeddings,
	namespace=PINECONE_NAMESPACE,
)

tavily_extract = TavilyExtract(extract_depth="advanced", format="markdown")


def _documentation_urls() -> list[str]:
	"""Return configured documentation URLs or the curated defaults."""
	configured_urls = os.getenv("DOCUMENTATION_URLS", "")

	if not configured_urls.strip():
		return list(DEFAULT_DOCUMENTATION_URLS)

	return [
		url.strip()
		for url in configured_urls.split(",")
		if url.strip()
	]


def _extract_documents(urls: list[str]) -> list[Document]:
	"""Extract clean Markdown content and keep compact source metadata."""
	response = tavily_extract.invoke({"urls": urls})
	documents = []

	for result in response.get("results", []):
		content = result.get("raw_content")
		source = str(result.get("url") or "").strip()

		if not isinstance(content, str) or len(content.strip()) < MIN_DOCUMENT_LENGTH:
			log_warning(f"Skipping empty or too-short document: {source or 'Unknown'}")
			continue

		documents.append(
			Document(
				page_content=content.strip(),
				metadata={"source": source, "url": source},
			)
		)

	for failure in response.get("failed_results", []):
		log_warning(f"Tavily failed to extract: {failure}")

	return documents


def _assign_document_ids(documents: list[Document]) -> None:
	"""Assign stable IDs so rerunning ingestion replaces identical chunks."""
	for document in documents:
		source = str(document.metadata.get("source") or "")
		payload = f"{source}\n{document.page_content}".encode("utf-8")
		document.id = hashlib.sha256(payload).hexdigest()


async def index_documents_async(documents: List[Document], batch_size: int = 50):
	"""Process documents in batches asynchronously."""
	log_header("VECTOR STORAGE PHASE")
	log_info(
		f"📚 VectorStore Indexing: Preparing to add {len(documents)} documents to vector store",
		Colors.DARKCYAN,
	)

	# Create batches
	batches = [
		documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
	]

	log_info(
		f"📦 VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
	)

	async def add_batch(batch: List[Document], batch_num: int) -> bool:
		try:
			await vectorstore.aadd_documents(
				batch,
				ids=[document.id for document in batch],
			)
			log_success(
				f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
			)
		except Exception as e:
			log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
			return False
		return True

	results = []

	for batch_num, batch in enumerate(batches, start=1):
		results.append(await add_batch(batch, batch_num))

	successful = sum(results)

	if successful == len(batches):
		log_success(
			f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})"
		)
	else:
		message = (
			f"VectorStore Indexing: Processed {successful}/{len(batches)} "
			"batches successfully"
		)
		log_error(message)
		raise RuntimeError(message)

async def main():
	log_header("Starting ingestion")

	urls = _documentation_urls()
	log_info(f"Extracting {len(urls)} documentation pages with Tavily", Colors.PURPLE)
	all_docs = _extract_documents(urls)

	if not all_docs:
		raise RuntimeError("Tavily did not return any usable documentation content")

	log_success(f"Extracted {len(all_docs)} documents")

	log_info("TEXT SPLITTING", Colors.CYAN)
	text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
	splitted_docs = text_splitter.split_documents(all_docs)
	_assign_document_ids(splitted_docs)
	log_success(f"Split into {len(splitted_docs)} chunks")

	await index_documents_async(splitted_docs, batch_size=500)

	log_header("PIPELINE COMPLETE")
	log_success("🎉 Documentation ingestion pipeline finished successfully!")
	log_info("📊 Summary:", Colors.BOLD)
	log_info(f"   • Documents extracted: {len(all_docs)}")
	log_info(f"   • Chunks created: {len(splitted_docs)}")
	log_info(f"   • Pinecone namespace: {PINECONE_NAMESPACE}")

if __name__ == "__main__":
	asyncio.run(main())
