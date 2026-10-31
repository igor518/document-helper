import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import (Colors, log_error, log_header, log_info, log_success, log_warning)

load_dotenv()

ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

PINECONE_INDEX_NAME = os.getenv(
	"PINECONE_INDEX_NAME",
	"langchain-docs-vectors-2026",
)


embeddings = OpenAIEmbeddings(
	model='text-embedding-3-small',
	show_progress_bar=False,
	chunk_size=50,
	retry_min_seconds=10
)

vectorstore = PineconeVectorStore(
	index_name=PINECONE_INDEX_NAME,
	embedding=embeddings
)

tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=10000)
tavily_crawl = TavilyCrawl()

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

	# Process all batches concurrently
	async def add_batch(batch: List[Document], batch_num: int):
		try:
			await vectorstore.aadd_documents(batch)
			log_success(
				f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
			)
		except Exception as e:
			log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
			return False
		return True

	# Process batches concurrently
	tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
	results = await asyncio.gather(*tasks, return_exceptions=True)

	# Count successful batches
	successful = sum(1 for result in results if result is True)

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

    log_info("Crawling Tavily", Colors.PURPLE)
    res = tavily_crawl.invoke({
        "url": "https://python.langchain.com/",
        "max_depth": 2,
        "extract_depth": "advanced",
        "instructions": "content on ai agents"
    })
    all_docs = [Document(page_content=result['raw_content'], metadata=result) for result in res["results"]]
    log_success(f"Crawled {len(all_docs)} documents")

    # Split documents info chunks
    log_info("TEXT SPLITTING", Colors.CYAN)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(f"Split into {len(splitted_docs)} chunks")

    # Process documents asynchronously
    await index_documents_async(splitted_docs, batch_size=500)

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")

if __name__ == "__main__":
    asyncio.run(main())
