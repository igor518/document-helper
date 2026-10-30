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


embeddings = OpenAIEmbeddings(
    model='text-embeddings-3-small',
    show_progress_bar=False,
    chunk_size=50,
    retry_min_seconds=10
)

vectorstore = PineconeVectorStore(
    index_name="langchain-docs-2026",
    embedding=embeddings
)

tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=10000)
tavily_crawl = TavilyCrawl()

async def main():
    log_header("Starting ingestion")

    log_info("Crawling Tavily", Colors.PURPLE)
    res = tavily_crawl.invoke({
        "url": "https://python.langchain.com/",
        "max_depth": 1,
        "extract_depth": "advanced",
        "instructions": "content on ai agents"
    })
    all_docs = [Document(page_content=result['raw_content'], metadata=result) for result in res["results"]]
    log_success(f"Crawled {len(all_docs)} documents")



if __name__ == "__main__":
    asyncio.run(main())