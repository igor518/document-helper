import os
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
	system_prompt = (
		"You are a helpful AI assistant that answers questions about LangChain documentation. "
		"You have access to a tool that retrieves relevant documentation. "
		"Use the tool to find relevant information before answering questions. "
		"Always cite the sources you use in your answers. "
		"If you cannot find the answer in the retrieved documentation, say so."
	)
	agent = create_agent(
		model,
		tools=[retrieve_context],
		system_prompt=system_prompt,
		debug=True,
	)
	messages = [{"role": "user", "content": query}]
	response = agent.invoke({"messages": messages})
	answer = response["messages"][-1].content
	context_docs = []

	for message in response["messages"]:
		if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
			if isinstance(message.artifact, list):
				context_docs.extend(message.artifact)

	return {
		"answer": answer,
		"context": context_docs,
	}


if __name__ == "__main__":
	result = run_llm(query="What are deep agents?")
	print(result)
