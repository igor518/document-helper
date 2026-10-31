from typing import Any, Dict, List
from urllib.parse import urlparse

import streamlit as st
from backend.core import run_llm


def _format_sources(context_docs: List[Any]) -> List[str]:
	sources = []
	seen = set()

	for doc in context_docs or []:
		metadata = getattr(doc, "metadata", None) or {}
		source = str(metadata.get("source") or metadata.get("url") or "").strip()
		parsed_source = urlparse(source)

		if (
			parsed_source.scheme not in {"http", "https"}
			or not parsed_source.netloc
			or source in seen
		):
			continue

		seen.add(source)
		sources.append(source)

	return sources


st.set_page_config(page_title="LangChain Documentation Helper", layout="centered")
st.title("LangChain Documentation Helper")

with st.sidebar:
	st.subheader("Session")
	if st.button("Clear chat", use_container_width=True):
		st.session_state.pop("messages", None)
		st.rerun()

if "messages" not in st.session_state:
	st.session_state.messages = [
		{
			"role": "assistant",
			"content": "Ask me anything about LangChain docs. I’ll retrieve relevant context and cite sources.",
			"sources": [],
		}
	]

for msg in st.session_state.messages:
	with st.chat_message(msg["role"]):
		st.markdown(msg["content"])
		if msg.get("sources"):
			with st.expander("Sources"):
				for s in msg["sources"]:
					st.markdown(f"- {s}")

prompt = st.chat_input("Ask a question about LangChain…")
if prompt:
	chat_history = [
		{"role": message["role"], "content": message["content"]}
		for message in st.session_state.messages
		if message.get("role") in {"user", "assistant"}
	]
	st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
	with st.chat_message("user"):
		st.markdown(prompt)

	with st.chat_message("assistant"):
		try:
			with st.spinner("Generating response…"):
				result: Dict[str, Any] = run_llm(prompt, chat_history=chat_history)
				answer = str(result.get("answer", "")).strip() or "(No answer returned.)"
				sources = _format_sources(result.get("context", []))

			st.markdown(answer)
			if sources:
				with st.expander("Sources"):
					for s in sources:
						st.markdown(f"- {s}")

			st.session_state.messages.append(
				{"role": "assistant", "content": answer, "sources": sources}
			)
		except Exception as e:
			st.error("Failed to generate a response.")
			st.exception(e)
