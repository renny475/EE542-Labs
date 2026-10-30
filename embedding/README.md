# PDF Chatbot (Streamlit + LangChain + OpenAI)

Lab 9 Part 2: a Streamlit app that reads PDFs, splits the text into chunks,
embeds them with OpenAI, stores the vectors in an in-memory FAISS index and
answers questions with a conversational retrieval chain.

Based on the course template (PDFs Chatbot using LangChain), migrated to the
LangChain 1.x package layout.

## How it works
1. `get_pdf_text` reads every page of the uploaded PDFs with `pypdf`.
2. `get_text_chunks` splits the text with `CharacterTextSplitter`
   (separator `\n`, chunk size 500, overlap 100).
3. `get_vectorstore` embeds the chunks with `text-embedding-3-small` and
   builds a FAISS vector store in memory.
4. `get_conversation_chain` wires `gpt-4o-mini` (temperature 0), a
   `ConversationBufferMemory` and the FAISS retriever (k = 4) into a
   `ConversationalRetrievalChain`.
5. `handle_userinput` sends the question through the chain and renders the
   chat history with the templates in `htmlTemplates.py`.

## Setup
```bash
cd embedding
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then paste your OpenAI API key
streamlit run app.py
```

Open http://localhost:8501, upload one or more PDFs in the sidebar, click
**Process**, then ask questions.

## Notes
- The same OpenAI key is used for embeddings and chat; both are billed per
  token on the OpenAI account.
- The FAISS index lives only in memory; re-upload and re-process after a
  restart.
- `langchain_community.vectorstores.FAISS` prints a deprecation warning on
  LangChain 1.x. It still works and is kept to match the assignment.
