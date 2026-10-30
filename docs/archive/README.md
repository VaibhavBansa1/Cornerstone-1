# Archive: older documents (kept for the record)

These describe the project **as it was at the minor evaluation (October 2026)**.
They are not updated. For how the system works now, read
[../architecture/architecture.md](../architecture/architecture.md) and [../FEATURES.md](../FEATURES.md).

| File | What it is | What has changed since |
|---|---|---|
| `Synopsis_AI_Chatbot_MITS.docx` | Project synopsis (Shahid Khan, Vaibhav Bansal; guide Ms. Aradhana Saxena) | Embedding model is now multilingual-e5-small (was all-MiniLM-L6-v2); search is hybrid (keywords + meaning + reranker), not vectors only; scanned PDFs are read with OCR; every chunk is dated; more AI providers with automatic fallback |
| `Presentation_AI_Chatbot_MITS.pptx` | Minor-evaluation slides | Same as above, plus: 86 automated tests (was 16), 55 graded test questions, college-wide pages added to the IT pages, live progress in the chat |
| `architecture_v1_original.png` | First architecture diagram (4 layers) | Replaced by `../architecture/architecture.png`, which shows every step and design decision |

The milestone reports (Requirement Analysis, System Design, Implementation,
Integration & Testing) were submitted on SDMS as Google Drive links and are not
stored in this repository. Milestones 1–3 describe the *planned* stack (React,
Node.js, MongoDB, Pinecone); the project switched to Python early on because the
AI parts (embeddings, FAISS, OCR, rerankers) are Python-first. See D18 in the
architecture doc.
