# Aira + RAG: how it works and how to set it up

## 1. The idea in one minute
Before, Aira got a small, fixed snapshot of your data in every question. Now your **database is the knowledge base**:

```
voc.db  ──►  chunks (rag_chunks table)  ──►  search  ──►  top matches + ranked overview  ──►  GPT-5 mini  ──►  answer + "Based on: …"
 themes        1 short text per theme,        keyword (TF-IDF)       (only what is relevant         (answers ONLY
 requests      request, or unanalyzed         + semantic (optional    to the question)               from that data)
 feedback      feedback entry                  embeddings)
```

Words you will see:
- **Chunk**: one small piece of text built from one database row, e.g. *"Request #3: Support SSO. Category: Security … Status: Completed. Original feedback: …"*.
- **Embedding**: a list of numbers that captures a text's meaning, so "sign-in" can match "login". Optional here.
- **Retrieval**: finding the chunks most related to the user's question.
- **Generation**: the model writes the answer using only what retrieval found.

Why it is better: Aira can now answer about *any* request, including old, approved or completed ones and the original
wording of the feedback, instead of only the top few. The ranked overview is still always sent, so "what should I do first?" works.

## 2. What was added
| File | Purpose |
|---|---|
| `rag.py` | builds chunks from the DB, keeps the index up to date, searches it, exports the knowledge document, admin panel |
| `db.py` | new table `rag_chunks` (created automatically; nothing to migrate by hand) |
| `aira.py` | asks `rag.py` for matching chunks and sends them with the overview; shows "Based on: …" under answers |
| `app.py` | adds the "AI knowledge index (RAG)" panel to the Admin sidebar |

No new packages: it uses `scikit-learn` and `numpy`, which are already in `requirements.txt`.

## 3. Setup, step by step

**Step 1: nothing to install.** If you use the earlier zip, just replace the project files with this version.

**Step 2: choose a search mode.**
- *Keyword only (works immediately, free).* Do nothing. Matching is by words (TF-IDF), computed locally.
- *Keyword + semantic (recommended).* In Azure AI Foundry, deploy an embedding model (for example `text-embedding-3-small`)
  and note the **deployment name**. Then add one line to `.env`:
  ```
  AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=your-embedding-deployment-name
  ```
  This is the same setting the clustering already uses, so if it is set you are done. `.env` is re-read automatically.
  Chat models cannot create embeddings, which is why a separate deployment is needed.

**Step 3: start the app.**
```
streamlit run app.py
```
The database upgrades itself (adds `rag_chunks`).

**Step 4: build the index once.** Open **Admin Access → sidebar → "AI knowledge index (RAG)" → Rebuild AI index**.
You should see something like *"Indexed 120 entries (120 embedded)"* and a line with the counts of themes, requests and
unanalyzed feedback. After this the index **updates itself** whenever Aira is asked something: only rows that changed
(new feedback, a task approved or completed, a new analysis) are rewritten and re-embedded.

**Step 5: ask Aira.** Open the bubble and try:
- *"Which task should I do first and why?"* → answers from the ranked overview ("Based on: Ranked task overview")
- *"What did client1 ask for?"* → answers from matching requests ("Based on: Request #4 · …")
- *"Has anything about SSO been completed?"* → finds approved/completed work and says so
- *"What's the weather?"* → "I couldn't find anything in your feedback data… try another command"

**Step 6 (optional): download the knowledge document.** The same sidebar panel has **Download knowledge document (.md)**:
your whole database as one readable Markdown file. You can upload it to any other chat tool or RAG system.

## 4. What happens when someone asks a question
1. The app builds the chunks from the database and compares them with the index (hash check). Changed rows are rewritten; if embeddings are on, only those are re-embedded.
2. The question is searched two ways: **keywords** (local, free) and, if configured, **meaning** (1 embedding call). The two rankings are merged.
3. The top 8 chunks + the ranked overview of pending tasks + counts go to GPT-5 mini with the rule *"answer only from this data, otherwise say NO_DATA"*.
4. The answer appears with the list of database entries it was based on.

A short follow-up such as *"and SSO?"* is searched together with the previous question so it keeps its context.

## 5. Settings (all optional, in `.env`)
| Setting | Default | Meaning |
|---|---|---|
| `AIRA_RAG_TOP_K` | 8 | database entries retrieved per question (more = more detail, more tokens) |
| `AIRA_RAG_MAX_EMBED_BATCHES` | 5 | new/changed entries embedded per question (64 per batch), the rest waits for the next question |
| `AIRA_RAG_REBUILD_MAX_BATCHES` | 20 | same limit for the Rebuild button (20 × 64 = 1,280 entries per click; click again for more) |
| `AIRA_HISTORY_TURNS` | 6 | earlier chat messages sent for follow-ups |

## 6. Cost and safety
- A question costs **1 chat call**, plus **1 embedding call** when semantic search is on. Index updates are batched (64 entries per call) and only for changed rows.
- Everything goes through the existing budget guard: `MAX_CALLS_PER_RUN` and `MAX_CALLS_PER_DAY` still apply. If the cap is hit, Aira says so; keyword search keeps working.
- Aira only sees your database; the retrieved text is treated as data, never as instructions.

## 7. Troubleshooting
| Symptom | Fix |
|---|---|
| Sidebar says "keyword only" | `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME` is empty in `.env`. |
| "embeddings skipped: …" after Rebuild | The embedding deployment name is wrong, the key lacks access, or the daily cap is reached. Keyword search still works; fix and click Rebuild again. |
| "N still queued" | More entries than one click embeds. Click Rebuild again, or just keep using Aira (it continues in batches). |
| Aira says it couldn't find anything | The question is outside your data, or uses words that don't appear in it. Enable embeddings for meaning-based matching, or rephrase. |
| Answers feel thin | Raise `AIRA_RAG_TOP_K` (for example 12). |
| Changed the embedding model | Nothing to do: entries embedded with another model are re-embedded automatically. |

## 8. Limits (good to know)
The index lives in SQLite and is searched by brute force, which is comfortably fast for thousands of entries. If the tracker
grows to hundreds of thousands, move the `rag_chunks` table to a vector database; `rag.py` is the only file that would change.

---

# Part 2: Company policy document (policy score + policy answers)

## 9. What it does
You upload a policy / handbook (like the Anon Inc. handbook). Two things happen:
1. **Every theme gets a policy score (0-100)** = how strongly the document says that kind of work must be prioritised
   (security breach, outage, failing payments = high; usability suggestions, feature requests = low). It is blended into the
   normal priority: `final score = (1 - w) x base score + w x policy score`, with `w = POLICY_WEIGHT` (default 0.25 = 25%).
2. **Aira can answer policy questions** ("What is the first-response target for a P1 ticket?") from the document's own words.

## 10. Where it is stored
| Table | Content |
|---|---|
| `policy_docs` | the uploaded document (name, hash, upload time, size) |
| `policy_sections` | the document split into sections by its headings (1., 9.1, 15.2 ... or `#` headings) |
| `policy_scores` | cached model judgement per theme: score, priority label (e.g. P1), section used, reason |
| `themes` (new columns) | `base_score` (score before policy), `policy_score`, `policy_level`, `policy_reason`; `score` is the final blended value |
| `rag_chunks` | the sections are also indexed here (kind `policy`), which is how Aira finds them |

## 11. Setup, step by step
1. Install the PDF reader once: `pip install -r requirements.txt` (adds `pypdf`).
2. Start the app: `streamlit run app.py` (the new tables and columns are created automatically).
3. Open **Admin Access**, find **Company policy document**, upload your file (PDF, TXT or MD) and click **Use this document**.
   You will see how many sections were read and how many themes were scored (1 API call per 15 new themes).
4. The table below the buttons shows, per theme, base score, policy score, level, final score and the reason (with the section number).
5. Open **Feedback Tracker**: each task shows its *Policy fit* line, and the ranking already reflects it. **All Extracted Requests** has a *Policy score* column.
6. Ask Aira: *"What does our policy say about P0 response times?"* or *"Why is Support SSO ranked first?"*.

Tips for a good document: real text PDFs (scanned images need OCR first) and numbered or `#` headings, so sections split cleanly.
Open "Sections read from the document" in the panel to check the split.

## 12. When scores are (re)calculated
- On upload, and on every **Run analysis** (themes are rebuilt then). A theme already scored for this document costs nothing.
- A theme is scored again only if its name, description or main category changes, or the document changes.
- Deleting a feedback entry never calls the model (cached scores only).
- **Re-score themes** forces a fresh judgement; **Remove document** puts every score back to the base value.
- If the policy has no rule for a theme, it gets **no** policy influence (shown as "no policy rule found").
- If the model call fails or the daily cap is reached, themes simply keep their base score.

## 13. Settings (`.env`, optional)
| Setting | Default | Meaning |
|---|---|---|
| `POLICY_WEIGHT` | 0.25 | share of the final score from the policy (0 = ignore, 1 = policy only) |
| `POLICY_FULL_CONTEXT_CHARS` | 60000 | documents up to this size are sent whole when scoring; bigger ones send only the best-matching sections |
| `POLICY_MAX_CALLS` | 3 | max scoring calls per analysis (15 themes each) |
| `AIRA_RAG_POLICY_K` | 4 | policy sections Aira retrieves per question |

## 14. Good to know
- One policy document is active at a time; uploading a new one replaces it and discards old scores.
- The policy score is the model's reading of your document, shown with its reason and section so an admin can check it.
- Aira answers policy questions only from the uploaded text; if the document does not say it, she tells you to try another command.
- **Clients can ask Aira about the policy too** (Feedback Tracker and Home). Only upload a document that is fine for clients to read:
  its text can be quoted back to them. (Your sample handbook is marked "Internal", so check this first.)
- Aira's client version never shows who raised a request or the original feedback text; the admin version does.

