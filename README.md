# VOC Enhancement Tracker (local MVP, Azure OpenAI only)

Feedback -> Azure OpenAI v1 API (LangChain) extraction -> embeddings -> clustering -> priority ranking.
Streamlit + SQLite. Run: `streamlit run app.py`. Full steps: see SETUP_STEPS.txt.

## Credentials (.env)
    AZURE_OPENAI_ENDPOINT=...
    AZURE_OPENAI_DEPLOYMENT_NAME=...
    AZURE_OPENAI_API_KEY=...
Optional: AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME (chat models can't create embeddings; if empty, local TF-IDF is used).

## Protection against accidental API exhaustion
- Hard caps checked BEFORE every call: per run (MAX_CALLS_PER_RUN) and per day (MAX_CALLS_PER_DAY, stored in voc.db; "Reset data" does not clear it).
- Sequential processing, max N items per click, no automatic retry loops (SDK retries limited to 1).
- Circuit breaker: stops after 3 failures in a row; stops instantly on auth / wrong deployment / rate-limit / bad-request errors.
- Poison-item parking: an item failing 3 times is skipped until you click "Retry failed items".
- Duplicate feedback is rejected; input text is truncated; output tokens capped.
- Theme names are cached (re-running doesn't re-pay), single-request themes cost no call.
- Everything is logged in the terminal; sidebar shows calls used today.

## Files
db.py (SQLite + usage log) · llm.py (Azure/LangChain + budget guard + prompts) · pipeline.py (run, cluster, rank) · app.py (UI)

## Who raised it (roles)
Each entry has a raiser: user, client, developer or tester + a number (client2, tester1...). Manual entry has
a role dropdown and number box; CSV uses a `user` column (columns: source,user,text).
Role weights scale both volume and urgency in the theme score (defaults client x1.5, user x1.0, tester x0.8,
developer x0.7; edit ROLE_WEIGHTS in .env). The same text from different people is kept (extra evidence);
the same text from the same person is rejected as a duplicate.

AUTO NUMBERING
  Manual: leave "No." blank and the next free number for that role is used (client1, client2 -> client3).
  Type a number to reuse a person (the page shows how many entries that person already has).
  CSV: "client2" = that person; "client" alone = next free number (explicit numbers in the file are kept as-is).
  Numbers come from the entries currently stored, so deleting every entry of the highest number frees it.

DATE / TIME, AGE AND REPEAT BOOSTS
  Every entry has a date and time. Manual: defaults to now; tick "Set date and time manually" to choose one.
  CSV: a date column (date / datetime / date and time / timestamp; e.g. 2026-10-01 14:30) is used as the
  entry's date, NOT the upload time. Blank = now. Re-importing a CSV updates dates of matching entries.
  Urgency % = model urgency x role weight x (1 + age boost + repeat boost), capped at 100%.
    Age boost:    +5% per 10 days since the entry's date, max +30% (reached at 60 days).
    Repeat boost: +5% for each extra request on the same theme (any people), max +20% (reached at 5 requests).
  Theme score = 40% weighted volume + 40% final urgency + 20% negative share.
  Each theme shows a one-sentence "Why" (no API calls).

## Aira (chat assistant)
Red/white bubble at the bottom-right of every page (code: aira.py, one call added at the end of app.py).
Aira reads your ranked themes (the task list), who raised them, urgency, age, sentiment and unanalyzed feedback, and
tells you what to do first and why. Uses the same Azure OpenAI deployment and the same budget guard (1 API call per
question, counted in "API calls today"). If the data cannot answer a question she says so and suggests another command.
No call is made when the tracker is empty. Optional AIRA_* settings: see .env.example.

## Pages (Home / Feedback Tracker / Admin Access)
Files: ui.py (router, animated home + navigation, optional admin gate) · results.py (saved-results display shared by both
pages) · app.py (admin_page, client_page, router). The page is kept in the URL (?view=client / ?view=admin; the old ?view=user
still works), so a refresh stays put.
- Home: title + two options. Click the title in the top bar to come back.
- Feedback Tracker: loads the themes/requests already saved in voc.db and lets a client approve changes; never calls the AI. Shows
  separate loading, empty ("No results generated. Please upload data.") and error states.
- Admin Access: everything that was on the single page before (forms, CSV import, analysis, entries, results) + sidebar.
  Optional ADMIN_PASSWORD in .env puts a password in front of it.
- Aira is on all three pages (see "Aira audiences" below).

## Task workflow (Pending -> Approved -> Completed)
Stored in the `requests` table (columns status, approved_at, completed_at; added automatically on start-up). A database from the
previous workflow is converted once, automatically: old "Completed" -> Approved (its time becomes the approval time) and old
"Verified" -> Completed (its time becomes the completion time). Themes are rebuilt on every analysis, so the status lives on the
requests, and the buttons act on request ids.
- Feedback Tracker: under each task a **priority slider** (0-100) starts at the model's suggested priority. Unchanged -> **Approved**
  works directly. Changed -> click **Save Priority** first; Approved stays disabled until the latest value is saved (a saved value
  becomes the new "unchanged" value). Approving sets status Approved, saves "Task Approved At" and the priorities, and the task
  leaves the client's list. The server re-checks this rule, so a stale page cannot approve an unsaved value.
  The slider and Save Priority are a Streamlit fragment: they update only their own control (no full page refresh), and the slider
  keeps its value after visiting other pages. The task list and chart are ranked by the client's saved priority (otherwise the
  model's), so a new ranking shows at the next page load or after approving.
- Priorities are stored separately on each request: `model_priority` (filled once, never overwritten) and `client_priority`
  (latest saved value). Admin Access shows both in "Approved tasks waiting for completion" and in All Extracted Requests (open
  requests show the model's current suggestion until the client saves or approves).
- Admin Access cards: Feedback items, Analyzed, **Approved Items** (waiting for completion), **Completed Items**.
- Admin Access: section "Approved Items Waiting for Completion" (highest client priority first); table columns "Item Approved At" / "Item Completed At". with a **Complete** button -> status Completed, "Task Completed At"
  saved. Completed is final.
- Every change only applies to rows in the expected state, so double clicks / stale pages cannot create duplicates.
- The ranked-themes chart is shown on Feedback Tracker only; the All Extracted Requests table on Admin Access only.

## Landing-page charts (analytics.py)
Two charts on the Home page, read live from voc.db (needs `altair>=5`, listed in requirements.txt):
- Requests Raised (request creation date), Requests Approved (a client approved it, placed by its approval date) and Requests
  Resolved (status Completed, placed by its completion date), side by side. Colours: blue / gold / green.
- Requests by category (creation date), one fixed colour per category.
Shared controls: financial year (April-March, latest 10, newest first) and Monthly / Quarterly. Empty periods show 0.

## Theme (theme.py + .streamlit/config.toml)
EY-inspired light enterprise look. All colours live in theme.py (page #F8F8FC, cards #FFFFFF, borders #E5E7EB, text #111827 /
#374151, primary blue #2563EB, yellow #FFE600 accent for key KPIs). Category colours (UI/UX blue, Performance green,
Integration orange, Reporting purple, Mobile teal; Security red and Other gray added) and status colours (Pending gray,
Approved gold, Completed green) are used identically in charts, badges, table dots and KPI accents. Restart Streamlit after
editing .streamlit/config.toml.

## Aira with RAG (rag.py)
The database is Aira's knowledge base: themes, requests (with original feedback, status, dates) and unanalyzed feedback are
indexed as chunks in the `rag_chunks` table, searched by keywords (local TF-IDF) plus optional embeddings, and sent to the
model with a ranked overview. Admin sidebar: "AI knowledge index" (Rebuild, download as Markdown). Full guide: RAG_SETUP.md.

## Company policy document (policy.py)
Admin Access: upload a policy/handbook (PDF/TXT/MD). Every theme gets a policy score (0-100) that is blended into its priority
(POLICY_WEIGHT, default 25%); base score is kept in themes.base_score. Sections are stored in policy_sections and indexed for
Aira, who answers policy questions from them. Needs `pypdf`. Details: RAG_SETUP.md part 2.

## Aira audiences
Aira serves two audiences, chosen by the page: **client** (Home and Feedback Tracker) and **admin** (Admin Access, and only once it is
unlocked when ADMIN_PASSWORD is set). Clients get a warm, plain-language Aira with their own shortcuts ("Approve first?", "What
happens next?", "Our policy"), friendly error messages without technical details, and a reduced view of the data: no names of who
raised a request, no original feedback text, no unanalyzed feedback, no admin counters or score formulas. Both audiences can
ask policy questions; clients see policy answers only from the uploaded policy document. Chats are cleared when the audience
changes. Note: anything in an uploaded policy document can be asked about by clients, so upload only text meant for them.

## Export Report (Feedback Tracker)
At the bottom of the Feedback Tracker, **Export Report** downloads an Excel file (needs `openpyxl`, listed in requirements.txt) with the
pending items exactly as listed, in the order shown (approved and completed items are excluded). Columns: Priority Number, Priority
Score (the client's saved priority where set, otherwise the suggested one), Item Heading / Description, Request Count, Why,
Policy Fit & Explanation. Policy explanations are shown in plain words (no P0-P4 codes, no section sign).

