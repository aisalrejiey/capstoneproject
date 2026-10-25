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

## Pages (Home / User Access / Admin Access)
Files: ui.py (router, animated home + navigation, optional admin gate) · results.py (saved-results display shared by both
pages) · app.py (admin_page, user_page, router). The page is kept in the URL (?view=user / ?view=admin), so a refresh stays put.
- Home: title + two options. Click the title in the top bar to come back.
- User Access: read-only. Loads the themes/requests already saved in voc.db; never calls the AI. Shows separate loading,
  empty ("No results generated. Please upload data.") and error states.
- Admin Access: everything that was on the single page before (forms, CSV import, analysis, entries, results) + sidebar.
  Optional ADMIN_PASSWORD in .env puts a password in front of it.
- Aira is on all three pages.

## Task workflow (Pending -> Completed -> Verified)
Stored in the `requests` table (columns status, completed_at, verified_at; added automatically on start-up, existing data is kept
as Pending). Themes are rebuilt on every analysis, so the status lives on the requests, and the buttons act on request ids.
- User Access: **Completed** under each task -> status Completed, "Task Submitted At" saved, task leaves the active list.
- Admin Access: section "Completed tasks" with **Verify** (-> Verified + "Task Verified At") and **Repost** (-> Pending again;
  submission time kept, no verification time, feedback creation date untouched).
- Every change only applies to rows in the expected state, so double clicks / stale pages cannot create duplicates.
- If a task is completed a second time after a Repost, "Task Submitted At" is set to that newer click.
- The ranked-themes chart is shown on User Access only. Landing page = dark cherry red, other pages white; Aira's bubble is
  white on the landing page and cherry red elsewhere.

