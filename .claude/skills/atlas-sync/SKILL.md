---
name: atlas-sync
description: Sync the Survey Visual Atlas with the survey's reference list and the Drive notes folder, then build pages for newly added notes. Use when Faris says new notes were added, the survey changed, asks to update/refresh/sync the Atlas, or asks to build the new pages.
---

# Sync the Atlas, then build new pages

Since 2026-10-30 the **survey PDF is the source of truth** for which papers the Atlas lists (all of its references,
numbered as in the survey, organized by its Fig. 5 taxonomy). The Papers Master sheet is no longer read: its old
summaries are frozen in `data/legacy_sheet.json`. Drive is read-only. Never edit the notes. Never guess a mapping.

## 1. If the survey changed: rebuild `data/survey.json`
`python scripts/build_survey.py --pdf <NonConvex_Survey_Final.pdf>` (asks Faris for the file if it is not in Downloads).
It parses the reference list, where each reference is cited, and checks the Fig. 5 transcription in the script
against the PDF (update `TAXONOMY` there if the figure changed). Report any survey mistakes you notice.

## 2. Find new notes
List the Notes folder: `search_files` with `parentId = '1XmjqySjmFT-BPS04Uti5m6-ibsx0makK'` (≈5 results per page,
paginate with `pageToken`). Compare titles against `data/notes_map.json` (`file` field). File names follow no single
pattern ("Ref71_Carmon_Duchi_2019.pdf" uses an old working number, "41_fang2019sharp.pdf" another): read each new
notes PDF's title page to identify the paper, then find its **survey reference number** in `data/survey.json`.

## 3. Download and map
Download new notes **one or two at a time** (`download_file_content`; parallel downloads can land in the same
timestamped tool-results file). Decode: `python scripts/decode_mcp.py <tool-results dir> .cache/notes` and check every
requested title came out. Files over 10 MB cannot be downloaded through the connector.
For each new paper add by hand:
- `data/atlas_refs.json`: `"<survey #>": {"key": "<surname><year><firstword>", "link": <official page>, "pdf": <official pdf>}`
- `data/notes_map.json`: `"<key>": {"file": "<exact Drive title>", "drive_id": "<id>"}` (+ the id in `data/drive_notes_listing.json`)

## 4. Run the sync
`python scripts/sync.py --notes-dir .cache/notes`
It copies mapped notes to `public/notes/<key>.pdf`, lists UNMAPPED files, writes `data/papers.json` (all survey
references) and prints **"NEW WORK — notes but no visual yet"**.

## 5. Build each new page
For every key in that list run the **paper-visuals** skill. Read the original paper (official PDF) and the notes
fully; trust the notes' math but verify every claim against the paper; write the verified findings (with theorem
numbers, the survey number, and anything the survey or notes get wrong) into `research/findings.md`.
Then run the **study-notes** skill for the same key, so its notes appear in the page (`papers/<key>/notes.md`) instead
of the PDF viewer.
For a large batch (more than ~10 pages) split it across parallel sessions, one git worktree and branch each, with a
shared brief like `research/round3/BRIEF.md` (own port, own findings/links files, no shared files touched); then
merge the branches, fold the findings and links in, and do step 6 once on main.

## 6. Thumbnails, resync, check, ship
- `python scripts/serve.py` (or preview "survey-atlas"), open `http://localhost:8570/thumbs.html?only=<k1>,<k2>`, wait for "done".
- `python scripts/sync.py` again (picks up visual flags and thumbnails).
- Audit every tab of each new page on localhost: `window.__atlas.tick(n)`, `window.__atlasErrors` empty, readouts match the
  check script; check the page at 375 px for horizontal overflow.
- `python scripts/check-math/run_all.py` must pass.
- `git add -A && git commit` (which papers were added) and `git push`; confirm the live page loads.
- Tell Faris what was added, anything unmapped, and anything a check could not confirm or that contradicts the survey.
