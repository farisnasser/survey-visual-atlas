---
name: atlas-sync
description: Sync the Survey Visual Atlas from Google Drive (Papers Master sheet + Notes PDFs) and build pages for newly added papers. Use when Faris says new papers or notes were added, asks to update/refresh/sync the Atlas, or asks to build the new pages.
---

# Sync the Atlas from Drive, then build new pages

Drive is read-only. Never edit the sheet or the notes. Never invent a paper or a mapping.

## 1. Export the sheet
`download_file_content` on the Papers Master (`1DzVJJDWxBM6I5YbaMa21Yr5rlE5duIX5dK-sWnmGS94`) with
`exportMimeType = application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`. The result is saved as a
`tool-results/...download_file_content-<ts>.txt` file. Decode it:
`python scripts/decode_mcp.py <tool-results dir> .cache/drive` then copy `.cache/drive/Papers Master` to `.cache/papers_master.xlsx`.

## 2. Find and download new notes
List the Notes folder: `search_files` with `parentId = '1XmjqySjmFT-BPS04Uti5m6-ibsx0makK'` (paginate with `pageToken`).
Compare against `data/notes_map.json`. Download only new or modified files (`modifiedTime`), **one or two at a time**:
parallel downloads can land in the same timestamped file and overwrite each other. Decode into `.cache/notes/` and check
every requested title came out. Files over 10 MB cannot be downloaded through the connector — get the paper from its
official URL (sheet "PDF" column) instead.

## 3. Run the sync
`python scripts/sync.py --xlsx .cache/papers_master.xlsx --notes-dir .cache/notes`
It prints each notes → Key match, any UNMATCHED file (fix by renaming in your copy or adding to `data/notes_map.json`
by hand — ask Faris if the right Key is unclear), and the list **"NEW WORK — notes but no visual yet"**.

## 4. Build each new page
For every Key in that list, run the **paper-visuals** skill. Read the paper PDF (download from the sheet's official
URL) and the notes PDF fully enough to be certain; trust the notes' math but verify every claim you put on the page
against the paper; record the verified findings in `research/findings.md`.

## 5. Thumbnails, resync, check, ship
- `python scripts/serve.py`, open `http://localhost:8570/thumbs.html?only=<key1>,<key2>` and wait for "done".
- `python scripts/sync.py --xlsx .cache/papers_master.xlsx` again (picks up visual flags and thumbnails).
- Audit each new page on localhost: every tab, `window.__atlas.tick(2000)`, no entries in `window.__atlasErrors`.
- `git add -A && git commit` (message: which papers were added) and `git push`. Check the live page loads.
- Tell Faris what was added, anything unmatched, and anything a check script could not confirm.
