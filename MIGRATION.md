# Migrating CAS to Claude Code

Goal: move from "Claude in a chat window + zip files" to Claude Code working
directly in your repo, with zero knowledge gaps. The trick is that **the
knowledge has to live in the repo**, because Claude Code starts every session
fresh and only knows what it can read from your files.

This folder already contains `CLAUDE.md` — the single most important file.
Claude Code reads it automatically at the start of every session. It captures
the decisions, constraints, and gotchas that aren't visible in the code itself
(why query_expansion, why caching is off, the Amazon schema quirks, the Gemini
quota trap, etc.).

## One-time setup (~10 minutes)

### 1. Install Claude Code
```bash
npm install -g @anthropic-ai/claude-code
# or follow https://code.claude.com for the latest installer
```
Requires Node.js. Verify: `claude --version`.

### 2. Put the project under version control
From your project root (`C:\Users\AnnamneediSaiSiddard\Documents\Context-Aware-Search`):
```bash
git init
git add .
git commit -m "Baseline: CAS v3 before Claude Code migration"
```
Git isn't strictly required, but it makes Claude Code far safer — you can review
diffs and roll back. Strongly recommended.

### 3. Drop the new files in
Copy from this zip into your project root, preserving paths:
- `CLAUDE.md`            → project root (NEXT TO app/, eval/, etc.)
- `MIGRATION.md`         → project root (this file, for your reference)
- `.claude/rules/*.md`   → optional modular rules (see below)

Your tree should look like:
```
Context-Aware-Search/
├── CLAUDE.md            ← Claude Code reads this every session
├── MIGRATION.md
├── .claude/
│   └── rules/
│       ├── safety.md
│       └── eval.md
├── app/
├── eval/
├── scripts/
├── data/
├── ui.py
├── requirements.txt
└── .env                 ← your real keys (gitignored, NOT in the zip)
```

### 4. Make sure .env exists locally
The zip never contains your real keys. Confirm your project root still has a
`.env` with `GOOGLE_API_KEY=...` (and any backups). `.gitignore` already
excludes it.

### 5. Launch and verify
```bash
cd Context-Aware-Search
claude
```
Then, in the Claude Code session, run:
```
/memory
```
This shows what Claude has loaded. You should see `CLAUDE.md` listed. Ask it a
test question that only CLAUDE.md would answer, e.g.:

> "What translator mode are we using and why?"

If it answers "query_expansion, because it beat HyDE and hybrid in the
2026-06-10 benchmark," the knowledge transfer worked.

## What NOT to do
- **Don't paste our whole chat history into Claude Code.** It's noisy and burns
  context. Everything load-bearing is already distilled into `CLAUDE.md`.
- **Don't bloat CLAUDE.md.** The practical limit is ~120 lines of high-signal
  content. If you add to it, ask of each line: "would removing this cause a
  mistake?" If not, cut it. A bloated file makes Claude ignore rules.
- **Don't document what the code already shows.** Claude reads the code. Only
  capture the WHY and the things that aren't in the code (decisions, external
  constraints, gotchas).

## Recommended first session in Claude Code

1. `/init` — optional. It generates its own CLAUDE.md from the codebase. Since
   you already have a curated one, you can skip this OR run it and merge any
   useful command/structure details it finds. Don't let it overwrite the
   decision/gotcha sections.
2. Ask: "Read CLAUDE.md and the eval/ directory, then summarize the current
   state of the project and the open work." This warms up its understanding and
   lets you confirm there are no gaps.
3. Start on the next real task (e.g. the P2 diversity/dedup work).

## Keeping memory fresh over time
- When you make a new architectural decision, add a one-line entry under
  "Decisions already made" in CLAUDE.md. That's how the next session inherits it.
- Claude Code also keeps its own auto-memory of things it learns
  (`/memory` to view). You don't need to duplicate those in CLAUDE.md.
- Review CLAUDE.md every couple of weeks; prune stale lines.

## Modular rules (.claude/rules/) — optional
For rules that are long or only relevant sometimes, put them in
`.claude/rules/*.md` instead of bloating CLAUDE.md. Two starters are included:
- `safety.md` — child-safety / content rules if the search ever faces end users.
- `eval.md` — the full eval methodology and metric-interpretation notes.
Claude Code loads these on demand.

## If something feels "forgotten" mid-project
Run `/memory` first. The rule may be in a nested file not yet loaded, or
CLAUDE.md may have grown too long and a rule is getting dropped. Prune and
re-test by observing whether behaviour actually changes.
