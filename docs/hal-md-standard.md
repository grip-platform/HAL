# HAL.md — Polymorphic Instruction File Standard

## Problem

Every agentic harness has its own instruction file:
- Claude Code: `CLAUDE.md`
- Cursor: `.cursorrules`
- GitHub Copilot: `.github/copilot-instructions.md`
- Windsurf: `.windsurfrules`
- Codex: `AGENTS.md`

Projects that work across harnesses need multiple files with duplicated content.

## Solution

HAL introduces `HAL.md` as a polymorphic instruction file. The format is identical
to CLAUDE.md (plain markdown with instructions). The filename signals ownership:

- `HAL.md` — HAL-specific instructions (read by HAL, ignored by others)
- `CLAUDE.md` — Claude Code instructions (read by CC, also read by HAL as fallback)
- `.cursorrules` — Cursor instructions (read by Cursor only)

HAL reads ALL instruction files it finds, with HAL.md taking priority.

## Resolution Chain

HAL's `context_assembler.py` resolves instructions in this order:

```
1. <project>/.hal/HAL.md          # Project-specific HAL instructions
2. <project>/.claude/CLAUDE.md    # Project-specific CC instructions
3. <project>/CLAUDE.md            # Legacy location
4. ~/.hal/HAL.md                  # Global HAL instructions
5. ~/.hal/CLAUDE.md               # Global fallback
```

All found files are concatenated into the system prompt. First match at each
level takes priority — so project-level overrides global-level.

## Format

HAL.md uses the same format as CLAUDE.md:

```markdown
# Project Instructions

## Rules
- Rule 1
- Rule 2

## Architecture
Description of project architecture.

## Commands
Custom slash commands or conventions.
```

No frontmatter required. No special syntax. Plain markdown.

## Adoption Path

1. **New projects**: Create `HAL.md` alongside or instead of `CLAUDE.md`
2. **Existing projects**: Keep `CLAUDE.md`, HAL reads it as fallback
3. **Multi-harness projects**: Keep both — HAL reads `HAL.md`, CC reads `CLAUDE.md`
4. **Migration**: Rename `CLAUDE.md` to `HAL.md` when ready

## For Other Harnesses

Any harness can adopt the pattern:
- Read your own file first (e.g., `.cursorrules`)
- Fall back to `CLAUDE.md` or `HAL.md`
- The format is always plain markdown

HAL.md is not a competing standard — it's a recognition that instruction
files are harness-specific but format-identical.

## Implementation

`~/.hal/lib/hal/context_assembler.py` lines 92-106:

```python
def _load_claude_md_chain(self) -> list[str]:
    """Load instructions from project -> user -> global. Supports CLAUDE.md and HAL.md."""
    paths = [
        os.path.join(self._project_dir, ".hal", "HAL.md"),
        os.path.join(self._project_dir, ".claude", "CLAUDE.md"),
        os.path.join(self._project_dir, "CLAUDE.md"),
        os.path.join(_HAL_HOME, "HAL.md"),
        os.path.join(_HAL_HOME, "CLAUDE.md"),
    ]
    chain = []
    for path in paths:
        content = _read_file(path)
        if content:
            chain.append(content)
    return chain
```
