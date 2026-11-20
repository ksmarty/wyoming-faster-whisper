#!/usr/bin/env python3
"""Keep the fork's only divergence in .github/workflows/publish.yml.

Upstream owns that workflow and keeps changing it, so the fork does not edit it
by hand: every job gets an `if:` guard naming the upstream repo, which makes the
whole workflow a no-op in the fork (it has no DOCKERHUB_* secrets and publishes
its own image from build-gpu.yml instead). The Sync Upstream workflow takes
upstream's copy of the file and then runs this script, so the guard survives
upstream edits instead of turning every sync into a merge conflict.
"""

from pathlib import Path

WORKFLOW = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "publish.yml"
GUARD = "if: github.repository_owner == 'OHF-Voice'"
NOTE = "    # Fork policy: upstream only; re-applied by script/fork_policy.py."


def is_job_key(line: str) -> bool:
    """True for a `  job-id:` line (a mapping key directly under `jobs:`)."""
    if not line.startswith("  ") or line.startswith("   ") or not line.endswith(":"):
        return False
    name = line.strip()[:-1]
    return bool(name) and name.replace("-", "").replace("_", "").isalnum()


def main() -> None:
    lines = WORKFLOW.read_text(encoding="utf-8").splitlines()
    try:
        jobs_at = lines.index("jobs:")
    except ValueError:
        raise SystemExit(f"{WORKFLOW}: no top-level `jobs:` key")

    job_keys: list[int] = []
    for index in range(jobs_at + 1, len(lines)):
        line = lines[index]
        if line.strip() and not line.startswith(" "):
            break  # next top-level key; jobs: ended
        if is_job_key(line):
            job_keys.append(index)

    added = 0
    # Insert bottom-up so earlier indices stay valid.
    for index in reversed(job_keys):
        # The guard goes above the job's own comments, so a re-run finds it in
        # the first line of actual config and leaves the file alone.
        nxt = index + 1
        while nxt < len(lines) and (not lines[nxt].strip() or lines[nxt].lstrip().startswith("#")):
            nxt += 1
        if nxt < len(lines) and GUARD in lines[nxt]:
            continue

        if nxt < len(lines) and lines[nxt].lstrip().startswith("if:"):
            # Upstream added its own condition: ours replaces it, because two
            # `if:` keys in one mapping is invalid YAML and this job must not
            # run in the fork at all.
            lines[nxt : nxt + 1] = [NOTE, "    " + GUARD]
        else:
            lines[index + 1 : index + 1] = [NOTE, "    " + GUARD]
        added += 1

    WORKFLOW.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"publish.yml: {added} job guard(s) added")


if __name__ == "__main__":
    main()
