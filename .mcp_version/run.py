"""
Entry point that loops through classes.json and runs the MCP agent for each
(main_type, sub_type) combination, passing the full methods dict as the prompt.
Checkpointing allows resuming from the last completed task.
"""

import asyncio
import json
from pathlib import Path

import dotenv
dotenv.load_dotenv()

import weave
from client import run_agent, WEAVE_PROJECT

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

CHECKPOINT_FILE = Path(__file__).parent / "checkpoint.json"
CLASSES_FILE = Path(__file__).parent.parent / "classes.json"


def load_checkpoint() -> set:
    try:
        return set(json.loads(CHECKPOINT_FILE.read_text()))
    except (FileNotFoundError, json.JSONDecodeError):
        return set()


def save_checkpoint(completed: set) -> None:
    CHECKPOINT_FILE.write_text(json.dumps(sorted(completed), indent=2))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    weave.init(WEAVE_PROJECT)

    classes = json.loads(CLASSES_FILE.read_text())
    completed = load_checkpoint()
    print(f"[CHECKPOINT] {len(completed)} tasks already completed, resuming.")

    for main_type in classes:
        for sub_type in classes[main_type]:
            key = f"{main_type}:{sub_type}"
            if key in completed:
                print(f"[SKIP] {key}")
                continue

            prompt = (
                f"main_type: {main_type}, "
                f"sub_type: {sub_type} - {json.dumps(classes[main_type][sub_type], ensure_ascii=False)}"
            )
            print(f"\n[RUN] {key}")
            asyncio.run(run_agent(prompt))
            completed.add(key)
            save_checkpoint(completed)
            print(f"[CHECKPOINT] Saved: {key}")


if __name__ == "__main__":
    main()
