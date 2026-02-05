import json
import uuid
from datetime import datetime

EXPERIMENT_LOG = "experiments.jsonl"


def log_experiment(
    system_prompt,
    user_prompt,
    full_prompt,
    temperature,
    response_format,
    model,
    response,
    token_count
):
    record = {
        "experiment_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "model": model,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "full_prompt": full_prompt,
        "temperature": temperature,
        "response_format": response_format,
        "token_count": token_count,
        "response": response
    }

    with open(EXPERIMENT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
