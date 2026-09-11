#!/usr/bin/env python3
import json
from pathlib import Path
root = Path(__file__).resolve().parents[1] / "results"
rows = []
for p in sorted(root.glob("*/*.json")):
    # Legacy runs have no common protocol fingerprint. Do not count backups as cities.
    if p.name not in ("NYC.json", "TKY.json"):
        continue
    d = json.loads(p.read_text())
    t = d.get("test") or {}
    if not t:
        continue
    rows.append({
        "model": d.get("model", p.parent.name),
        "city": d.get("city", p.stem),
        "acc@1": t.get("acc@1"),
        "acc@5": t.get("acc@5"),
        "acc@10": t.get("acc@10"),
        "recall@5": t.get("recall@5"),
        "recall@10": t.get("recall@10"),
        "mrr": t.get("mrr"),
        "n": t.get("num_samples"),
        "protocol_status": "legacy_unverified_do_not_rank_across_protocols",
        "source": str(p.relative_to(root)),
    })
print(json.dumps(rows, indent=2))
out = root / "summary.json"
out.write_text(json.dumps(rows, indent=2))
print("wrote", out)
