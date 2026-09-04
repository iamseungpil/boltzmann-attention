import gzip, json, sys
p = sys.argv[1]
with gzip.open(p, 'rt', encoding='utf-8') as f:
    d = json.load(f)
print(json.dumps(d.get("tasks"), ensure_ascii=False, indent=1)[:12000])
