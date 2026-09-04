import gzip, json, sys, re
p = sys.argv[1]; pat = sys.argv[2]; span=int(sys.argv[3]) if len(sys.argv)>3 else 400
with gzip.open(p, 'rt', encoding='utf-8', errors='replace') as f:
    raw = f.read()
idxs=[m.start() for m in re.finditer(re.escape(pat), raw)]
print("HITS", len(idxs), "in", p)
seen=set()
for i in idxs:
    s=raw[max(0,i-span):i+span]
    k=s[:120]
    if k in seen: continue
    seen.add(k)
    print("~~~~"); print(s.replace("\n","\n"))
