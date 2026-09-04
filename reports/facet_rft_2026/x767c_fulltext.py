import gzip, json, sys, re
p = sys.argv[1]; pat = sys.argv[2]
with gzip.open(p, 'rt', encoding='utf-8') as f:
    raw = f.read()
print("BYTES", len(raw))
idxs = [m.start() for m in re.finditer(re.escape(pat), raw)]
print("HITS", len(idxs))
for i in idxs[:40]:
    print("....", raw[max(0,i-300):i+400].replace("\n","\n"))
    print("~~~~~~~~~~~~~~~~")
