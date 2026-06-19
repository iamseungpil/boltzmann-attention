import re
p=r"C:\Users\승원\.claude\projects\C--workspace\e085d28a-9d78-42e5-ad4a-4082c2e0a832\tool-results\webfetch-1781883339945-pi6jus.pdf"
txt=""
err=""
try:
    from pypdf import PdfReader
    r=PdfReader(p)
    for pg in r.pages:
        txt+=pg.extract_text() or ""
except Exception as e:
    err+="pypdf:"+str(e)+"\n"
    try:
        import fitz
        d=fitz.open(p)
        for pg in d:
            txt+=pg.get_text()
    except Exception as e2:
        err+="fitz:"+str(e2)+"\n"
out=open(r"C:\workspace\ba-frft\_pdf_report.txt","w",encoding="utf-8")
out.write("ERR:"+err+"\nLEN:"+str(len(txt))+"\n")
for kw in ["o4-mini","o4 mini","o1","o3-mini","o3 ","reasoning","deepseek-r1","DeepSeek-R1","superior","SOP","procedure"]:
    idxs=[m.start() for m in re.finditer(re.escape(kw), txt, re.IGNORECASE)]
    out.write(f"KW {kw!r}: {len(idxs)}\n")
# dump context around 'reasoning' and 'o4'
for kw in ["o4-mini","reasoning model","superior"]:
    for m in re.finditer(re.escape(kw), txt, re.IGNORECASE):
        s=max(0,m.start()-300); e=min(len(txt),m.start()+300)
        out.write("\n=== ctx "+kw+" ===\n"+txt[s:e].replace("\n"," ")+"\n")
out.close()
