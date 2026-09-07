# -*- coding: utf-8 -*-
import io, ast, shutil, subprocess
SRC = "/home/woori/scratch/repo_rep2/scripts/distill/tau2/t2_gate_patch.py"
P   = "/home/woori/scratch/repo_rep3/scripts/distill/tau2/t2_gate_patch.py"
shutil.copyfile(SRC, P)
ast.parse(io.open(P, encoding="utf-8").read())
print("복구 OK (rep2 원본에서)")

s = io.open(P, encoding="utf-8").read()
ANCHOR = "    _g = next((g for g in _gs if g not in _done and g not in _degen), None)"
assert s.count(ANCHOR) == 1, "앵커 %d개" % s.count(ANCHOR)

NEW = "\n".join([
 "    # LB7 재료 배달 (2026-09-08 · x829 격리): 퇴화 축에서 침묵 대신 선언된 문서 제목을 준다.",
 "    #   x829: 재료 없음 0/8 · 우리 군 이름 0/8(8회 전부 같은 틀린 군) · 그 군 제목 47건 8/8.",
 "    #   => 결손은 능력이 아니라 재료다([[78]]). 종전에는 여기서 빈 문자열을 돌려주고 끝났다.",
 "    #   출처: id 는 A3 doc_index[군][계열] 에 이미 list 로 선언돼 있고, 제목은 env(KB json title).",
 "    #   엔진은 나열만 한다 - 순위·최댓값·'정답은 X' 0, 고르는 것은 모델 몫([[62]] ③④·[[10]]).",
 "    #   [[70]] 파는 것 = 컨텍스트(제목 N줄). LB6(부하 축소)과 짝으로만 판정한다.",
 "    if _skip_degen and not [g for g in _gs if g not in _done and g not in _degen]:",
 "        _nl7 = chr(10)",
 "        _idx7 = (_po or {}).get('doc_index') or {}",
 "        _ids7 = []",
 "        for _g7 in _skip_degen:",
 "            for _s7, _dl7 in (_idx7.get(_g7) or {}).items():",
 "                _ids7 += [d for d in (_dl7 or ()) if d]",
 "        _dd7 = os.environ.get('T2_KB_DOCS_DIR') or ''",
 "        _ln7 = []",
 "        for _d7 in _ids7:",
 "            _t7 = ''",
 "            if _dd7:",
 "                try:",
 "                    with open(os.path.join(_dd7, _d7 + '.json'), encoding='utf-8') as _fh7:",
 "                        _t7 = str(json.load(_fh7).get('title') or '')",
 "                except Exception:",
 "                    _t7 = ''",
 "            _ln7.append(('%s  %s' % (_d7, _t7)) if _t7 else _d7)",
 "        if _ln7:",
 "            print('[T2_DEGEN_TITLES] deliver group=%s docs=%d titled=%d'",
 "                  % (','.join(_skip_degen), len(_ln7), sum(1 for x in _ln7 if '  ' in x)),",
 "                  file=sys.stderr, flush=True)",
 "            return ('[KB INDEX] These are all knowledge-base documents declared for this area, '",
 "                    'with their titles. None of them has been read in this conversation. Pick '",
 "                    'the one that answers your current question and read it before you decide.'",
 "                    + _nl7 + _nl7.join(_ln7))",
 ANCHOR])

s = s.replace(ANCHOR, NEW, 1)
io.open(P, "w", encoding="utf-8").write(s)
ast.parse(io.open(P, encoding="utf-8").read())
print("LB7 패치 적용 + 문법 OK")
print("삽입 확인:", subprocess.run(["grep","-c","T2_DEGEN_TITLES",P],capture_output=True,text=True).stdout.strip(), "회")
print("rep2 와의 diff 줄수:", subprocess.run(["bash","-c","diff %s %s | wc -l"%(SRC,P)],capture_output=True,text=True).stdout.strip())
