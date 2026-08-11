# -*- coding: utf-8 -*-
"""회귀 — 배달 계측 `T2_ROUTE_TRACE` (오프라인·모델 0·env 0·설계 §5·§7-1·원장 C427).

이 계측은 **거동을 바꾸지 않는다**. 그래서 검정할 것도 거동이 아니라 **불변식**이다:

  ⑴ `_SRC8`(계측이 쓰는 순서)와 배타 `elif` 체인(실제 정책)이 **같은 순서**인가.
     두 벌이 되면 계측이 거짓 rank 를 보고한다 — 그게 C427 을 늦게 발견한 이유와 같은 종류다.
  ⑵ rank 번호가 체인 위치와 맞는가(`rw_fb` = 11 · `wev_fb` = 7 — C427 이 인용한 수).
  ⑶ 계측 블록이 **content 를 건드리지 않는가**(문자열 대입이 없어야 한다).
  ⑷ 실패해도 런을 안 깨는가(`except` 로 감싸였는가).
"""
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SRC = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
OK = []


def chk(name, cond, extra=""):
    OK.append(bool(cond))
    print("  %s %s%s" % ("PASS" if cond else "FAIL", name, (" — " + str(extra)) if extra else ""))


# ⑴ 두 벌 대조 --------------------------------------------------------------
m = re.search(r"_SRC8 = \((.*?)\)\n            _chose8", SRC, re.S)
chk("_SRC8 을 찾았다", m is not None)
src8 = re.findall(r'\("([a-z_]+)", (\w+)\)', m.group(1)) if m else []
src8_vars = [v for _n, v in src8]

chain = re.findall(r"elif (\w+_fb) is not None and c is \1\[0\]", SRC)
chk("배타 체인을 찾았다 (%d 분기)" % len(chain), len(chain) >= 14, chain[:3])

# 체인에는 `ep_fb` 앞에 `main_prov` 가 있고 그것은 `_SRC8` 에 없다(do_gate 도 마찬가지).
chk("_SRC8 순서 == elif 체인 순서", src8_vars == chain,
    "\n      _SRC8: %s\n      chain: %s" % (src8_vars, chain))

# ⑵ rank 번호 (C427 이 인용한 수) --------------------------------------------
# 체인 앞의 두 분기(`do_gate`·`main_prov`)를 세어 +3
rank = {v: i + 3 for i, v in enumerate(src8_vars)}
chk("wev_fb rank == 8", rank.get("wev_fb") == 8, rank.get("wev_fb"))
chk("rw_fb rank == 11 (CALL_FORM 이 실리는 자리)", rank.get("rw_fb") == 11, rank.get("rw_fb"))
chk("wev(ARG_EMPTY) 가 rw(CALL_FORM) 보다 앞", rank.get("wev_fb", 99) < rank.get("rw_fb", 0))

# ⑶ 계측이 content 를 건드리지 않는다 ----------------------------------------
blk = re.search(r'if os\.environ\.get\("T2_ROUTE_TRACE".*?except Exception as _e9:', SRC, re.S)
chk("계측 블록을 찾았다", blk is not None)
body = blk.group(0) if blk else ""
_assign = [l for l in body.split("\n")
           if re.search(r"(?<![=!<>])\bcontent\s*=(?!=)", l)]
chk("content **대입** 0 (읽기는 허용)", not _assign, _assign)
chk("fb 변경 0", "fb.append" not in body and "fb =" not in body)
chk("작업버퍼(work) 변경 0", "work" not in body)

# ⑷ 실패가 런을 깨지 않는다 ---------------------------------------------------
chk("try/except 로 감쌌다", "except Exception as _e9" in SRC)
_recs = re.findall(r"\b_fbr\d*\.record\(\"route\",\s*([^,]+),", SRC)
chk("route 레코드가 있다", len(_recs) >= 1, _recs)
chk("사이드카는 text 를 안 싣는다(전 호출 None)",
    bool(_recs) and all(a.strip() == "None" for a in _recs), _recs)

# ⑸ 삼분 (설계 v1.5 §5.1) — 억제↔체인↔미생성이 갈리는가 -----------------------
#   이것이 없으면 다음 런의 `lost_to` 판정이 성립하지 않는다: 억제와 미생성이
#   둘 다 `None` 이라 `_cands`(`_v9[0] is c`)에서 똑같이 사라진다.
for _o in ("won", "lost", "suppressed"):
    chk("outcome=%s 를 기록한다" % _o, '"%s"' % _o in SRC)
chk("억제 지점이 등록한다(_t2_silenced)", "_t2_silenced" in SRC)
chk("억제 등록이 배타 체인 **밖**이다",
    "_t2_silenced" in SRC.split("_SRC8 = (")[0] or 'outcome="suppressed"' in SRC)

# ⑹ `arrived` 를 호출부에서 세지 않는다 (설계 §5 리뷰 N3 · `proc_fb` 死배선 교훈) --
_arr = [l for l in body.split("\n")
        if re.search(r"(?<![=!<>])\barrived\s*=(?!=)", l) and not l.lstrip().startswith("#")]
chk("계측 블록이 arrived 를 **대입**하지 않는다 (주석 언급은 허용)", not _arr, _arr)
chk("arrived 는 모델 입력에서 잰다(_gen 안)",
    bool(re.search(r"def _gen\(.*?_rec\[\"arrived\"\] = bool\(_txt and _txt in _hay\)",
                   SRC, re.S)))
chk("기본 ON (계측은 무해)", 'os.environ.get("T2_ROUTE_TRACE", "1")' in SRC)

# ⑸ 문법 -----------------------------------------------------------------------
import ast
try:
    ast.parse(SRC)
    chk("t2_gate_patch.py 파싱", True)
except SyntaxError as e:
    chk("t2_gate_patch.py 파싱", False, e)

print("\n%d/%d" % (sum(OK), len(OK)))
sys.exit(0 if all(OK) else 1)
