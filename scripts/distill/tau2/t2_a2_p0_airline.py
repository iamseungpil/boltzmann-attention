#!/usr/bin/env python
"""P-A2-0: frontier(Fable-5) 컴파일 airline GATE_SPEC의 replay 검증 (A2_FRONTEND_DISTILL §6).

airline gold actions 전수를 spec의 결정론 게이트(G3/G4/G5/G6 — db_check 의미를 본 검증기가
구현)로 replay — **over-deny=0**이 frontier-컴파일 정확도의 1차 통화 (Guard-2 동형).
G1/G2는 대화-전용이라 replay 불가(retail과 동일 한계 — observed-proxy 주의).

Run: cd /home/woori/scratch/tau2-bench && PYTHONPATH=src \
  /home/woori/venvs/seka_env/bin/python $REPO/scripts/distill/tau2/t2_a2_p0_airline.py
"""
from datetime import datetime, timedelta

NOW = datetime.fromisoformat("2024-05-15T15:00:00")
FLOWN = {"flying", "landed"}


def seg_status(db, seg):
    f = db.flights.get(seg.flight_number)
    if not f:
        return None
    d = f.dates.get(seg.date)
    return getattr(d, "status", None) if d else None


def main():
    from tau2.domains.airline.environment import get_environment, get_tasks
    env = get_environment()
    db = env.tools.db
    tasks = get_tasks(None)

    stats = {g: [0, []] for g in ("G3", "G4", "G5", "G6")}  # [checked, over_deny list]

    def deny(g, tid, name, why):
        stats[g][1].append((tid, name, why))

    for t in tasks:
        actions = (t.evaluation_criteria.actions if t.evaluation_criteria else None) or []
        # GT 유저: user_id 인자 + reservation owner 합의
        gt = set()
        for a in actions:
            args = a.arguments or {}
            if args.get("user_id"):
                gt.add(args["user_id"])
            if args.get("reservation_id"):
                r = db.reservations.get(args["reservation_id"])
                if r:
                    gt.add(r.user_id)
        gt_user = gt.pop() if len(gt) == 1 else None

        # ★상태-추적 replay: 선행 gold WRITE의 효과(cabin/flights 변경)를 적용하며 진행
        # (task 7·32 교훈: cabin 업그레이드 후 취소/추가 변경은 새 상태 기준 적격 — static-DB 평가는 가짜 over-deny)
        st = {}  # rid -> {"cabin": str, "flights": set[(fn,date)]}

        def cur(res):
            s = st.get(res.reservation_id, {})
            cab = s.get("cabin", str(res.cabin))
            fls = s.get("flights", {(x.flight_number, x.date) for x in res.flights})
            return cab, fls

        for a in actions:
            args = a.arguments or {}
            rid = args.get("reservation_id")
            res = db.reservations.get(rid) if rid else None

            if a.name in ("cancel_reservation", "update_reservation_baggages",
                          "update_reservation_flights", "update_reservation_passengers"):
                if res and gt_user:
                    stats["G3"][0] += 1
                    if res.user_id != gt_user:
                        deny("G3", t.id, a.name, f"owner {res.user_id} != gt {gt_user}")

            if a.name == "cancel_reservation" and res:
                stats["G4"][0] += 1
                cab, fls = cur(res)
                sts = [seg_status(db, type("S", (), {"flight_number": fn, "date": dt})())
                       for fn, dt in fls]
                flown = any(s in FLOWN for s in sts)
                try:
                    within24 = (NOW - datetime.fromisoformat(res.created_at)) < timedelta(hours=24)
                except (ValueError, TypeError):
                    within24 = False
                eligible = (not flown) and (
                    within24 or any(s == "cancelled" for s in sts)
                    or cab == "business" or str(res.insurance) == "yes")
                if not eligible:
                    deny("G4", t.id, a.name,
                         f"flown={flown} 24h={within24} cabin={cab} ins={res.insurance} sts={sts}")

            if a.name == "update_reservation_flights" and res:
                stats["G5"][0] += 1
                cab0, fls0 = cur(res)
                newf = {(f.get("flight_number"), f.get("date")) if isinstance(f, dict)
                        else (f.flight_number, f.date) for f in (args.get("flights") or [])}
                cab = args.get("cabin")
                sts = [seg_status(db, type("S", (), {"flight_number": fn, "date": dt})())
                       for fn, dt in fls0]
                if newf != fls0 and cab0 == "basic_economy":
                    deny("G5", t.id, a.name, "flight-change on basic_economy")
                if cab and cab != cab0 and any(s in FLOWN for s in sts):
                    deny("G5", t.id, a.name, "cabin-change with flown segment")
                # 상태 적용 (gold는 정책-적합 가정 — 효과 반영)
                st[rid] = {"cabin": cab or cab0, "flights": newf or fls0}
            if a.name == "update_reservation_passengers" and res:
                stats["G5"][0] += 1
                if len(args.get("passengers") or []) != len(res.passengers):
                    deny("G5", t.id, a.name,
                         f"count {len(args.get('passengers') or [])} != {len(res.passengers)}")
            if a.name == "update_reservation_baggages" and res:
                stats["G5"][0] += 1
                tb = args.get("total_baggages")
                if tb is not None and int(tb) < int(res.total_baggages):
                    deny("G5", t.id, a.name, f"baggage remove {res.total_baggages}->{tb}")

            if a.name in ("book_reservation", "update_reservation_flights"):
                uid = args.get("user_id") or gt_user
                user = db.users.get(uid) if uid else None
                pm = []
                if a.name == "book_reservation":
                    pm = [p.get("payment_id") if isinstance(p, dict) else p
                          for p in (args.get("payment_methods") or [])]
                elif args.get("payment_id"):
                    pm = [args["payment_id"]]
                if user and pm:
                    stats["G6"][0] += 1
                    srcs = []
                    for pid in pm:
                        m = user.payment_methods.get(pid)
                        if m is None:
                            deny("G6", t.id, a.name, f"{pid} not in profile")
                            srcs = None
                            break
                        srcs.append(str(m.source))
                    if srcs is not None:
                        if (srcs.count("certificate") > 1 or srcs.count("credit_card") > 1
                                or srcs.count("gift_card") > 3):
                            deny("G6", t.id, a.name, f"composition {srcs}")
                        if a.name == "update_reservation_flights" and srcs and \
                                srcs[0] not in ("gift_card", "credit_card"):
                            deny("G6", t.id, a.name, f"flight-change pay via {srcs[0]}")

    total_over = 0
    for g, (n, dl) in stats.items():
        print(f"{g}: checked={n} OVER_DENY={len(dl)}")
        for row in dl[:6]:
            print("   OVER:", row)
        total_over += len(dl)
    print(f"[P-A2-0 verdict] tasks={len(tasks)} total OVER_DENY={total_over} "
          f"({'PASS - frontier compile replay-clean' if total_over == 0 else 'inspect'})")


if __name__ == "__main__":
    main()
