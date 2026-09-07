import json,glob,os
KB="/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/documents"
ENUM=["fraud_or_security_concern","account_closure_request","deceased_account_holder",
"legal_or_regulatory_matter","account_ownership_dispute","complex_billing_dispute",
"abusive_customer_behavior","third_party_inquiry","technical_system_error",
"unconfirmed_external_communication","customer_demands_after_unavailable_offer_refusal",
"kb_search_unsuccessful_customer_requests_transfer","specialized_department_required",
"accessibility_or_special_needs","customer_frustrated_demands_human",
"supervisor_request_service_complaint","customer_requests_human_no_specific_reason",
"request_completed_customer_wants_human_followup","other"]
hits=[]
for p in sorted(glob.glob(KB+"/*.json")):
    c=open(p,encoding="utf-8",errors="ignore").read()
    n=sum(1 for e in ENUM if e!="other" and e in c)
    if n>=2: hits.append((os.path.basename(p),n,json.load(open(p)).get("title")))
print("표적 문서 집합 (enum 코드 2개 이상 포함) = %d건"%len(hits))
for f,n,t in hits: print("   %-52s 코드%2d  %s"%(f,n,t))
json.dump([f for f,_,_ in hits],open("/home/woori/scratch/x829_targets.json","w"))
