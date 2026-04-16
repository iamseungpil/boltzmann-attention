#!/usr/bin/env python3
"""Build 3 variants of banking_knowledge ontology for domain rebalancing study.

Variant A: Split (7 fine-grained categories)
Variant B: Regroup (4 balanced semantic clusters, manual)
Variant C: Auto-cluster (K-means on docstring embeddings, k=4)
"""
import argparse
import json
from pathlib import Path


# ---- Load source classifications from existing ontology ----
SOURCE = "reports/tau2_2026_04_17/tau2_banking_knowledge_ontology.json"


# ---- Variant A: Split user into fine categories ----
VARIANT_A_DOMAIN = {
    # credit_card domain
    "apply_for_credit_card": "credit_card",
    "get_credit_card_accounts_by_user": "credit_card",
    "get_credit_card_transactions_by_user": "credit_card",
    # profile domain
    "change_user_email": "profile",
    "get_user_information_by_id": "profile",
    "get_user_information_by_name": "profile",
    "get_user_information_by_email": "profile",
    # verification
    "log_verification": "verification",
    # referral
    "get_referrals_by_user": "referral",
    "submit_referral": "referral",
    # transaction
    "submit_transaction": "transaction",
    # escalation
    "transfer_to_human_agents": "escalation",
    "request_human_agent_transfer": "escalation",
    # discoverable_agent
    "unlock_discoverable_agent_tool": "discoverable_agent",
    "call_discoverable_agent_tool": "discoverable_agent",
    "list_discoverable_agent_tools": "discoverable_agent",
    # discoverable_user
    "give_discoverable_user_tool": "discoverable_user",
    "call_discoverable_user_tool": "discoverable_user",
    "list_discoverable_user_tools": "discoverable_user",
    # utility
    "get_current_time": "utility",
}


# ---- Variant B: Regroup into 4 semantic clusters ----
VARIANT_B_DOMAIN = {
    # customer_service: profile management, verification, escalation
    "change_user_email":            "customer_service",
    "get_user_information_by_id":   "customer_service",
    "get_user_information_by_name": "customer_service",
    "get_user_information_by_email": "customer_service",
    "log_verification":             "customer_service",
    "transfer_to_human_agents":     "customer_service",
    "request_human_agent_transfer": "customer_service",
    # financial: credit card, transactions
    "apply_for_credit_card":                   "financial",
    "get_credit_card_accounts_by_user":        "financial",
    "get_credit_card_transactions_by_user":    "financial",
    "submit_transaction":                      "financial",
    # social: referrals
    "get_referrals_by_user": "social",
    "submit_referral":       "social",
    # agent_system: all discoverable tools
    "unlock_discoverable_agent_tool": "agent_system",
    "call_discoverable_agent_tool":   "agent_system",
    "list_discoverable_agent_tools":  "agent_system",
    "give_discoverable_user_tool":    "agent_system",
    "call_discoverable_user_tool":    "agent_system",
    "list_discoverable_user_tools":   "agent_system",
    # utility
    "get_current_time": "utility",
}


# ---- Variant C: Auto-cluster via embeddings ----
def variant_c_domain(classifications, k=4):
    """K-means on docstring embeddings using sentence-transformers."""
    try:
        from sklearn.cluster import KMeans
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("[warn] sklearn or sentence_transformers not available.")
        print("[warn] Falling back to simple K-means on TF-IDF")
        return variant_c_tfidf(classifications, k=k)

    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    names = [item["name"] for item in classifications]
    texts = [item["docstring"] for item in classifications]
    embeddings = encoder.encode(texts, convert_to_numpy=True)
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(embeddings)
    labels = km.labels_
    return {name: f"cluster_{label}" for name, label in zip(names, labels)}


def variant_c_tfidf(classifications, k=4):
    """Fallback: K-means on TF-IDF of docstrings."""
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    names = [item["name"] for item in classifications]
    texts = [item["docstring"] for item in classifications]
    tfidf = TfidfVectorizer(stop_words="english", max_features=300).fit_transform(texts)
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(tfidf)
    labels = km.labels_
    return {name: f"cluster_{label}" for name, label in zip(names, labels)}


def build_variant(source_data, domain_map, out_path):
    """Apply domain_map to source classifications and rebuild ontology JSON."""
    new_data = json.loads(json.dumps(source_data))  # deep copy

    # Update classifications
    for item in new_data["classifications"]:
        item["domain"] = domain_map.get(item["name"], "utility")

    # Rebuild domain category → sentences list
    new_domains = {}
    for item in new_data["classifications"]:
        d = item["domain"]
        # Use tool docstring as the sentence for this facet
        short_desc = item["docstring"].split("\n")[0][:200].strip()
        new_domains.setdefault(d, []).append(short_desc)

    # Update ontology block
    new_data["ontology"]["domain"] = new_domains

    # Update classification_distribution
    from collections import Counter
    dist = Counter(item["domain"] for item in new_data["classifications"])
    new_data["classification_distribution"]["domain"] = dict(dist)

    # Save
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)

    print(f"\n[variant] wrote {out_path}")
    print(f"  domain distribution: {dict(dist)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=SOURCE)
    p.add_argument("--out-dir", default="reports/tau2_2026_04_17")
    args = p.parse_args()

    source_data = json.load(open(args.source))
    classifications = source_data["classifications"]

    # Variant A
    print("=" * 60)
    print("VARIANT A: Split (7 fine-grained categories)")
    print("=" * 60)
    build_variant(source_data, VARIANT_A_DOMAIN,
                  f"{args.out_dir}/tau2_banking_variantA.json")

    # Variant B
    print("\n" + "=" * 60)
    print("VARIANT B: Regroup (4 semantic clusters, manual)")
    print("=" * 60)
    build_variant(source_data, VARIANT_B_DOMAIN,
                  f"{args.out_dir}/tau2_banking_variantB.json")

    # Variant C
    print("\n" + "=" * 60)
    print("VARIANT C: Auto-cluster (K-means, k=4)")
    print("=" * 60)
    c_map = variant_c_domain(classifications, k=4)
    print(f"  Auto-clusters:")
    from collections import defaultdict
    cluster_tools = defaultdict(list)
    for name, cluster in c_map.items():
        cluster_tools[cluster].append(name)
    for cluster, tools in sorted(cluster_tools.items()):
        print(f"    {cluster}: {tools}")
    build_variant(source_data, c_map,
                  f"{args.out_dir}/tau2_banking_variantC.json")


if __name__ == "__main__":
    main()
