import time
from collections import Counter
from itertools import combinations

import pandas as pd
import streamlit as st


@st.cache_data(show_spinner="Mining frequent itemsets & association rules…", max_entries=8, ttl=3600)
def compute_association_rules(_df: pd.DataFrame, data_version: tuple = (), min_support_pct: float = 0.5):
    _t0 = time.time()
    invoice_products = _df.groupby("Invoice")["Description"].unique().apply(lambda a: sorted(set(a)))
    invoice_products = invoice_products[invoice_products.apply(len) >= 2]
    n_invoices = len(invoice_products)
    if n_invoices < 20:
        return {"ok": False, "reason": "Not enough multi-item baskets in the current filter to mine association rules."}

    item_counts = Counter()
    for prods in invoice_products:
        item_counts.update(prods)

    min_support_count = max(5, int((min_support_pct / 100) * n_invoices))
    frequent_items = {p for p, c in item_counts.items() if c >= min_support_count}

    pair_counter: Counter = Counter()
    for prods in invoice_products:
        filt = [p for p in prods if p in frequent_items]
        if len(filt) >= 2:
            pair_counter.update(combinations(filt, 2))

    rules = []
    for (a, b), c in pair_counter.items():
        supp = c / n_invoices
        rules.append((a, b, c, supp, c / item_counts[a], c * n_invoices / (item_counts[a] * item_counts[b])))
        rules.append((b, a, c, supp, c / item_counts[b], c * n_invoices / (item_counts[a] * item_counts[b])))

    rules_df = pd.DataFrame(rules, columns=["antecedent", "consequent", "co_occurrence", "support", "confidence", "lift"])
    if not rules_df.empty:
        rules_df = rules_df.sort_values("lift", ascending=False).head(20000)
    st.sidebar.caption(f"⏱️ compute_association_rules compute: {time.time()-_t0:.2f}s")
    return {
        "ok": True,
        "rules_df": rules_df,
        "n_invoices": n_invoices,
        "n_frequent_items": len(frequent_items),
        "min_support_count": min_support_count,
    }
