import plotly.express as px
import streamlit as st

from src.config import PALETTE, PLOTLY_LAYOUT
from src.models.recommendations import compute_association_rules
from src.ui.components import fmt_num, kpi_card, section_title
from src.ui.sidebar import render_sidebar_and_header

dff, data_version = render_sidebar_and_header()

section_title("\"Customers Also Bought\" — Apriori Association Rule Mining", "Frequent itemset mining over shopping baskets, scored with support, confidence, and lift (the standard association-rule metrics).")

with st.expander("How to read these numbers"):
    st.markdown(
        "- **Support** — how often this pair appears together, as a share of all multi-item baskets.\n"
        "- **Confidence** — of the baskets containing the *antecedent*, what share also contained the *consequent*.\n"
        "- **Lift** — how much more likely the pair is to co-occur than if the two products were bought independently. Lift > 1 means a genuine association, not just two popular items."
    )

min_support_pct = st.slider("Minimum item support (%) — Apriori pruning threshold", 0.5, 5.0, 0.5, 0.1)
ar = compute_association_rules(dff, data_version, min_support_pct=min_support_pct)

if not ar["ok"]:
    st.info(ar["reason"])
else:
    rules_df = ar["rules_df"]
    c1, c2, c3 = st.columns(3)
    kpi_card(c1, "Multi-Item Baskets", fmt_num(ar["n_invoices"]), "🧺", color=PALETTE[0])
    kpi_card(c2, "Frequent Items (post-pruning)", fmt_num(ar["n_frequent_items"]), "📦", color=PALETTE[1])
    kpi_card(c3, "Rules Generated", fmt_num(len(rules_df)), "🔗", color=PALETTE[3])

    st.write("")
    all_products = sorted(rules_df["antecedent"].unique().tolist())
    default_idx = all_products.index("WHITE HANGING HEART T-LIGHT HOLDER") if "WHITE HANGING HEART T-LIGHT HOLDER" in all_products else 0
    col_a, col_b = st.columns([2, 1])
    with col_a:
        product = st.selectbox("Choose a product", all_products, index=default_idx if all_products else 0)
    with col_b:
        rank_by = st.selectbox("Rank recommendations by", ["lift", "confidence"], index=0)

    recs = rules_df[rules_df["antecedent"] == product].sort_values(rank_by, ascending=False).head(5)
    if not recs.empty:
        rec_plot = recs.sort_values(rank_by)
        fig = px.bar(rec_plot, x=rank_by, y="consequent", orientation="h", color_discrete_sequence=[PALETTE[2]],
                     hover_data=["support", "confidence", "lift"])
        fig.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False, yaxis_title="")
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("No strong association rule found for this product at the current support threshold — try lowering it.")

    st.write("")
    section_title("Top 15 Rules Overall (by lift)")
    top_rules = rules_df.sort_values("lift", ascending=False).drop_duplicates(subset=["support", "lift"]).head(15)
    display_rules = top_rules[["antecedent", "consequent", "support", "confidence", "lift"]].copy()
    display_rules["support"] = (display_rules["support"] * 100).round(2).astype(str) + "%"
    display_rules["confidence"] = (display_rules["confidence"] * 100).round(1).astype(str) + "%"
    display_rules["lift"] = display_rules["lift"].round(1)
    st.dataframe(display_rules, width='stretch', hide_index=True)
    if st.button("Prepare download", key="btn_prep_rules"):
        st.session_state["prep_rules"] = True
    if st.session_state.get("prep_rules"):
        st.download_button("⬇️ Download all association rules (CSV)", data=rules_df.to_csv(index=False).encode("utf-8"),
                            file_name="association_rules.csv", mime="text/csv")
