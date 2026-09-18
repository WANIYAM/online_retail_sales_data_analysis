import streamlit as st

from src.ui.components import section_title
from src.ui.sidebar import render_sidebar_and_header

dff, data_version = render_sidebar_and_header()

section_title("Data Explorer", "Search, filter, and export the cleaned transaction-level data.")

c1, c2, c3 = st.columns(3)
search = c1.text_input("Search product description")
cust_filter = c2.text_input("Filter by Customer ID")
sort_by = c3.selectbox("Sort by", ["InvoiceDate", "Revenue", "Quantity", "Price"], index=0)

view = dff.copy()
if search:
    view = view[view["Description"].str.contains(search, case=False, na=False)]
if cust_filter:
    try:
        view = view[view["Customer ID"] == int(cust_filter)]
    except ValueError:
        st.caption("⚠️ Customer ID must be numeric.")
view = view.sort_values(sort_by, ascending=False)

st.caption(f"Showing {len(view):,} of {len(dff):,} rows")
st.dataframe(
    view[["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Revenue", "Customer ID", "Country"]].head(2000),
    width='stretch', hide_index=True, height=500,
)
if st.button("Prepare download", key="btn_prep_explorer"):
    st.session_state["prep_explorer"] = True
if st.session_state.get("prep_explorer"):
    st.download_button("⬇️ Download filtered data (CSV)", data=view.head(50000).to_csv(index=False).encode("utf-8"),
                        file_name="filtered_transactions.csv", mime="text/csv")
