import streamlit as st


st.markdown("""
    <h1 class='center'>Nền tảng kết nối những người khiếm thính với thế giới</h1>
""", unsafe_allow_html=True)
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    trangchu_button = st.button('Trang Chủ')
with col2:
    huongdan_button = st.button("Hướng dẫn")
with col3:
    donate_button = st.button("Donate")

# Kiểm tra xem nút nào được nhấn và điều hướng tương ứng
if donate_button:
    st.query_params["page"] = "donate_button"
elif huongdan_button:
    st.query_params["page"] = "huongdan_button"

# Điều hướng dựa trên tham số truy vấn
query_params = st.query_params
if "page" in query_params:
    page = query_params["page"]
    if page == "donate_button":
        with open("donate.py", encoding="utf-8") as f:
            exec(f.read())
    elif page == "huongdan_button":
        with open("huong_dan_dung.py", encoding="utf-8") as f:
            exec(f.read())
