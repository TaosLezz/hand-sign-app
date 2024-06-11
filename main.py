
import streamlit as st

page = st.sidebar.selectbox("Menu:", ["Trang Chủ","Giao Tiếp", "Donate", "Hướng dẫn"])


# Xử lý khi người dùng thay đổi trang
if page == "Trang Chủ":
    st.query_params["page"] = "trangchu"
elif page == "Giao Tiếp":
    st.query_params["page"] = "communicate"
elif page == "Donate":
    st.query_params["page"] = "donate"
elif page == "Hướng dẫn":
    st.query_params["page"] = "huong_dan"

# Điều hướng dựa trên tham số truy vấn
query_params = st.query_params
if "page" in query_params:
    page = query_params["page"]
    if page == "trangchu":
        with open("trangchu.py", encoding="utf-8") as f:
            exec(f.read())
    elif page == "communicate":
        with open("communicate.py", encoding="utf-8") as f:
            exec(f.read())
    elif page == "donate":
        with open("donate.py", encoding="utf-8") as f:
            exec(f.read())
    elif page == "huong_dan":
        with open("huong_dan_dung.py", encoding="utf-8") as f:
            exec(f.read())
