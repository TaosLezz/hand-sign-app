import streamlit as st

st.title("Donate")
st.write("Nhằm giúp chúng tôi có thêm kinh phí để vận hành cũng như phát triển "
             "dự án tốt hơn. Bạn có thể donate cho chúng tôi thông qua tài khoản ngân hàng này")
col1, col2 = st.columns([2, 1])
with col1:
    st.image("donate-anim.png", use_column_width=True)
with col2:
    st.markdown("""
        <h1 style='font-size: 30px;'>QRCode</h1>
    """, unsafe_allow_html=True)
    st.write("Ủng hộ chúng tôi qua mã qrcode này kèm lời nhắn:")
    st.write("`Donate signbridge`.")
    st.markdown("<style>", unsafe_allow_html=True)
    st.image("qr.png")
