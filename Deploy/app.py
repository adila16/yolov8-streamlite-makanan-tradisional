import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ===== Styling =====
def local_css():
    st.markdown("""
    <style>
    body {
        background-color: #fff9f4;
    }
    h1, h4 {
        text-align: center;
        color: #e67300;
    }
    .stButton > button {
        background-color: #e67300;
        color: white;
    }
    .deteksi-box {
        padding: 10px;
        background-color: #fef0e6;
        border-radius: 6px;
        margin-bottom: 6px;
        border-left: 5px solid #e67300;
    }
    .navbar {
        display: flex;
        justify-content: center;
        gap: 40px;
        margin-top: 10px;
        margin-bottom: 25px;
        flex-wrap: wrap;
    }
    .nav-item {
        font-weight: bold;
        text-decoration: none;
        color: #e67300;
        font-size: 18px;
        padding: 8px 16px;
        border-radius: 8px;
        transition: background-color 0.3s ease;
    }
    .nav-item:hover {
        background-color: #ffe8d6;
        cursor: pointer;
    }
    .selected {
        background-color: #e67300;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

local_css()

# ===== Navbar Logic =====
nav_options = ["Beranda", "Tentang Aplikasi", "Deteksi Makanan"]
if "nav_selected" not in st.session_state:
    st.session_state["nav_selected"] = "Deteksi Makanan"

query_nav = st.query_params.get("nav", [st.session_state["nav_selected"]])[0]
nav_param = query_nav if query_nav in nav_options else "Deteksi Makanan"
st.session_state["nav_selected"] = nav_param

nav_html = '<div class="navbar">'
for nav in nav_options:
    class_attr = "nav-item selected" if nav == nav_param else "nav-item"
    nav_html += f'<a class="{class_attr}" href="?nav={nav}">{nav}</a>'
nav_html += '</div>'
st.markdown(nav_html, unsafe_allow_html=True)

# ===== Halaman: Beranda =====
if nav_param == "Beranda":
    st.title("Selamat Datang di Aplikasi Deteksi Makanan 🍱")
    st.markdown("Silakan pilih menu di atas untuk mulai mengenali makanan tradisional Indonesia melalui gambar.")

# ===== Halaman: Tentang Aplikasi =====
elif nav_param == "Tentang Aplikasi":
    st.title("Tentang Aplikasi 💡")
    st.markdown("""
Aplikasi ini dirancang untuk mendeteksi makanan tradisional Indonesia dari gambar dan menampilkan estimasi kalorinya.  
Ini merupakan proyek visual berbasis **YOLOv8 + Streamlit**, cocok untuk edukasi gizi dan eksplorasi budaya kuliner Nusantara.

#### 🔥 Fitur:
- Deteksi cerdas menggunakan model YOLOv8
- Estimasi kalori tiap makanan
- Tampilan menarik dengan emoji & warna hangat
- Antarmuka seperti website modern, ringan dan cepat

#### ⚙️ Teknologi:
- YOLOv8 (Ultralytics)
- Python (PIL, Streamlit)
- CSS Custom Theme

#### 👩‍💻 Developer:
**Adilla** – antusias di bidang 3D modeling, AR, dan visualisasi ilmiah.  
Project ini jadi wadah eksplorasi kreatif dan pengenalan AI dalam budaya lokal 🇮🇩
""")

# ===== Halaman: Deteksi Makanan =====
elif nav_param == "Deteksi Makanan":
    st.markdown("<h1>🍱 Deteksi Makanan Tradisional Nusantara</h1>", unsafe_allow_html=True)
    st.markdown("<h4>Kenali makanan lokal dan informasi kalorinya dengan bantuan YOLOv8</h4>", unsafe_allow_html=True)
    st.markdown("---")

    model = YOLO("best.pt")
    kelas_makanan = {
        0: "GADO-GADO (137 KALORI)", 1: "KERAK TELOR (244 KALORI)", 2: "KUE PANCONG (296 KALORI)",
        3: "KUE DONGKAL (191 KALORI)", 4: "KUE ONGOL-ONGOL (111 KALORI)",
        5: "KUE PUTU MAYANG (121 KALORI)", 6: "NASI UDUK (253 KALORI)", 7: "SOTO BETAWI (135 KALORI)"
    }
    kelas_valid = set(kelas_makanan.keys())
    emoji_map = {
        "GADO-GADO": "🥗", "KERAK TELOR": "🍳", "KUE PANCONG": "🍰", "KUE DONGKAL": "🥮",
        "KUE ONGOL-ONGOL": "🧁", "KUE PUTU MAYANG": "🍡", "NASI UDUK": "🍚", "SOTO BETAWI": "🍲"
    }

    uploaded_file = st.file_uploader("📸 Upload Gambar Makanan", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼️ Gambar yang Diupload", use_container_width=True)

        with st.spinner("🔍 Mendeteksi objek makanan..."):
            results = model.predict(image, conf=0.3)
            result = results[0]
            boxes = result.boxes

            st.markdown("---")
            col1, col2 = st.columns([1, 1.3])
            detected_names = []

            with col1:
                if boxes and len(boxes.cls) > 0:
                    for box, cls_id, conf in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist()):
                        cls_id_int = int(cls_id)
                        if cls_id_int in kelas_valid:
                            nama = kelas_makanan[cls_id_int]
                            nama_bersih = nama.split(" (")[0]
                            detected_names.append(nama_bersih)
                            st.markdown(f"""
                                <div class='deteksi-box'>
                                    🍴 <b>{nama}</b><br>
                                    Confidence: {conf:.2f}
                                </div>
                            """, unsafe_allow_html=True)

                    if detected_names:
                        st.markdown("### 🎊 Makanan Terdeteksi 🎊")
                        efek_makanan = " ".join(emoji_map.get(n.upper(), "🍽️") for n in detected_names)
                        st.markdown(f"<h1 style='text-align:center'>{efek_makanan}</h1>", unsafe_allow_html=True)
                    else:
                        st.warning("⚠️ Makanan yang terdeteksi belum dikenali.")
                else:
                    st.warning("⚠️ Tidak ada objek yang dikenali.")

            with col2:
                st.image(result.plot(), caption="🎯 Hasil Deteksi", use_container_width=True)