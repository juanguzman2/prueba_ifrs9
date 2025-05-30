import streamlit as st


st.set_page_config(layout="wide", page_title="Prueba_IFRS9", page_icon=":rocket:")

st.title("Juan Esteban Guzmán")

st.sidebar.success("Menú de navegación")

pages = {
    "Solucion Problema": [
        st.Page("paginas/solucion_problema.py", title="Solucion Problema", icon="📤"),
                ],
    "Pronostico de la probabilidad de default": [
        st.Page("paginas/predict.py", title="Pronostico de la probabilidad de default", icon="🤖"),
                ],
    }

pg = st.navigation(pages,position="sidebar")
pg.run()