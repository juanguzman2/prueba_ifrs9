import streamlit as st


st.set_page_config(layout="wide", page_title="Prueba_IFRS9", page_icon=":rocket:")

st.title("Juan Esteban Guzmán")

st.sidebar.success("Menú de navegación")

pages = {
    # "Carga de archivos": [
    #     st.Page("paginas/carga_archivos.py", title="Carga de archivos", icon="📤"),
    #             ],
    # "Resumen Ejecutivo": [
    #     st.Page("paginas/visualizacion.py", title="Resumen Ejecutivo", icon="📊"),
    #             ],
    "Pronostico certificados invalidos": [
        st.Page("paginas/predict.py", title="Pronostico certificados invalidos", icon="🤖"),
                ],
    # "Desarrollo Técnico": [
    #     st.Page("paginas/desarollo_tecnico.py", title="Desarrollo Técnico", icon="📚")],
    # "Ejercicio 2 - Automatización del Reporte": [
    #     st.Page("paginas/ejercicio_2.py", title="Ejercicio 2 - Automatización del Reporte", icon="⚙️"),
    #             ],
    # "Ejercicio 3 - Validación Inteligente de Certificados Médicos": [
    #     st.Page("paginas/ejercicio_3.py", title="Ejercicio 3 - Validación Inteligente de Certificados Médicos", icon="🧾"),
    #             ]
    }

pg = st.navigation(pages,position="sidebar")
pg.run()