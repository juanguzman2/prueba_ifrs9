# app_streamlit.py
import streamlit as st
import pandas as pd
import requests
import io

API_URL = "https://prueba-ifrs9.onrender.com/predict/"

st.title("📊 Clasificador de Riesgo de Crédito")

st.markdown("""
Carga un archivo `.csv` (separado por pipe `|`) para obtener las probabilidades de incumplimiento y la clasificación en grupos de riesgo `t1` a `t8`.
""")

uploaded_file = st.file_uploader("📁 Subir archivo base_prueba.csv", type=["csv"])

if uploaded_file:
    if st.button("🚀 Enviar a la API"):
        with st.spinner("Enviando a la API..."):
            try:
                # Enviar archivo a la API
                files = {'file': (uploaded_file.name, uploaded_file.getvalue())}
                response = requests.post(API_URL, files=files)

                if response.status_code == 200:
                    data = response.json()
                    df_result = pd.DataFrame(data)

                    # Mostrar tabla
                    st.success("✅ Clasificación completada")
                    st.dataframe(df_result)

                    # Mostrar resumen
                    resumen = (
                        df_result.groupby("grupo_riesgo")
                        .size()
                        .reset_index(name="cantidad")
                        .sort_values(by="grupo_riesgo")
                    )
                    st.subheader("📈 Resumen por grupo de riesgo")
                    st.table(resumen)

                    # Descargar como CSV
                    csv = df_result.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Descargar resultados como CSV",
                        data=csv,
                        file_name="predicciones_riesgo.csv",
                        mime="text/csv"
                    )

                else:
                    st.error(f"❌ Error {response.status_code}: {response.json().get('error')}")

            except Exception as e:
                st.error(f"⚠️ Error al conectar con la API: {e}")
