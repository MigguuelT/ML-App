import streamlit as st
import pandas as pd
import joblib
import os
from automl_agent import AutoMLAgentPro

# Configuração da Página
st.set_page_config(page_title="AutoML Agent Pro", page_icon="🤖", layout="wide")

st.title("🤖 Agente de Machine Learning Automatizado")
st.markdown("""
Este agente analisa seus dados, trata outliers, seleciona as melhores features
e treina o modelo ideal (Classificação ou Regressão) automaticamente.
""")

# --- BARRA LATERAL ---
st.sidebar.header("1. Upload de Dados")
uploaded_file = st.sidebar.file_uploader("Carregue seu arquivo CSV", type=["csv"])

st.sidebar.header("2. Configurações")
target_col = st.sidebar.text_input("Nome da Coluna Alvo (Target)", value="")
description = st.sidebar.text_area("Descrição do Problema (Opcional)", placeholder="Ex: Prever preço de imóveis")
btn_train = st.sidebar.button("🚀 Iniciar Treinamento")

# --- ÁREA PRINCIPAL ---
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("### 📊 Pré-visualização dos Dados")
        st.dataframe(df.head())
        
        if target_col and target_col not in df.columns:
            st.error(f"Erro: A coluna '{target_col}' não foi encontrada no arquivo.")
            
    except Exception as e:
        st.error(f"Erro ao ler arquivo: {e}")

    # --- LÓGICA DE TREINAMENTO ---
    if btn_train and target_col in df.columns:
        st.divider()
        st.subheader("⚙️ Treinando Modelo Inteligente...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            agent = AutoMLAgentPro()
            
            status_text.text("Analisando dados, tratando outliers e otimizando hiperparâmetros...")
            progress_bar.progress(20)
            
            # TREINO + CAPTURA DE MÉTRICAS
            metrics = agent.train(df, target_column=target_col, description=description)
            
            progress_bar.progress(80)
            status_text.text("Gerando relatório final...")
            
            # Salvar modelo
            model_filename = "meu_modelo_treinado.pkl"
            agent.save_model(model_filename)
            
            progress_bar.progress(100)
            status_text.empty()
            
            # --- DASHBOARD DE RESULTADOS ---
            st.success("✅ Treinamento Concluído com Sucesso!")
            
            st.markdown("### 🏆 Melhor Modelo Encontrado")
            col_info1, col_info2 = st.columns(2)
            col_info1.info(f"**Algoritmo Vencedor:** {agent.best_model.steps[-1][1].__class__.__name__}")
            col_info2.info(f"**Tipo de Problema:** {agent.problem_type.upper()}")

            # --- VISUALIZAÇÃO DE MÉTRICAS ---
            st.markdown("### 📊 Performance nos Dados de Teste")
            
            if agent.problem_type == 'classification':
                # Métricas Classificação
                m_col1, m_col2 = st.columns(2)
                m_col1.metric("Acurácia (Accuracy)", f"{metrics['accuracy']:.2%}")
                
                # Tabela detalhada
                st.markdown("#### Detalhes por Classe (Precision/Recall)")
                report_df = pd.DataFrame(metrics['report']).transpose()
                # Remove as linhas de média se quiser limpar a view, ou mantém
                st.dataframe(report_df.style.highlight_max(axis=0))
                
            else:
                # Métricas Regressão
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("R² Score (Explicação)", f"{metrics['r2']:.4f}")
                m_col2.metric("Erro Médio (MAE)", f"{metrics['mae']:.4f}")
                m_col3.metric("RMSE", f"{metrics['rmse']:.4f}")
                
                if metrics['r2'] > 0.80:
                    st.caption("🌟 Excelente! O modelo explica muito bem a variação dos dados.")
                elif metrics['r2'] < 0.50:
                    st.caption("⚠️ Atenção: O modelo teve dificuldade. Considere adicionar mais dados ou features.")

            # --- DOWNLOAD E PARÂMETROS ---
            st.divider()
            col_down1, col_down2 = st.columns(2)
            
            with col_down1:
                st.write("### 📥 Baixar Modelo Pronto")
                with open(model_filename, "rb") as f:
                    st.download_button(
                        label="Download Modelo (.PKL)",
                        data=f,
                        file_name=model_filename,
                        mime="application/octet-stream"
                    )
            
            with col_down2:
                with st.expander("🔍 Ver Hiperparâmetros Técnicos"):
                    st.json(agent.best_params)

        except Exception as e:
            st.error(f"Ocorreu um erro crítico: {e}")

elif btn_train:
    st.warning("Por favor, faça o upload de um arquivo CSV primeiro.")
else:
    st.info("👈 Comece carregando seus dados na barra lateral.")