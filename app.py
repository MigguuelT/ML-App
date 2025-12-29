import streamlit as st
import pandas as pd
import joblib
import os

# Importa a sua classe (certifique-se que o arquivo automl_agent.py está na mesma pasta)
from automl_agent import AutoMLAgentPro

# Configuração da Página
st.set_page_config(page_title="AutoML Agent Pro", page_icon="🤖", layout="wide")

st.title("🤖 Agente de Machine Learning Automatizado")
st.markdown("""
Este agente analisa seus dados, escolhe o melhor modelo (Classificação ou Regressão), 
trata outliers e treina a inteligência artificial automaticamente.
""")

# --- BARRA LATERAL (Inputs) ---
st.sidebar.header("1. Upload de Dados")
uploaded_file = st.sidebar.file_uploader("Carregue seu arquivo CSV", type=["csv"])

st.sidebar.header("2. Configurações")
target_col = st.sidebar.text_input("Nome da Coluna Alvo (Target)", value="")
description = st.sidebar.text_area("Descrição do Problema (Opcional)", placeholder="Ex: Prever rotatividade de clientes")
btn_train = st.sidebar.button("🚀 Iniciar Treinamento")

# --- ÁREA PRINCIPAL ---
if uploaded_file is not None:
    # Ler o arquivo
    try:
        df = pd.read_csv(uploaded_file)
        st.write("### 📊 Pré-visualização dos Dados")
        st.dataframe(df.head())
        
        # Verificar colunas
        if target_col and target_col not in df.columns:
            st.error(f"Erro: A coluna '{target_col}' não foi encontrada no arquivo.")
        
    except Exception as e:
        st.error(f"Erro ao ler arquivo: {e}")

    # --- LÓGICA DE TREINAMENTO ---
    if btn_train and target_col in df.columns:
        st.divider()
        st.subheader("⚙️ Processando...")
        
        # Barra de progresso visual
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # 1. Instanciar Agente
            agent = AutoMLAgentPro()
            
            # Como o print() não sai no site, vamos capturar logs ou confiar no resultado final
            # Para um app real, idealmente alteraríamos a classe para retornar textos em vez de printar
            # Mas aqui vamos rodar direto:
            
            status_text.text("Analisando tipo de problema e tratando dados...")
            progress_bar.progress(20)
            
            # Redirecionando prints para o console do servidor (logs)
            # E executando o treino
            agent.train(df, target_column=target_col, description=description)
            
            progress_bar.progress(80)
            status_text.text("Finalizando validação cruzada...")
            
            # Salvar modelo temporariamente para download
            model_filename = "meu_modelo_treinado.pkl"
            agent.save_model(model_filename)
            
            progress_bar.progress(100)
            status_text.text("Concluído!")
            
            # --- EXIBIÇÃO DOS RESULTADOS ---
            st.success("Treinamento Finalizado com Sucesso!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"**Tipo de Problema Detectado:** {agent.problem_type.upper()}")
                st.info(f"**Modelo Vencedor:** {agent.best_model.steps[-1][1].__class__.__name__}")
                
                # Exibir melhores hiperparâmetros
                with st.expander("Ver Melhores Hiperparâmetros"):
                    st.json(agent.best_params)

            with col2:
                st.write("### 📥 Baixar Modelo")
                with open(model_filename, "rb") as f:
                    st.download_button(
                        label="Download do Modelo (.pkl)",
                        data=f,
                        file_name=model_filename,
                        mime="application/octet-stream"
                    )

            st.warning("Nota: Para ver as métricas detalhadas (Acurácia/RMSE), verifique o log do terminal ou adapte a classe para retornar esses valores.")

        except Exception as e:
            st.error(f"Ocorreu um erro durante o treinamento: {e}")

elif btn_train:
    st.warning("Por favor, faça o upload de um arquivo CSV primeiro.")

else:
    st.info("👈 Comece carregando um arquivo na barra lateral.")