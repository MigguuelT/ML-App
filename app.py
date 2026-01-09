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

# --- BARRA LATERAL: 1. UPLOAD ---
st.sidebar.header("1. Upload de Dados")

# Opção extra para garantir leitura correta de CSVs brasileiros (ponto e vírgula)
sep_option = st.sidebar.selectbox("Separador do CSV", options=[", (Vírgula)", "; (Ponto e Vírgula)"])
separator = "," if sep_option == ", (Vírgula)" else ";"

uploaded_file = st.sidebar.file_uploader("Carregue seu arquivo CSV", type=["csv"])

# --- INICIALIZAÇÃO DE VARIÁVEIS ---
df = None
target_col = None
btn_train = False
description = ""

# --- LEITURA DO ARQUIVO E CONFIGURAÇÃO DINÂMICA ---
if uploaded_file is not None:
    try:
        # Lê o arquivo com o separador escolhido
        df = pd.read_csv(uploaded_file, sep=separator)
        
        # CORREÇÃO CRÍTICA: Remove espaços em branco antes e depois dos nomes das colunas
        # Ex: " Sex " vira "Sex"
        df.columns = df.columns.str.strip()
        
        # --- BARRA LATERAL: 2. CONFIGURAÇÕES (Só aparecem após upload) ---
        st.sidebar.divider()
        st.sidebar.header("2. Configurações")
        
        # DROPDOWN: O usuário escolhe a coluna da lista (Evita erros de digitação)
        all_columns = df.columns.tolist()
        target_col = st.sidebar.selectbox("Escolha a Coluna Alvo (Target)", options=all_columns)
        
        description = st.sidebar.text_area("Descrição do Problema (Opcional)", placeholder="Ex: Prever sobreviventes do Titanic")
        
        # Botão de treino
        btn_train = st.sidebar.button("🚀 Iniciar Treinamento")
        
        # --- ÁREA PRINCIPAL: PREVIEW ---
        st.write("### 📊 Pré-visualização dos Dados")
        st.write(f"Dimensões do Dataset: {df.shape[0]} linhas x {df.shape[1]} colunas")
        st.dataframe(df.head())
        
    except Exception as e:
        st.error(f"Erro ao ler o arquivo. Tente mudar o 'Separador do CSV' na barra lateral.\nDetalhe do erro: {e}")

else:
    st.info("👈 Comece carregando seus dados na barra lateral.")


# --- LÓGICA DE TREINAMENTO ---
if btn_train and df is not None:
    st.divider()
    st.subheader(f"⚙️ Treinando Modelo para prever: **{target_col}**")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        agent = AutoMLAgentPro()
        
        status_text.text("Analisando dados, tratando outliers e otimizando hiperparâmetros...")
        progress_bar.progress(20)
        
        # TREINO + CAPTURA DE MÉTRICAS
        # Passamos description apenas se o usuário tiver digitado algo
        desc_final = description if description else f"Previsão de {target_col}"
        metrics = agent.train(df, target_column=target_col, description=desc_final)
        
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
            st.markdown("#### Detalhes por Classe")
            report_df = pd.DataFrame(metrics['report']).transpose()
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
        st.error(f"Ocorreu um erro crítico durante o treino: {e}")