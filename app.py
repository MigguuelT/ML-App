import streamlit as st
import pandas as pd
import joblib
from automl_agent import AutoMLAgentPro

# Configuração da Página
st.set_page_config(page_title="AutoML Agent Pro", page_icon="🤖", layout="wide")

st.title("🤖 Agente de Machine Learning Automatizado")

# --- CRIAÇÃO DAS ABAS ---
tab1, tab2 = st.tabs(["🏋️‍♂️ Treinamento (Criar IA)", "🔮 Previsão (Usar IA)"])

# ====================================================
# ABA 1: TREINAMENTO
# ====================================================
with tab1:
    st.markdown("### Treine um novo modelo a partir dos seus dados históricos")
    
    # --- BARRA LATERAL (Exclusiva da Aba 1, mas renderizada na sidebar comum) ---
    st.sidebar.header("📁 Configuração de Arquivos")
    sep_option = st.sidebar.selectbox("Separador do CSV", options=[", (Vírgula)", "; (Ponto e Vírgula)"])
    separator = "," if sep_option == ", (Vírgula)" else ";"
    
    # Upload Dados de Treino
    train_file = st.sidebar.file_uploader("Upload Dados de TREINO (CSV)", type=["csv"], key="train_uploader")

    # Variáveis de controle
    df_train = None
    target_col = None
    
    if train_file is not None:
        try:
            df_train = pd.read_csv(train_file, sep=separator)
            df_train.columns = df_train.columns.str.strip() # Limpeza de nomes
            
            st.write("#### Pré-visualização dos Dados de Treino")
            st.dataframe(df_train.head())
            
            # Seleção de Colunas
            st.sidebar.divider()
            st.sidebar.subheader("⚙️ Config. Treino")
            
            all_columns = df_train.columns.tolist()
            target_col = st.sidebar.selectbox("Coluna Alvo (Target)", options=all_columns)
            description = st.sidebar.text_area("Descrição (Opcional)", placeholder="Ex: Prever vendas")
            
            btn_train = st.sidebar.button("🚀 Iniciar Treinamento")

            # --- LÓGICA DE TREINO ---
            if btn_train:
                st.divider()
                st.subheader(f"⚙️ Treinando Modelo para: {target_col}")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    agent = AutoMLAgentPro()
                    
                    status_text.text("Analisando e treinando...")
                    progress_bar.progress(20)
                    
                    desc_final = description if description else f"Previsão de {target_col}"
                    metrics = agent.train(df_train, target_column=target_col, description=desc_final)
                    
                    progress_bar.progress(80)
                    status_text.text("Finalizando...")
                    
                    # Salvar modelo
                    model_filename = "modelo_treinado.pkl"
                    agent.save_model(model_filename)
                    
                    progress_bar.progress(100)
                    status_text.empty()
                    
                    st.success("✅ Treinamento Concluído!")
                    
                    # Exibir Métricas
                    col1, col2 = st.columns(2)
                    col1.info(f"**Algoritmo:** {agent.best_model.steps[-1][1].__class__.__name__}")
                    col1.info(f"**Tipo:** {agent.problem_type.upper()}")
                    
                    if agent.problem_type == 'classification':
                        col2.metric("Acurácia", f"{metrics['accuracy']:.2%}")
                    else:
                        col2.metric("R² Score", f"{metrics['r2']:.4f}")

                    # Download do Modelo
                    with open(model_filename, "rb") as f:
                        st.download_button(
                            label="⬇️ Baixar Modelo (.pkl)",
                            data=f,
                            file_name=model_filename,
                            mime="application/octet-stream"
                        )
                        
                except Exception as e:
                    st.error(f"Erro no treino: {e}")
                    
        except Exception as e:
            st.error(f"Erro ao ler arquivo de treino: {e}")
    else:
        st.info("👈 Faça upload do arquivo CSV na barra lateral para começar o treino.")


# ====================================================
# ABA 2: PREVISÃO (USO DO MODELO)
# ====================================================
with tab2:
    st.markdown("### Fazer previsões em novos dados usando um modelo salvo")
    
    col_upload_model, col_upload_data = st.columns(2)
    
    # 1. Upload do Modelo (.pkl)
    with col_upload_model:
        st.subheader("1. Carregar Modelo (.pkl)")
        uploaded_model = st.file_uploader("Arraste o arquivo .pkl aqui", type=["pkl"])

    # 2. Upload dos Novos Dados (.csv)
    with col_upload_data:
        st.subheader("2. Carregar Novos Dados (.csv)")
        uploaded_new_data = st.file_uploader("Arraste o CSV com novos dados", type=["csv"])

    # Lógica de Previsão
    if uploaded_model is not None and uploaded_new_data is not None:
        st.divider()
        try:
            # Carregar Modelo
            model = joblib.load(uploaded_model)
            st.success("Modelo carregado com sucesso!")
            
            # Carregar Dados (Usando o mesmo separador configurado na sidebar para consistência)
            df_new = pd.read_csv(uploaded_new_data, sep=separator)
            df_new.columns = df_new.columns.str.strip()
            
            st.write("#### Dados carregados (Primeiras linhas):")
            st.dataframe(df_new.head())
            
            # Botão de Previsão
            if st.button("🔮 Gerar Previsões"):
                with st.spinner("Calculando previsões..."):
                    try:
                        # Fazer a previsão
                        # O pipeline cuida de tratar nulos e encodings automaticamente!
                        predictions = model.predict(df_new)
                        
                        # Adicionar resultado ao DataFrame
                        df_result = df_new.copy()
                        df_result['PREVISAO_IA'] = predictions
                        
                        st.balloons()
                        st.write("### ✅ Resultado das Previsões:")
                        
                        # Destacar a coluna de previsão
                        st.dataframe(df_result.style.apply(lambda x: ['background-color: #d1e7dd' if x.name == 'PREVISAO_IA' else '' for i in x], axis=0))
                        
                        # Converter para CSV para download
                        csv = df_result.to_csv(index=False).encode('utf-8')
                        
                        st.download_button(
                            label="⬇️ Baixar Planilha com Previsões (.csv)",
                            data=csv,
                            file_name="resultado_previsoes.csv",
                            mime="text/csv",
                        )
                        
                    except Exception as pred_error:
                        st.error(f"Erro ao prever: {pred_error}")
                        st.warning("Dica: Verifique se o novo arquivo CSV tem as mesmas colunas (nomes) que foram usadas no treino.")
                        
        except Exception as e:
            st.error(f"Erro ao carregar arquivos: {e}")