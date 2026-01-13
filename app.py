import streamlit as st
import pandas as pd
import joblib
from automl_agent import AutoMLAgentPro

st.set_page_config(page_title="AutoML Agent Pro", page_icon="🤖", layout="wide")
st.title("🤖 Agente de Machine Learning Automatizado")

tab1, tab2 = st.tabs(["🏋️‍♂️ Treinamento (Criar IA)", "🔮 Previsão (Usar IA)"])

# ====================================================
# ABA 1: TREINAMENTO
# ====================================================
with tab1:
    st.markdown("### Treine um novo modelo")
    st.sidebar.header("📁 Configuração de Arquivos")
    sep_option = st.sidebar.selectbox("Separador do CSV", options=[", (Vírgula)", "; (Ponto e Vírgula)"])
    separator = "," if sep_option == ", (Vírgula)" else ";"
    
    train_file = st.sidebar.file_uploader("Upload Dados de TREINO (CSV)", type=["csv"], key="train_uploader")

    if train_file is not None:
        try:
            df_train = pd.read_csv(train_file, sep=separator)
            df_train.columns = df_train.columns.str.strip()
            
            st.write("#### Pré-visualização")
            st.dataframe(df_train.head())
            
            st.sidebar.divider()
            
            # 1. ESCOLHA DO TARGET
            all_cols = df_train.columns.tolist()
            target_col = st.sidebar.selectbox("Coluna Alvo (Target)", options=all_cols)
            
            # 2. FILTRO DE COLUNAS (NOVIDADE CRÍTICA)
            # Removemos a target da lista de possíveis exclusões para não dar erro
            cols_possible_drop = [c for c in all_cols if c != target_col]
            drop_cols = st.sidebar.multiselect(
                "Remover Colunas (IDs, Nomes, Vazamento)", 
                options=cols_possible_drop,
                help="Selecione colunas que não ajudam na previsão ou que são 'spoilers' (ex: ID, Nome, Data de Cancelamento)."
            )
            
            description = st.sidebar.text_area("Descrição (Opcional)", placeholder="Ex: Prever vendas")
            
            if st.sidebar.button("🚀 Iniciar Treinamento"):
                st.divider()
                st.subheader(f"⚙️ Treinando para: {target_col}")
                
                # --- REMOÇÃO DAS COLUNAS SELECIONADAS ---
                if drop_cols:
                    st.warning(f"Removendo colunas: {drop_cols}")
                    df_train_final = df_train.drop(columns=drop_cols)
                else:
                    df_train_final = df_train
                # ----------------------------------------
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    agent = AutoMLAgentPro()
                    
                    status_text.text("Limpando dados e otimizando modelo...")
                    progress_bar.progress(20)
                    
                    metrics = agent.train(df_train_final, target_column=target_col, description=description)
                    
                    progress_bar.progress(80)
                    agent.save_model("modelo_treinado.pkl")
                    progress_bar.progress(100)
                    status_text.empty()
                    
                    st.success("✅ Treinamento Concluído!")
                    
                    # --- VISUALIZAÇÃO ONE-HOT ---
                    if agent.problem_type == 'classification' and agent.target_mapping:
                        with st.expander("🔍 Ver Transformação do Alvo (Target Encoding)"):
                            st.write("O modelo converteu suas classes de texto para números internos:")
                            df_target_map = pd.DataFrame(list(agent.target_mapping.items()), columns=['Classe Original', 'Código Interno'])
                            st.dataframe(df_target_map, hide_index=True)

                    # --- VISUALIZAÇÃO DAS FEATURES ---
                    try:
                        sample_data = df_train_final.drop(columns=[target_col]).head(5)
                        encoding_examples = agent.get_encoding_examples(sample_data)
                        
                        if encoding_examples:
                            with st.expander("🔍 Ver Transformação das Variáveis de Entrada"):
                                for col_name, df_example in encoding_examples.items():
                                    st.markdown(f"**Origem: {col_name}**")
                                    st.dataframe(df_example.style.background_gradient(cmap='Blues'))
                        else:
                            with st.expander("ℹ️ Sobre as Variáveis de Entrada"):
                                st.write("Todas as variáveis de entrada foram identificadas como numéricas.")
                                st.write(f"Numéricas: {len(agent.numeric_features)} | Texto: {len(agent.categorical_features)}")
                                
                    except Exception as viz_error:
                        st.warning(f"Erro visualização: {viz_error}")

                    col1, col2 = st.columns(2)
                    col1.info(f"**Algoritmo:** {agent.best_model.steps[-1][1].__class__.__name__}")
                    col1.info(f"**Tipo:** {agent.problem_type.upper()}")
                    
                    if agent.problem_type == 'classification':
                        col2.metric("Acurácia", f"{metrics['accuracy']:.2%}")
                    else:
                        col2.metric("R² Score", f"{metrics['r2']:.4f}")

                    with open("modelo_treinado.pkl", "rb") as f:
                        st.download_button("⬇️ Baixar Modelo (.pkl)", data=f, file_name="modelo_treinado.pkl")
                        
                except Exception as e:
                    st.error(f"Erro no treino: {e}")
        except Exception as e:
            st.error(f"Erro ao ler arquivo: {e}")

# ====================================================
# ABA 2: PREVISÃO
# ====================================================
with tab2:
    st.markdown("### Fazer previsões com modelo salvo")
    
    c1, c2 = st.columns(2)
    uploaded_model = c1.file_uploader("Carregar Modelo (.pkl)", type=["pkl"])
    uploaded_new_data = c2.file_uploader("Carregar Novos Dados (.csv)", type=["csv"])

    if uploaded_model and uploaded_new_data:
        st.divider()
        try:
            model = joblib.load(uploaded_model)
            st.success("Modelo carregado!")
            
            df_new = pd.read_csv(uploaded_new_data, sep=separator)
            df_new.columns = df_new.columns.str.strip()
            
            # Limpeza na Previsão
            temp_agent = AutoMLAgentPro()
            df_clean = temp_agent.clean_data_types(df_new)
            
            st.write("#### Dados (Processados):")
            st.dataframe(df_clean.head())
            
            if st.button("🔮 Gerar Previsões"):
                try:
                    # Tenta prever. Se houver colunas extras (como ID) que não estavam no treino,
                    # o sklearn vai dar erro. Por isso precisamos dropar aqui também ou confiar
                    # que o usuário suba o CSV já limpo, ou que o ColumnTransformer ignore o resto.
                    # O ColumnTransformer ignora colunas não especificadas SE o resto for passthrough,
                    # mas aqui filtramos por tipo. 
                    
                    # MELHOR ABORDAGEM: O modelo espera EXATAMENTE as colunas de treino.
                    # Se o CSV novo tiver coluna "ID" e o modelo foi treinado sem "ID", vai dar erro?
                    # O ColumnTransformer seleciona pelo nome. Então se tiver colunas A MAIS, não tem problema.
                    # Se tiver colunas A MENOS, dá erro.
                    
                    predictions = model.predict(df_clean)
                    
                    df_result = df_new.copy()
                    df_result['PREVISAO_IA'] = predictions
                    
                    st.write("### ✅ Resultado:")
                    st.dataframe(df_result.head())
                    
                    csv = df_result.to_csv(index=False).encode('utf-8')
                    st.download_button("⬇️ Baixar Resultado", data=csv, file_name="previsoes.csv", mime="text/csv")
                    
                except Exception as pred_error:
                    st.error(f"Erro ao prever: {pred_error}")
                    st.warning("Verifique se o CSV novo contém as colunas usadas no treino.")
                    
        except Exception as e:
            st.error(f"Erro geral: {e}")