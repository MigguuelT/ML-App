"""
Como usar este script:
Coloque os arquivos na mesma pasta:
O script (fazer_previsao_arquivo.py).
O seu modelo salvo (meu_modelo_treinado.pkl).
A planilha com os dados novos (ex: novos_clientes.csv ou .xlsx).
Edite as configurações (Opcional):
Abra o script e edite as variáveis no topo (ARQUIVO_NOVOS_DADOS, etc) se seus arquivos tiverem nomes diferentes.
Execute:
Bash:
python fazer_previsao_arquivo.py
Melhorias incluídas:
Suporte Híbrido: Aceita tanto .csv quanto Excel (.xlsx).
Detector de Separador: Se for CSV, ele tenta detectar automaticamente se é vírgula ou ponto e vírgula (comum no Brasil).
Limpeza de Colunas: Remove espaços em branco dos cabeçalhos automaticamente (.strip()), evitando o erro de "Coluna não encontrada".
Saída Compatível: O arquivo final é salvo com separador ;, para que você possa clicar duas vezes e abrir direto no Excel em português sem ficar tudo bagunçado.

"""


import joblib
import pandas as pd
import os

# =========================================================
# CONFIGURAÇÕES DO USUÁRIO (EDITE AQUI)
# =========================================================

# 1. Nome do arquivo do modelo treinado (.pkl)
ARQUIVO_MODELO = 'meu_modelo_treinado.pkl'

# 2. Nome da planilha com os novos dados para previsão (.csv ou .xlsx)
ARQUIVO_NOVOS_DADOS = 'novos_clientes.csv' 

# 3. Nome do arquivo onde o resultado será salvo
ARQUIVO_SAIDA = 'resultado_previsoes.csv'

# =========================================================
# SCRIPT DE PREVISÃO AUTOMÁTICA
# =========================================================

def carregar_dados_inteligente(filepath):
    """
    Tenta ler CSV (com separador , ou ;) ou Excel (.xlsx).
    """
    extensao = os.path.splitext(filepath)[1].lower()
    
    if extensao == '.csv':
        try:
            # Tenta ler com vírgula (padrão)
            df = pd.read_csv(filepath, sep=',')
            # Se ler tudo em uma coluna só, tenta ponto e vírgula
            if df.shape[1] == 1:
                df = pd.read_csv(filepath, sep=';')
        except Exception as e:
            print(f"Erro ao ler CSV: {e}")
            return None
    elif extensao in ['.xls', '.xlsx']:
        try:
            df = pd.read_excel(filepath)
        except Exception as e:
            print(f"Erro ao ler Excel: {e}")
            return None
    else:
        print("Formato de arquivo não suportado. Use .csv ou .xlsx")
        return None
    
    # Limpeza crucial: remove espaços dos nomes das colunas (ex: " Idade " -> "Idade")
    df.columns = df.columns.str.strip()
    return df

def executar():
    print("--- INICIANDO PROCESSO DE INFERÊNCIA ---")

    # 1. Verificar se arquivos existem
    if not os.path.exists(ARQUIVO_MODELO):
        print(f"[ERRO] Modelo '{ARQUIVO_MODELO}' não encontrado.")
        return
    if not os.path.exists(ARQUIVO_NOVOS_DADOS):
        print(f"[ERRO] Arquivo de dados '{ARQUIVO_NOVOS_DADOS}' não encontrado.")
        return

    # 2. Carregar Modelo
    print(f"1. Carregando modelo: {ARQUIVO_MODELO}...")
    modelo = joblib.load(ARQUIVO_MODELO)

    # 3. Carregar Novos Dados
    print(f"2. Lendo planilha: {ARQUIVO_NOVOS_DADOS}...")
    df_novos = carregar_dados_inteligente(ARQUIVO_NOVOS_DADOS)
    
    if df_novos is None:
        return # Para se deu erro na leitura

    print(f"   > Dados carregados: {df_novos.shape[0]} linhas e {df_novos.shape[1]} colunas.")

    # 4. Fazer Previsão
    print("3. Executando previsões...")
    try:
        # O pipeline já trata nulos, texto e normalização.
        # Só precisamos garantir que as colunas existam.
        previsoes = modelo.predict(df_novos)
        
        # 5. Salvar Resultado
        df_resultado = df_novos.copy()
        df_resultado['PREVISAO_IA'] = previsoes
        
        df_resultado.to_csv(ARQUIVO_SAIDA, index=False, sep=';') # Salva com ; para abrir fácil no Excel BR
        
        print("\n" + "="*40)
        print(f"SUCESSO! Previsões salvas em: {ARQUIVO_SAIDA}")
        print("="*40)
        print("Primeiras 5 linhas do resultado:")
        print(df_resultado[['PREVISAO_IA']].head()) # Mostra só a coluna nova para confirmar
        
    except Exception as e:
        print(f"\n[ERRO CRÍTICO NA PREVISÃO]: {e}")
        print("Dica: Verifique se a planilha nova tem as mesmas colunas (nomes) usadas no treino.")

if __name__ == "__main__":
    executar()