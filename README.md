# AutoML Agent Pro 🤖

Este repositório contém uma aplicação de **AutoML (Automated Machine Learning)** desenvolvida em Python e utilizando **Streamlit** para a interface gráfica. O sistema é capaz de detectar automaticamente o tipo de problema (Classificação ou Regressão), pré-processar os dados, selecionar o melhor algoritmo de Machine Learning, otimizar hiperparâmetros e gerar um modelo pronto para uso.

## ✨ Funcionalidades

O **AutoML Agent Pro** automatiza grande parte do pipeline de ciência de dados:

1.  **Detecção Automática do Problema**:
    *   Analisa a coluna alvo (target) para determinar se é um problema de **Regressão** (valores contínuos) ou **Classificação** (categorias).

2.  **Pré-processamento Inteligente**:
    *   **Tratamento de Dados Faltantes**: Preenchimento com mediana (numéricos) e moda (categóricos).
    *   **Tratamento de Outliers**: Utilização do `RobustScaler` para lidar com valores discrepantes.
    *   **Codificação**: `OneHotEncoder` para variáveis categóricas.

3.  **Seleção de Modelos e Otimização**:
    *   Testa múltiplos algoritmos potentes:
        *   *Classificação*: HistGradientBoostingClassifier, RandomForestClassifier, LogisticRegression.
        *   *Regressão*: HistGradientBoostingRegressor, RandomForestRegressor, Ridge Regression.
    *   **Seleção de Features**: Utiliza `SelectKBest` para identificar as variáveis mais relevantes.
    *   **Otimização de Hiperparâmetros**: Executa `GridSearchCV` com validação cruzada (`StratifiedKFold` ou `KFold`) para encontrar a melhor configuração.

4.  **Interface Amigável**:
    *   Upload fácil de arquivos CSV.
    *   Visualização dos dados.
    *   Download do modelo treinado (`.pkl`).

## 🚀 Como Executar

### Pré-requisitos

Certifique-se de ter o Python instalado. É recomendado criar um ambiente virtual.

### Instalação

Clone o repositório e instale as dependências listadas no arquivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Executando a Aplicação

### Opção 1: Web App (Interface Visual)

Para iniciar a interface web do Streamlit, execute o seguinte comando no terminal:

```bash
streamlit run app.py
```

O navegador será aberto automaticamente (geralmente em `http://localhost:8501`).

### Opção 2: Script de Previsão (Automação)
Para fazer previsões em lote sem abrir interface gráfica:
1. Coloque seu modelo treinado (.pkl) e seus dados (.csv ou .xlsx) na pasta.
2. Edite o arquivo `fazer_previsao_arquivo.py` com os nomes dos arquivos.
3. Execute:
python fazer_previsao_arquivo.py

## 📂 Estrutura do Projeto

*   **`app.py`**: Arquivo principal da aplicação Streamlit. Gerencia a interface do usuário, upload de arquivos e interação com o agente de ML.
*   **`automl_agent.py`**: Contém a classe `AutoMLAgentPro`, que encapsula toda a lógica de Machine Learning (pré-processamento, treinamento, avaliação e salvamento).
*   **`fazer_previsao_arquivo.py`**: (Local): Esse é um script independente para você rodar no seu computador (terminal) quando não quiser abrir o navegador.
*   **`requirements.txt`**: Lista das bibliotecas Python necessárias.

## 🛠️ Tecnologias Utilizadas

*   [Streamlit](https://streamlit.io/) - Framework para Web Apps de Dados
*   [Scikit-Learn](https://scikit-learn.org/) - Biblioteca de Machine Learning
*   [Pandas](https://pandas.pydata.org/) - Manipulação de Dados
*   [Joblib](https://joblib.readthedocs.io/) - Serialização de Modelos

## 📝 Uso
### Aba Treinamento:
1.  Abra a aplicação.
2.  Na barra lateral, faça o upload do seu dataset em formato **CSV**.
3.  Informe o nome da **Coluna Alvo (Target)** que deseja prever.
4.  (Opcional) Forneça uma descrição do problema para contexto.
5.  Clique em **"Iniciar Treinamento"**.
6.  Aguarde o processamento e baixe o modelo final otimizado!
7.  
### Aba Previsão:
1.  No campo 1, você sobe o arquivo .pkl que acabou de baixar (ou um antigo).
2.  No campo 2, você sobe um CSV novo (ex: novos clientes que você quer saber se vão cancelar o serviço).
**Importante:** Esse CSV novo precisa ter as mesmas colunas de características (features) que o arquivo original, mas não precisa ter a coluna Target (a resposta), nem precisa estar tratado (o modelo trata nulos e texto automaticamente).
3.  Clique em Gerar Previsões e baixe a planilha pronta com a coluna nova PREVISAO_IA.

---
*Desenvolvido com ❤️ para simplificar o Machine Learning.*
