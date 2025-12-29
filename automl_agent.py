import numpy as np
import pandas as pd
import warnings
import joblib  # Para salvar/carregar modelos
import os

# Sklearn Imports
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

# Modelos (Incluindo HistGradientBoosting - SOTA do Scikit-Learn)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge

# Métricas
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, mean_absolute_error

warnings.filterwarnings('ignore')

class AutoMLAgentPro:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.problem_type = None
        self.problem_description = ""
        self.best_model = None
        self.best_params = None
        self.target_encoder = None
        self.results = {}
        
    def _detect_problem_type(self, y):
        """Detecção automática do tipo de problema baseada no alvo."""
        if pd.api.types.is_float_dtype(y):
            return 'regression'
        elif pd.api.types.is_object_dtype(y) or pd.api.types.is_string_dtype(y):
            return 'classification'
        elif pd.api.types.is_integer_dtype(y):
            if y.nunique() < 20 or (y.nunique() / len(y) < 0.05):
                return 'classification'
            return 'regression'
        return 'regression'

    def _get_preprocessor(self, X):
        """
        MELHORIA 1: Tratamento Robusto.
        Usa Mediana e RobustScaler para mitigar outliers.
        """
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns

        # Tratamento Numérico: Mediana + RobustScaler
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler()) 
        ])

        # Tratamento Categórico
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        return preprocessor

    def _get_model_candidates(self):
        """
        MELHORIA 2 e 3: Seleção de Features e Modelos Modernos.
        Inclui HistGradientBoosting (rápido e preciso) e SelectKBest.
        """
        
        # Define função de pontuação para seleção de features
        if self.problem_type == 'classification':
            selector_score_func = f_classif
            scoring = 'accuracy'
            cv_split = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            
            # Modelos de Classificação
            models = [
                {
                    'name': 'HistGradientBoosting (LightGBM style)',
                    'estimator': HistGradientBoostingClassifier(random_state=self.random_state),
                    'params': {
                        'model__learning_rate': [0.01, 0.1],
                        'model__max_iter': [100, 200],
                        'selector__k': ['all', 10] # Tenta usar todas features ou só as 10 melhores
                    }
                },
                {
                    'name': 'Random Forest Classifier',
                    'estimator': RandomForestClassifier(random_state=self.random_state),
                    'params': {
                        'model__n_estimators': [100],
                        'model__max_depth': [None, 10],
                        'selector__k': ['all', 10]
                    }
                },
                {
                    'name': 'Logistic Regression (Robust)',
                    'estimator': LogisticRegression(random_state=self.random_state, max_iter=1000),
                    'params': {
                        'model__C': [0.1, 1, 10],
                        'selector__k': ['all', 10]
                    }
                }
            ]
            
        else: # Regression
            selector_score_func = f_regression
            scoring = 'r2'
            cv_split = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
            
            # Modelos de Regressão
            models = [
                {
                    'name': 'HistGradientBoosting Regressor',
                    'estimator': HistGradientBoostingRegressor(random_state=self.random_state),
                    'params': {
                        'model__learning_rate': [0.01, 0.1],
                        'model__max_iter': [100, 200],
                        'selector__k': ['all', 10]
                    }
                },
                {
                    'name': 'Random Forest Regressor',
                    'estimator': RandomForestRegressor(random_state=self.random_state),
                    'params': {
                        'model__n_estimators': [100],
                        'model__max_depth': [None, 10],
                        'selector__k': ['all', 10]
                    }
                },
                {
                    'name': 'Ridge Regression',
                    'estimator': Ridge(),
                    'params': {
                        'model__alpha': [0.1, 1.0],
                        'selector__k': ['all', 10]
                    }
                }
            ]
            
        return models, scoring, cv_split, selector_score_func

    def train(self, df, target_column, description=None):
        """
        Método principal de treino.
        MELHORIA 4: Input de usuário para contexto.
        """
        print("\n" + "#"*60)
        print("   INICIANDO AGENTE AUTOML PRO 2.0   ")
        print("#"*60)

        # 0. Coleta de Contexto (Prompt do Usuário)
        if description:
            self.problem_description = description
        else:
            print("\n>>> INPUT NECESSÁRIO: Por favor, descreva brevemente o problema.")
            print("Ex: 'Prever churn de clientes' ou 'Estimar preço de casas'.")
            self.problem_description = input("Descrição: ")
        
        print(f"\nCONTEXTO DO PROBLEMA: {self.problem_description}")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # 1. Determinar tipo
        self.problem_type = self._detect_problem_type(y)
        print(f"TIPO DETECTADO: {self.problem_type.upper()}")

        # Encoding de target se necessário
        if self.problem_type == 'classification' and y.dtype == 'object':
            self.target_encoder = LabelEncoder()
            y = self.target_encoder.fit_transform(y)

        # 2. Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state)

        # 3. Preprocessor
        preprocessor = self._get_preprocessor(X)

        # 4. Configuração de Candidatos
        candidates, scoring_metric, cv, selector_func = self._get_model_candidates()
        
        best_score_overall = -np.inf
        best_model_overall = None
        
        print(f"\n>>> Otimizando modelos usando métrica: {scoring_metric.upper()}")
        
        for candidate in candidates:
            # PIPELINE ATUALIZADO: Preprocess -> Selector -> Model
            pipe = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('selector', SelectKBest(score_func=selector_func)),
                ('model', candidate['estimator'])
            ])
            
            print(f"   > Treinando: {candidate['name']}...")
            
            grid = GridSearchCV(pipe, param_grid=candidate['params'], cv=cv, scoring=scoring_metric, n_jobs=-1)
            grid.fit(X_train, y_train)
            
            print(f"     Melhor Validação: {grid.best_score_:.4f}")
            
            if grid.best_score_ > best_score_overall:
                best_score_overall = grid.best_score_
                best_model_overall = grid.best_estimator_
                self.best_params = grid.best_params_

        self.best_model = best_model_overall
        print(f"\n>>> MODELO VENCEDOR: {self.best_model.steps[-1][1].__class__.__name__}")
        print(f">>> Score Validação Cruzada: {best_score_overall:.4f}")

        # 5. Avaliação Final
        print("\n" + "-"*30)
        print(" RELATÓRIO FINAL (DADOS DE TESTE)")
        print("-" * 30)
        y_pred = self.best_model.predict(X_test)
        
        if self.problem_type == 'classification':
            print(classification_report(y_test, y_pred))
        else:
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE:  {mae:.4f}")
            print(f"R2:   {r2:.4f}")

    def save_model(self, filename='automl_model.pkl'):
        """MELHORIA 5: Persistência do Modelo."""
        if self.best_model:
            joblib.dump(self.best_model, filename)
            print(f"\n[SISTEMA] Modelo salvo com sucesso em '{filename}'")
        else:
            print("\n[ERRO] Nenhum modelo treinado para salvar.")

    def load_model(self, filename='automl_model.pkl'):
        if os.path.exists(filename):
            self.best_model = joblib.load(filename)
            print(f"\n[SISTEMA] Modelo carregado de '{filename}'")
        else:
            print(f"\n[ERRO] Arquivo '{filename}' não encontrado.")

# ==========================================
# EXEMPLE DE EXECUÇÃO
# ==========================================

if __name__ == "__main__":
    from sklearn.datasets import make_regression
    
    # 1. Gerar dados sujos (com outliers)
    print("Gerando dados de teste...")
    X, y = make_regression(n_samples=500, n_features=10, noise=0.5, random_state=42)
    df = pd.DataFrame(X, columns=[f'f{i}' for i in range(10)])
    
    # Introduzindo Outliers propositalmente
    df.loc[0:10, 'f0'] = 100000.0 
    df['target'] = y
    
    # 2. Instanciar Agente
    agent = AutoMLAgentPro()
    
    # 3. Treinar (Aqui ele vai pedir input se não passarmos description)
    # Para teste automático, passamos a descrição direta
    agent.train(df, target_column='target', description="Previsão de valores com Outliers Extremos")
    
    # 4. Salvar
    agent.save_model("modelo_teste.pkl")