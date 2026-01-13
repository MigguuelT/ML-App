import numpy as np
import pandas as pd
import warnings
import joblib
import os

# Sklearn Imports
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

# Modelos
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
        self.categorical_features = [] 
        self.numeric_features = []
        
    def _detect_problem_type(self, y):
        """Detecção automática do tipo de problema baseada no alvo."""
        if pd.api.types.is_float_dtype(y):
            return 'regression'
        elif pd.api.types.is_object_dtype(y) or pd.api.types.is_string_dtype(y):
            return 'classification'
        elif pd.api.types.is_integer_dtype(y):
            # Heurística: Se tiver menos de 20 valores únicos ou menos de 5% de variabilidade
            if y.nunique() < 20 or (y.nunique() / len(y) < 0.05):
                return 'classification'
            return 'regression'
        return 'regression'

    def _get_preprocessor(self, X):
        """Configura o tratamento de dados (Limpeza + Escalonamento + Encoding)."""
        # Salvar nomes das colunas para uso posterior (Visualização)
        self.numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        # Tratamento Numérico: Mediana + RobustScaler (Anti-Outliers)
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler()) 
        ])

        # Tratamento Categórico: Moda + OneHotEncoder
        # sparse_output=False é crucial para conseguirmos visualizar os dados depois
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) 
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        return preprocessor

    def _get_model_candidates(self):
        """Define os algoritmos e o grid de hiperparâmetros a serem testados."""
        if self.problem_type == 'classification':
            selector_score_func = f_classif
            scoring = 'accuracy'
            cv_split = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            
            models = [
                {
                    'name': 'HistGradientBoosting (LightGBM style)',
                    'estimator': HistGradientBoostingClassifier(random_state=self.random_state),
                    'params': {
                        'model__learning_rate': [0.01, 0.1], 
                        'model__max_iter': [100, 200], 
                        'selector__k': ['all', 10]
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
        """Método principal de treinamento."""
        if description:
            self.problem_description = description
        
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # 1. Determinar tipo
        self.problem_type = self._detect_problem_type(y)
        print(f"Tipo Detectado: {self.problem_type}")
        
        # Encoding de target se for classificação com texto
        if self.problem_type == 'classification' and y.dtype == 'object':
            self.target_encoder = LabelEncoder()
            y = self.target_encoder.fit_transform(y)

        # 2. Split Treino/Teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state)

        # 3. Preparação
        preprocessor = self._get_preprocessor(X)
        candidates, scoring_metric, cv, selector_func = self._get_model_candidates()
        
        best_score_overall = -np.inf
        best_model_overall = None
        
        # 4. Loop de Otimização
        for candidate in candidates:
            # Pipeline: Preprocess -> Select -> Model
            pipe = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('selector', SelectKBest(score_func=selector_func)),
                ('model', candidate['estimator'])
            ])
            
            print(f"Treinando {candidate['name']}...")
            
            # --- CORREÇÃO IMPORTANTE: n_jobs=1 ---
            # Define n_jobs=1 para evitar estouro de memória no Streamlit Cloud
            grid = GridSearchCV(
                pipe, 
                param_grid=candidate['params'], 
                cv=cv, 
                scoring=scoring_metric, 
                n_jobs=1 
            )
            
            grid.fit(X_train, y_train)
            
            if grid.best_score_ > best_score_overall:
                best_score_overall = grid.best_score_
                best_model_overall = grid.best_estimator_
                self.best_params = grid.best_params_

        self.best_model = best_model_overall
        print(f"Melhor modelo: {self.best_model.steps[-1][1].__class__.__name__}")

        # 5. Avaliação Final e Métricas
        y_pred = self.best_model.predict(X_test)
        final_metrics = {}

        if self.problem_type == 'classification':
            final_metrics['accuracy'] = accuracy_score(y_test, y_pred)
            final_metrics['report'] = classification_report(y_test, y_pred, output_dict=True)
        else:
            final_metrics['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
            final_metrics['mae'] = mean_absolute_error(y_test, y_pred)
            final_metrics['r2'] = r2_score(y_test, y_pred)

        return final_metrics

    def save_model(self, filename='automl_model.pkl'):
        if self.best_model:
            joblib.dump(self.best_model, filename)

    def load_model(self, filename='automl_model.pkl'):
        if os.path.exists(filename):
            self.best_model = joblib.load(filename)

    def get_encoding_examples(self, X_sample):
        """
        Gera um dicionário mostrando 'Antes vs Depois' para cada coluna categórica.
        Recebe X_sample (um DataFrame com algumas linhas raw).
        """
        if not self.best_model or len(self.categorical_features) == 0:
            return {}
        
        results = {}
        
        try:
            # 1. Acessa o preprocessor treinado
            preprocessor = self.best_model.named_steps['preprocessor']
            
            # 2. Transforma a amostra
            transformed_array = preprocessor.transform(X_sample)
            
            # 3. Pega os nomes das colunas de saída
            feature_names = preprocessor.get_feature_names_out()
            
            # 4. Cria um DataFrame com os dados transformados
            df_transformed = pd.DataFrame(transformed_array, columns=feature_names, index=X_sample.index)
            
            # 5. Para cada coluna categórica original, montamos a comparação
            for col_orig in self.categorical_features:
                related_cols = [c for c in feature_names if f"cat__{col_orig}_" in c]
                
                if related_cols:
                    comparison_df = X_sample[[col_orig]].copy()
                    comparison_df.columns = ["VALOR ORIGINAL"]
                    
                    subset_transformed = df_transformed[related_cols].copy()
                    
                    # Limpa os nomes das colunas
                    clean_names = [c.replace("cat__", "") for c in related_cols]
                    subset_transformed.columns = clean_names
                    
                    final_df = pd.concat([comparison_df, subset_transformed], axis=1)
                    results[col_orig] = final_df
            
            return results

        except Exception as e:
            print(f"Erro ao gerar exemplos de encoding: {e}")
            return {}