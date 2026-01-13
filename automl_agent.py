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
        
    def _smart_type_conversion(self, df):
        """
        CORREÇÃO CRÍTICA: Tenta converter colunas 'object' que parecem números.
        Resolve o problema de '10,5' (BR) ser lido como texto.
        """
        df_clean = df.copy()
        for col in df_clean.columns:
            # Se a coluna é do tipo objeto (texto)
            if df_clean[col].dtype == 'object':
                try:
                    # 1. Tenta substituir vírgula por ponto (Padrão BR -> US)
                    # convertemos para string primeiro para garantir
                    series_fixed = df_clean[col].astype(str).str.replace(',', '.')
                    
                    # 2. Tenta converter para número
                    # errors='raise' garante que só converte se TODOS os valores forem numéricos
                    # Se tiver um "Masculino" no meio, ele falha e vai para o 'except', mantendo como texto
                    series_numeric = pd.to_numeric(series_fixed)
                    
                    # Se chegou aqui, funcionou! Substitui a coluna.
                    df_clean[col] = series_numeric
                    print(f"   [AutoML] Coluna '{col}' convertida de Texto para Numérico com sucesso.")
                except:
                    # Se deu erro (tem letras misturadas), mantém como texto original
                    pass
        return df_clean

    def _detect_problem_type(self, y):
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
        self.numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        print(f"   > Numéricas: {len(self.numeric_features)} | Categóricas: {len(self.categorical_features)}")

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler()) 
        ])

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
        # Configuração Otimizada para Performance
        FOLDS = 3 
        
        if self.problem_type == 'classification':
            selector_score_func = f_classif
            scoring = 'accuracy'
            cv_split = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=self.random_state)
            
            models = [
                {
                    'name': 'HistGradientBoosting',
                    'estimator': HistGradientBoostingClassifier(random_state=self.random_state),
                    'params': {
                        'model__learning_rate': [0.1], 
                        'model__max_iter': [100], 
                        'selector__k': ['all', 10]
                    }
                },
                {
                    'name': 'Random Forest',
                    'estimator': RandomForestClassifier(random_state=self.random_state),
                    'params': {
                        'model__n_estimators': [50], 
                        'model__max_depth': [10], 
                        'selector__k': ['all']
                    }
                },
                {
                    'name': 'Logistic Regression',
                    'estimator': LogisticRegression(random_state=self.random_state, max_iter=1000),
                    'params': {'model__C': [1], 'selector__k': ['all']}
                }
            ]
            
        else: # Regression
            selector_score_func = f_regression
            scoring = 'r2'
            cv_split = KFold(n_splits=FOLDS, shuffle=True, random_state=self.random_state)
            
            models = [
                {
                    'name': 'HistGradientBoosting Regressor',
                    'estimator': HistGradientBoostingRegressor(random_state=self.random_state),
                    'params': {
                        'model__learning_rate': [0.1],
                        'model__max_iter': [100],
                        'selector__k': ['all', 10]
                    }
                },
                {
                    'name': 'Random Forest Regressor',
                    'estimator': RandomForestRegressor(random_state=self.random_state),
                    'params': {
                        'model__n_estimators': [50],
                        'model__max_depth': [10],
                        'selector__k': ['all']
                    }
                },
                {
                    'name': 'Ridge Regression',
                    'estimator': Ridge(),
                    'params': {'model__alpha': [1.0], 'selector__k': ['all']}
                }
            ]
            
        return models, scoring, cv_split, selector_score_func

    def train(self, df, target_column, description=None):
        if description: self.problem_description = description
        
        # 1. LIMPEZA INTELIGENTE (Correção Decimal BR)
        print("Iniciando conversão inteligente de tipos...")
        df = self._smart_type_conversion(df)

        X = df.drop(columns=[target_column])
        y = df[target_column]

        self.problem_type = self._detect_problem_type(y)
        print(f"Tipo Detectado: {self.problem_type}")
        
        if self.problem_type == 'classification' and y.dtype == 'object':
            self.target_encoder = LabelEncoder()
            y = self.target_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state)

        preprocessor = self._get_preprocessor(X)
        candidates, scoring_metric, cv, selector_func = self._get_model_candidates()
        
        best_score_overall = -np.inf
        best_model_overall = None
        
        for candidate in candidates:
            pipe = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('selector', SelectKBest(score_func=selector_func)),
                ('model', candidate['estimator'])
            ])
            
            print(f"Treinando {candidate['name']}...")
            
            # n_jobs=1 para memória, verbose=3 para log
            grid = GridSearchCV(
                pipe, 
                param_grid=candidate['params'], 
                cv=cv, 
                scoring=scoring_metric, 
                n_jobs=1,
                verbose=3 
            )
            
            grid.fit(X_train, y_train)
            
            if grid.best_score_ > best_score_overall:
                best_score_overall = grid.best_score_
                best_model_overall = grid.best_estimator_
                self.best_params = grid.best_params_

        self.best_model = best_model_overall
        print(f"Melhor modelo: {self.best_model.steps[-1][1].__class__.__name__}")

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
        if self.best_model: joblib.dump(self.best_model, filename)

    def load_model(self, filename='automl_model.pkl'):
        if os.path.exists(filename): self.best_model = joblib.load(filename)

    def get_encoding_examples(self, X_sample):
        # O método de visualização precisa tratar os dados da mesma forma que o treino
        # Aplicamos a conversão inteligente na amostra também para não dar erro
        X_sample_clean = self._smart_type_conversion(X_sample)

        if not self.best_model or len(self.categorical_features) == 0:
            return {}
        
        results = {}
        try:
            preprocessor = self.best_model.named_steps['preprocessor']
            transformed_array = preprocessor.transform(X_sample_clean)
            feature_names = preprocessor.get_feature_names_out()
            df_transformed = pd.DataFrame(transformed_array, columns=feature_names, index=X_sample_clean.index)
            
            for col_orig in self.categorical_features:
                related_cols = [c for c in feature_names if f"cat__{col_orig}_" in c]
                if related_cols:
                    comparison_df = X_sample_clean[[col_orig]].copy()
                    comparison_df.columns = ["VALOR ORIGINAL"]
                    subset_transformed = df_transformed[related_cols].copy()
                    clean_names = [c.replace("cat__", "") for c in related_cols]
                    subset_transformed.columns = clean_names
                    final_df = pd.concat([comparison_df, subset_transformed], axis=1)
                    results[col_orig] = final_df
            return results
        except Exception as e:
            print(f"Erro encoding: {e}")
            return {}