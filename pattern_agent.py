"""
Agente Multi-Modelo para Detecção de Padrões em Sequências Numéricas

Este agente combina múltiplas bibliotecas de machine learning e análise de séries temporais
para detectar padrões em sequências de números e sugerir o próximo número.

Bibliotecas utilizadas:
- STUMPY: Detecção de padrões repetidos (motifs)
- tsfresh: Extração automática de features
- tslearn: Clustering de sequências similares
- PyTorch: Deep Learning (LSTM)
- statsmodels: Análise estatística
- Darts: Forecasting de séries temporais
- Prophet: Detecção de tendências
- scikit-learn: Modelos clássicos de ML
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import warnings
import json
import os
warnings.filterwarnings('ignore')

# Bibliotecas de análise de padrões
try:
    import stumpy
except ImportError:
    stumpy = None
    print("Aviso: STUMPY não instalado")

try:
    from tsfresh import extract_features, select_features
    from tsfresh.utilities.dataframe_functions import impute
except ImportError:
    extract_features = None
    print("Aviso: tsfresh não instalado")

try:
    from tslearn.clustering import TimeSeriesKMeans
    from tslearn.preprocessing import TimeSeriesScalerMeanVariance
except ImportError:
    TimeSeriesKMeans = None
    print("Aviso: tslearn não instalado")

try:
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
except ImportError:
    sm = None
    print("Aviso: statsmodels não instalado")

try:
    from darts import TimeSeries
    from darts.models import LSTM, ExponentialSmoothing, Prophet as DartsProphet
except ImportError:
    TimeSeries = None
    print("Aviso: darts não instalado")

try:
    from prophet import Prophet
except ImportError:
    Prophet = None
    print("Aviso: prophet não instalado")

# Deep Learning
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    torch = None
    print("Aviso: PyTorch não instalado")

# Scikit-learn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class SequenceDataset(Dataset):
    """Dataset PyTorch para sequências numéricas"""
    def __init__(self, sequences, window_size=10):
        self.sequences = sequences
        self.window_size = window_size
        self.X, self.y = self._create_windows()
    
    def _create_windows(self):
        X, y = [], []
        for seq in self.sequences:
            for i in range(len(seq) - self.window_size):
                X.append(seq[i:i+self.window_size])
                y.append(seq[i+self.window_size])
        return torch.FloatTensor(X), torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMPredictor(nn.Module):
    """Modelo LSTM para predição de sequências"""
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x.unsqueeze(-1))
        out = self.fc(out[:, -1, :])
        return out


class PatternAgent:
    """
    Agente Multi-Modelo para Detecção de Padrões
    
    Combina múltiplas abordagens de ML para analisar sequências e sugerir próximo número.
    """
    
    def __init__(self, min_value=1, max_value=25, sequence_length=15):
        self.min_value = min_value
        self.max_value = max_value
        self.sequence_length = sequence_length
        
        # Armazenamento de dados
        self.all_sequences = []
        self.processed_sequences = []
        
        # Modelos
        self.lstm_model = None
        self.rf_model = None
        self.gb_model = None
        self.clusterer = None
        self.scaler = StandardScaler()
        
        # Cache de análises
        self.motifs_cache = {}
        self.features_cache = {}
        self.clusters_cache = {}
        
        # Histórico de predições
        self.prediction_history = []
        
        # Feedback do último confronto: evitar repetir erros, favorecer os que faltaram
        self.last_avoid_numbers = []   # números que geramos mas estavam errados
        self.last_prefer_numbers = []  # números corretos que não geramos
        
        # Performance tracking por modelo (para ajuste dinâmico de pesos)
        self.model_performance = {
            'lstm': {'hits': 0, 'total': 0, 'accuracy': 0.0},
            'stumpy': {'hits': 0, 'total': 0, 'accuracy': 0.0},
            'random_forest': {'hits': 0, 'total': 0, 'accuracy': 0.0},
            'gradient_boosting': {'hits': 0, 'total': 0, 'accuracy': 0.0},
            'arima': {'hits': 0, 'total': 0, 'accuracy': 0.0},
            'darts': {'hits': 0, 'total': 0, 'accuracy': 0.0},
            'prophet': {'hits': 0, 'total': 0, 'accuracy': 0.0}
        }
        
        # Pesos dinâmicos (carrega pesos salvos ou usa valores padrão)
        self.weights_file = 'model_weights.json'
        # Carrega pesos salvos (se existirem) ou usa padrão
        self.model_weights = self._load_model_weights()
        
        # Calibração por OBJETIVO: sequência completa (próxima sequência de 15), não "próximo número"
        # True = acerto quando a predição do modelo está no CONJUNTO da próxima sequência (in-set)
        self.use_calibration_by_sequence = True
        # Legado: só usado quando correct_set não é passado (calibração por "próximo número")
        self.use_exact_match_for_calibration = True
        # Geração por sequência inteira: analisa e gera os 15 de uma vez (objetivo 13+/14+)
        self.use_whole_sequence_generation = True
        
        # Correlação entre linhas (7.2): K e pesos configuráveis para usar mais o histórico
        self.k_whole_sequence = 200   # Número de sequências similares no whole-sequence (aumentar com muitas linhas, ex.: 300)
        self.k_knn = 80               # Número de vizinhos no K-NN do fallback (aumentar com muitas linhas, ex.: 120)
        self.weight_successor_sequence = 1.5   # Peso da "sequência seguinte" ( > 1 = mais influência do que costuma vir depois)
        self.weight_knn_suggestions = 1.4      # Peso das sugestões K-NN no fallback ( > 1 = mais influência de linhas parecidas)
        self.auto_k_for_large_history = True   # Se True, com muitas linhas (>400) usa K_whole=300 e K_knn=120 automaticamente
        self.n_suggested_sequences = 3         # Número de sequências alternativas a gerar (mais opções = mais chances de acerto)
        
    def load_sequences_from_excel(self, file_path: str, sheet_name: Optional[str] = None) -> List[List[int]]:

        try:
            # Se sheet_name é None, lê a primeira planilha (padrão)
            # Se for um dicionário (múltiplas planilhas), pega a primeira
            excel_data = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
            
            # Se retornou um dicionário (múltiplas planilhas), pega a primeira
            if isinstance(excel_data, dict):
                df = list(excel_data.values())[0]
            else:
                df = excel_data
            
            sequences = []
            
            for _, row in df.iterrows():
                # Filtra valores válidos e converte para int
                seq = []
                for x in row.values:
                    if pd.notna(x):
                        try:
                            seq.append(int(float(x)))  # Converte para float primeiro para lidar com decimais
                        except (ValueError, TypeError):
                            continue
                
                # Aceita sequências com tamanho exato ou próximo
                if len(seq) >= self.sequence_length:
                    sequences.append(seq[:self.sequence_length])
                elif len(seq) > 0:
                    # Se a sequência é menor mas tem dados válidos, completa com o último valor
                    while len(seq) < self.sequence_length:
                        seq.append(seq[-1] if seq else 1)
                    sequences.append(seq[:self.sequence_length])
            
            self.all_sequences = sequences
            print(f"[OK] Carregadas {len(sequences)} sequencias de {self.sequence_length} numeros")
            return sequences
        
        except Exception as e:
            import traceback
            print(f"Erro ao carregar sequências: {e}")
            print(f"Detalhes: {traceback.format_exc()}")
            return []
    
    def preprocess_sequences(self):
        """Normaliza e prepara sequências para análise"""
        if not self.all_sequences:
            return
        
        # Normalização para análise (mantém valores originais também)
        self.processed_sequences = []
        for seq in self.all_sequences:
            # Normaliza para [0, 1]
            normalized = [(x - self.min_value) / (self.max_value - self.min_value) 
                         for x in seq]
            self.processed_sequences.append(normalized)
        
        print(f"[OK] {len(self.processed_sequences)} sequencias pre-processadas")
    
    def detect_motifs_stumpy(self, sequence: List[float], m: int = 5) -> Dict:
        """
        Detecta padrões repetidos usando STUMPY (Matrix Profile)
        
        Args:
            sequence: Sequência normalizada
            m: Tamanho do padrão a buscar
        
        Returns:
            Dicionário com informações sobre padrões encontrados
        """
        if stumpy is None or len(sequence) < m * 2:
            return {'found': False, 'next_candidates': []}
        
        try:
            mp = stumpy.stump(sequence, m=m)
            matrix_profile = mp[:, 0]
            
            # Encontra os melhores matches (menores distâncias)
            if len(matrix_profile) > 0:
                min_idx = np.argmin(matrix_profile)
                min_dist = matrix_profile[min_idx]
                
                # Se encontrou um padrão similar
                if min_dist < 0.3:  # Threshold de similaridade
                    # Busca o próximo número após o padrão encontrado
                    pattern_end = min_idx + m
                    if pattern_end < len(sequence):
                        next_val = sequence[pattern_end]
                        # Converte de volta para o range original
                        original_val = int(next_val * (self.max_value - self.min_value) + self.min_value)
                        return {
                            'found': True,
                            'next_candidates': [original_val],
                            'confidence': 1.0 - min_dist,
                            'pattern_position': min_idx
                        }
            
            return {'found': False, 'next_candidates': []}
        
        except Exception as e:
            return {'found': False, 'next_candidates': [], 'error': str(e)}
    
    def extract_features_tsfresh(self, sequence: List[float]) -> Dict:
        """
        Extrai features automáticas usando tsfresh
        
        Args:
            sequence: Sequência normalizada
        
        Returns:
            Dicionário com features extraídas
        """
        if extract_features is None:
            return {}
        
        try:
            # Prepara dados no formato tsfresh
            df = pd.DataFrame({
                'id': [0] * len(sequence),
                'time': range(len(sequence)),
                'value': sequence
            })
            
            # Extrai features
            extracted = extract_features(df, column_id='id', column_sort='time', 
                                        column_value='value', impute_function=impute)
            
            if not extracted.empty:
                # Retorna as features mais relevantes
                features = extracted.iloc[0].to_dict()
                return features
            
            return {}
        
        except Exception as e:
            return {'error': str(e)}
    
    def cluster_sequences_tslearn(self, sequences: List[List[float]], n_clusters: int = 5) -> Dict:

        if TimeSeriesKMeans is None or len(sequences) < n_clusters:
            return {}
        
        try:
            # Prepara dados
            X = np.array(sequences)
            
            # Clustering com DTW
            km = TimeSeriesKMeans(n_clusters=min(n_clusters, len(sequences)), 
                                 metric="dtw", max_iter=10, random_state=42)
            labels = km.fit_predict(X)
            
            # Retorna informações dos clusters
            cluster_info = {}
            for i in range(min(n_clusters, len(sequences))):
                cluster_seqs = [seq for seq, label in zip(sequences, labels) if label == i]
                if cluster_seqs:
                    # Média do último número de cada sequência no cluster
                    last_values = [seq[-1] for seq in cluster_seqs]
                    avg_last = np.mean(last_values)
                    original_val = int(avg_last * (self.max_value - self.min_value) + self.min_value)
                    cluster_info[i] = {
                        'size': len(cluster_seqs),
                        'predicted_next': original_val,
                        'confidence': len(cluster_seqs) / len(sequences)
                    }
            
            return cluster_info
        
        except Exception as e:
            return {'error': str(e)}
    
    def train_lstm(self, sequences: List[List[float]], epochs: int = 50, batch_size: int = 32):
        """
        Treina modelo LSTM para predição
        
        Args:
            sequences: Lista de sequências normalizadas
            epochs: Número de épocas
            batch_size: Tamanho do batch
        """
        if torch is None or len(sequences) < 10:
            return
        
        try:
            # Prepara dados
            dataset = SequenceDataset(sequences, window_size=10)
            if len(dataset) == 0:
                return
            
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Cria modelo
            self.lstm_model = LSTMPredictor(input_size=1, hidden_size=64, num_layers=2)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=0.001)
            
            # Treina
            self.lstm_model.train()
            for epoch in range(epochs):
                total_loss = 0
                for X_batch, y_batch in dataloader:
                    optimizer.zero_grad()
                    outputs = self.lstm_model(X_batch)
                    loss = criterion(outputs.squeeze(), y_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                if (epoch + 1) % 10 == 0:
                    print(f"  LSTM Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
            
            print("[OK] Modelo LSTM treinado")
        
        except Exception as e:
            print(f"Erro ao treinar LSTM: {e}")
    
    def predict_with_lstm(self, sequence: List[float]) -> Optional[int]:
        """Prediz próximo número usando LSTM"""
        if self.lstm_model is None or len(sequence) < 10:
            return None
        
        try:
            self.lstm_model.eval()
            with torch.no_grad():
                # Usa últimos 10 números
                window = sequence[-10:]
                x = torch.FloatTensor(window).unsqueeze(0)
                pred = self.lstm_model(x).item()
                
                # Converte de volta
                pred = max(self.min_value, min(self.max_value, 
                      int(pred * (self.max_value - self.min_value) + self.min_value)))
                return pred
        except:
            return None
    
    def train_sklearn_models(self, sequences: List[List[float]]):
        """Treina modelos Random Forest e Gradient Boosting"""
        if len(sequences) < 10:
            return
        
        try:
            X, y = [], []
            for seq in sequences:
                for i in range(len(seq) - 1):
                    X.append(seq[:i+1])
                    y.append(seq[i+1])
            
            # Padding para tamanho fixo
            max_len = max(len(x) for x in X)
            X_padded = []
            for x in X:
                padded = x + [0] * (max_len - len(x))
                X_padded.append(padded[:max_len])
            
            X_array = np.array(X_padded)
            y_array = np.array(y)
            
            # Treina Random Forest
            self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            self.rf_model.fit(X_array, y_array)
            
            # Treina Gradient Boosting
            self.gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            self.gb_model.fit(X_array, y_array)
            
            print("[OK] Modelos scikit-learn treinados")
        
        except Exception as e:
            print(f"Erro ao treinar modelos sklearn: {e}")
    
    def predict_with_sklearn(self, sequence: List[float]) -> List[int]:
        """Prediz usando modelos sklearn"""
        predictions = []
        
        if self.rf_model is not None:
            try:
                # Prepara entrada
                max_len = self.rf_model.n_features_in_
                padded = sequence + [0] * (max_len - len(sequence))
                padded = padded[:max_len]
                
                pred_rf = self.rf_model.predict([padded])[0]
                pred_rf = max(self.min_value, min(self.max_value, 
                      int(pred_rf * (self.max_value - self.min_value) + self.min_value)))
                predictions.append(pred_rf)
            except:
                pass
        
        if self.gb_model is not None:
            try:
                max_len = self.gb_model.n_features_in_
                padded = sequence + [0] * (max_len - len(sequence))
                padded = padded[:max_len]
                
                pred_gb = self.gb_model.predict([padded])[0]
                pred_gb = max(self.min_value, min(self.max_value, 
                      int(pred_gb * (self.max_value - self.min_value) + self.min_value)))
                predictions.append(pred_gb)
            except:
                pass
        
        return predictions
    
    def predict_with_arima(self, sequence: List[float]) -> Optional[int]:
        """Prediz usando ARIMA (statsmodels)"""
        if sm is None or len(sequence) < 5:
            return None
        
        try:
            # Converte para série temporal
            ts = pd.Series(sequence)
            
            # Tenta ARIMA(1,1,1)
            model = ARIMA(ts, order=(1, 1, 1))
            fitted = model.fit()
            
            # Prediz próximo valor
            forecast = fitted.forecast(steps=1)
            pred = forecast.iloc[0] if hasattr(forecast, 'iloc') else forecast[0]
            
            # Converte de volta
            pred = max(self.min_value, min(self.max_value, 
                  int(pred * (self.max_value - self.min_value) + self.min_value)))
            return pred
        except:
            return None
    
    def predict_with_darts(self, sequence: List[float]) -> Optional[int]:
        """Prediz usando Darts"""
        if TimeSeries is None or len(sequence) < 10:
            return None
        
        try:
            # Cria TimeSeries
            ts = TimeSeries.from_values(np.array(sequence))
            
            # Usa Exponential Smoothing (rápido)
            model = ExponentialSmoothing()
            model.fit(ts)
            
            # Prediz
            forecast = model.predict(1)
            pred = forecast.values()[0][0]
            
            # Converte de volta
            pred = max(self.min_value, min(self.max_value, 
                  int(pred * (self.max_value - self.min_value) + self.min_value)))
            return pred
        except:
            return None
    
    def predict_with_prophet(self, sequence: List[float]) -> Optional[int]:
        """Prediz usando Prophet"""
        if Prophet is None or len(sequence) < 10:
            return None
        
        try:
            # Prepara dados no formato Prophet
            df = pd.DataFrame({
                'ds': pd.date_range('2024-01-01', periods=len(sequence), freq='D'),
                'y': sequence
            })
            
            model = Prophet()
            model.fit(df)
            
            # Prediz próximo valor
            future = model.make_future_dataframe(periods=1)
            forecast = model.predict(future)
            pred = forecast['yhat'].iloc[-1]
            
            # Converte de volta
            pred = max(self.min_value, min(self.max_value, 
                  int(pred * (self.max_value - self.min_value) + self.min_value)))
            return pred
        except:
            return None
    
    def analyze_all_patterns(self, sequence: List[int]) -> Dict:
        """
        Analisa uma sequência usando TODOS os modelos disponíveis
        
        Args:
            sequence: Sequência original (valores de 1 a 25)
        
        Returns:
            Dicionário com todas as predições e análises
        """
        # Normaliza sequência
        normalized = [(x - self.min_value) / (self.max_value - self.min_value) 
                     for x in sequence]
        
        results = {
            'sequence': sequence,
            'predictions': {},
            'analyses': {}
        }
        
        # 1. STUMPY - Detecção de motifs
        motifs = self.detect_motifs_stumpy(normalized)
        if motifs.get('found'):
            results['predictions']['stumpy'] = motifs['next_candidates'][0]
            results['analyses']['stumpy'] = motifs
        
        # 2. LSTM
        lstm_pred = self.predict_with_lstm(normalized)
        if lstm_pred:
            results['predictions']['lstm'] = lstm_pred
        
        # 3. Scikit-learn
        sklearn_preds = self.predict_with_sklearn(normalized)
        if sklearn_preds:
            results['predictions']['random_forest'] = sklearn_preds[0] if len(sklearn_preds) > 0 else None
            results['predictions']['gradient_boosting'] = sklearn_preds[1] if len(sklearn_preds) > 1 else None
        
        # 4. ARIMA
        arima_pred = self.predict_with_arima(normalized)
        if arima_pred:
            results['predictions']['arima'] = arima_pred
        
        # 5. Darts
        darts_pred = self.predict_with_darts(normalized)
        if darts_pred:
            results['predictions']['darts'] = darts_pred
        
        # 6. Prophet
        prophet_pred = self.predict_with_prophet(normalized)
        if prophet_pred:
            results['predictions']['prophet'] = prophet_pred
        
        # 7. tsfresh features (análise) - DESABILITADO para performance (muito lento)
        # features = self.extract_features_tsfresh(normalized)
        # if features:
        #     results['analyses']['tsfresh_features'] = features
        
        return results
    
    def ensemble_predict(self, sequence: List[int], method: str = 'weighted_vote') -> Dict:
        """
        Combina predições de todos os modelos usando ensemble
        
        Args:
            sequence: Sequência original
            method: Método de combinação ('weighted_vote', 'average', 'majority')
        
        Returns:
            Predição final combinada com confiança
        """
        analysis = self.analyze_all_patterns(sequence)
        predictions = analysis['predictions']
        
        if not predictions:
            return {
                'prediction': None,
                'confidence': 0.0,
                'method': method,
                'individual_predictions': {}
            }
        
        # Coleta todas as predições
        all_preds = [v for v in predictions.values() if v is not None]
        
        if method == 'weighted_vote':
            # Usa pesos dinâmicos ajustados durante o treinamento
            weights = self.model_weights
            
            # Conta votos ponderados
            vote_count = {}
            for model, pred in predictions.items():
                if pred is not None:
                    weight = weights.get(model, 0.1)
                    vote_count[pred] = vote_count.get(pred, 0) + weight
            
            if vote_count:
                best_pred = max(vote_count.items(), key=lambda x: x[1])
                confidence = min(1.0, best_pred[1] / sum(weights.values()))
                
                return {
                    'prediction': best_pred[0],
                    'confidence': confidence,
                    'method': method,
                    'individual_predictions': predictions,
                    'vote_weights': vote_count
                }
        
        elif method == 'average':
            avg_pred = int(np.mean(all_preds))
            std_pred = np.std(all_preds)
            confidence = max(0.0, 1.0 - (std_pred / (self.max_value - self.min_value)))
            
            return {
                'prediction': avg_pred,
                'confidence': confidence,
                'method': method,
                'individual_predictions': predictions
            }
        
        elif method == 'majority':
            from collections import Counter
            counter = Counter(all_preds)
            most_common = counter.most_common(1)[0]
            
            return {
                'prediction': most_common[0],
                'confidence': most_common[1] / len(all_preds),
                'method': method,
                'individual_predictions': predictions
            }
        
        return {
            'prediction': all_preds[0] if all_preds else None,
            'confidence': 0.5,
            'method': method,
            'individual_predictions': predictions
        }
    
    def train_full_system(self, sequences: Optional[List[List[int]]] = None, progress_callback=None):

        if sequences is None:
            sequences = self.all_sequences
        
        if not sequences:
            print("Nenhuma sequência disponível para treino")
            return
        
        print(f"\n[INFO] Iniciando treinamento do sistema completo com {len(sequences)} sequencias...")
        
        # Pré-processa
        self.preprocess_sequences()
        
        # 1. Treina modelos de ML
        print("\n[INFO] Treinando modelos de Machine Learning...")
        self.train_sklearn_models(self.processed_sequences)
        
        # 2. Treina LSTM
        print("\n[INFO] Treinando modelo LSTM...")
        self.train_lstm(self.processed_sequences, epochs=30, batch_size=32)
        
        # 3. Clustering de sequências
        print("\n[INFO] Analisando clusters de sequencias...")
        if len(self.processed_sequences) >= 10:
            clusters = self.cluster_sequences_tslearn(self.processed_sequences, n_clusters=min(10, len(sequences)//10))
            self.clusters_cache = clusters
            print(f"[OK] {len(clusters)} clusters identificados")
        
        # 4. Calibra pesos com o máximo de sequências possível, para a geração usar TODO o aprendizado
        num_for_feedback = max(0, len(sequences) - 10)
        if num_for_feedback > 0:
            try:
                self.simulate_feedback_on_history(num_sequences=num_for_feedback, progress_callback=progress_callback)
            except Exception as e:
                print(f"[AVISO] Erro na simulação de feedback: {e}. Continuando com pesos padrão.")
        else:
            try:
                # Com poucas sequências: avalia em TODAS para que os pesos reflitam todo o padrão aprendido
                self.evaluate_models_performance(sequences)
                self.update_model_weights()
            except Exception as e:
                print(f"[AVISO] Erro na avaliação: {e}. Continuando com pesos padrão.")
        
        print("\n[OK] Sistema treinado com sucesso!")
        print("\n[PESOS] Pesos ajustados baseados na performance:")
        for model, weight in sorted(self.model_weights.items(), key=lambda x: x[1], reverse=True):
            perf = self.model_performance[model]
            if perf['total'] > 0:
                print(f"  {model}: {weight:.3f} (acurácia: {perf['accuracy']:.2%})")
    
    def evaluate_models_performance(self, sequences: List[List[int]], test_size: int = 20):
        """
        Avalia cada modelo por PARES (sequência anterior → próxima sequência).
        Acerto = a predição do modelo (dado o contexto da seq. anterior) está no CONJUNTO
        da próxima sequência (objetivo: próxima sequência completa, não próximo número).
        Atualiza model_performance e os pesos são usados na geração (weighted_vote).
        """
        # Zera performance para esta avaliação refletir só as sequências atuais
        for model_name in self.model_performance:
            self.model_performance[model_name]['hits'] = 0
            self.model_performance[model_name]['total'] = 0
            self.model_performance[model_name]['accuracy'] = 0.0

        # Lista de sequências com 15+ números para formar pares (anterior, próxima)
        all_valid = [s for s in sequences if len(s) >= 15]
        if len(all_valid) < 2:
            print("[AVALIAÇÃO] Precisa de pelo menos 2 sequências para calibrar por pares (anterior → próxima).")
            return

        # Pares (contexto, próxima sequência): ordem temporal = linha N → linha N+1
        if len(all_valid) <= 50:
            pairs = [(all_valid[i - 1], all_valid[i]) for i in range(1, len(all_valid))]
            print(f"[AVALIAÇÃO] Calibração por SEQUÊNCIA: {len(pairs)} pares (anterior → próxima).")
        else:
            # Amostra dos pares mais recentes
            use = all_valid[-max(test_size, 100):]
            pairs = [(use[i - 1], use[i]) for i in range(1, len(use))]
            if len(pairs) > 200:
                pairs = pairs[-200:]
            print(f"[AVALIAÇÃO] Calibração por SEQUÊNCIA: {len(pairs)} pares (anterior → próxima, amostra).")

        n_pairs = len(pairs)
        for idx, (context_seq, next_seq) in enumerate(pairs, 1):
            if idx % 50 == 0:
                print(f"[AVALIAÇÃO] Processando par {idx}/{n_pairs}...")
            # Contexto = sequência anterior (completa ou pelo menos 3 números)
            context = context_seq[:15] if len(context_seq) >= 3 else next_seq[:10]
            if len(context) < 3:
                continue
            correct_set = set(next_seq[:15])
            analysis = self.analyze_all_patterns(context)
            self.update_weights_incremental(analysis['predictions'], correct_set=correct_set)

        # Recalcula acurácia final por modelo (update_weights_incremental já atualiza, mas garantir)
        for model_name in self.model_performance:
            perf = self.model_performance[model_name]
            if perf['total'] > 0:
                perf['accuracy'] = perf['hits'] / perf['total']
            else:
                perf['accuracy'] = 0.0
    
    def simulate_feedback_on_history(self, num_sequences: int = 200, progress_callback=None):

        if not self.all_sequences or num_sequences <= 0:
            return
        for model_name in self.model_performance:
            self.model_performance[model_name]['hits'] = 0
            self.model_performance[model_name]['total'] = 0
            self.model_performance[model_name]['accuracy'] = 0.0
        to_use = [s for s in self.all_sequences[-num_sequences:] if len(s) >= 15]
        pairs = [(to_use[i - 1], to_use[i]) for i in range(1, len(to_use))]
        total_pairs = len(pairs)
        if total_pairs == 0:
            print("[SIMULAÇÃO] Poucos pares (anterior → próxima) para calibrar.")
            return
        print(f"[SIMULAÇÃO] Calibração por PRÓXIMA SEQUÊNCIA: {total_pairs} pares (anterior → próxima)...")
        if progress_callback:
            progress_callback(0, total_pairs, f"Iniciando: 0/{total_pairs} pares")
        for idx, (context_seq, next_seq) in enumerate(pairs):
            context = context_seq[:15] if len(context_seq) >= 3 else next_seq[:10]
            if len(context) < 3:
                continue
            correct_set = set(next_seq[:15])
            analysis = self.analyze_all_patterns(context)
            self.update_weights_incremental(analysis['predictions'], correct_set=correct_set)
            current = idx + 1
            if current % 100 == 0:
                print(f"[SIMULAÇÃO] Processados {current}/{total_pairs} pares...")
            if progress_callback:
                progress_callback(current, total_pairs, f"Simulação: {current}/{total_pairs} pares")
        self.update_model_weights()
        self.save_model_weights()
        print(f"[SIMULAÇÃO] Concluída: {total_pairs} pares (pesos calibrados para 'próxima sequência completa').")
        if progress_callback:
            progress_callback(total_pairs, total_pairs, "Simulação concluída.")

    def update_model_weights(self):
        """
        Atualiza os pesos dos modelos baseado na performance
        Modelos com maior acurácia recebem mais peso
        """
        # Coleta acurácias
        accuracies = {}
        for model_name, perf in self.model_performance.items():
            accuracies[model_name] = perf['accuracy']
        
        # Se nenhum modelo foi avaliado, mantém pesos padrão
        if all(acc == 0.0 for acc in accuracies.values()):
            return
        
        # Normaliza acurácias (adiciona pequeno valor para evitar zero)
        min_acc = min(accuracies.values())
        max_acc = max(accuracies.values())
        
        if max_acc == min_acc:
            # Todos têm a mesma performance, mantém pesos iguais
            equal_weight = 1.0 / len(accuracies)
            for model_name in self.model_weights:
                self.model_weights[model_name] = equal_weight
        else:
            # Normaliza para 0-1 e aplica suavização
            normalized = {}
            for model_name, acc in accuracies.items():
                # Normaliza entre 0 e 1
                norm_acc = (acc - min_acc) / (max_acc - min_acc) if max_acc > min_acc else 0.5
                # Aplica suavização (evita pesos muito extremos)
                # Usa exponencial para dar mais peso aos melhores, mas não muito extremo
                normalized[model_name] = (norm_acc ** 0.7) + 0.1  # Suavização
            
            # Normaliza para somar 1.0
            total = sum(normalized.values())
            for model_name in self.model_weights:
                self.model_weights[model_name] = normalized.get(model_name, 0.1) / total
        
        # Garante que pesos mínimos não sejam muito baixos (mínimo 2%)
        min_weight = 0.02
        for model_name in self.model_weights:
            if self.model_weights[model_name] < min_weight:
                self.model_weights[model_name] = min_weight
        
        # Renormaliza após ajuste de mínimo
        total = sum(self.model_weights.values())
        for model_name in self.model_weights:
            self.model_weights[model_name] /= total
        
        # Salva pesos atualizados
        self.save_model_weights()
    
    def _load_model_weights(self) -> Dict[str, float]:
        """
        Carrega pesos salvos de arquivo JSON, ou retorna pesos padrão se não existir
        """
        default_weights = {
            'lstm': 0.25,
            'stumpy': 0.20,
            'random_forest': 0.15,
            'gradient_boosting': 0.15,
            'arima': 0.10,
            'darts': 0.10,
            'prophet': 0.05
        }
        
        weights_file_path = 'model_weights.json'  # Usa caminho fixo
        if os.path.exists(weights_file_path):
            try:
                with open(weights_file_path, 'r', encoding='utf-8') as f:
                    saved_weights = json.load(f)
                    # Valida e normaliza pesos carregados
                    if isinstance(saved_weights, dict):
                        # Garante que todos os modelos estão presentes
                        for model in default_weights:
                            if model not in saved_weights:
                                saved_weights[model] = default_weights[model]
                        # Normaliza para somar 1.0
                        total = sum(saved_weights.values())
                        if total > 0:
                            for model in saved_weights:
                                saved_weights[model] /= total
                            return saved_weights
            except Exception as e:
                print(f"[AVISO] Erro ao carregar pesos salvos: {e}. Usando pesos padrão.")
        
        return default_weights
    
    def save_model_weights(self):
        """
        Salva pesos atuais em arquivo JSON
        """
        try:
            weights_file_path = getattr(self, 'weights_file', 'model_weights.json')
            with open(weights_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.model_weights, f, indent=2)
        except Exception as e:
            print(f"[AVISO] Erro ao salvar pesos: {e}")
    
    def update_weights_incremental(self, predictions: Dict[str, Optional[int]], true_value: Optional[int] = None, correct_set: Optional[set] = None):
        """
        Atualiza pesos incrementalmente baseado em uma predição e seu resultado.
        
        Objetivo = próxima SEQUÊNCIA (conjunto de 15): use correct_set.
        Legado (próximo número): use true_value.
        
        Args:
            predictions: Dicionário com predições de cada modelo
            true_value: Valor real do "próximo número" (usado só se correct_set não for passado)
            correct_set: Conjunto dos 15 números da próxima sequência correta. Acerto = pred in correct_set.
        """
        if correct_set is not None:
            # Calibração por sequência: acerto = o número sugerido está na próxima sequência completa
            for model_name, pred in predictions.items():
                if pred is not None and model_name in self.model_performance:
                    self.model_performance[model_name]['total'] += 1
                    if pred in correct_set:
                        self.model_performance[model_name]['hits'] += 1
                    perf = self.model_performance[model_name]
                    if perf['total'] > 0:
                        perf['accuracy'] = perf['hits'] / perf['total']
        elif true_value is not None:
            # Legado: calibração por "próximo número" (exato ou tolerância)
            hit_fn = (lambda p, t: p == t) if getattr(self, 'use_exact_match_for_calibration', True) else (lambda p, t: abs(p - t) <= 2)
            for model_name, pred in predictions.items():
                if pred is not None and model_name in self.model_performance:
                    self.model_performance[model_name]['total'] += 1
                    if hit_fn(pred, true_value):
                        self.model_performance[model_name]['hits'] += 1
                    perf = self.model_performance[model_name]
                    if perf['total'] > 0:
                        perf['accuracy'] = perf['hits'] / perf['total']
        else:
            return
        
        # Atualiza pesos a cada 10 avaliações (para não ser muito frequente)
        total_evaluations = sum(p['total'] for p in self.model_performance.values())
        if total_evaluations > 0 and total_evaluations % 10 == 0:
            self.update_model_weights()
            print(f"[PESOS] Pesos atualizados após {total_evaluations} avaliações")
    
    def suggest_next_number(self, sequence: List[int], method: str = 'weighted_vote', allow_empty: bool = False) -> Dict:
        """
        Gera a próxima sequência completa de 15 números combinando todos os modelos.
        Com allow_empty=True, aceita sequência vazia para gerar uma sequência nova só dos padrões históricos.
        """
        if len(sequence) < 3 and not (allow_empty and hasattr(self, 'all_sequences') and self.all_sequences):
            return {
                'prediction': None,
                'next_sequence': None,
                'confidence': 0.0,
                'message': 'Sequência muito curta (mínimo 3 números)'
            }
        
        # Gerar 1 a n_suggested_sequences sequências (ex.: 3 opções para aumentar chances)
        list_sequences = self.generate_next_sequence(sequence)
        generated_sequence = list_sequences[0] if list_sequences else []

        # Análise para confiança (usa primeira sequência)
        seq_for_analysis = sequence if len(sequence) >= 3 else (generated_sequence[:10] if generated_sequence else [])
        analysis = self.analyze_all_patterns(seq_for_analysis)
        predictions = analysis['predictions']

        all_preds = [v for v in predictions.values() if v is not None]
        if all_preds:
            matches = sum(1 for pred in all_preds if pred in generated_sequence)
            confidence = matches / len(all_preds) if all_preds else 0.0
        else:
            confidence = 0.5

        n_models = len([v for v in predictions.values() if v is not None])
        n_opts = len(list_sequences)
        msg = (f'Nova sequência gerada a partir dos padrões históricos ({len(self.all_sequences)} referências)' 
               if (allow_empty and not sequence) else f'Sequência gerada combinando {n_models} modelos')
        if n_opts > 1:
            msg += f' — {n_opts} opções geradas para ampliar as chances.'
        return {
            'prediction': generated_sequence[0] if generated_sequence else None,
            'next_sequence': generated_sequence,
            'next_sequences': list_sequences,
            'confidence': confidence,
            'method': method,
            'individual_predictions': predictions,
            'input_sequence': sequence,
            'models_used': n_models,
            'message': msg
        }
    
    def _get_effective_k_whole(self) -> int:
        """K efetivo para whole-sequence: configurável + auto-aumento quando há muitas linhas."""
        k = getattr(self, 'k_whole_sequence', 200)
        if getattr(self, 'auto_k_for_large_history', True) and self.all_sequences and len(self.all_sequences) > 400:
            k = min(300, max(k, 300))
        return k

    def _get_effective_k_knn(self) -> int:
        """K efetivo para K-NN: configurável + auto-aumento quando há muitas linhas."""
        k = getattr(self, 'k_knn', 80)
        if getattr(self, 'auto_k_for_large_history', True) and self.all_sequences and len(self.all_sequences) > 400:
            k = min(120, max(k, 120))
        return k

    def _get_ranked_successor_sequences(self, context_sequence: Optional[List[int]] = None, K: Optional[int] = None) -> List[Tuple[List[int], float]]:
        """
        Padrão nas SEQUÊNCIAS: retorna as sequências completas que REALMENTE ocorreram
        após contextos similares (sucessoras), ordenadas por peso (similaridade do contexto).
        Não usa "números que mais aparecem" — usa "qual sequência inteira veio depois".
        Retorno: [(seq_15_numeros, peso), ...] ordenado por peso decrescente.
        """
        if not self.all_sequences or len(self.all_sequences) < 2:
            return []
        if K is None:
            K = self._get_effective_k_whole()
        # Contexto: o que o usuário passou ou a última sequência do histórico
        context = context_sequence[:15] if (context_sequence and len(context_sequence) >= 3) else None
        if not context and self.all_sequences:
            context = self.all_sequences[-1][:15]
        if not context:
            return []
        context_set = set(context)
        # Sequências como conjuntos para overlap; guardamos a lista original para pegar o sucessor
        all_sets = [set(seq[:15]) for seq in self.all_sequences if len(seq) >= 15]
        if len(all_sets) < 2:
            return []
        # K sequências mais parecidas com o contexto (por overlap)
        indexed = [(len(context_set & all_sets[i]), i) for i in range(len(all_sets))]
        indexed.sort(key=lambda x: (-x[0], x[1]))
        top_k_idx = [i for _, i in indexed[:min(K, len(indexed))]]
        weight_succ = getattr(self, 'weight_successor_sequence', 1.0)
        # Pesos por SEQUÊNCIA SUCESSORA (chave = tuple ordenado do conjunto, valor = peso acumulado)
        from collections import defaultdict
        successor_weight = defaultdict(float)
        successor_example = {}  # uma sequência (lista) representante para retornar
        for pos, i in enumerate(top_k_idx):
            overlap = indexed[pos][0]
            w = (overlap / 15.0) ** 1.2 * weight_succ
            if i + 1 < len(self.all_sequences):
                succ = self.all_sequences[i + 1][:15]
                if len(succ) == 15:
                    key = tuple(sorted(succ))
                    successor_weight[key] += w
                    successor_example[key] = succ
        if not successor_weight:
            return []
        # Ordenar por peso decrescente; retornar listas (sequência completa)
        ranked = sorted(successor_weight.items(), key=lambda x: -x[1])
        return [(successor_example[k], weight) for k, weight in ranked]

    def _generate_whole_sequence_from_history(self, context_sequence: Optional[List[int]] = None, K: Optional[int] = None) -> List[int]:
        """
        Gera a próxima sequência com base em PADRÃO DE SEQUÊNCIAS: retorna a sequência
        completa que mais vezes (ponderada por similaridade) ocorreu DEPOIS de contextos
        parecidos com o atual. Não é "números com maior score".
        """
        ranked = self._get_ranked_successor_sequences(context_sequence, K)
        if ranked:
            return list(ranked[0][0])
        return []

    def _generate_whole_sequence_variants(self, context_sequence: Optional[List[int]] = None, K: Optional[int] = None, n_variants: int = 3) -> List[List[int]]:
        """
        Gera n_variants opções com base em PADRÃO DE SEQUÊNCIAS: são as n_variants
        sequências completas que mais vezes (ponderado por similaridade) ocorreram
        DEPOIS de contextos parecidos. Cada opção é uma sequência real do histórico,
        não combinação por "números que mais aparecem".
        """
        ranked = self._get_ranked_successor_sequences(context_sequence, K)
        if not ranked or n_variants < 1:
            return []
        out = [list(seq) for seq, _ in ranked[:n_variants]]
        return out

    def generate_next_sequence(self, sequence: List[int], n_sequences: Optional[int] = None) -> List[List[int]]:
        """
        Retorna uma lista de 1 a n_sequences sugestões (cada uma é lista de 15 números).
        Com n_sequences=3, a 1ª é o top 15 por score; a 2ª e 3ª são variações (2 e 3 trocas no ranking).
        """
        n_sequences = n_sequences if n_sequences is not None else getattr(self, 'n_suggested_sequences', 1)
        n_sequences = max(1, min(n_sequences, 5))  # entre 1 e 5

        use_whole = getattr(self, 'use_whole_sequence_generation', True)
        if use_whole and self.all_sequences:
            context = sequence[:15] if sequence else []
            if n_sequences > 1:
                list_seqs = self._generate_whole_sequence_variants(context_sequence=context, K=None, n_variants=n_sequences)
                if list_seqs:
                    return list_seqs
            final = self._generate_whole_sequence_from_history(context_sequence=context, K=None)
            if len(final) == 15:
                return [final]
        # Fallback: método número a número
        historical_patterns = self._analyze_historical_patterns()
        current_seq = sequence.copy() if len(sequence) > 0 else []
        generated = []
        if len(current_seq) >= 15:
            current_seq = current_seq[:10]
        while len(generated) < 15 - len(current_seq):
            # Usa a sequência atual + números já gerados
            working_seq = current_seq + generated
            
            # Analisa com todos os modelos (que já foram treinados com TODO o histórico)
            analysis = self.analyze_all_patterns(working_seq)
            predictions = analysis['predictions']
            
            # Combina predições dos modelos com padrões históricos
            historical_suggestions = self._get_historical_suggestions(working_seq, historical_patterns)
            # k-NN no histórico: sequências mais parecidas (por conjunto); K configurável
            k_knn = self._get_effective_k_knn()
            knn_suggestions = self._get_knn_suggestions(working_seq, K=k_knn)
            
            # Coleta todas as predições válidas dos modelos
            all_preds = [v for v in predictions.values() if v is not None and 
                        self.min_value <= v <= self.max_value]
            
            # Adiciona sugestões históricas e k-NN às predições (para fallback e para votação)
            if historical_suggestions:
                all_preds.extend(historical_suggestions)
            if knn_suggestions:
                all_preds.extend([num for num, _ in knn_suggestions[:10]])
            
            if not all_preds:
                # Fallback: usa frequência dos números na sequência atual
                from collections import Counter
                counter = Counter(working_seq)
                # Pega números menos frequentes (mais diversidade)
                least_common = [num for num, count in counter.most_common()[-5:]]
                if least_common:
                    next_num = least_common[0]
                else:
                    # Último recurso: número aleatório no range
                    next_num = np.random.randint(self.min_value, self.max_value + 1)
            else:
                # Usa pesos dinâmicos ajustados durante o treinamento
                weights = self.model_weights
                
                # Conta votos ponderados dos modelos
                vote_count = {}
                for model, pred in predictions.items():
                    if pred is not None and self.min_value <= pred <= self.max_value:
                        weight = weights.get(model, 0.1)
                        vote_count[pred] = vote_count.get(pred, 0) + weight
                
                # Adiciona sugestões históricas com peso menor (complementam modelos)
                if historical_suggestions:
                    hist_weight = 0.3  # Peso menor para padrões históricos
                    for hist_num in historical_suggestions[:3]:  # Top 3 sugestões históricas
                        if self.min_value <= hist_num <= self.max_value:
                            vote_count[hist_num] = vote_count.get(hist_num, 0) + hist_weight
                
                # k-NN: histórico imenso usado diretamente (sequências similares por conjunto); peso configurável
                weight_knn = getattr(self, 'weight_knn_suggestions', 1.0)
                if knn_suggestions:
                    max_s = max((s for _, s in knn_suggestions[:15]), default=1)
                    for num, score in knn_suggestions[:15]:
                        if self.min_value <= num <= self.max_value:
                            vote_count[num] = vote_count.get(num, 0) + min(1.2 * weight_knn * (score / max_s), 1.2 * weight_knn)
                
                # Feedback do último confronto: evita repetir erros, favorece os que faltaram
                avoid = getattr(self, 'last_avoid_numbers', [])
                prefer = getattr(self, 'last_prefer_numbers', [])
                for num in avoid:
                    if self.min_value <= num <= self.max_value:
                        vote_count[num] = vote_count.get(num, 0) - 0.6  # penaliza
                for num in prefer:
                    if self.min_value <= num <= self.max_value:
                        vote_count[num] = vote_count.get(num, 0) + 0.5  # favorece
                
                if vote_count:
                    # Pega o número com maior voto, mas evita repetições
                    sorted_votes = sorted(vote_count.items(), key=lambda x: x[1], reverse=True)
                    for num, _ in sorted_votes:
                        if num not in working_seq + generated:
                            next_num = num
                            break
                    else:
                        # Se todos já estão na sequência, pega o mais votado mesmo
                        next_num = sorted_votes[0][0]
                else:
                    # Fallback: usa sugestões históricas ou média
                    if historical_suggestions:
                        next_num = historical_suggestions[0]
                    else:
                        next_num = int(np.round(np.mean(all_preds))) if all_preds else np.random.randint(self.min_value, self.max_value + 1)
                    next_num = max(self.min_value, min(self.max_value, next_num))
            
            # Garante que não repete números na sequência
            if next_num not in working_seq + generated:
                generated.append(next_num)
            else:
                # Se repetiu, tenta outro número do range que não está na sequência
                available = [n for n in range(self.min_value, self.max_value + 1) 
                           if n not in working_seq + generated]
                if available:
                    # Escolhe aleatoriamente entre os disponíveis para diversidade
                    generated.append(np.random.choice(available))
                else:
                    # Se não há mais números disponíveis (raro, mas possível),
                    # adiciona o mais votado mesmo (aceita repetição como último recurso)
                    generated.append(next_num)
        
        # Retorna sequência completa (sempre 15 números); repete n_sequences vezes para manter API
        final_sequence = (current_seq + generated)[:15]
        return [final_sequence] * n_sequences

    def _analyze_historical_patterns(self) -> Dict:

        if not self.all_sequences or len(self.all_sequences) == 0:
            return {}
        
        patterns = {
            'number_frequency': {},  
            'pair_frequency': {},    
            'common_numbers': [],    
            'rare_numbers': []      
        }
        
        # Analisa frequência de números em TODAS as sequências
        from collections import Counter
        all_numbers = []
        for seq in self.all_sequences:
            all_numbers.extend(seq)
        
        number_counter = Counter(all_numbers)
        patterns['number_frequency'] = dict(number_counter)
        
        # Identifica números mais e menos comuns
        if number_counter:
            patterns['common_numbers'] = [num for num, _ in number_counter.most_common(10)]
            patterns['rare_numbers'] = [num for num, _ in number_counter.most_common()[-10:]]
        
        # Analisa co-ocorrência de números (pares que aparecem juntos)
        pair_counter = Counter()
        for seq in self.all_sequences:
            if len(seq) >= 2:
                # Cria pares de números adjacentes e não-adjacentes na sequência
                for i in range(len(seq)):
                    for j in range(i+1, len(seq)):
                        pair = tuple(sorted([seq[i], seq[j]]))
                        pair_counter[pair] += 1
        
        patterns['pair_frequency'] = dict(pair_counter)
        
        return patterns
    
    def _get_historical_suggestions(self, current_seq: List[int], patterns: Dict) -> List[int]:

        suggestions = []
        
        if not patterns or not current_seq:
            return suggestions
        
        # 1. Sugere números que aparecem frequentemente com números já na sequência
        current_set = set(current_seq)
        pair_freq = patterns.get('pair_frequency', {})
        
        # Encontra números que co-ocorrem frequentemente com números atuais
        cooccurrence_scores = {}
        for pair, freq in pair_freq.items():
            num1, num2 = pair
            if num1 in current_set and num2 not in current_set:
                cooccurrence_scores[num2] = cooccurrence_scores.get(num2, 0) + freq
            elif num2 in current_set and num1 not in current_set:
                cooccurrence_scores[num1] = cooccurrence_scores.get(num1, 0) + freq
        
        # Adiciona números com maior co-ocorrência
        if cooccurrence_scores:
            sorted_cooc = sorted(cooccurrence_scores.items(), key=lambda x: x[1], reverse=True)
            suggestions.extend([num for num, _ in sorted_cooc[:5]])
        
        # 2. Se ainda faltam sugestões, adiciona números comuns que não estão na sequência
        common_numbers = patterns.get('common_numbers', [])
        for num in common_numbers:
            if num not in current_set and num not in suggestions:
                suggestions.append(num)
            if len(suggestions) >= 10:
                break
        
        # 3. Se ainda faltam, adiciona números raros para diversidade
        if len(suggestions) < 5:
            rare_numbers = patterns.get('rare_numbers', [])
            for num in rare_numbers:
                if num not in current_set and num not in suggestions:
                    suggestions.append(num)
                if len(suggestions) >= 10:
                    break
        
        return suggestions[:10]  # Retorna até 10 sugestões
    
    def _get_knn_suggestions(self, working_seq: List[int], K: int = 80) -> List[Tuple[int, float]]:

        from collections import Counter
        if not self.all_sequences or len(working_seq) == 0:
            return []
        working_set = set(working_seq)
        # Para sequência vazia, usa as mais frequentes no histórico
        if len(working_set) == 0:
            all_nums = []
            for seq in self.all_sequences:
                if len(seq) == 15:
                    all_nums.extend(seq)
            cnt = Counter(all_nums)
            return [(num, float(freq)) for num, freq in cnt.most_common(15)]
        # Overlap de conjunto: quantos números de working_seq estão em cada sequência histórica
        scored = []
        for seq in self.all_sequences:
            if len(seq) < 15:
                continue
            s = set(seq[:15])
            overlap = len(working_set & s)
            scored.append((overlap, s))
        # Top K sequências mais similares (maior overlap)
        scored.sort(key=lambda x: x[0], reverse=True)
        top_k_sets = [s for _, s in scored[:K]]
        # Números que aparecem nas top K e não estão em working_seq; pontua por frequência
        candidate_scores = Counter()
        for s in top_k_sets:
            for num in s:
                if num not in working_set and self.min_value <= num <= self.max_value:
                    candidate_scores[num] += 1
        return [(num, float(score)) for num, score in candidate_scores.most_common(20)]
    
    def learn_incrementally(self, new_sequence: List[int], true_next: Optional[int] = None):

        if len(new_sequence) == self.sequence_length:
            self.all_sequences.append(new_sequence)
            self.preprocess_sequences()
            
            # Se temos o valor real, avalia predições anteriores e atualiza pesos
            if true_next is not None and len(new_sequence) >= 14:
                input_seq = new_sequence[:14]
                analysis = self.analyze_all_patterns(input_seq)
                predictions = analysis['predictions']
                # Atualiza pesos incrementalmente
                self.update_weights_incremental(predictions, true_next)
            
            # Re-treina modelos periodicamente (a cada 100 novas sequências)
            if len(self.all_sequences) % 100 == 0:
                print(f" Re-treinando modelos com {len(self.all_sequences)} sequências...")
                self.train_full_system()

    def compare_sequences_set(self, generated: List[int], correct: List[int]) -> Dict:

        if len(correct) != 15 or len(generated) != 15:
            return {'error': 'Sequências devem ter 15 números'}
        gen_set = set(generated)
        cor_set = set(correct)
        in_both = sorted(gen_set & cor_set)
        only_generated = sorted(gen_set - cor_set)
        only_correct = sorted(cor_set - gen_set)
        hits = len(in_both)
        return {
            'hits': hits,
            'total': 15,
            'accuracy_pct': round(100 * hits / 15, 1),
            'in_both': in_both,
            'only_generated': only_generated,
            'only_correct': only_correct,
            'message': f'{hits}/15 números coincidem (mesmos números, sem considerar ordem).'
        }

    def retrain_with_correct_sequence(self, correct: List[int], last_generated: Optional[List[int]] = None, full_retrain: bool = False) -> Dict:

        if len(correct) != 15:
            return {'error': 'A sequência deve ter 15 números'}
        # Apenas dica para a próxima sugestão (evitar repetir erros, favorecer os que faltaram)
        if last_generated is not None and len(last_generated) == 15:
            gen_set = set(last_generated)
            cor_set = set(correct)
            self.last_avoid_numbers = sorted(gen_set - cor_set)
            self.last_prefer_numbers = sorted(cor_set - gen_set)
        # Incorpora a sequência correta ao histórico (próximas sugestões usam mais dados)
        self.all_sequences.append(correct)
        self.preprocess_sequences()
        self.save_model_weights()
        # Opcional: re-treinar tudo incluindo esta nova sequência (recalibra pesos pela tabela inteira)
        if full_retrain:
            self.train_full_system()
        return {
            'success': True,
            'message': 'Sequência correta adicionada ao histórico.' + (' Re-treino completo executado.' if full_retrain else '')
        }


if __name__ == "__main__":

    agent = PatternAgent(min_value=1, max_value=25, sequence_length=15)
    

    sequences = agent.load_sequences_from_excel('tabela_original.xlsx')
    
    if sequences:
        # Treina sistema completo
        agent.train_full_system()
        
        # Testa predição
        test_sequence = sequences[0][:14]  # Primeiros 14 números
        print(f"\n[TESTE] Testando predicao para sequencia: {test_sequence}")
        
        result = agent.suggest_next_number(test_sequence)
        print(f"\n[PREDICAO] Proximo numero sugerido: {result['prediction']}")
        print(f"[CONFIANCA] {result['confidence']:.2%}")
        print(f"[MODELOS] Modelos utilizados: {result['models_used']}")
        seqs = result.get('next_sequences', [result.get('next_sequence')])
        if seqs:
            for i, s in enumerate(seqs, 1):
                print(f"\n[OPCAO {i}] {' '.join(map(str, s))}")
        print(f"\n[DETALHES] Predicoes individuais:")
        for model, pred in result['individual_predictions'].items():
            if pred:
                print(f"  - {model}: {pred}")
