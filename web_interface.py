"""
Interface Web Simples para o Pattern Agent

Permite upload de arquivo Excel, mostra progresso e resultados das predições.
"""

from flask import Flask, request, render_template_string, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
import json
from pattern_agent import PatternAgent
import threading
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max
app.config['SECRET_KEY'] = 'pattern-agent-secret-key'

# Cria pasta de uploads se não existir
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Estado global do agente
agent_state = {
    'agent': None,
    'status': 'idle',  # idle, loading, training, ready, error
    'message': '',
    'sequences_count': 0,
    'training_progress': 0,
    'training_total': 0,
    'training_message': '',
    'results': None
}

ALLOWED_EXTENSIONS = {'xlsx', 'xls'}

def _make_json_serializable(obj):
    """Converte tipos NumPy (int32, int64, float32, etc.) para tipos nativos Python."""
    try:
        import numpy as np
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(x) for x in obj]
    return obj

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pattern Agent - Detecção de Padrões</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Crimson+Text:ital,wght@0,400;0,600;1,400&display=swap');
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Crimson Text', 'Times New Roman', serif;
            background: #3d2817;
            background-image: 
                radial-gradient(circle at 20% 50%, rgba(139, 90, 43, 0.3) 0%, transparent 50%),
                radial-gradient(circle at 80% 80%, rgba(101, 67, 33, 0.3) 0%, transparent 50%),
                repeating-linear-gradient(
                    45deg,
                    transparent,
                    transparent 2px,
                    rgba(101, 67, 33, 0.05) 2px,
                    rgba(101, 67, 33, 0.05) 4px
                );
            min-height: 100vh;
            padding: 20px;
            color: #2c1810;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: #f4e8d8;
            background-image: 
                url('data:image/svg+xml,<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg"><rect width="100" height="100" fill="%23f4e8d8"/><circle cx="50" cy="50" r="1" fill="%23d4b896" opacity="0.3"/></svg>'),
                linear-gradient(to bottom, rgba(212, 184, 150, 0.1) 0%, transparent 100%);
            border: 8px solid #8b5a2b;
            border-style: double;
            box-shadow: 
                0 0 0 4px #6b4423,
                0 0 0 8px #8b5a2b,
                0 20px 60px rgba(0, 0, 0, 0.5),
                inset 0 0 100px rgba(139, 90, 43, 0.1);
            overflow: hidden;
            position: relative;
        }
        
        .container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: 
                repeating-linear-gradient(
                    0deg,
                    transparent,
                    transparent 2px,
                    rgba(139, 90, 43, 0.03) 2px,
                    rgba(139, 90, 43, 0.03) 4px
                );
            pointer-events: none;
            opacity: 0.5;
        }
        
        .header {
            background: linear-gradient(135deg, #8b5a2b 0%, #6b4423 100%);
            background-image: 
                repeating-linear-gradient(
                    45deg,
                    transparent,
                    transparent 10px,
                    rgba(0, 0, 0, 0.1) 10px,
                    rgba(0, 0, 0, 0.1) 20px
                );
            color: #f4e8d8;
            padding: 40px 30px;
            text-align: center;
            border-bottom: 6px solid #5a3419;
            position: relative;
            box-shadow: inset 0 -10px 20px rgba(0, 0, 0, 0.3);
        }
        
        .header::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: repeating-linear-gradient(
                90deg,
                #f4e8d8 0px,
                #f4e8d8 10px,
                transparent 10px,
                transparent 20px
            );
        }
        
        .header h1 {
            font-family: 'Playfair Display', 'Times New Roman', serif;
            font-size: 3em;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 4px;
            text-shadow: 
                3px 3px 0px #5a3419,
                6px 6px 10px rgba(0, 0, 0, 0.5);
            font-weight: 900;
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.95;
            font-style: italic;
            letter-spacing: 2px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }
        
        .content {
            padding: 40px;
            position: relative;
            z-index: 1;
        }
        
        .upload-section {
            background: #e8dcc6;
            background-image: 
                repeating-linear-gradient(
                    0deg,
                    transparent,
                    transparent 20px,
                    rgba(139, 90, 43, 0.05) 20px,
                    rgba(139, 90, 43, 0.05) 21px
                );
            border: 4px double #8b5a2b;
            border-style: double;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 
                inset 0 0 20px rgba(139, 90, 43, 0.2),
                0 5px 15px rgba(0, 0, 0, 0.3);
            position: relative;
        }
        
        .upload-section::before {
            content: '';
            position: absolute;
            top: 10px;
            left: 10px;
            right: 10px;
            bottom: 10px;
            border: 1px solid rgba(139, 90, 43, 0.3);
            pointer-events: none;
        }
        
        .upload-section.dragover {
            background: #d4b896;
            border-color: #6b4423;
        }
        
        .upload-section h2 {
            font-family: 'Playfair Display', serif;
            font-size: 1.8em;
            color: #5a3419;
            margin-bottom: 20px;
            text-transform: uppercase;
            letter-spacing: 3px;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
        }
        
        .file-input-wrapper {
            position: relative;
            display: inline-block;
        }
        
        input[type="file"] {
            display: none;
        }
        
        .file-label {
            display: inline-block;
            padding: 12px 35px;
            background: #8b5a2b;
            color: #f4e8d8;
            border: 3px solid #6b4423;
            cursor: pointer;
            font-size: 1.1em;
            font-family: 'Playfair Display', serif;
            text-transform: uppercase;
            letter-spacing: 2px;
            transition: all 0.3s;
            box-shadow: 
                0 4px 8px rgba(0, 0, 0, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.2);
            position: relative;
        }
        
        .file-label:hover {
            background: #6b4423;
            border-color: #5a3419;
            transform: translateY(-1px);
            box-shadow: 
                0 6px 12px rgba(0, 0, 0, 0.4),
                inset 0 1px 0 rgba(255, 255, 255, 0.2);
        }
        
        .file-label:active {
            transform: translateY(0);
            box-shadow: 
                0 2px 4px rgba(0, 0, 0, 0.3),
                inset 0 2px 4px rgba(0, 0, 0, 0.2);
        }
        
        .btn {
            padding: 12px 35px;
            background: #6b4423;
            color: #f4e8d8;
            border: 3px solid #5a3419;
            font-size: 1.1em;
            font-family: 'Playfair Display', serif;
            text-transform: uppercase;
            letter-spacing: 2px;
            cursor: pointer;
            transition: all 0.3s;
            margin-top: 20px;
            box-shadow: 
                0 4px 8px rgba(0, 0, 0, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.2);
        }
        
        .btn:hover:not(:disabled) {
            background: #5a3419;
            border-color: #4a2815;
            transform: translateY(-1px);
            box-shadow: 
                0 6px 12px rgba(0, 0, 0, 0.4),
                inset 0 1px 0 rgba(255, 255, 255, 0.2);
        }
        
        .btn:disabled {
            background: #8b7a6b;
            border-color: #6b5a4b;
            color: #a89a8a;
            cursor: not-allowed;
            opacity: 0.6;
        }
        
        .status-section {
            margin-top: 30px;
            padding: 25px;
            background: #e8dcc6;
            background-image: 
                repeating-linear-gradient(
                    0deg,
                    transparent,
                    transparent 20px,
                    rgba(139, 90, 43, 0.05) 20px,
                    rgba(139, 90, 43, 0.05) 21px
                );
            border: 3px double #8b5a2b;
            min-height: 200px;
            box-shadow: 
                inset 0 0 20px rgba(139, 90, 43, 0.2),
                0 3px 10px rgba(0, 0, 0, 0.2);
        }
        
        .status-section h2 {
            font-family: 'Playfair Display', serif;
            font-size: 1.6em;
            color: #5a3419;
            margin-bottom: 20px;
            text-transform: uppercase;
            letter-spacing: 2px;
            border-bottom: 2px solid #8b5a2b;
            padding-bottom: 10px;
        }
        
        .status-item {
            padding: 15px 20px;
            margin: 12px 0;
            background: #f4e8d8;
            border: 2px solid #d4b896;
            border-left: 6px solid #8b5a2b;
            display: flex;
            align-items: center;
            gap: 15px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
            font-size: 1.05em;
        }
        
        .status-item.success {
            border-left-color: #6b7a3a;
            background: #f0ead8;
        }
        
        .status-item.error {
            border-left-color: #8b4a2b;
            background: #f4e0d8;
        }
        
        .status-item.processing {
            border-left-color: #8b7a2b;
            background: #f4e8d0;
        }
        
        .spinner {
            border: 3px solid #d4b896;
            border-top: 3px solid #8b5a2b;
            border-radius: 50%;
            width: 20px;
            height: 20px;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .results-section {
            margin-top: 30px;
            display: none;
        }
        
        .results-section.show {
            display: block;
        }
        
        .results-section h2 {
            font-family: 'Playfair Display', serif;
            font-size: 1.8em;
            color: #5a3419;
            margin-bottom: 20px;
            text-transform: uppercase;
            letter-spacing: 3px;
            border-bottom: 3px double #8b5a2b;
            padding-bottom: 15px;
        }
        
        .result-card {
            background: #e8dcc6;
            background-image: 
                repeating-linear-gradient(
                    0deg,
                    transparent,
                    transparent 20px,
                    rgba(139, 90, 43, 0.05) 20px,
                    rgba(139, 90, 43, 0.05) 21px
                );
            border: 4px double #8b5a2b;
            padding: 30px;
            margin: 20px 0;
            box-shadow: 
                inset 0 0 20px rgba(139, 90, 43, 0.2),
                0 5px 15px rgba(0, 0, 0, 0.3);
        }
        
        .result-card h3 {
            font-family: 'Playfair Display', serif;
            color: #5a3419;
            margin-bottom: 20px;
            font-size: 1.4em;
            text-transform: uppercase;
            letter-spacing: 2px;
            border-bottom: 2px solid #8b5a2b;
            padding-bottom: 10px;
        }
        
        .prediction-box {
            background: linear-gradient(135deg, #8b5a2b 0%, #6b4423 100%);
            background-image: 
                repeating-linear-gradient(
                    45deg,
                    transparent,
                    transparent 10px,
                    rgba(0, 0, 0, 0.1) 10px,
                    rgba(0, 0, 0, 0.1) 20px
                );
            color: #f4e8d8;
            padding: 40px;
            border: 4px double #5a3419;
            text-align: center;
            margin: 25px 0;
            box-shadow: 
                inset 0 0 30px rgba(0, 0, 0, 0.3),
                0 5px 15px rgba(0, 0, 0, 0.4);
        }
        
        .prediction-number {
            font-family: 'Playfair Display', serif;
            font-size: 5em;
            font-weight: 900;
            margin: 25px 0;
            text-shadow: 
                3px 3px 0px #5a3419,
                6px 6px 10px rgba(0, 0, 0, 0.5);
            letter-spacing: 8px;
        }
        
        .confidence-bar {
            width: 100%;
            height: 35px;
            background: rgba(90, 52, 25, 0.5);
            border: 2px solid #5a3419;
            overflow: hidden;
            margin: 20px 0;
            box-shadow: inset 0 2px 5px rgba(0, 0, 0, 0.3);
        }
        
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #6b7a3a 0%, #8b9a4a 100%);
            transition: width 0.5s;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #f4e8d8;
            font-weight: bold;
            font-size: 1.1em;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
        }
        
        .model-predictions {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 25px;
        }
        
        .model-item {
            background: #f4e8d8;
            padding: 20px;
            border: 2px solid #d4b896;
            text-align: center;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        }
        
        .model-name {
            font-family: 'Playfair Display', serif;
            font-weight: bold;
            color: #5a3419;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-size: 0.9em;
            border-bottom: 1px solid #8b5a2b;
            padding-bottom: 8px;
        }
        
        .model-value {
            font-family: 'Playfair Display', serif;
            font-size: 2em;
            color: #6b4423;
            font-weight: 700;
        }
        
        .log-container {
            background: #2c1810;
            background-image: 
                repeating-linear-gradient(
                    0deg,
                    transparent,
                    transparent 2px,
                    rgba(139, 90, 43, 0.1) 2px,
                    rgba(139, 90, 43, 0.1) 4px
                );
            color: #d4b896;
            padding: 25px;
            border: 3px double #8b5a2b;
            font-family: 'Courier New', monospace;
            font-size: 0.95em;
            max-height: 400px;
            overflow-y: auto;
            margin-top: 25px;
            box-shadow: 
                inset 0 0 20px rgba(0, 0, 0, 0.5),
                0 3px 10px rgba(0, 0, 0, 0.3);
        }
        
        .log-line {
            margin: 6px 0;
            padding: 4px 8px;
            line-height: 1.6;
        }
        
        .log-line.success { 
            color: #8b9a4a; 
            text-shadow: 0 0 3px rgba(139, 154, 74, 0.5);
        }
        .log-line.error { 
            color: #b85a3a; 
            text-shadow: 0 0 3px rgba(184, 90, 58, 0.5);
        }
        .log-line.info { 
            color: #8b7a5a; 
        }
        .log-line.warning { 
            color: #b89a5a; 
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Gerador sequencial</h1>
            <p>Detecção Inteligente de Padrões em Sequências Numéricas</p>
        </div>
        
        <div class="content">
            <div class="upload-section" id="uploadSection">
                <h2>Upload do Arquivo Excel</h2>
                <p style="margin: 20px 0; color: #5a3419; font-style: italic;">Selecione um arquivo .xlsx com sequências numéricas</p>
                <div class="file-input-wrapper">
                    <input type="file" id="fileInput" accept=".xlsx,.xls" />
                    <label for="fileInput" class="file-label">Escolher Arquivo</label>
                </div>
                <p id="fileName" style="margin-top: 15px; color: #6b4423; font-weight: bold; font-style: italic;"></p>
                <button class="btn" id="processBtn" onclick="processFile()" disabled>Processar Arquivo</button>
            </div>
            
            <div class="status-section">
                <h2>Status do Processamento</h2>
                <div id="progressBarSection" style="display: none; margin: 15px 0;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 6px; color: #5a3419; font-size: 0.95em;">
                        <span id="progressBarText">0/0</span>
                        <span id="progressBarPct">0%</span>
                    </div>
                    <div style="height: 24px; background: #e8dcc6; border: 2px solid #8b5a2b; border-radius: 4px; overflow: hidden;">
                        <div id="progressBarFill" style="height: 100%; width: 0%; background: linear-gradient(90deg, #8b5a2b, #6b4423); transition: width 0.3s ease;"></div>
                    </div>
                    <p id="progressBarMessage" style="margin-top: 6px; color: #6b4423; font-size: 0.9em;"></p>
                </div>
                <div id="statusContainer"></div>
            </div>
            
            <div class="results-section" id="resultsSection">
                <h2>Resultados</h2>
                <div id="resultsContainer"></div>
                <div id="compareSection" class="compare-section" style="display: none;">
                    <p style="margin-top: 25px; color: #5a3419;">Depois de comparar e retreinar, gere outra sequência (sem reprocessar o arquivo):</p>
                    <button type="button" class="btn" id="generateNewBtn" onclick="generateNewSequence()" style="margin-bottom: 20px;">Gerar nova sequência</button>
                    <h3 style="margin-top: 30px; color: #5a3419; border-bottom: 2px solid #8b5a2b; padding-bottom: 10px;">Confrontar com sequência correta</h3>
                    <p style="color: #5a3419; margin: 15px 0;">Informe a sequência correta (15 números). Comparação é por conjunto (não considera ordem).</p>
                    <input type="text" id="correctSequenceInput" placeholder="Ex: 3, 7, 12, 1, 25, 9, 14, 6, 18, 22, 4, 11, 19, 8, 15" style="width: 100%; max-width: 500px; padding: 12px; font-size: 1em; border: 2px solid #8b5a2b; border-radius: 4px; margin-bottom: 12px;" />
                    <button type="button" class="btn" id="compareBtn" onclick="compareWithCorrect()">Comparar</button>
                    <button type="button" class="btn" id="retrainBtn" onclick="retrainWithCorrect()" style="margin-left: 10px;">Retreinar</button>
                    <label style="display: inline-block; margin-left: 15px; color: #5a3419; cursor: pointer;">
                        <input type="checkbox" id="fullRetrainCheck" /> Retreinamento completo (re-treina LSTM e modelos; mais lento)
                    </label>
                    <div id="compareResult" style="margin-top: 20px;"></div>
                </div>
            </div>
            
            <div class="log-container" id="logContainer" style="display: none;">
                <div id="logContent"></div>
            </div>
        </div>
    </div>
    
    <script>
        let statusInterval;
        let logInterval;
        
        document.getElementById('fileInput').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                document.getElementById('fileName').textContent = 'Arquivo selecionado: ' + file.name;
                document.getElementById('processBtn').disabled = false;
            }
        });
        
        function addLog(message, type = 'info') {
            const logContainer = document.getElementById('logContainer');
            const logContent = document.getElementById('logContent');
            logContainer.style.display = 'block';
            
            const logLine = document.createElement('div');
            logLine.className = 'log-line ' + type;
            logLine.textContent = '[' + new Date().toLocaleTimeString() + '] ' + message;
            logContent.appendChild(logLine);
            logContent.scrollTop = logContent.scrollHeight;
        }
        
        function addStatus(message, type = 'info') {
            const container = document.getElementById('statusContainer');
            const item = document.createElement('div');
            item.className = 'status-item ' + type;
            
            if (type === 'processing') {
                item.innerHTML = '<div class="spinner"></div><span>' + message + '</span>';
            } else {
                item.innerHTML = '<span>' + message + '</span>';
            }
            
            container.appendChild(item);
            container.scrollTop = container.scrollHeight;
        }
        
        function clearStatus() {
            document.getElementById('statusContainer').innerHTML = '';
            document.getElementById('logContent').innerHTML = '';
        }
        
        function processFile() {
            const fileInput = document.getElementById('fileInput');
            const file = fileInput.files[0];
            
            if (!file) {
                alert('Por favor, selecione um arquivo!');
                return;
            }
            
            clearStatus();
            document.getElementById('processBtn').disabled = true;
            document.getElementById('resultsSection').classList.remove('show');
            
            const formData = new FormData();
            formData.append('file', file);
            
            addLog('Iniciando upload do arquivo...', 'info');
            addStatus('Enviando arquivo...', 'processing');
            
            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(async response => {
                const text = await response.text();
                let data = null;
                try {
                    data = text ? JSON.parse(text) : null;
                } catch (e) {
                    addLog('Resposta inválida do servidor (não é JSON). Status: ' + response.status, 'error');
                    addStatus('Erro: servidor retornou resposta inválida', 'error');
                    document.getElementById('processBtn').disabled = false;
                    return;
                }
                if (!response.ok) {
                    const errMsg = (data && data.error) ? data.error : ('HTTP ' + response.status + (text ? ': ' + text.substring(0, 200) : ''));
                    addLog('Erro no upload: ' + errMsg, 'error');
                    addStatus('Erro: ' + errMsg, 'error');
                    document.getElementById('processBtn').disabled = false;
                    return;
                }
                if (data && data.success) {
                    addLog('Arquivo carregado com sucesso!', 'success');
                    addStatus('Arquivo carregado: ' + data.filename, 'success');
                    startProcessing(data.filename);
                } else {
                    addLog('Erro ao carregar arquivo: ' + (data && data.error ? data.error : 'Resposta inesperada'), 'error');
                    addStatus('Erro: ' + (data && data.error ? data.error : 'Resposta inesperada'), 'error');
                    document.getElementById('processBtn').disabled = false;
                }
            })
            .catch(error => {
                addLog('Erro na requisição: ' + (error.message || error), 'error');
                addStatus('Erro na comunicação com servidor (rede ou CORS)', 'error');
                document.getElementById('processBtn').disabled = false;
            });
        }
        
        function startProcessing(filename) {
            addLog('Iniciando processamento...', 'info');
            addStatus('Processando arquivo... (aguarde o tempo que for necessário)', 'processing');
            
            // Inicia polling de status
            statusInterval = setInterval(checkStatus, 1000);
            logInterval = setInterval(checkLogs, 500);
            
            // Sem timeout: espera a resposta do servidor pelo tempo que demorar
            fetch('/process', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({filename: filename})
            })
            .then(async response => {
                const text = await response.text();
                let data = null;
                try {
                    data = text ? JSON.parse(text) : null;
                } catch (e) {
                    clearInterval(statusInterval);
                    clearInterval(logInterval);
                    document.getElementById('processBtn').disabled = false;
                    addLog('Resposta do servidor inválida (não é JSON). Status: ' + response.status, 'error');
                    addStatus('Erro: servidor retornou resposta inválida', 'error');
                    return;
                }
                if (!response.ok) {
                    clearInterval(statusInterval);
                    clearInterval(logInterval);
                    document.getElementById('processBtn').disabled = false;
                    const errMsg = (data && data.error) ? data.error : ('HTTP ' + response.status);
                    addLog('Erro no processamento: ' + errMsg, 'error');
                    addStatus('Erro: ' + errMsg, 'error');
                    return;
                }
                if (data && data.success) {
                    addLog('Processamento concluído!', 'success');
                    showResults(data.results);
                } else {
                    addLog('Erro no processamento: ' + (data && data.error ? data.error : 'Resposta inesperada'), 'error');
                    addStatus('Erro no processamento', 'error');
                }
                clearInterval(statusInterval);
                clearInterval(logInterval);
                document.getElementById('processBtn').disabled = false;
            })
            .catch(error => {
                addLog('Erro de rede: ' + (error.message || error), 'error');
                addStatus('Erro na comunicação (verifique se o servidor está rodando)', 'error');
                clearInterval(statusInterval);
                clearInterval(logInterval);
                document.getElementById('processBtn').disabled = false;
            });
        }
        
        function checkStatus() {
            fetch('/status')
            .then(response => response.json())
            .then(data => {
                const total = data.training_total || 0;
                const current = data.training_progress || 0;
                const msg = data.training_message || '';
                const section = document.getElementById('progressBarSection');
                if (total > 0 && data.status === 'training') {
                    section.style.display = 'block';
                    const pct = Math.round((current / total) * 100);
                    document.getElementById('progressBarText').textContent = current + ' / ' + total + ' sequências';
                    document.getElementById('progressBarPct').textContent = pct + '%';
                    document.getElementById('progressBarFill').style.width = pct + '%';
                    document.getElementById('progressBarMessage').textContent = msg;
                } else {
                    section.style.display = 'none';
                }
            })
            .catch(err => console.error('Erro ao buscar status:', err));
        }
        
        let lastLogCount = 0;
        
        function checkLogs() {
            fetch('/logs')
            .then(response => response.json())
            .then(data => {
                if (data.logs && data.logs.length > lastLogCount) {
                    // Adiciona apenas logs novos
                    const newLogs = data.logs.slice(lastLogCount);
                    newLogs.forEach(log => {
                        addLog(log.message, log.type);
                        // Também adiciona ao status se for importante
                        if (log.type === 'success' || log.type === 'error') {
                            addStatus(log.message, log.type);
                        }
                    });
                    lastLogCount = data.logs.length;
                }
            })
            .catch(err => console.error('Erro ao buscar logs:', err));
        }
        
        function showResults(results) {
            const container = document.getElementById('resultsContainer');
            container.innerHTML = '';

            const listSeqs = results.next_sequences || (results.next_sequence ? [results.next_sequence] : null);
            const sequence = results.next_sequence || (results.prediction ? [results.prediction] : null);

            if (results && listSeqs && listSeqs.length > 0) {
                listSeqs.forEach((seq, optIndex) => {
                    const sequenceHtml = seq.map((num) =>
                        `<span style="display: inline-block; padding: 8px 12px; margin: 4px; background: linear-gradient(135deg, #8b5a2b 0%, #6b4423 100%); color: #f4e8d8; border: 2px solid #5a3419; border-radius: 4px; font-weight: bold; font-size: 1.1em; min-width: 40px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.3);">${num}</span>`
                    ).join('');
                    const card = document.createElement('div');
                    card.className = 'result-card';
                    const title = listSeqs.length > 1 ? `Opção ${optIndex + 1} de ${listSeqs.length}` : 'Próxima Sequência Completa (15 números)';
                    card.innerHTML = `
                        <h3>${title}</h3>
                        <div class="prediction-box">
                            <div style="font-size: 1.3em; line-height: 2; text-align: center; padding: 20px;">
                                ${sequenceHtml}
                            </div>
                            ${optIndex === 0 ? `
                                <p style="font-size: 1.2em; margin-top: 20px; font-style: italic; text-align: center;">
                                    Confiança: ${(results.confidence * 100).toFixed(2)}%
                                </p>
                                <div class="confidence-bar">
                                    <div class="confidence-fill" style="width: ${results.confidence * 100}%">
                                        ${(results.confidence * 100).toFixed(1)}%
                                    </div>
                                </div>
                            ` : ''}
                            ${optIndex === 0 && results.input_sequence && results.input_sequence.length ? `
                                <div style="margin-top: 30px; padding-top: 20px; border-top: 2px solid #8b5a2b;">
                                    <p style="font-size: 1em; color: #6b4423; margin-bottom: 10px;"><strong>Sequência de entrada:</strong></p>
                                    <div style="font-size: 1.1em; line-height: 1.8;">
                                        ${results.input_sequence.map((num) =>
                                            `<span style="display: inline-block; padding: 6px 10px; margin: 3px; background: #e8dcc6; color: #5a3419; border: 1px solid #8b5a2b; border-radius: 3px; font-weight: normal;">${num}</span>`
                                        ).join('')}
                                    </div>
                                </div>
                            ` : ''}
                        </div>
                    `;
                    container.appendChild(card);
                });
                const cardExtra = document.createElement('div');
                cardExtra.className = 'result-card';
                cardExtra.innerHTML = `
                    <h3 style="margin-top: 15px;">Predições por Modelo</h3>
                    <div class="model-predictions">
                        ${Object.entries(results.individual_predictions || {})
                            .filter(([k, v]) => v !== null)
                            .map(([model, pred]) => `
                                <div class="model-item">
                                    <div class="model-name">${model}</div>
                                    <div class="model-value">${pred}</div>
                                </div>
                            `).join('')}
                    </div>
                    <p style="margin-top: 25px; color: #5a3419; font-size: 1.05em; line-height: 1.8;">
                        <strong>Método usado:</strong> ${results.method || 'weighted_vote'}<br>
                        <strong>Modelos utilizados:</strong> ${results.models_used || 0}<br>
                        ${results.message ? `<strong>Info:</strong> ${results.message}` : ''}
                    </p>
                `;
                container.appendChild(cardExtra);
                document.getElementById('compareSection').style.display = 'block';
                document.getElementById('compareResult').innerHTML = '';
                document.getElementById('correctSequenceInput').value = '';
            } else {
                container.innerHTML = '<p style="color: #8b4a2b; font-size: 1.1em; font-style: italic;">Não foi possível gerar predição.</p>';
                document.getElementById('compareSection').style.display = 'none';
            }

            document.getElementById('resultsSection').classList.add('show');
        }
        
        function generateNewSequence() {
            const btn = document.getElementById('generateNewBtn');
            btn.disabled = true;
            btn.textContent = 'Gerando...';
            fetch('/generate_new', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' })
            .then(response => response.json())
            .then(data => {
                btn.disabled = false;
                btn.textContent = 'Gerar nova sequência';
                if (data.success && data.results) {
                    showResults(data.results);
                    document.getElementById('compareResult').innerHTML = '';
                } else {
                    alert(data.error || 'Erro ao gerar nova sequência.');
                }
            })
            .catch(err => {
                btn.disabled = false;
                btn.textContent = 'Gerar nova sequência';
                alert('Erro: ' + err.message);
            });
        }
        
        function parseCorrectSequence() {
            const input = document.getElementById('correctSequenceInput').value.trim();
            const parts = input.split(/[,\s]+/).filter(s => s.length > 0).map(s => parseInt(s, 10));
            if (parts.length !== 15 || parts.some(n => isNaN(n) || n < 1 || n > 25)) {
                alert('Digite exatamente 15 números entre 1 e 25, separados por vírgula ou espaço.');
                return null;
            }
            return parts;
        }
        
        function compareWithCorrect() {
            const parts = parseCorrectSequence();
            if (parts === null) return;
            const btn = document.getElementById('compareBtn');
            btn.disabled = true;
            btn.textContent = 'Comparando...';
            fetch('/feedback_compare', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ correct_sequence: parts })
            })
            .then(response => response.json())
            .then(data => {
                btn.disabled = false;
                btn.textContent = 'Comparar';
                const container = document.getElementById('compareResult');
                if (data.success && data.comparison) {
                    const c = data.comparison;
                    const inBoth = (c.in_both && c.in_both.length) ? c.in_both.join(', ') : '—';
                    const onlyGen = (c.only_generated && c.only_generated.length) ? c.only_generated.join(', ') : '—';
                    const onlyCor = (c.only_correct && c.only_correct.length) ? c.only_correct.join(', ') : '—';
                    container.innerHTML = `
                        <div class="result-card" style="margin-top: 15px;">
                            <p style="font-size: 1.2em; margin-bottom: 10px;"><strong>${c.hits}/15 números coincidem (${c.accuracy_pct}%)</strong></p>
                            <p style="color: #5a3419; margin-bottom: 15px;">${c.message}</p>
                            <p style="margin: 8px 0;"><strong>Números em ambos (acertos):</strong> ${inBoth}</p>
                            <p style="margin: 8px 0;"><strong>Gerados que não estão na correta:</strong> ${onlyGen}</p>
                            <p style="margin: 8px 0;"><strong>Na correta que não foram gerados:</strong> ${onlyCor}</p>
                        </div>
                    `;
                } else {
                    container.innerHTML = '<p style="color: #8b4a2b;">' + (data.error || 'Erro ao comparar.') + '</p>';
                }
            })
            .catch(err => {
                btn.disabled = false;
                btn.textContent = 'Comparar';
                document.getElementById('compareResult').innerHTML = '<p style="color: #8b4a2b;">Erro: ' + err.message + '</p>';
            });
        }
        
        function retrainWithCorrect() {
            const parts = parseCorrectSequence();
            if (parts === null) return;
            const btn = document.getElementById('retrainBtn');
            const fullRetrain = document.getElementById('fullRetrainCheck').checked;
            btn.disabled = true;
            btn.textContent = fullRetrain ? 'Retreinando (completo)...' : 'Retreinando...';
            fetch('/feedback_retrain', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ correct_sequence: parts, full_retrain: fullRetrain })
            })
            .then(response => response.json())
            .then(data => {
                btn.disabled = false;
                btn.textContent = 'Retreinar';
                const container = document.getElementById('compareResult');
                if (data.success) {
                    container.innerHTML = '<div class="result-card" style="margin-top: 15px;"><p style="color: #228b22; font-size: 1.1em;">' + (data.message || 'Sequência adicionada ao histórico e modelos reajustados.') + '</p></div>';
                } else {
                    container.innerHTML = '<p style="color: #8b4a2b;">' + (data.error || 'Erro ao retreinar.') + '</p>';
                }
            })
            .catch(err => {
                btn.disabled = false;
                btn.textContent = 'Retreinar';
                document.getElementById('compareResult').innerHTML = '<p style="color: #8b4a2b;">Erro: ' + err.message + '</p>';
            });
        }
    </script>
</body>
</html>
"""

# Logs em memória
logs = []
log_lock = threading.Lock()

def add_log(message, log_type='info'):
    """Adiciona log à lista"""
    with log_lock:
        logs.append({
            'message': message,
            'type': log_type,
            'timestamp': time.time()
        })
        # Mantém apenas últimos 200 logs
        if len(logs) > 200:
            logs.pop(0)

# Classe para capturar prints do agente
class LogCapture:
    def __init__(self):
        self.buffer = []
    
    def write(self, text):
        if text.strip():
            # Remove caracteres de controle e formata
            clean_text = text.strip().replace('\n', '').replace('\r', '')
            if clean_text:
                # Detecta tipo de log
                if '[OK]' in clean_text or 'Carregadas' in clean_text or 'treinado' in clean_text.lower():
                    log_type = 'success'
                elif '[ERRO]' in clean_text or 'Erro' in clean_text or 'ERROR' in clean_text:
                    log_type = 'error'
                elif '[INFO]' in clean_text or 'INFO' in clean_text:
                    log_type = 'info'
                elif 'Epoch' in clean_text or 'Loss' in clean_text:
                    log_type = 'info'
                else:
                    log_type = 'info'
                
                add_log(clean_text, log_type)
                self.buffer.append(clean_text)
    
    def flush(self):
        pass
    
    def get_logs(self):
        return self.buffer

@app.errorhandler(413)
def request_entity_too_large(e):
    """Arquivo maior que o limite (50MB). Retorna JSON para o frontend tratar."""
    return jsonify({'success': False, 'error': 'Arquivo muito grande. Limite: 50MB'}), 413

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Endpoint para upload de arquivo"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'Nenhum arquivo enviado'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Nenhum arquivo selecionado'})
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            add_log(f'Arquivo salvo: {filename}', 'success')
            return jsonify({'success': True, 'filename': filename})
        else:
            return jsonify({'success': False, 'error': 'Tipo de arquivo não permitido'})
    
    except Exception as e:
        add_log(f'Erro no upload: {str(e)}', 'error')
        return jsonify({'success': False, 'error': str(e)})

@app.route('/process', methods=['POST'])
def process_file():
    """Processa o arquivo e treina o agente"""
    import sys
    
    try:
        data = request.get_json(silent=True)
        if not data or not isinstance(data, dict):
            return jsonify({'success': False, 'error': 'Corpo da requisição deve ser JSON com campo "filename"'})
        filename = data.get('filename')
        if not filename or not isinstance(filename, str) or not filename.strip():
            return jsonify({'success': False, 'error': 'Campo "filename" obrigatório'})
        filename = filename.strip()
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'Arquivo não encontrado: ' + filename})
        
        add_log('Criando agente...', 'info')
        agent_state['status'] = 'loading'
        agent_state['message'] = 'Carregando sequências...'
        
        # Captura prints do agente
        log_capture = LogCapture()
        old_stdout = sys.stdout
        
        try:
            # Redireciona stdout para capturar logs
            sys.stdout = log_capture
            
            # Cria agente
            agent = PatternAgent(min_value=1, max_value=25, sequence_length=15)
            
            add_log('Carregando sequências do Excel...', 'info')
            sequences = agent.load_sequences_from_excel(filepath)
            
            if not sequences:
                add_log('Nenhuma sequência encontrada!', 'error')
                sys.stdout = old_stdout
                return jsonify({'success': False, 'error': 'Nenhuma sequência válida encontrada'})
            
            agent_state['sequences_count'] = len(sequences)
            agent_state['status'] = 'training'
            agent_state['message'] = 'Treinando modelos...'
            agent_state['training_progress'] = 0
            agent_state['training_total'] = 0
            agent_state['training_message'] = ''
            
            def _progress_callback(current, total, message):
                agent_state['training_progress'] = current
                agent_state['training_total'] = total
                agent_state['training_message'] = message or ''
            
            # Treina sistema (os prints serão capturados automaticamente)
            agent.train_full_system(progress_callback=_progress_callback)
            
            agent_state['training_total'] = 0
            agent_state['status'] = 'ready'
            agent_state['message'] = 'Sistema pronto!'
            agent_state['agent'] = agent
            
            # Gera uma sequência NOVA baseada nos padrões dos ~3500 treinados (não completa a primeira)
            if len(sequences) > 0:
                add_log('Gerando nova sequência a partir dos padrões aprendidos...', 'info')
                # Gera os 15 números usando só padrões/modelos (sequência nova, não cópia da primeira)
                result = agent.suggest_next_number([], allow_empty=True)

                agent_state['results'] = _make_json_serializable(result)

                return jsonify({
                    'success': True,
                    'results': _make_json_serializable(result),
                    'sequences_count': len(sequences)
                })
            
            return jsonify({'success': True, 'results': None})
        
        finally:
            # Restaura stdout
            sys.stdout = old_stdout
    
    except Exception as e:
        import traceback
        error_msg = str(e) + '\n' + traceback.format_exc()
        add_log(f'Erro no processamento: {error_msg}', 'error')
        agent_state['status'] = 'error'
        agent_state['message'] = str(e)
        return jsonify({'success': False, 'error': str(e)})

@app.route('/status')
def get_status():

    serializable_state = {
        'agent': None if agent_state['agent'] is None else 'loaded',
        'status': agent_state['status'],
        'message': agent_state['message'],
        'sequences_count': agent_state['sequences_count'],
        'training_progress': agent_state['training_progress'],
        'training_total': agent_state.get('training_total', 0),
        'training_message': agent_state.get('training_message', ''),
        'results': _make_json_serializable(agent_state['results'])
    }
    return jsonify(serializable_state)

@app.route('/logs')
def get_logs():
    """Retorna logs recentes"""
    with log_lock:
        # Retorna apenas logs novos (últimos 50)
        recent_logs = logs[-50:] if len(logs) > 50 else logs
        return jsonify({'logs': recent_logs})

@app.route('/predict', methods=['POST'])
def predict():
    """Faz predição para uma sequência"""
    try:
        data = request.json
        sequence = data.get('sequence', [])
        
        if not agent_state['agent']:
            return jsonify({'success': False, 'error': 'Agente não treinado'})
        
        result = agent_state['agent'].suggest_next_number(sequence)
        return jsonify({'success': True, 'result': _make_json_serializable(result)})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/generate_new', methods=['POST'])
def generate_new():
    """Gera uma nova sequência com o agente atual (sem reprocessar o arquivo)."""
    try:
        if not agent_state['agent']:
            return jsonify({'success': False, 'error': 'Agente não treinado. Processe o arquivo primeiro.'})
        result = agent_state['agent'].suggest_next_number([], allow_empty=True)
        agent_state['results'] = _make_json_serializable(result)
        return jsonify({'success': True, 'results': agent_state['results']})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/feedback_compare', methods=['POST'])
def feedback_compare():
    """Compara a sequência gerada com a correta (apenas conjunto de números, sem ordem)."""
    try:
        data = request.json
        correct_sequence = data.get('correct_sequence', [])
        
        if not agent_state['agent']:
            return jsonify({'success': False, 'error': 'Agente não treinado'})
        
        last_results = agent_state.get('results')
        if not last_results or not last_results.get('next_sequence'):
            return jsonify({'success': False, 'error': 'Nenhuma sequência gerada para confrontar. Gere uma sequência primeiro.'})
        
        generated = last_results['next_sequence']
        if len(correct_sequence) != 15:
            return jsonify({'success': False, 'error': 'A sequência correta deve ter exatamente 15 números.'})
        
        correct_sequence = [int(x) for x in correct_sequence]
        result = agent_state['agent'].compare_sequences_set(generated, correct_sequence)
        if result.get('error'):
            return jsonify({'success': False, 'error': result['error']})
        return jsonify({'success': True, 'comparison': _make_json_serializable(result)})
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e) + '\n' + traceback.format_exc()})


@app.route('/feedback_retrain', methods=['POST'])
def feedback_retrain():
    """Adiciona a sequência correta ao histórico e reajusta os modelos. Opcional: retreinamento completo."""
    try:
        data = request.json or {}
        correct_sequence = data.get('correct_sequence', [])
        full_retrain = data.get('full_retrain', False)
        
        if not agent_state['agent']:
            return jsonify({'success': False, 'error': 'Agente não treinado'})
        
        if len(correct_sequence) != 15:
            return jsonify({'success': False, 'error': 'A sequência correta deve ter exatamente 15 números.'})
        
        correct_sequence = [int(x) for x in correct_sequence]
        last_generated = None
        if agent_state.get('results') and agent_state['results'].get('next_sequence'):
            last_generated = agent_state['results']['next_sequence']
        result = agent_state['agent'].retrain_with_correct_sequence(correct_sequence, last_generated=last_generated, full_retrain=full_retrain)
        if result.get('error'):
            return jsonify({'success': False, 'error': result['error']})
        return jsonify({'success': True, 'message': result.get('message', '')})
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e) + '\n' + traceback.format_exc()})

if __name__ == '__main__':
    print("=" * 60)
    print("Interface Web do Pattern Agent")
    print("=" * 60)
    print("Acesse: http://localhost:5000")
    print("=" * 60)
    # use_reloader=False evita OSError WinError 10038 no Windows (reloader mexe em soquete)
    # Processamento pode demorar vários minutos; manter conexão estável
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
