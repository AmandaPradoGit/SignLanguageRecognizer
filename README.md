# SignLanguageRecognizer

Sistema para reconhecimento de gestos em Libras (Língua Brasileira de Sinais), utilizando a biblioteca MediaPipe do Python. O projeto permite coletar dados de gestos, treinar um modelo de machine learning e realizar reconhecimento em tempo real via câmera.

## Funcionalidades
- Coleta de Dados: Capture gestos das mãos para criar um dataset.<br><br>
- Treinamento do Modelo: Treine um classificador Random Forest com validação cruzada.<br><br>
- Reconhecimento em Tempo Real: Identifique gestos ao vivo com estabilização de predições.

## Requisitos 
- Python 3.8 ou superior<br><br>
- Webcam compatível

## Instalações Necessárias

```bash
python3 -m pip install opencv-python mediapipe scikit-learn pandas joblib
# SignLanguageRecognizer — instruções rápidas

Instruções mínimas para executar coleta de dados, treino e reconhecimento em tempo real.

1) Criar e ativar ambiente virtual (Linux/macOS):

```bash
python3 -m venv venv
source venv/bin/activate
```

2) Instalar dependências a partir de `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3) Coletar dados (usa webcam):

```bash
python3 coletar_dados.py
```

4) Treinar o modelo a partir do dataset:

```bash
python3 treinar_modelo.py
```

5) Executar reconhecimento em tempo real (webcam):

```bash
python3 reconhecer_tempo_real.py
```

Observações:
- Garanta que o ambiente virtual (`venv`) esteja ativado antes de rodar os comandos acima.
- Se ocorrerem erros de importação com `mediapipe`, `scikit-learn` ou `joblib`, recrie o `venv` e reinstale com `pip install -r requirements.txt`.

Arquivos de interesse no repositório: `dataset.csv`, `datasetoficial.csv`, `modelo_alfabeto.pkl`, `static/hand_landmarker.task`.
