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
```

```bash
pip install opencv-python mediapipe scikit-learn pandas joblib
```

```bash
pip install mediapipe opencv-python tensorflow numpy
```

```bash
sudo apt install python3-venv python3-full
```

```bash
sudo apt install python3-xyz
```

```bash
pip install mediapipe opencv-python tensorflow scikit-learn numpy pandas
```

```bash
sudo apt install python3-pip
```

```bash
pip install seaborn matplotlib
```

## Como Usar
Entrar no ambiente venv
```bash
source venv/bin/activate
```
Coletar dados
```bash
python3 coletar_dados.py
```
Treinar Modelo
```bash
python3 treinar_modelo.py
```

Reconhecer em tempo real
```bash
python3 reconhecer_tempo_real.py
```