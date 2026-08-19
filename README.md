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
# SignLanguageRecognizer

Projeto para reconhecimento de gestos em Libras (Língua Brasileira de Sinais) usando MediaPipe e scikit-learn.

Este repositório contém scripts para coletar dados de gestos, treinar um classificador e executar reconhecimento em tempo real via webcam.

**Requisitos mínimos**
- Python 3.8+
- Webcam compatível

**Arquivos importantes no repositório**
- `dataset.csv`, `datasetoficial.csv`: datasets (exemplos de dados coletados).
- `modelo_alfabeto.pkl`: modelo treinado (incluído no repositório).
- `static/hand_landmarker.task`: recurso do MediaPipe usado pelo pipeline.
- `coletar_dados.py`, `treinar_modelo.py`, `reconhecer_tempo_real.py`: scripts principais.

**Instalação recomendada (passo a passo)**

1. Clonar o repositório:

```bash
git clone <repo-url>
cd SignLanguageRecognizer
```

2. Criar e ativar um ambiente virtual:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Instalar dependências (use o `requirements.txt` fornecido):

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Se preferir instalar manualmente pacotes adicionais úteis:

```bash
pip install seaborn matplotlib tensorflow
```

**Uso básico**

- Coletar dados (executa captura via webcam e salva em CSV):

```bash
python3 coletar_dados.py
```

- Treinar o modelo a partir do dataset:

```bash
python3 treinar_modelo.py
```

- Executar reconhecimento em tempo real (usa a webcam):

```bash
python3 reconhecer_tempo_real.py
```

**Observações sobre os arquivos**
- `modelo_alfabeto.pkl` já existe no repositório e pode ser usado diretamente por `reconhecer_tempo_real.py`.
- `static/hand_landmarker.task` é requerimento do MediaPipe para alguns pipelines — o arquivo está incluído.

**Formato esperado do dataset**
- Os scripts usam CSVs com colunas de features extraídas (landmarks) e uma coluna alvo (label). Use `checar_csv.py` para validar o CSV antes de treinar.

**Resolução de problemas comuns**
- Erro de import (e.g. `No module named mediapipe`): verifique que o venv está ativado e execute `pip install -r requirements.txt`.
- Webcam não detectada: verifique permissões e teste com `ffmpeg -f video4linux2 -list_formats all -i /dev/video0` ou outro software.
- Versões incompatíveis do MediaPipe/OpenCV: use as versões em `requirements.txt`.

**Contribuição e Issues**
- Abra uma issue para bugs ou pedidos de feature.

**Licença**
- Este projeto está licenciado sob MIT. Veja o arquivo LICENSE (se aplicável) ou adicione uma licença conforme necessário.

---

Se quiser, eu posso:
- atualizar `requirements.txt` com versões recomendadas;
- adicionar um `LICENSE`;
- ou incluir instruções mais detalhadas para `coletar_dados.py` e `treinar_modelo.py`.
