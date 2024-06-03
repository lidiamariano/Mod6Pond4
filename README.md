# Módulo 6 - Ponderada 4 - Utilizando Perceptrons Multi Camada

## 🔍 Descrição
Este repositório contém implementações de um Perceptron Multicamadas (MLP) para resolver o problema da porta lógica XOR, tanto em Python puro quanto utilizando PyTorch.
![Mod6Pond4](https://github.com/lidiamariano/Mod6Pond4/assets/123901342/bd72da84-2b51-4ea8-9af1-d468e88e4a4f)

## 💻 Estrutura do Projeto
- `mlp.py`: Implementa um Perceptron Multicamadas (MLP) em Python puro com uma camada escondida, utilizando a função sigmoide e um algoritmo de retropropagação para treinar o modelo na resolução do problema da porta XOR.
- `mlp_pytorch.py`: Implementa um Perceptron Multicamadas (MLP) usando a biblioteca PyTorch, com uma camada escondida e a função sigmoide, treinando o modelo para resolver o problema da porta XOR através de retropropagação usando o otimizador SGD.

## ⚙️ Instrução de execução
### Pré-requisitos:
- Python 
- NumPy
- PyTorch
### Instalação:
1. Clonagem do repositório
`git clone https://github.com/lidiamariano/Mod6Pond4`
2. Instalação das Dependências
`pip install -r requirements.txt`
### Execução 
1. Vá até o caminho relativo do código
`cd src`
2. Para rodar a implementação em Python puro
`python3 mlp.py`
3. Para rodar a implementação em PyTorch
`python3 mlp_pytorch.py`
