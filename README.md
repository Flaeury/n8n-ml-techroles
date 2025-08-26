# n8n-ml-techroles: Classificação de Cargos em Dados de TI com Machine Learning

Este projeto utiliza uma abordagem de ponta a ponta para classificar perfis de profissionais de TI em categorias de cargos. A metodologia empregada integra o **Processo de Descoberta de Conhecimento em Bancos de Dados (KDD)** com a ferramenta de automação **n8n** e modelos de **Machine Learning** para criar um sistema de classificação automatizado e eficiente.

## 🎯 Objetivo

O objetivo principal é processar dados brutos de perfis profissionais, extrair características relevantes e classificá-los em uma das categorias de cargos definidas. Este processo automatizado pode ser aplicado a tarefas como recrutamento, análise de mercado e segmentação de público.

## ⚙️ Arquitetura do Projeto

A solução é construída em três pilares principais, conforme ilustrado no fluxo de trabalho:

### 1. Pré-processamento e Treinamento do Modelo (KDD)

Esta etapa segue o rigoroso processo KDD para transformar os dados brutos em um formato que pode ser usado para treinar um modelo de Machine Learning.

* **Coleta e Limpeza:** Os dados são extraídos de uma base de dados e passam por um processo de limpeza para remover inconsistências e preencher valores ausentes.
* **Transformação:** Os dados textuais (como "bancos de dados" e "cargos") são codificados em formatos numéricos que os algoritmos de ML podem entender, como a contagem de bancos SQL/NoSQL e o `LabelEncoder`.
* **Mineração de Dados:** Um modelo de classificação é treinado para aprender a relação entre as características do perfil (idade, gênero, experiência, etc.) e o cargo correspondente. O projeto explora dois modelos:
    * **Random Forest Classifier:** Um modelo de árvore de decisão robusto e de alto desempenho.
    * **MLP Classifier (Rede Neural):** Para explorar uma abordagem mais avançada de deep learning.
* **Avaliação:** A performance dos modelos é validada usando métricas como **Acurácia**, **Precisão**, **Recall** e a **Matriz de Confusão** para garantir que eles sejam confiáveis.
* **Serialização:** O modelo treinado e os objetos de pré-processamento (como o `LabelEncoder`) são salvos usando `joblib`, permitindo que o modelo seja reutilizado e implantado sem a necessidade de novo treinamento.

### 2. Automação do Fluxo (n8n)

A ferramenta **n8n** é utilizada para orquestrar o fluxo de trabalho de ponta a ponta, conectando as diferentes etapas de forma visual e sem a necessidade de código complexo.

* **Trigger (WebHook):** O fluxo é iniciado por uma chamada HTTP externa, permitindo que ele seja integrado a outras plataformas ou formulários.
* **Node de Pré-processamento:** Um nó de código Python executa as etapas de limpeza e transformação dos dados de entrada, garantindo que o novo perfil de usuário esteja no formato correto para a predição.
* **Node de Predição (Modelo):** O modelo de Machine Learning (`modelo_cargos.pkl`) é carregado e utilizado para prever o cargo do perfil processado.
* **Node de Resultado:** A predição do modelo é enviada como uma resposta, que pode ser consumida por outro serviço ou exibida em uma interface.

### 3. Integração e Implantação

O fluxo completo é exposto como um endpoint de API, permitindo que qualquer aplicação (como um formulário de cadastro, um sistema de RH, etc.) envie dados de um novo perfil e receba o cargo previsto instantaneamente.

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python
* **Bibliotecas de ML:** Scikit-learn, Pandas, Joblib
* **Automação:** n8n
* **Visualização:** Matplotlib (para análise)

## 🚀 Como Executar o Projeto

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/Flaeury/n8n-ml-techroles.git](https://github.com/Flaeury/n8n-ml-techroles.git)
    cd n8n-ml-techroles
    ```
2.  **Instale as dependências Python:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Execute o script de treinamento:**
    ```bash
    python treinar_modelo.py
    ```
    Isso irá gerar o arquivo `modelo_cargos.pkl`.
4.  **Configure o n8n:**
    * Importe o fluxo `fluxo-n8n.json` (ou recrie o fluxo manualmente conforme a imagem).
    * Certifique-se de que o nó de código Python tenha acesso ao seu modelo.

## 📄 Link para teste:

 * **FrontEnd:** [https://agent-inteligent-n8n-forms.vercel.app](https://agent-inteligent-n8n-forms.vercel.app)