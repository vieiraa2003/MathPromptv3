
# 🧮 Conversor de Frases para Expressões Matemáticas

Este projeto converte frases matemáticas simples escritas em **português** para **expressões matemáticas** e calcula o resultado.

Foi desenvolvido com **Hugging Face Transformers**, **Gradio** para interface e suporta tradução automática para inglês com o modelo **NLLB**.

---

## ✨ Funcionalidades

- Tradução automática de frases em português para inglês (modelo `facebook/nllb-200-distilled-600M`)
- Geração de expressão matemática com `google/flan-t5-base`
- Resolução segura das expressões (sem `eval` inseguro)
- Interface gráfica com Gradio
- Avaliação sob demanda da acurácia do modelo

---

## 🚀 Como rodar o projeto

1. Clone o repositório:

```bash
https://github.com/vieiraa2003/MathPromptv3
```

2. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. Rode o app com Gradio:

```bash
python app_gradio.py
```

Acesse no navegador: [http://localhost:7860](http://localhost:7860)

---

## 📊 Avaliação de Acurácia

Clique em **"Avaliar Acurácia"** na interface para verificar o desempenho atual do modelo com base em frases simples.

---

## 📁 Estrutura do Projeto

```
.
├── app_gradio.py         # Código principal com interface Gradio
├── requirements.txt      # Dependências do projeto
├── README.md             # Este arquivo
└── .gitignore            # Ignora arquivos desnecessários no Git
```

---

## ✅ Exemplos de entrada válidos

- "o dobro de 8"
- "a soma de 2 mais 2"
- "a metade de 6"
- "a diferença entre 7 e 3"
- "2 mais 2"
- "3 vezes 4"

---

## 🧠 Modelos usados

- `facebook/nllb-200-distilled-600M` — Tradução português → inglês
- `google/flan-t5-base` — Geração de expressões matemáticas

---

## 🔒 Segurança

A resolução das expressões é feita com verificação por regex para evitar execução de código malicioso.

---

## 📃 Licença

Este projeto é open-source e pode ser usado livremente com atribuição.
