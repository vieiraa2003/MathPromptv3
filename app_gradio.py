!pip install gradio
!pip install transformers

import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import re

# === DICIONÁRIO DE SINÔNIMOS PARA NORMALIZAÇÃO EM PORTUGUÊS ===
sinonimos_operadores = {
    "+": ["mais", "soma", "adição", "adicionar", "somar"],
    "-": ["menos", "subtração", "diminuir", "diferença", "subtrair"],
    "*": ["vezes", "multiplicado", "multiplicação", "produto", "dobro", "triplo"],
    "/": ["dividido", "divisão", "metade", "terço"]
}

def normalizar_operadores(frase):
    frase = frase.lower()
    for operador, sinonimos in sinonimos_operadores.items():
        for palavra in sinonimos:
            frase = re.sub(rf"\b{palavra}\b", operador, frase)
    return frase

# === MODELO DE TRADUÇÃO (PT → EN) COM NLLB ===
nllb_model_name = "facebook/nllb-200-distilled-600M"
nllb_tokenizer = AutoTokenizer.from_pretrained(nllb_model_name)
nllb_model = AutoModelForSeq2SeqLM.from_pretrained(nllb_model_name)

def traduzir_para_ingles(texto_pt):
    texto_pt = normalizar_operadores(texto_pt)
    nllb_tokenizer.src_lang = "por_Latn"
    inputs = nllb_tokenizer(texto_pt, return_tensors="pt", padding=True)
    eng_token_id = nllb_tokenizer.convert_tokens_to_ids(">>eng_Latn<<")
    translated_tokens = nllb_model.generate(**inputs, forced_bos_token_id=eng_token_id)
    texto_en = nllb_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return texto_en

# === MODELO DE GERAÇÃO DE EXPRESSÃO ===
flan_model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(flan_model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(flan_model_name)
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

def texto_para_expressao(texto):
    prompt = (
        "Convert the sentence into a math expression using only numbers and operators.\n"
        "Sentence: Double 8 → 2 * 8\n"
        "Sentence: The sum of 2 and 2 → 2 + 2\n"
        "Sentence: The difference between 7 and 3 → 7 - 3\n"
        "Sentence: Three times 4 → 3 * 4\n"
        "Sentence: 10 divided by 2 → 10 / 2\n"
        "Sentence: Half of 6 → 6 / 2\n"
        "Sentence: 4 plus 4 → 4 + 4\n"
        "Sentence: 9 minus 5 → 9 - 5\n"
        "Sentence: Multiply 6 and 3 → 6 * 3\n"
        "Sentence: What is 12 divided by 4? → 12 / 4\n"
        f"Sentence: {texto}\nExpression:"
    )
    result = generator(prompt, max_new_tokens=30, do_sample=False)
    expressao = result[0]["generated_text"]
    return expressao.strip()

# === RESOLUÇÃO DA EXPRESSÃO ===
def resolver_expressao(expr):
    try:
        if re.fullmatch(r"[0-9\+\-\*/\(\) ]+", expr):
            return eval(expr)
        else:
            return "Expressão inválida"
    except Exception as e:
        return f"Erro: {e}"

# === FUNÇÃO PRINCIPAL ===
def processar(frase):
    frase_en = traduzir_para_ingles(frase)
    expressao = texto_para_expressao(frase_en)
    resultado = resolver_expressao(expressao)
    return expressao, resultado

# === AVALIAÇÃO DE ACURÁCIA COM COMPARAÇÃO DE VALORES ===
testes = [
    ("o dobro de 8", "2 * 8"),
    ("a soma de 2 mais 2", "2 + 2"),
    ("a metade de 6", "6 / 2"),
    ("a diferença entre 7 e 3", "7 - 3"),
    ("3 vezes 4", "3 * 4"),
    ("4 mais 4", "4 + 4"),
    ("6 menos 2", "6 - 2"),
    ("10 dividido por 2", "10 / 2"),
    ("quanto é 9 menos 3", "9 - 3"),
    ("qual o produto de 5 e 5", "5 * 5"),
    ("qual a divisão de 8 por 4", "8 / 4")
]

def avaliar_acuracia():
    acertos = 0
    total = len(testes)
    resultados = []

    for entrada, esperado in testes:
        frase_en = traduzir_para_ingles(entrada)
        gerado = texto_para_expressao(frase_en)
        esperado_formatado = esperado.replace(" ", "")
        gerado_formatado = gerado.replace(" ", "")
        try:
            acertou = eval(esperado_formatado) == eval(gerado_formatado)
        except:
            acertou = False
        resultados.append(f"{entrada} => {gerado} ({'✅' if acertou else '❌'})")
        if acertou:
            acertos += 1

    acuracia = round((acertos / total) * 100, 2)
    resumo = f"Acurácia: {acertos}/{total} = {acuracia}%\n\n" + "\n".join(resultados)
    return resumo

# === INTERFACE COM GRADIO ===
with gr.Blocks() as demo:
    gr.Markdown("# 🧮 Math Prompt")

    with gr.Row():
        entrada = gr.Textbox(label="Digite uma frase matemática (em português)")
        botao = gr.Button("Converter")
    
    expressao = gr.Textbox(label="Expressão Gerada")
    resultado = gr.Textbox(label="Resultado da Expressão")

    botao.click(fn=processar, inputs=entrada, outputs=[expressao, resultado])

    with gr.Accordion("📊 Avaliar Acurácia", open=False):
        botao_avaliar = gr.Button("Avaliar Acurácia do Modelo")
        saida_avaliacao = gr.Textbox(lines=10, label="Relatório de Acurácia")
        botao_avaliar.click(fn=avaliar_acuracia, outputs=saida_avaliacao)

demo.launch()
