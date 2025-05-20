import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import re

# === MODELO DE TRADUÇÃO (PT → EN) COM NLLB ===
nllb_model_name = "facebook/nllb-200-distilled-600M"
nllb_tokenizer = AutoTokenizer.from_pretrained(nllb_model_name)
nllb_model = AutoModelForSeq2SeqLM.from_pretrained(nllb_model_name)

def traduzir_para_ingles(texto_pt):
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
        "Example 1:\nSentence: Double 8\nExpression: 2 * 8\n"
        "Example 2:\nSentence: The sum of 2 and 2\nExpression: 2 + 2\n"
        "Example 3:\nSentence: The difference between 7 and 3\nExpression: 7 - 3\n"
        "Example 4:\nSentence: Three times 4\nExpression: 3 * 4\n"
        f"Sentence: {texto}\nExpression:"
    )
    result = generator(prompt, max_new_tokens=30, do_sample=False)
    expressao = result[0]["generated_text"]
    return expressao.strip()

# === RESOLUÇÃO DA EXPRESSÃO ===
def resolver_expressao(expr):
    try:
        # Permite apenas números e operadores seguros
        if re.fullmatch(r"[0-9\+\-\* \(\)]+", expr):
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

# === AVALIAÇÃO DE ACURÁCIA ===
testes = [
    ("o dobro de 8", "2 * 8"),
    ("a soma de 2 mais 2", "2 + 2"),
    ("a metade de 6", "2 * 3"),  # tratado como multiplicação
    ("a diferença entre 7 e 3", "7 - 3"),
    ("3 vezes 4", "3 * 4"),
    ("4 mais 4", "4 + 4"),
    ("6 menos 2", "6 - 2")
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
        acertou = esperado_formatado == gerado_formatado
        resultados.append(f"{entrada} => {gerado} ({'✅' if acertou else '❌'})")
        if acertou:
            acertos += 1

    acuracia = round((acertos / total) * 100, 2)
    resumo = f"Acurácia: {acertos}/{total} = {acuracia}%\n\n" + "\n".join(resultados)
    return resumo

# === INTERFACE COM GRADIO ===
with gr.Blocks() as demo:
    gr.Markdown("# 🧮 Conversor de Frases para Expressões Matemáticas")

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
