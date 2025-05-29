import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import re
import random

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

# === GERAÇÃO DE TESTES ALEATÓRIOS ===

def gerar_teste_aleatorio():
    operadores = {
        "+": ["a soma de {} e {}", "quanto é {} mais {}", "adicione {} a {}", "{} mais {}"],
        "-": ["a diferença entre {} e {}", "quanto é {} menos {}", "subtraia {} de {}", "{} menos {}"],
        "*": ["o produto de {} e {}", "{} vezes {}", "multiplique {} por {}", "o triplo de {}", "o dobro de {}"],
        "/": ["a divisão de {} por {}", "{} dividido por {}", "a metade de {}", "um terço de {}"]
    }

    op_simbolo = random.choice(list(operadores.keys()))
    templates = operadores[op_simbolo]
    template = random.choice(templates)

    num1 = random.randint(1, 20)
    num2 = random.randint(1, 20)

    if "dobro de" in template:
        frase = template.format(num1)
        expressao = f"2 * {num1}"
    elif "triplo de" in template:
        frase = template.format(num1)
        expressao = f"3 * {num1}"
    elif "metade de" in template:
        frase = template.format(num1)
        expressao = f"{num1} / 2"
    elif "terço de" in template:
        # Para garantir que seja um número inteiro para o terço, podemos ajustar aqui ou aceitar float
        # Por simplicidade, vamos permitir float no resultado.
        frase = template.format(num1)
        expressao = f"{num1} / 3"
    elif op_simbolo == '/':
        if num2 == 0:
            num2 = random.randint(1, 20) # Garante que num2 não seja zero
        # Opcional: Garanta que num1 seja múltiplo de num2 para divisões exatas
        if num1 % num2 != 0:
            num1 = num2 * random.randint(1, 5)
            if num1 == 0: num1 = num2 # Evita 0 / X
        frase = template.format(num1, num2)
        expressao = f"{num1} {op_simbolo} {num2}"
    else:
        frase = template.format(num1, num2)
        expressao = f"{num1} {op_simbolo} {num2}"

    return (frase, expressao)

# === AVALIAÇÃO DE ACURÁCIA COM COMPARAÇÃO DE VALORES ===

def avaliar_acuracia_aleatoria(num_testes=20):
    acertos = 0
    total = num_testes
    resultados = []

    testes_aleatorios = [gerar_teste_aleatorio() for _ in range(num_testes)]

    for entrada, esperado in testes_aleatorios:
        frase_en = traduzir_para_ingles(entrada)
        gerado = texto_para_expressao(frase_en)
        esperado_formatado = esperado.replace(" ", "")
        gerado_formatado = gerado.replace(" ", "")
        try:
            # Para comparação de floats, usar uma pequena tolerância pode ser melhor
            # do que igualdade exata, especialmente para divisões.
            # No entanto, para este caso, onde esperamos inteiros após as divisões forçadas,
            # a igualdade exata geralmente funciona. Se precisar de floats, considere:
            # abs(eval(esperado_formatado) - eval(gerado_formatado)) < 1e-9
            acertou = eval(esperado_formatado) == eval(gerado_formatado)
        except Exception as e:
            acertou = False
        resultados.append(f"'{entrada}' => '{gerado}' (Esperado: '{esperado}') ({'✅' if acertou else '❌'})")
        if acertou:
            acertos += 1

    acuracia = round((acertos / total) * 100, 2)
    # Apenas alteramos a string de resumo para destacar a porcentagem
    resumo = f"## Acurácia: {acertos}/{total} = **{acuracia}%**\n\n" + "\n".join(resultados)
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
        gr.Markdown("Gera testes aleatórios para avaliar o desempenho do modelo.")
        num_testes_input = gr.Slider(minimum=10, maximum=100, step=5, value=20, label="Número de Testes Aleatórios")
        botao_avaliar = gr.Button("Avaliar Acurácia do Modelo (Aleatório)")
        saida_avaliacao = gr.Textbox(lines=15, label="Relatório de Acurácia", interactive=False) # Adicionado interactive=False
        
        botao_avaliar.click(fn=avaliar_acuracia_aleatoria, inputs=num_testes_input, outputs=saida_avaliacao)

demo.launch()
