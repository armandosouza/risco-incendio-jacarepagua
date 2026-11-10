import pandas as pd
import numpy as np
import pickle
import gradio as gr
import os

try:
    with open('modelo_arvore_fma.pkl', 'rb') as f:
        modelo = pickle.load(f)
    print("-> Modelo 'modelo_arvore_fma.pkl' carregado com sucesso!")
except FileNotFoundError:
    print("ERRO: O arquivo 'modelo_arvore_fma.pkl' não foi encontrado.")

# Função de predição e regras de sinalização
def prever_risco_incendio(temp_max, umid_min, vento_max, dias_sem_chuva):
    entrada = pd.DataFrame([[temp_max, umid_min, vento_max, dias_sem_chuva]],
                           columns=['Temp_Max', 'Umid_Min', 'Vento_Max', 'Dias_Sem_Chuva'])

    # Classificação pela Árvore de Decisão
    risco = modelo.predict(entrada)[0]

    # Definição das saídas visuais e orientações comunitárias
    if risco == 'Baixo':
        semaforo = "🟢 RISCO BAIXO"
        orientacao = (
            "✅ CONDIÇÕES FAVORÁVEIS:\n"
            "• Remova folhas secas, galhos e lixo acumulado ao redor de sua casa.\n"
            "• Chame um serviço especializado para realizar a poda das árvores.\n"
            "• Deixe faixas de terra sem vegetação ao redor de propriedades e matas.\n"
            "• Mapeie lugares com fontes de água para que os bombeiros ou brigadistas possam usar.\n"
        )
    elif risco == 'Médio':
        semaforo = "🟡 RISCO MÉDIO"
        orientacao = (
            "⚠️ ATENÇÃO ELEVADA:\n"
            "• PROIBIDO qualquer atividade de queima de lixo, folhas secas ou entulho na comunidade.\n"
            "• Não descarte em vias públicas cigarros e garrafas de vidro, para não ocasionar iniciar o fogo.\n"
            "• Fique de olho nos canais da Defesa Civil."
        )
    else:  # Alto
        semaforo = "🔴 RISCO ALTO / PERIGOSO"
        orientacao = (
            "🚨 ALERTA MÁXIMO DE EMERGÊNCIA:\n"
            "• Fica estritamente proibido o uso de fogo para qualquer finalidade ao ar livre.\n"
            "• Tenha um kit de emergência pronto dentro de casa.\n"
            "• Se possível, molhe as áreas de vegetação seca mais próximas às residências durante os horários mais quentes do dia.\n"
            "• Em caso de fumaça ou fogo, LIGUE IMEDIATAMENTE: Bombeiros (193) ou Defesa Civil (199)."
        )

    return semaforo, orientacao

# Construção do painel através de Gradio Blocks

with gr.Blocks(title="Painel de Prevenção a Incêndios - Jacarepaguá", theme=gr.themes.Soft(
    font=[gr.themes.GoogleFont("Poppins"), "Arial", "sans-serif"]
)) as demo:
    gr.Markdown(
        """
        # Painel de Prevenção a Incêndios Florestais para a região de Jacarepaguá<br>
        ### Estação Meteorológica A636 (Jacarepaguá / INMET) — Entorno do Maciço da Pedra Branca
        *Sistema de alerta preditivo baseado em Aprendizado de Máquina e na Fórmula de Monte Alegre.
        """
    )

    with gr.Row():
        # Coluna da Esquerda: Controles dos Sensores / Sliders
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Condições Climáticas:")
            temp_max = gr.Slider(minimum=15.0, maximum=45.0, value=32.0, step=0.5, label="Temperatura Máxima Prevista (°C)")
            umid_min = gr.Slider(minimum=10.0, maximum=100.0, value=40.0, step=1.0, label="Umidade Relativa Mínima (%)")
            vento_max = gr.Slider(minimum=0.0, maximum=30.0, value=6.0, step=0.5, label="Rajada Máxima de Vento (m/s)")
            dias_sem_chuva = gr.Number(minimum=0, value=5, precision=0, label="Dias Consecutivos Sem Chuva (Estiagem)")

            btn_prever = gr.Button("🔍 Calcular Risco de Incêndio", variant="primary")

        # Coluna da Direita: Semáforo e Orientações
        with gr.Column(scale=1):
            gr.Markdown("### 🚦 Diagnóstico do Semáforo e Ações Comunitárias")
            saida_semaforo = gr.Textbox(label="Nível de Risco Previsto pelo Modelo", interactive=False)
            saida_orientacao = gr.Textbox(label="Protocolo de Ação e Recomendações", lines=8, interactive=False)

    # Conexões para atualizar os resultados tanto via botão quanto ao arrastar os sliders
    inputs_list = [temp_max, umid_min, vento_max, dias_sem_chuva]
    outputs_list = [saida_semaforo, saida_orientacao]

    btn_prever.click(fn=prever_risco_incendio, inputs=inputs_list, outputs=outputs_list)
    temp_max.change(fn=prever_risco_incendio, inputs=inputs_list, outputs=outputs_list)
    umid_min.change(fn=prever_risco_incendio, inputs=inputs_list, outputs=outputs_list)
    vento_max.change(fn=prever_risco_incendio, inputs=inputs_list, outputs=outputs_list)
    dias_sem_chuva.change(fn=prever_risco_incendio, inputs=inputs_list, outputs=outputs_list)

    gr.Markdown("---")
    gr.Markdown("📍 *PROJETO INTEGRADOR 3: CLASSIFICAÇÃO DO RISCO DE INCÊNDIOS FLORESTAIS EM ÁREAS MUNICIPAIS POR MEIO DE APRENDIZAGEM DE MÁQUINA*")

port = int(os.environ.get("PORT", 7860))

# 2. Inicializa o Gradio liberando o IP 0.0.0.0 e travando a porta do Render
demo.launch(
    server_name="0.0.0.0",
    server_port=port,
    prevent_thread_lock=False
)
