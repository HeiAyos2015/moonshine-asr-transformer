"""
App interactiva de inferencia con Moonshine (Transformer encoder-decoder para voz a texto).

Proyecto final - Procesamiento de Datos Secuenciales con Deep Learning
Autora: Heidy Romero Ayos

Artículo base:
    N. Jeffries et al., "Moonshine: Speech Recognition for Live Transcription
    and Voice Commands", arXiv:2410.15608, 2024.

Cómo ejecutar:
    pip install -r requirements.txt
    streamlit run app.py
"""

import io
import re
import time

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from transformers import AutoProcessor, MoonshineForConditionalGeneration

SR = 16000                # Moonshine trabaja con audio a 16 kHz
TOKENS_POR_SEGUNDO = 6.5  # límite recomendado de tokens por segundo de audio

st.set_page_config(page_title="Moonshine: voz a texto", page_icon="🎙️", layout="wide")


# ---------------------------------------------------------------------------
# 1. Carga del modelo (se guarda en caché para no recargarlo en cada interacción)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando el modelo preentrenado...")
def cargar_modelo(nombre_modelo: str):
    """Descarga (la primera vez) y carga el procesador y el modelo Moonshine."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(nombre_modelo)
    # attn_implementation="eager" permite que el modelo devuelva las matrices de atención
    model = MoonshineForConditionalGeneration.from_pretrained(
        nombre_modelo, attn_implementation="eager"
    ).to(device)
    model.eval()
    return processor, model, device


# ---------------------------------------------------------------------------
# 2. Funciones de apoyo
# ---------------------------------------------------------------------------
def cargar_audio(datos: bytes) -> np.ndarray:
    """Lee un audio (bytes) y lo devuelve como vector numpy a 16 kHz, mono."""
    audio, _ = librosa.load(io.BytesIO(datos), sr=SR, mono=True)
    return audio.astype(np.float32)


def transcribir(audio, processor, model, device):
    """Ejecuta la inferencia: audio -> ids de tokens -> texto. Devuelve también el tiempo."""
    entradas = processor(audio, sampling_rate=SR, return_tensors="pt").to(device)
    max_tokens = max(10, int(len(audio) / SR * TOKENS_POR_SEGUNDO))
    inicio = time.time()
    with torch.no_grad():
        ids = model.generate(**entradas, max_length=max_tokens)
    if device == "cuda":
        torch.cuda.synchronize()
    tiempo = time.time() - inicio
    texto = processor.decode(ids[0], skip_special_tokens=True)
    return texto, ids, entradas, tiempo


def analizar(entradas, ids, model):
    """Pasada completa (teacher forcing) para obtener atenciones y tensores Q, K, V
    de la primera capa del encoder y del decoder mediante hooks."""
    capturas, ganchos = {}, []

    def crear_gancho(nombre):
        def gancho(_modulo, _entrada, salida):
            capturas[nombre] = tuple(salida.shape)
        return gancho

    for nombre, modulo in model.named_modules():
        if ".layers.0." in nombre and nombre.split(".")[-1] in ("q_proj", "k_proj", "v_proj"):
            ganchos.append(modulo.register_forward_hook(crear_gancho(nombre)))

    with torch.no_grad():
        salida = model(**entradas, decoder_input_ids=ids,
                       output_attentions=True, use_cache=False)
    for g in ganchos:
        g.remove()
    return salida, capturas


def tipo_atencion(nombre_modulo: str):
    """Clasifica el módulo y explica de dónde vienen sus entradas."""
    tensor = nombre_modulo.split(".")[-1][0].upper()  # q_proj -> Q
    if "encoder_attn" in nombre_modulo:
        origen = "Decoder (tokens)" if tensor == "Q" else "Encoder (audio)"
        return "Cross-attention", tensor, origen
    if ".decoder." in nombre_modulo:
        return "Self-attention enmascarada (decoder)", tensor, "Decoder (tokens)"
    return "Self-attention (encoder)", tensor, "Encoder (audio)"


def limpiar_token(tok: str) -> str:
    return tok.replace("▁", " ").strip() or tok


def normalizar(texto: str) -> str:
    """Normaliza texto para calcular el WER (minúsculas, sin puntuación, abreviaturas)."""
    texto = re.sub(r"[^a-z0-9' ]", " ", texto.lower())
    for patron, completo in {r"\bmr\b": "mister", r"\bmrs\b": "missus",
                             r"\bdr\b": "doctor", r"\bst\b": "saint"}.items():
        texto = re.sub(patron, completo, texto)
    return re.sub(r"\s+", " ", texto).strip()


def mapa_calor(matriz, titulo, xlabel, ylabel, etiquetas_y=None, etiquetas_x=None,
               extent=None, cmap="magma", alto=4.5):
    fig, ax = plt.subplots(figsize=(10, alto))
    im = ax.imshow(matriz, aspect="auto", cmap=cmap, extent=extent)
    if etiquetas_y is not None:
        ax.set_yticks(range(len(etiquetas_y)))
        ax.set_yticklabels(etiquetas_y, fontsize=8)
    if etiquetas_x is not None:
        ax.set_xticks(range(len(etiquetas_x)))
        ax.set_xticklabels(etiquetas_x, rotation=90, fontsize=8)
    ax.set_title(titulo)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, label="Peso de atención")
    fig.tight_layout()
    return fig


def seleccionar(atencion_capa, cabeza):
    """atencion_capa: (batch, cabezas, consultas, claves). Devuelve promedio o una cabeza."""
    m = atencion_capa[0]
    return (m.mean(dim=0) if cabeza == "Promedio" else m[int(cabeza)]).cpu().numpy()


# ---------------------------------------------------------------------------
# 3. Interfaz
# ---------------------------------------------------------------------------
st.title("🎙️ Moonshine: reconocimiento de voz con un Transformer encoder–decoder")
st.caption("Proyecto final · Procesamiento de Datos Secuenciales con Deep Learning · Heidy Romero Ayos")

with st.sidebar:
    st.header("Configuración")
    nombre_modelo = st.selectbox(
        "Modelo preentrenado",
        ["UsefulSensors/moonshine-base", "UsefulSensors/moonshine-tiny"],
        help="'tiny' es más pequeño y rápido; 'base' es más preciso.",
    )
    referencia = st.text_area(
        "Transcripción de referencia (opcional)",
        help="Si escribes lo que realmente se dice en el audio, la app calcula el WER.",
    )
    st.markdown("---")
    st.markdown(
        "**Artículo base:** [Moonshine (arXiv:2410.15608)](https://arxiv.org/abs/2410.15608)  \n"
        "**Pesos:** [Hugging Face](https://huggingface.co/UsefulSensors/moonshine-base)"
    )
    st.info("El modelo transcribe audio en **inglés**. Se recomiendan audios de menos de 30 segundos.")

processor, model, device = cargar_modelo(nombre_modelo)
st.sidebar.success(f"Modelo cargado en: **{device.upper()}**")

# --- Entrada de datos: subir archivo o grabar con el micrófono ---
st.subheader("1. Cargar el audio")
col_subir, col_grabar = st.columns(2)
with col_subir:
    archivo = st.file_uploader("Sube un archivo de audio", type=["wav", "mp3", "flac", "ogg", "m4a"])
with col_grabar:
    grabacion = st.audio_input("…o graba tu voz") if hasattr(st, "audio_input") else None

fuente = grabacion if grabacion is not None else archivo
if fuente is None:
    st.info("Sube o graba un audio para comenzar.")
    st.stop()

try:
    audio = cargar_audio(fuente.getvalue())
except Exception as e:
    st.error(f"No se pudo leer el audio ({e}). Prueba con un archivo .wav o .mp3.")
    st.stop()

duracion = len(audio) / SR
if duracion < 0.2:
    st.warning("El audio es demasiado corto.")
    st.stop()
if duracion > 30:
    st.warning(f"El audio dura {duracion:.1f} s; con audios largos el modelo puede perder precisión.")

st.audio(fuente.getvalue())
fig, ax = plt.subplots(figsize=(10, 2))
ax.plot(np.arange(len(audio)) / SR, audio, linewidth=0.5)
ax.set_xlabel("Tiempo (s)")
ax.set_ylabel("Amplitud")
ax.set_title(f"Forma de onda · {duracion:.2f} s · {len(audio)} muestras a 16 kHz")
fig.tight_layout()
st.pyplot(fig)

# --- Inferencia ---
st.subheader("2. Transcripción")
# Streamlit vuelve a ejecutar el script cada vez que se mueve un control. Para no repetir
# la inferencia, guardamos el resultado en session_state según el audio y el modelo.
clave = (nombre_modelo, hash(fuente.getvalue()))
if st.session_state.get("clave") != clave:
    with st.spinner("Transcribiendo..."):
        if st.session_state.get("calentado") != nombre_modelo:   # la primera inferencia es más lenta
            transcribir(audio[: SR], processor, model, device)
            st.session_state["calentado"] = nombre_modelo
        texto, ids, entradas, tiempo = transcribir(audio, processor, model, device)
    with st.spinner("Calculando atenciones y tensores Q, K, V..."):
        salida, capturas = analizar(entradas, ids, model)
    st.session_state.update(clave=clave, resultado=(texto, ids, entradas, tiempo, salida, capturas))

texto, ids, entradas, tiempo, salida, capturas = st.session_state["resultado"]

st.success(texto if texto else "(sin texto)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Duración del audio", f"{duracion:.2f} s")
c2.metric("Tiempo de inferencia", f"{tiempo:.3f} s")
c3.metric("RTF", f"{tiempo / duracion:.3f}", help="Tiempo de inferencia ÷ duración. Menor que 1 = más rápido que el tiempo real.")
c4.metric("Tokens generados", ids.shape[1])

if referencia.strip():
    import jiwer
    wer = jiwer.wer(normalizar(referencia), normalizar(texto))
    st.metric("WER (Word Error Rate)", f"{wer * 100:.2f} %")

tokens = processor.tokenizer.convert_ids_to_tokens(ids[0])
etiquetas = [limpiar_token(t) for t in tokens]
with st.expander("Ver la salida como tokens (antes de convertirla en texto)"):
    st.write(pd.DataFrame({"id": ids[0].tolist(), "token": tokens}).T)

# --- Análisis interno ---
st.subheader("3. ¿Qué pasa dentro del modelo?")

n_audio = salida.encoder_last_hidden_state.shape[1]
n_capas_dec = len(salida.cross_attentions)
n_cabezas = salida.cross_attentions[0].shape[1]
opciones_cabeza = ["Promedio"] + [str(h) for h in range(n_cabezas)]

tab_cross, tab_dec, tab_enc, tab_qkv, tab_arq = st.tabs(
    ["🔗 Cross-attention", "✍️ Decoder (enmascarada)", "🎧 Encoder", "🧮 Q, K, V", "🏗️ Arquitectura"]
)

with tab_cross:
    st.markdown(
        "Cada fila es un token generado (**Query**, viene del decoder) y cada columna un instante "
        "del audio (**Keys** y **Values**, vienen del encoder). La diagonal muestra el "
        "**alineamiento** entre lo que se dice y lo que se escribe."
    )
    col_a, col_b = st.columns(2)
    capa = col_a.slider("Capa del decoder", 1, n_capas_dec, n_capas_dec, key="capa_cross")
    cabeza = col_b.selectbox("Cabeza de atención", opciones_cabeza, key="cab_cross")
    matriz = seleccionar(salida.cross_attentions[capa - 1], cabeza)
    st.pyplot(mapa_calor(matriz, f"Cross-attention · capa {capa} · cabeza {cabeza}",
                         "Tiempo del audio (s)", "Token generado", etiquetas_y=etiquetas,
                         extent=[0, duracion, len(etiquetas) - 0.5, -0.5],
                         alto=max(4, 0.25 * len(etiquetas))))

with tab_dec:
    st.markdown(
        "Self-attention del decoder con **máscara causal**: cada token solo puede mirar los tokens "
        "anteriores. Por eso la matriz es **triangular inferior**."
    )
    col_a, col_b = st.columns(2)
    capa = col_a.slider("Capa del decoder", 1, n_capas_dec, n_capas_dec, key="capa_dec")
    cabeza = col_b.selectbox("Cabeza de atención", opciones_cabeza, key="cab_dec")
    matriz = seleccionar(salida.decoder_attentions[capa - 1], cabeza)
    st.pyplot(mapa_calor(matriz, f"Decoder: self-attention enmascarada · capa {capa}",
                         "Token al que presta atención (Key)", "Token que consulta (Query)",
                         etiquetas_y=etiquetas, etiquetas_x=etiquetas, cmap="Blues",
                         alto=max(4, 0.25 * len(etiquetas))))

with tab_enc:
    n_capas_enc = len(salida.encoder_attentions)
    st.markdown(
        f"Self-attention del encoder: cada uno de los **{n_audio} vectores de audio** "
        f"(~{n_audio / duracion:.0f} por segundo) mira a todos los demás. Con **RoPE**, la atención "
        "depende de la distancia relativa en el tiempo."
    )
    col_a, col_b = st.columns(2)
    capa = col_a.slider("Capa del encoder", 1, n_capas_enc, 1, key="capa_enc")
    cabeza = col_b.selectbox("Cabeza de atención", opciones_cabeza, key="cab_enc")
    matriz = seleccionar(salida.encoder_attentions[capa - 1], cabeza)
    st.pyplot(mapa_calor(matriz, f"Encoder: self-attention · capa {capa}",
                         "Tiempo (s) — Key", "Tiempo (s) — Query",
                         extent=[0, duracion, duracion, 0], cmap="viridis", alto=5))

with tab_qkv:
    st.markdown(
        r"En cada capa de atención: $Q = XW_Q$, $K = XW_K$, $V = XW_V$ (capas `q_proj`, `k_proj`, "
        r"`v_proj`), y luego $\text{softmax}\left(QK^T/\sqrt{d_k}\right)V$. "
        "Formas capturadas en la **primera capa**, como `(batch, longitud, dimensión)`:"
    )
    filas = []
    for nombre, forma in capturas.items():
        tipo, tensor, origen = tipo_atencion(nombre)
        filas.append({"Tipo de atención": tipo, "Tensor": tensor, "Viene de": origen,
                      "Forma": str(forma), "Módulo": nombre})
    st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)
    st.markdown(
        f"En la **cross-attention**, Q tiene la longitud de los **tokens** ({ids.shape[1]}), "
        f"mientras que K y V tienen la longitud del **audio** ({n_audio}): cada token pregunta "
        "qué parte del audio le sirve para decidir la siguiente palabra."
    )

with tab_arq:
    cfg = model.config
    datos = {k: getattr(cfg, k) for k in [
        "hidden_size", "intermediate_size", "encoder_num_hidden_layers",
        "decoder_num_hidden_layers", "encoder_num_attention_heads",
        "decoder_num_attention_heads", "vocab_size", "partial_rotary_factor",
        "rope_theta"] if hasattr(cfg, k)}
    datos["parámetros totales"] = f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f} M"
    st.table(pd.DataFrame(datos.items(), columns=["Hiperparámetro", "Valor"]).astype(str))
    st.markdown(
        "**Flujo:** audio crudo (16 kHz) → **stem convolucional** (3 convoluciones) → "
        "**encoder** (self-attention + RoPE) → **decoder** (self-attention enmascarada + "
        "cross-attention + RoPE) → capa lineal + softmax → siguiente token (autorregresivo)."
    )
    with st.expander("Ver la estructura completa del modelo"):
        st.code(str(model))
