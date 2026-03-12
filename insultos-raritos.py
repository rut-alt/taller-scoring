import streamlit as st
import random

# --- Listas de palabras ---
adjetivos = [
    "desorientado", "extraño", "confundido", "torpe",
    "insoportable", "ridículo", "fantasmagórico", "peculiar"
]

sustantivos = [
    "patán", "gnomo", "murciélago", "troll",
    "cactus", "pingüino", "ornitorrinco", "caballo de peluche"
]

frases_finales = [
    "que no encuentra sus calcetines.",
    "con más ideas que sentido común.",
    "que habla con los electrodomésticos.",
    "que confunde la sal con el azúcar.",
    "que baila sin música."
]

# --- Función para generar insulto ---
def generar_insulto():
    adj = random.choice(adjetivos)
    sust = random.choice(sustantivos)
    frase = random.choice(frases_finales)
    return f"¡Eres un {adj} {sust} {frase}"

# --- Streamlit ---
st.title("🦄 Generador de Insultos Raros")
st.write("Haz clic en el botón para generar un insulto absurdo y divertido:")

if st.button("Generar insulto"):
    insulto = generar_insulto()
    st.success(insulto)
