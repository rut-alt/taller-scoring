# app.py
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Dict

import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class CategoryResult:
    j: int
    x: float
    contribution_pct: float
    delta_from_prev_pct: float


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def normalize_list_len(values: List[float], n: int, fill: float = 0.0) -> List[float]:
    vals = list(values or [])
    if len(vals) < n:
        vals += [fill] * (n - len(vals))
    if len(vals) > n:
        vals = vals[:n]
    return vals


def gaps_to_x(k: int, gaps: List[float], cap_mode: str = "clip") -> Dict:
    """
    Construye x(j) con x(k)=1 y gaps (penalizaciones) no negativas entre categorías.
    gaps: lista de longitud k-1, interpretada como:
      gap_t = caída al pasar de (t+1) -> t (de mejor a peor), acumulativa.
    Entonces:
      x_k = 1
      x_{k-1} = 1 - gap_{k-1}
      x_{k-2} = 1 - gap_{k-1} - gap_{k-2}
      ...
      x_1 = 1 - sum(gaps)

    Restricción deseada: sum(gaps) <= 1 (saldo).
    cap_mode:
      - "clip": si suma>1, recorta el último gap para que suma=1
      - "scale": si suma>1, reescala todos los gaps para que suma=1
    """
    if k < 2:
        raise ValueError("k debe ser >= 2.")

    gaps = normalize_list_len(gaps, k - 1, fill=0.0)
    gaps = [max(0.0, float(g)) for g in gaps]
    s = sum(gaps)

    if s > 1.0 + 1e-12:
        if cap_mode == "scale":
            gaps = [g / s for g in gaps]  # ahora suma 1
        else:
            # clip: recorta el último gap para ajustar saldo
            excess = s - 1.0
            gaps[-1] = max(0.0, gaps[-1] - excess)

    s_eff = sum(gaps)
    remaining = max(0.0, 1.0 - s_eff)

    # Construir x desde arriba: x_k=1
    xs = [0.0] * k
    xs[-1] = 1.0
    acc = 0.0
    for idx in range(k - 2, -1, -1):
        acc += gaps[idx]
        xs[idx] = clamp01(1.0 - acc)

    xs[-1] = 1.0

    return {"x_values": xs, "gaps_eff": gaps, "sum_gaps": s_eff, "remaining": remaining}


def generate_scale_fixed_weight(
    peso_pct: float,
    x_values: List[float],
) -> Dict:
    k = len(x_values)
    xs = [clamp01(x) for x in x_values]

    results: List[CategoryResult] = []
    prev = 0.0
    for j in range(1, k + 1):
        x = xs[j - 1]
        contrib = float(peso_pct) * x
        delta = contrib - prev if j > 1 else 0.0
        results.append(
            CategoryResult(
                j=j,
                x=round(x, 6),
                contribution_pct=round(contrib, 4),
                delta_from_prev_pct=round(delta, 4),
            )
        )
        prev = contrib

    return {"peso_pct": float(peso_pct), "k": k, "categories": results, "x_values": xs}


def scale_to_df(scale: dict, labels: List[str]) -> pd.DataFrame:
    rows = []
    for idx, r in enumerate(scale["categories"]):
        rows.append(
            {
                "K (j)": r.j,
                "Etiqueta (texto libre)": labels[idx] if idx < len(labels) else "",
                "x(j)": r.x,
                "Suma al score total % (peso*x)": r.contribution_pct,
                "Δ vs prev %": r.delta_from_prev_pct,
            }
        )
    return pd.DataFrame(rows)


# =========================
# Streamlit App
# =========================

st.set_page_config(page_title="Taller scoring por tarjetas", layout="wide")
st.title("Taller scoring — pesos fijos + 'saldo' para repartir (mejor=1)")


def init_model():
    raw_pct = [
        ("Antigüedad 1ª contratación", 7.5),
        ("Vinculación: Nº de Ramos con nosotros", 7.5),
        ("Rentabilidad de la póliza actual", 7.5),
        ("Descuentos o Recargos aplicados sobre tarifa", 5.5),
        ("Morosidad: Históricos sin incidencia en devolución (Anotaciones de póliza)", 5.0),
        ("Engagement comercial / Uso de canales propios (App / Área cliente / Web privada)", 4.5),
        ("Frecuencia uso de coberturas complementarias que no emiten siniestralidad.", 4.5),
        ("Total de asegurados en el Total de sus pólizas - Media de asegurados por póliza", 4.5),
        ("Edad", 4.5),
        ("Rentabilidad de la póliza retrospectivo/histórica (LTV)", 4.5),
        ("Tipo de distribución", 4.5),
        ("Vinculación: Coberturas complementarias opcionales", 4.5),
        ("Contactabilidad: Más de X Campos de datos (Tiene App, Teléfono, Mail, etc.).", 4.0),
        ("Edad del asegurado más mayor", 4.0),
        ("Vinculación familiar", 3.0),
        ("Prescriptor", 3.0),
        ("Exposición a comunicaciones de marca (RRSS, mailing…)", 3.0),
        ("Descendencia", 3.0),
        ("Medio de pago", 2.5),
        ("Frecuencia de pago (Periodicidad)", 2.0),
        ("Probabilidad de desglose", 1.5),
        ("Tipo de producto", 1.5),
        ("NPS", 1.5),
        ("Mascotas", 1.5),
        ("Localización (enfocado a potencial de compra)", 1.5),
        ("Autónomo", 1.0),
        ("Siniestralidad (Salud)", 1.0),
        ("Grado de digitalización de la póliza", 0.5),
        ("Profesión", 0.5),
        ("Nivel educativo", 0.5),
        ("Sexo", 0.0),
    ]

    variables = []
    for idx, (name, peso_pct) in enumerate(raw_pct, start=1):
        preset_labels = ["", "", ""]
        preset_gaps = [0.5, 0.5]  # suma 1 → x1=0, x2=0.5, x3=1

        if "Nº de Ramos con nosotros" in name:
            preset_labels = ["0 ramos", "1-2 ramos", "3 o más ramos"]
            preset_gaps = [0.4, 0.6]  # x=[0,0.6,1]

        variables.append(
            {
                "id": f"var_{idx:02d}",
                "name": name,
                "peso_pct": float(peso_pct),  # fijo
                "k": 3,
                "labels": preset_labels,
                "gaps": preset_gaps,
                "notes": "",
            }
        )

    return {
        "variables": variables,
        "settings": {
            "cap_mode": "clip",
        },
    }


if "model" not in st.session_state:
    st.session_state.model = init_model()


def normalize_labels(var: dict):
    k = int(var["k"])
    labels = normalize_list_len(var.get("labels") or [], k, fill="")
    var["labels"] = labels


def normalize_gaps(var: dict):
    k = int(var["k"])
    gaps = normalize_list_len(var.get("gaps") or [], k - 1, fill=0.0)
    var["gaps"] = [max(0.0, float(g)) for g in gaps]


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.image("LOGOTIPO-AES-05.png", use_container_width=True)

    st.subheader("Controles globales")

    st.session_state.model["settings"]["cap_mode"] = st.selectbox(
        "Si te pasas del saldo (Σ gaps > 1):",
        options=["clip", "scale"],
        index=0 if st.session_state.model["settings"].get("cap_mode", "clip") == "clip" else 1,
        help=(
            "clip = recorta el último gap para que Σ=1.\n"
            "scale = reescala todos los gaps proporcionalmente para que Σ=1."
        ),
    )

    st.divider()
    total_pesos = sum(float(v.get("peso_pct", 0.0)) for v in st.session_state.model["variables"])
    st.metric("Suma pesos fijos", f"{total_pesos:.2f}%")

    st.caption("Aquí solo tocáis la escala interna (x), no los pesos.")

    st.divider()

    st.caption("Exporta el estado del taller (k, gaps, etiquetas, notas).")
    st.download_button(
        "⬇️ Descargar JSON del taller",
        data=pd.Series(st.session_state.model).to_json(force_ascii=False, indent=2).encode("utf-8"),
        file_name="taller_scoring.json",
        mime="application/json",
        use_container_width=True,
    )


# =========================
# Tarjetas
# =========================

vars_list = st.session_state.model["variables"]

col1, col2 = st.columns(2, gap="large")
cols = [col1, col2]

cap_mode = st.session_state.model["settings"].get("cap_mode", "clip")

for i, var in enumerate(vars_list):
    normalize_labels(var)
    normalize_gaps(var)

    with cols[i % 2]:
        with st.container(border=True):
            st.subheader(var["name"])
            peso_pct = float(var.get("peso_pct", 0.0))
            st.metric("Peso fijo (%)", f"{peso_pct:.2f}")

            var["k"] = int(
                st.number_input(
                    "k (nº de subcategorías)",
                    min_value=2,
                    max_value=10,
                    value=int(var["k"]),
                    step=1,
                    key=f"k_{var['id']}",
                )
            )
            normalize_labels(var)
            normalize_gaps(var)

            st.markdown("**Etiquetas por categoría (texto libre)**")
            left, right = st.columns(2)
            for j in range(1, int(var["k"]) + 1):
                target = left if j % 2 == 1 else right
                var["labels"][j - 1] = target.text_input(
                    f"K = {j}",
                    value=var["labels"][j - 1],
                    key=f"lbl_{var['id']}_{j}",
                )

            st.markdown("**Reparto del saldo (gaps) — x(mejor)=1 fijo**")
            st.caption("Los gaps son las caídas entre categorías al bajar de nivel. Σ gaps ≤ 1.")

            # Editas k-1 gaps
            lg, rg = st.columns(2)
            for t in range(1, int(var["k"])):  # 1..k-1
                target = lg if t % 2 == 1 else rg
                var["gaps"][t - 1] = target.number_input(
                    f"gap {t} (caída entre K={t} y K={t+1})",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(var["gaps"][t - 1]),
                    step=0.01,
                    key=f"gap_{var['id']}_{t}",
                )

            # >>> AVISOS <<<
            raw_gaps = [float(g) for g in (var.get("gaps") or [])]
            sum_raw = sum(raw_gaps)

            if any(g > 1.0 for g in raw_gaps):
                st.error(
                    "⚠️ Hay algún gap > 1. Recuerda: los gaps se introducen en escala 0–1. "
                    "Ejemplo: 0.40 = 40% del saldo."
                )

            if sum_raw > 1.0 + 1e-12:
                st.warning(
                    f"⚠️ Te has pasado de saldo: Σ gaps = {sum_raw:.3f} (> 1). "
                    f"Se aplicará el modo '{cap_mode}' para ajustarlo."
                )
            # <<< AVISOS <<<

            # Convertimos gaps -> x
            conv = gaps_to_x(k=int(var["k"]), gaps=var["gaps"], cap_mode=cap_mode)
            xs = conv["x_values"]
            var["gaps"] = conv["gaps_eff"]  # por si hubo clip/scale
                        # === Aviso de discriminación de la variable ===
            eps = 1e-6
            x_min = min(xs)
            discr_range = 1.0 - x_min  # porque x_max = 1

            # Clasificación de la capacidad discriminatoria
            if discr_range >= 0.80:
                discr_label = "muy discriminatoria"
            elif discr_range >= 0.50:
                discr_label = "medianamente discriminatoria"
            else:
                discr_label = "poco discriminatoria"

            if conv["sum_gaps"] < 1.0 - eps:
                st.warning(
                    "🟡 **Variable menos discriminatoria (Σ gaps < 1)**\n\n"
                    f"- **Σ gaps usado** = {conv['sum_gaps']:.3f} → **saldo sin usar** = {conv['remaining']:.3f}\n"
                    f"- **Rango real de discriminación** Δx = x_max − x_min = 1 − {x_min:.3f} = **{discr_range:.3f}**\n"
                    f"- Interpretación: la variable es **{discr_label}** (cuanto menor es Δx, menos separa perfiles).\n\n"
                    "📐 **Nota:** en este esquema **Δx = Σ gaps**, porque **x(mejor)=1** y **x(peor)=1−Σ gaps**.\n\n"
                    "**Recomendación habitual en scoring:** usar **Σ gaps = 1** para que la peor categoría sea **x=0** "
                    "y la variable utilice toda la escala **0–1**."
                )

            elif abs(conv["sum_gaps"] - 1.0) <= eps:
                st.success(
                    "🟢 **Escala completa (Σ gaps = 1)**\n\n"
                    f"- **Rango de discriminación** Δx = **{discr_range:.3f}** (máximo).\n"
                    "- La peor categoría llega a **x=0** y la mejor queda en **x=1**.\n\n"
                    "📐 **Nota:** en este esquema **Δx = Σ gaps**."
                )

            st.info(f"Saldo usado: {conv['sum_gaps']:.3f} | Saldo restante: {conv['remaining']:.3f} | x(mejor)=1")

            var["notes"] = st.text_area(
                "Notas / criterio (opcional)",
                value=var.get("notes", ""),
                key=f"notes_{var['id']}",
            )

            scale = generate_scale_fixed_weight(peso_pct=peso_pct, x_values=xs)
            df = scale_to_df(scale, var["labels"])

            st.caption(f"Máximo de la variable (mejor): {peso_pct:.2f}% (porque x=1)")
            st.dataframe(df, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Resumen del modelo")
summary = []
for v in st.session_state.model["variables"]:
    k = int(v["k"])
    conv = gaps_to_x(k=k, gaps=v.get("gaps") or [], cap_mode=cap_mode)
    xs = conv["x_values"]
    summary.append(
        {
            "Variable": v["name"],
            "Peso %": round(float(v.get("peso_pct", 0.0)), 2),
            "k": k,
            "Σ gaps": round(conv["sum_gaps"], 3),
            "x (preview)": " | ".join([f"{x:.2f}" for x in xs[:3]]) + (" ..." if len(xs) > 3 else ""),
        }
    )

            # === Aviso de discriminación (Σ gaps != 1) ===
            eps = 1e-6
            x_min = min(xs)
            discr_range = 1.0 - x_min  # como x_max=1 fijo

            # Clasificación simple (ajusta umbrales si quieres)
            if discr_range >= 0.80:
                discr_label = "muy discriminatoria"
            elif discr_range >= 0.50:
                discr_label = "medianamente discriminatoria"
            else:
                discr_label = "poco discriminatoria"

            # Si no se usa todo el saldo (Σ gaps < 1) => pierde discriminación
            if conv["sum_gaps"] < 1.0 - eps:
                st.warning(
                    "🟡 **Variable menos discriminatoria (Σ gaps < 1)**\n\n"
                    f"- **Σ gaps (usado)** = {conv['sum_gaps']:.3f} → **saldo sin usar** = {conv['remaining']:.3f}\n"
                    f"- **Rango real de discriminación** Δx = x_max − x_min = 1 − {x_min:.3f} = **{discr_range:.3f}**\n"
                    f"- Interpretación: la variable es **{discr_label}** (cuanto menor es Δx, menos separa perfiles).\n\n"
                    "👉 **No recomendado** en scoring si quieres máxima separación: lo habitual es ajustar a **Σ gaps = 1** "
                    "para que la peor categoría llegue a **x=0** y la variable use toda la escala 0–1."
                )

            # Si se pasa de saldo (Σ gaps > 1) ya lo avisas antes, pero aquí puedes reforzarlo si quieres
            elif conv["sum_gaps"] > 1.0 + eps:
                st.warning(
                    "⚠️ **Σ gaps > 1**: se ha ajustado con el modo seleccionado "
                    f"('{cap_mode}'). Rango final Δx = **{discr_range:.3f}**."
                )

            # Si está perfecto (Σ gaps == 1)
            else:
                st.success(
                    "🟢 **Escala completa (Σ gaps = 1)**\n\n"
                    f"- **Rango de discriminación** Δx = **{discr_range:.3f}** (máximo)\n"
                    "- La peor categoría llega a **x=0** y la mejor queda en **x=1**."
                )
            # === Fin aviso discriminación ===
st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
