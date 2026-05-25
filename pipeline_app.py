"""
ML Pipeline App — Streamlit
Funciones: Exploración de datos, Normalización, Encoding, PCA, Regresión Logística, KNN
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import (
    LabelEncoder, StandardScaler, MinMaxScaler,
    RobustScaler, OrdinalEncoder
)
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Pipeline Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# ESTILOS CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Space Mono', monospace !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
        border-right: 1px solid #2a2a4a;
    }
    [data-testid="stSidebar"] * { color: #e0e0ff !important; }

    /* Tarjetas de métricas */
    .metric-card {
        background: linear-gradient(135deg, #1e1e3a 0%, #252545 100%);
        border: 1px solid #3a3a6a;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-card .label {
        font-size: 0.75rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #8888cc;
        margin-bottom: 0.3rem;
    }
    .metric-card .value {
        font-family: 'Space Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        color: #a78bfa;
    }

    /* Sección header */
    .section-header {
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
        border-radius: 10px;
        padding: 0.7rem 1.2rem;
        margin: 1.5rem 0 1rem;
        font-family: 'Space Mono', monospace;
        font-size: 1rem;
        font-weight: 700;
        color: white !important;
        letter-spacing: 0.03em;
    }

    /* Tabla de info */
    .info-table { border-collapse: collapse; width: 100%; }
    .info-table th {
        background: #2a2a4a;
        color: #a78bfa;
        padding: 8px 12px;
        text-align: left;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .info-table td {
        padding: 7px 12px;
        border-bottom: 1px solid #2a2a4a;
        font-size: 0.9rem;
    }
    .info-table tr:hover td { background: #1a1a3a; }

    /* Badges de tipo */
    .badge-num { background:#1d4ed8; color:white; padding:2px 8px; border-radius:20px; font-size:0.75rem; }
    .badge-cat { background:#7c3aed; color:white; padding:2px 8px; border-radius:20px; font-size:0.75rem; }
    .badge-bool { background:#0f766e; color:white; padding:2px 8px; border-radius:20px; font-size:0.75rem; }
    .badge-date { background:#b45309; color:white; padding:2px 8px; border-radius:20px; font-size:0.75rem; }

    /* Alerta info */
    .info-box {
        background: #1e2a4a;
        border-left: 4px solid #4f46e5;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #c0c0f0;
    }
    .success-box {
        background: #1a3a2a;
        border-left: 4px solid #10b981;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #a0f0c0;
    }

    /* Estilo general de tabs */
    button[data-baseweb="tab"] {
        font-family: 'Space Mono', monospace !important;
        font-size: 0.85rem !important;
    }

    div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def classify_type(dtype):
    if pd.api.types.is_numeric_dtype(dtype):
        return "Numérica", "badge-num"
    elif pd.api.types.is_bool_dtype(dtype):
        return "Booleana", "badge-bool"
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return "Fecha", "badge-date"
    else:
        return "Categórica", "badge-cat"


def metric_card(label, value):
    return f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
    </div>
    """


def section_header(icon, text):
    st.markdown(f'<div class="section-header">{icon} &nbsp; {text}</div>', unsafe_allow_html=True)


def plot_style():
    plt.rcParams.update({
        "figure.facecolor": "#0f0f1a",
        "axes.facecolor": "#1a1a2e",
        "axes.edgecolor": "#3a3a6a",
        "axes.labelcolor": "#c0c0f0",
        "xtick.color": "#8888cc",
        "ytick.color": "#8888cc",
        "text.color": "#e0e0ff",
        "grid.color": "#2a2a4a",
        "grid.linestyle": "--",
        "font.family": "monospace",
    })


# ─────────────────────────────────────────────────────────────────────────────
# ESTADO DE SESIÓN
# ─────────────────────────────────────────────────────────────────────────────
if "df_original" not in st.session_state:
    st.session_state.df_original = None
if "df_processed" not in st.session_state:
    st.session_state.df_processed = None
if "df_pca" not in st.session_state:
    st.session_state.df_pca = None
if "pca_model" not in st.session_state:
    st.session_state.pca_model = None
if "scaler_fitted" not in st.session_state:
    st.session_state.scaler_fitted = None


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — NAVEGACIÓN
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 ML Pipeline Studio")
    st.markdown("---")
    st.markdown("### Navegación")
    page = st.radio(
        "",
        [
            "📁 1. Carga y Exploración",
            "📐 2. Encoding Categórico",
            "⚙️ 3. Preprocesamiento",
            "🔭 4. Reducción PCA",
            "🤖 5. Modelos ML",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    if st.session_state.df_original is not None:
        df_orig = st.session_state.df_original
        st.markdown(f"**Dataset cargado**")
        st.markdown(f"🗂️ `{df_orig.shape[0]}` filas × `{df_orig.shape[1]}` cols")
    if st.session_state.df_processed is not None:
        st.markdown(f"✅ Preprocesado listo")
    if st.session_state.df_pca is not None:
        st.markdown(f"🔭 PCA listo")
    st.markdown("---")
    st.markdown("<small style='opacity:0.5'>ML Pipeline Studio v1.0<br>Powered by Streamlit + scikit-learn</small>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — CARGA Y EXPLORACIÓN
# ═════════════════════════════════════════════════════════════════════════════
if page == "📁 1. Carga y Exploración":
    st.title("📁 Carga y Exploración de Datos")

    uploaded = st.file_uploader("Sube tu archivo CSV", type=["csv"])

    if uploaded:
        sep = st.selectbox("Separador", [",", ";", "|", "\t"], index=0)
        enc = st.selectbox("Encoding", ["utf-8", "latin-1", "cp1252"], index=0)
        try:
            data = pd.read_csv(uploaded, sep=sep, encoding=enc)
            st.session_state.df_original = data
            st.session_state.df_processed = data.copy()
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")
            st.stop()

        st.markdown('<div class="success-box">✅ Archivo cargado correctamente</div>', unsafe_allow_html=True)

        # ── Dimensiones
        section_header("📏", "Dimensiones del Dataset")
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(metric_card("Filas", f"{data.shape[0]:,}"), unsafe_allow_html=True)
        with c2: st.markdown(metric_card("Columnas", f"{data.shape[1]}"), unsafe_allow_html=True)
        with c3: st.markdown(metric_card("Filas Duplicadas", f"{data.duplicated().sum()}"), unsafe_allow_html=True)
        with c4: st.markdown(metric_card("Celdas Nulas", f"{data.isnull().sum().sum():,}"), unsafe_allow_html=True)

        # ── Estructura de columnas
        section_header("🗃️", "Estructura de Variables")

        rows_html = ""
        for col in data.columns:
            dtype = data[col].dtype
            tipo_text, badge_class = classify_type(dtype)
            nulos = data[col].isnull().sum()
            pct_nulos = (nulos / len(data) * 100)
            unicos = data[col].nunique()
            rows_html += f"""
            <tr>
                <td><b>{col}</b></td>
                <td><code>{dtype}</code></td>
                <td><span class="{badge_class}">{tipo_text}</span></td>
                <td>{nulos} <small style="color:#8888cc">({pct_nulos:.1f}%)</small></td>
                <td>{unicos}</td>
            </tr>
            """
        st.markdown(f"""
        <table class="info-table">
            <thead><tr>
                <th>Columna</th><th>Dtype</th><th>Tipo</th>
                <th>Nulos (%)</th><th>Únicos</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)

        # ── Nulos
        section_header("🕳️", "Mapa de Nulos por Columna")
        nulos_df = pd.DataFrame({
            "Columna": data.columns,
            "Nulos": data.isnull().sum().values,
            "% Nulos": (data.isnull().sum().values / len(data) * 100).round(2),
        })
        nulos_df = nulos_df[nulos_df["Nulos"] > 0].sort_values("% Nulos", ascending=False)

        if nulos_df.empty:
            st.markdown('<div class="success-box">✅ No hay valores nulos en el dataset</div>', unsafe_allow_html=True)
        else:
            plot_style()
            fig, ax = plt.subplots(figsize=(10, max(3, len(nulos_df) * 0.45)))
            bars = ax.barh(nulos_df["Columna"], nulos_df["% Nulos"],
                           color=["#7c3aed" if v < 30 else "#dc2626" for v in nulos_df["% Nulos"]])
            ax.set_xlabel("% Nulos")
            ax.axvline(30, color="#f59e0b", ls="--", lw=1.2, label="30% umbral")
            ax.legend()
            ax.set_title("Porcentaje de Nulos por Columna", pad=12)
            fig.tight_layout()
            st.pyplot(fig)
            st.dataframe(nulos_df, use_container_width=True)

        # ── Estadísticas
        section_header("📊", "Resumen Estadístico")
        desc = data.describe().round(4)
        st.dataframe(desc, use_container_width=True)

        # ── Vista previa
        section_header("👀", "Vista Previa del Dataset")
        n_rows = st.slider("Filas a mostrar", 5, 50, 10)
        st.dataframe(data.head(n_rows), use_container_width=True)

        # ── Distribuciones
        section_header("📈", "Distribuciones")
        num_cols = data.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            sel_cols = st.multiselect("Selecciona columnas numéricas", num_cols, default=num_cols[:min(4, len(num_cols))])
            if sel_cols:
                plot_style()
                ncols_plot = min(3, len(sel_cols))
                nrows_plot = (len(sel_cols) + ncols_plot - 1) // ncols_plot
                fig, axes = plt.subplots(nrows_plot, ncols_plot,
                                         figsize=(5 * ncols_plot, 3.5 * nrows_plot))
                axes = np.array(axes).flatten()
                palette = ["#7c3aed", "#4f46e5", "#0ea5e9", "#10b981", "#f59e0b", "#ef4444"]
                for i, col in enumerate(sel_cols):
                    c = palette[i % len(palette)]
                    axes[i].hist(data[col].dropna(), bins=30, color=c, edgecolor="#0f0f1a", alpha=0.85)
                    axes[i].set_title(col, fontsize=10)
                    axes[i].set_xlabel("")
                for j in range(len(sel_cols), len(axes)):
                    axes[j].set_visible(False)
                fig.tight_layout()
                st.pyplot(fig)

    else:
        st.markdown('<div class="info-box">👆 Sube un archivo CSV para comenzar el análisis.</div>', unsafe_allow_html=True)
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Jupyter_logo.svg/207px-Jupyter_logo.svg.png",
                 width=60, caption="")


# ═════════════════════════════════════════════════════════════════════════════
# PÁGINA 3 — PREPROCESAMIENTO / NORMALIZACIÓN
# ═════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ 3. Preprocesamiento":
    st.title("⚙️ Preprocesamiento y Normalización")

    if st.session_state.df_original is None:
        st.warning("⚠️ Primero carga un dataset en la sección 1.")
        st.stop()

    data = st.session_state.df_processed.copy()
    num_cols = data.select_dtypes(include=np.number).columns.tolist()
    cat_cols = data.select_dtypes(exclude=np.number).columns.tolist()

    # ── Manejo de Nulos
    section_header("🕳️", "Manejo de Valores Nulos")
    if data.isnull().sum().sum() == 0:
        st.markdown('<div class="success-box">✅ No hay nulos — no se requiere imputación.</div>', unsafe_allow_html=True)
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Variables Numéricas**")
            num_imp = st.selectbox("Imputar numéricos con:", ["No imputar", "Media", "Mediana", "Cero"])
        with col2:
            st.markdown("**Variables Categóricas**")
            cat_imp = st.selectbox("Imputar categóricos con:", ["No imputar", "Moda", "Valor 'Unknown'"])

        if st.button("Aplicar imputación"):
            for c in num_cols:
                if data[c].isnull().sum() > 0:
                    if num_imp == "Media":
                        data[c] = data[c].fillna(data[c].mean())
                    elif num_imp == "Mediana":
                        data[c] = data[c].fillna(data[c].median())
                    elif num_imp == "Cero":
                        data[c] = data[c].fillna(0)
            for c in cat_cols:
                if data[c].isnull().sum() > 0:
                    if cat_imp == "Moda":
                        data[c] = data[c].fillna(data[c].mode()[0])
                    elif cat_imp == "Valor 'Unknown'":
                        data[c] = data[c].fillna("Unknown")
            st.session_state.df_processed = data
            st.success("✅ Imputación aplicada.")
            st.rerun()

    # ── Manejo de Duplicados
    section_header("🔁", "Manejo de Filas Duplicadas")
    dupes = data.duplicated().sum()
    st.markdown(f'<div class="info-box">🔁 Filas duplicadas encontradas: <b>{dupes}</b></div>', unsafe_allow_html=True)
    if dupes > 0:
        if st.button("Eliminar duplicados"):
            data = data.drop_duplicates()
            st.session_state.df_processed = data
            st.success(f"✅ {dupes} duplicados eliminados.")
            st.rerun()

    # ── Normalización
    section_header("📐", "Normalización de Variables Numéricas")

    if not num_cols:
        st.info("No hay columnas numéricas para normalizar.")
    else:
        st.markdown('<div class="info-box">Selecciona las variables que <b>NO</b> quieres normalizar (ej. variable objetivo, IDs, flags binarios). El resto se normalizará.</div>', unsafe_allow_html=True)

        cols_excluir = st.multiselect(
            "🚫 Variables a EXCLUIR de la normalización",
            num_cols,
            default=[],
            placeholder="Elige columnas que no deben normalizarse...",
        )
        cols_to_scale = [c for c in num_cols if c not in cols_excluir]

        col_prev1, col_prev2 = st.columns(2)
        with col_prev1:
            st.markdown(f"**✅ Se normalizarán ({len(cols_to_scale)}):**")
            if cols_to_scale:
                st.code("\n".join(cols_to_scale))
            else:
                st.info("Ninguna columna seleccionada.")
        with col_prev2:
            st.markdown(f"**🚫 Se mantendrán sin cambio ({len(cols_excluir)}):**")
            if cols_excluir:
                st.code("\n".join(cols_excluir))
            else:
                st.info("Ninguna excluida.")

        scaler_method = st.radio(
            "Método de normalización",
            ["StandardScaler (Z-score)", "MinMaxScaler (0–1)", "RobustScaler (mediana–IQR)"],
            horizontal=True,
        )

        st.markdown("""
        | Método | Fórmula | Cuándo usarlo |
        |--------|---------|---------------|
        | StandardScaler | `(x - μ) / σ` | Datos aprox. normales |
        | MinMaxScaler | `(x - min) / (max - min)` | Sin outliers extremos |
        | RobustScaler | `(x - mediana) / IQR` | Con outliers |
        """)

        if st.button("✅ Aplicar Normalización", type="primary"):
            if cols_to_scale:
                if "Standard" in scaler_method:
                    scaler = StandardScaler()
                elif "MinMax" in scaler_method:
                    scaler = MinMaxScaler()
                else:
                    scaler = RobustScaler()

                data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])
                st.session_state.df_processed = data
                st.session_state.scaler_fitted = scaler
                st.success(f"✅ Normalización aplicada a {len(cols_to_scale)} columnas. {len(cols_excluir)} columna(s) sin modificar.")

                # Comparar antes/después
                plot_style()
                preview_cols = cols_to_scale[:min(3, len(cols_to_scale))]
                orig = st.session_state.df_original[preview_cols]
                norm = data[preview_cols]

                fig, axes = plt.subplots(2, len(preview_cols),
                                         figsize=(5 * len(preview_cols), 6))
                if len(preview_cols) == 1:
                    axes = np.array(axes).reshape(2, 1)
                for i, col in enumerate(preview_cols):
                    axes[0, i].hist(orig[col].dropna(), bins=25, color="#4f46e5", edgecolor="#0f0f1a", alpha=0.85)
                    axes[0, i].set_title(f"{col}\n(original)", fontsize=9)
                    axes[1, i].hist(norm[col].dropna(), bins=25, color="#10b981", edgecolor="#0f0f1a", alpha=0.85)
                    axes[1, i].set_title(f"{col}\n(normalizado)", fontsize=9)
                fig.suptitle("Antes vs Después de la Normalización", y=1.02)
                fig.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("⚠️ Todas las columnas están excluidas. Selecciona al menos una para normalizar.")

    st.markdown("---")
    section_header("📋", "Dataset Actual")
    st.dataframe(data.head(20), use_container_width=True)
    st.markdown(f"**Shape:** `{data.shape[0]} × {data.shape[1]}`")


# ═════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — ENCODING CATEGÓRICO
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📐 2. Encoding Categórico":
    st.title("📐 Encoding de Variables Categóricas")

    if st.session_state.df_processed is None:
        st.warning("⚠️ Primero carga y preprocesa el dataset.")
        st.stop()

    data = st.session_state.df_processed.copy()
    cat_cols = data.select_dtypes(exclude=np.number).columns.tolist()

    if not cat_cols:
        st.markdown('<div class="success-box">✅ No hay variables categóricas en el dataset actual.</div>', unsafe_allow_html=True)
        st.stop()

    section_header("🏷️", "Variables Categóricas Detectadas")
    for col in cat_cols:
        uniq = data[col].nunique()
        vals = data[col].value_counts().head(5).index.tolist()
        st.markdown(f"**`{col}`** — {uniq} categorías únicas &nbsp;&nbsp; `{vals}`")

    st.markdown("---")
    section_header("🔧", "Configurar Encoding por Variable")

    encoding_config = {}
    ordinal_orders = {}

    for col in cat_cols:
        with st.expander(f"📌 {col}  ({data[col].nunique()} categorías)"):
            method = st.selectbox(
                f"Método para `{col}`",
                ["No transformar", "Label Encoding", "One-Hot Encoding", "Ordinal Encoding"],
                key=f"enc_{col}",
            )
            encoding_config[col] = method

            if method == "Ordinal Encoding":
                uniq_vals = data[col].dropna().unique().tolist()
                st.markdown("Define el orden (de menor a mayor):")
                order_input = st.text_area(
                    f"Orden para `{col}` (una categoría por línea)",
                    value="\n".join([str(v) for v in uniq_vals]),
                    key=f"ord_{col}",
                    height=120,
                )
                ordinal_orders[col] = [v.strip() for v in order_input.strip().split("\n") if v.strip()]

    if st.button("✅ Aplicar Encoding", type="primary"):
        df_enc = data.copy()
        applied = []

        for col, method in encoding_config.items():
            if method == "Label Encoding":
                le = LabelEncoder()
                df_enc[col] = le.fit_transform(df_enc[col].astype(str))
                applied.append(f"✔ `{col}` → Label Encoding")

            elif method == "One-Hot Encoding":
                dummies = pd.get_dummies(df_enc[col], prefix=col, drop_first=True, dtype=int)
                df_enc = pd.concat([df_enc.drop(columns=[col]), dummies], axis=1)
                applied.append(f"✔ `{col}` → One-Hot ({dummies.shape[1]} nuevas cols)")

            elif method == "Ordinal Encoding":
                order = ordinal_orders.get(col, [])
                if order:
                    # Mapeo manual: más robusto que OrdinalEncoder ante valores no listados
                    order_map = {str(v).strip(): i for i, v in enumerate(order)}
                    df_enc[col] = df_enc[col].astype(str).str.strip().map(order_map)
                    # Valores no mapeados quedan NaN → rellenar con -1
                    unmapped = df_enc[col].isna().sum()
                    if unmapped > 0:
                        df_enc[col] = df_enc[col].fillna(-1)
                        st.warning(f"⚠️ `{col}`: {unmapped} valor(es) no encontrado(s) en el orden definido → asignados -1.")
                    df_enc[col] = df_enc[col].astype(float)
                    applied.append(f"✔ `{col}` → Ordinal Encoding ({len(order)} niveles)")
                else:
                    st.warning(f"No se definió orden para `{col}`. Se omite.")

        st.session_state.df_processed = df_enc
        st.success("✅ Encoding aplicado correctamente")
        for msg in applied:
            st.markdown(msg)

        st.markdown(f"\n**Nuevo shape:** `{df_enc.shape[0]} × {df_enc.shape[1]}`")
        st.dataframe(df_enc.head(10), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PÁGINA 4 — PCA
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔭 4. Reducción PCA":
    st.title("🔭 Reducción de Dimensionalidad — PCA")

    if st.session_state.df_processed is None:
        st.warning("⚠️ Primero completa las etapas anteriores.")
        st.stop()

    data = st.session_state.df_processed.copy()
    num_cols = data.select_dtypes(include=np.number).columns.tolist()

    if len(num_cols) < 2:
        st.error("Se necesitan al menos 2 variables numéricas para PCA.")
        st.stop()

    section_header("📌", "Selección de Variables para PCA")
    sel_vars = st.multiselect(
        "Variables numéricas a incluir en PCA",
        num_cols,
        default=num_cols,
    )

    if len(sel_vars) < 2:
        st.info("Selecciona al menos 2 variables.")
        st.stop()

    vars_per_component = 5
    max_comp = len(sel_vars)
    st.markdown(f'<div class="info-box">ℹ️ Cada componente se forma con las <b>{vars_per_component}</b> variables de mayor varianza. Máximo posible: <b>{max_comp}</b> componentes.</div>', unsafe_allow_html=True)

    n_components = st.slider(
        "Número de componentes PCA",
        min_value=1,
        max_value=max_comp,
        value=min(5, max_comp),
    )

    if st.button("🚀 Ejecutar PCA", type="primary"):
        X = data[sel_vars].dropna()

        # Escalar si no se hizo antes
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA completo para seleccionar variables
        pca_full = PCA(n_components=len(sel_vars))
        pca_full.fit(X_scaled)

        # Ordenar variables por varianza total explicada
        loadings = np.abs(pca_full.components_).sum(axis=0)
        var_order = np.argsort(loadings)[::-1]  # mayor a menor
        ordered_vars = [sel_vars[i] for i in var_order]

        # PCA final con n componentes solicitados
        pca_model = PCA(n_components=n_components)
        X_pca = pca_model.fit_transform(X_scaled)

        # Crear dataset PCA
        pca_cols = [f"PC{i+1}" for i in range(n_components)]
        df_pca = pd.DataFrame(X_pca, columns=pca_cols, index=X.index)

        # Añadir columnas no numéricas si existen
        non_num = data.select_dtypes(exclude=np.number).columns.tolist()
        for c in non_num:
            df_pca[c] = data[c].values[:len(df_pca)]

        st.session_state.df_pca = df_pca
        st.session_state.pca_model = pca_model
        st.session_state.pca_vars_ordered = ordered_vars

        explained = pca_model.explained_variance_ratio_
        cumulative = np.cumsum(explained)

        # ── Resultados
        section_header("📊", "Varianza Explicada por Componente")

        plot_style()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        colors = ["#7c3aed", "#4f46e5", "#0ea5e9", "#10b981", "#f59e0b",
                  "#ef4444", "#ec4899", "#8b5cf6", "#06b6d4", "#84cc16"]
        bar_colors = [colors[i % len(colors)] for i in range(n_components)]

        ax1.bar(pca_cols, explained * 100, color=bar_colors, edgecolor="#0f0f1a", width=0.6)
        ax1.set_ylabel("Varianza Explicada (%)")
        ax1.set_title("Varianza por Componente")
        ax1.set_ylim(0, max(explained * 100) * 1.15)
        for i, v in enumerate(explained):
            ax1.text(i, v * 100 + 0.3, f"{v*100:.1f}%", ha="center", fontsize=9, color="#e0e0ff")

        ax2.plot(pca_cols, cumulative * 100, marker="o", color="#10b981",
                 linewidth=2.5, markersize=8)
        ax2.fill_between(range(n_components), cumulative * 100,
                         alpha=0.2, color="#10b981")
        ax2.axhline(90, color="#f59e0b", ls="--", lw=1.5, label="90%")
        ax2.axhline(95, color="#ef4444", ls="--", lw=1.5, label="95%")
        ax2.set_xticks(range(n_components))
        ax2.set_xticklabels(pca_cols)
        ax2.set_ylabel("Varianza Acumulada (%)")
        ax2.set_title("Varianza Acumulada")
        ax2.legend()
        ax2.set_ylim(0, 105)

        fig.tight_layout()
        st.pyplot(fig)

        # ── Tabla resumen
        summary_df = pd.DataFrame({
            "Componente": pca_cols,
            "Varianza Explicada (%)": (explained * 100).round(2),
            "Varianza Acumulada (%)": (cumulative * 100).round(2),
        })
        st.dataframe(summary_df, use_container_width=True)

        # ── Contribución de variables por componente
        section_header("📌", "Variables por Componente (Top 5)")
        loadings_df = pd.DataFrame(
            pca_model.components_.T,
            columns=pca_cols,
            index=sel_vars,
        )

        cols_show = st.columns(min(n_components, 4))
        for i in range(n_components):
            with cols_show[i % min(n_components, 4)]:
                top = loadings_df[f"PC{i+1}"].abs().sort_values(ascending=False).head(vars_per_component)
                st.markdown(f"**PC{i+1}**")
                for var, val in top.items():
                    st.markdown(f"- `{var}` : {val:.3f}")

        # ── Heatmap de loadings
        section_header("🗺️", "Heatmap de Loadings")
        plot_style()
        fig2, ax = plt.subplots(figsize=(max(8, n_components * 1.5), max(6, len(sel_vars) * 0.4)))
        sns.heatmap(
            loadings_df,
            cmap="RdBu_r",
            center=0,
            annot=True if len(sel_vars) <= 15 else False,
            fmt=".2f",
            linewidths=0.5,
            linecolor="#0f0f1a",
            ax=ax,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title("Loadings — Contribución de Variables a Componentes")
        fig2.tight_layout()
        st.pyplot(fig2)

        # ── Nuevo dataset
        section_header("💾", "Nuevo Dataset PCA")
        st.dataframe(df_pca.head(15), use_container_width=True)
        st.markdown(f"**Shape:** `{df_pca.shape[0]} × {df_pca.shape[1]}`")

        csv_pca = df_pca.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Descargar Dataset PCA (.csv)",
            csv_pca,
            "dataset_pca.csv",
            "text/csv",
        )


# ═════════════════════════════════════════════════════════════════════════════
# PÁGINA 5 — MODELOS ML
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🤖 5. Modelos ML":
    st.title("🤖 Modelos de Machine Learning")

    if st.session_state.df_processed is None:
        st.warning("⚠️ Primero completa las etapas anteriores.")
        st.stop()

    df_proc = st.session_state.df_processed
    df_pca = st.session_state.df_pca

    results_lr = None
    results_knn = None

    algo = st.radio(
        "Selecciona el algoritmo",
        ["🔵 Regresión Logística", "🟣 KNN (K-Nearest Neighbors)", "⚡ Ambos (Comparar)"],
        horizontal=True,
    )

    # ─────────────── Helper: validar que Y sea discreta ───────────────
    def check_y_discreta(series, nombre):
        """Devuelve True si la variable es apta para clasificación, False si es continua."""
        if not np.issubdtype(np.array(series).dtype, np.number):
            return True   # categórica/string → ok
        unique_ratio = len(series.dropna().unique()) / max(len(series.dropna()), 1)
        es_continua = unique_ratio > 0.05 and np.issubdtype(series.dtype, np.floating)
        if es_continua:
            st.error(
                f"❌ La variable **{nombre}** es continua "
                f"({series.nunique()} valores únicos, tipo `{series.dtype}`). "
                f"Los clasificadores necesitan clases discretas. "
                f"👉 Ve a **⚙️ 3. Preprocesamiento** y excluye esta variable de la normalización, "
                f"o elige otra variable con pocas categorías como variable dependiente."
            )
            return False
        return True

    # ─────────────── Función de entrenamiento ───────────────
    def train_and_evaluate(X_train, X_test, y_train, y_test, model, model_name, feature_names=None):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        try:
            if len(np.unique(y_train)) == 2:
                y_prob = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_prob)
            else:
                y_prob = model.predict_proba(X_test)
                auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
        except Exception:
            auc = None
            y_prob = None

        return {
            "model": model,
            "name": model_name,
            "accuracy": acc,
            "report": report,
            "cm": cm,
            "y_pred": y_pred,
            "y_test": y_test,
            "auc": auc,
            "y_prob": y_prob,
            "X_train_shape": X_train.shape,
            "feature_names": feature_names or [],
        }


    def show_results(res):
        model_key = res["name"].replace(" ", "_").lower()

        section_header("📊", f"Resultados — {res['name']}")

        # ── Métricas base (con threshold=0.5 por defecto)
        y_test_arr = np.array(res["y_test"])
        classes = np.unique(y_test_arr)
        is_binary = (len(classes) == 2 and res["y_prob"] is not None)

        # ── Slider de Threshold (solo binario con probabilidades)
        if is_binary:
            section_header("🎚️", "Ajuste de Threshold de Clasificación")
            st.markdown('<div class="info-box">Mueve el threshold para ver cómo cambian las métricas en tiempo real. <b>0.5</b> es el valor por defecto.</div>', unsafe_allow_html=True)

            threshold = st.slider(
                "Threshold de decisión",
                min_value=0.01, max_value=0.99, value=0.50, step=0.01,
                key=f"thr_{model_key}",
                format="%.2f",
            )
            y_pred_thr = (res["y_prob"] >= threshold).astype(int)
            acc_thr = accuracy_score(y_test_arr, y_pred_thr)
            report_thr = classification_report(y_test_arr, y_pred_thr, output_dict=True, zero_division=0)
            cm_thr = confusion_matrix(y_test_arr, y_pred_thr)
        else:
            threshold = 0.5
            y_pred_thr = res["y_pred"]
            acc_thr = res["accuracy"]
            report_thr = res["report"]
            cm_thr = res["cm"]

        # ── Tarjetas de métricas dinámicas
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(metric_card("Accuracy", f"{acc_thr*100:.1f}%"), unsafe_allow_html=True)
        with c2:
            macro = report_thr.get("macro avg", {})
            st.markdown(metric_card("F1-Score (macro)", f"{macro.get('f1-score', 0)*100:.1f}%"), unsafe_allow_html=True)
        with c3:
            prec = report_thr.get("macro avg", {}).get("precision", 0)
            st.markdown(metric_card("Precisión (macro)", f"{prec*100:.1f}%"), unsafe_allow_html=True)
        with c4:
            auc_text = f"{res['auc']:.3f}" if res["auc"] is not None else "N/A"
            st.markdown(metric_card("AUC-ROC", auc_text), unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:
            # Matriz de confusión (reactiva al threshold)
            plot_style()
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_thr)
            disp.plot(ax=ax_cm, colorbar=False, cmap="Blues")
            thr_label = f" (thr={threshold:.2f})" if is_binary else ""
            ax_cm.set_title(f"Matriz de Confusión{thr_label}")
            fig_cm.tight_layout()
            st.pyplot(fig_cm)

        with col_b:
            # Reporte de clasificación
            report_df = pd.DataFrame(report_thr).T.round(3)
            report_df = report_df.drop(index=["accuracy"], errors="ignore")
            st.markdown("**Reporte de Clasificación**")
            st.dataframe(report_df, use_container_width=True)

        # ── Curva ROC + línea del threshold (binario)
        if is_binary:
            fpr, tpr, thresholds_roc = roc_curve(y_test_arr, res["y_prob"])
            # encontrar punto más cercano al threshold actual
            idx_thr = np.argmin(np.abs(thresholds_roc - threshold))

            plot_style()
            fig_roc, ax_roc = plt.subplots(figsize=(7, 4))
            ax_roc.plot(fpr, tpr, color="#7c3aed", lw=2.5,
                        label=f"AUC = {res['auc']:.3f}")
            ax_roc.plot([0, 1], [0, 1], color="#555577", ls="--", lw=1.5, label="Random")
            ax_roc.scatter(fpr[idx_thr], tpr[idx_thr], s=120, zorder=5,
                           color="#f59e0b", edgecolors="#0f0f1a",
                           label=f"Threshold = {threshold:.2f}")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title("Curva ROC")
            ax_roc.legend()
            fig_roc.tight_layout()
            st.pyplot(fig_roc)

        # ── Top-10 Coeficientes (solo Regresión Logística)
        if "Logística" in res["name"] or "Logistic" in res["name"]:
            model_obj = res.get("model")
            feat_names = res.get("feature_names", [])
            if model_obj is not None and hasattr(model_obj, "coef_") and feat_names:
                section_header("🏆", "Top 10 Coeficientes — Regresión Logística")
                st.markdown('<div class="info-box">Los coeficientes con mayor valor absoluto son los que más impacto tienen en la predicción. Positivo → aumenta la probabilidad de la clase; Negativo → la reduce.</div>', unsafe_allow_html=True)

                coef_arr = model_obj.coef_
                # Para multiclase tomar la media del valor absoluto entre clases
                if coef_arr.shape[0] > 1:
                    coef_vals = np.mean(np.abs(coef_arr), axis=0)
                    coef_signed = np.mean(coef_arr, axis=0)
                else:
                    coef_vals = np.abs(coef_arr[0])
                    coef_signed = coef_arr[0]

                top10_idx = np.argsort(coef_vals)[::-1][:10]
                top10_names = [feat_names[i] for i in top10_idx]
                top10_vals = coef_signed[top10_idx]
                top10_abs = coef_vals[top10_idx]

                coef_df = pd.DataFrame({
                    "Variable": top10_names,
                    "Coeficiente": top10_vals.round(4),
                    "|Coeficiente|": top10_abs.round(4),
                    "Dirección": ["↑ Positivo" if v > 0 else "↓ Negativo" for v in top10_vals],
                })
                st.dataframe(coef_df, use_container_width=True)

                # Gráfica de barras horizontales
                plot_style()
                fig_coef, ax_coef = plt.subplots(figsize=(9, 5))
                bar_colors = ["#10b981" if v > 0 else "#ef4444" for v in top10_vals]
                y_pos = np.arange(len(top10_names))
                ax_coef.barh(y_pos, top10_vals, color=bar_colors,
                             edgecolor="#0f0f1a", height=0.65)
                ax_coef.set_yticks(y_pos)
                ax_coef.set_yticklabels(top10_names, fontsize=10)
                ax_coef.axvline(0, color="#8888cc", lw=1.2)
                ax_coef.set_xlabel("Coeficiente")
                ax_coef.set_title("Top 10 Coeficientes por Valor Absoluto")
                # Etiquetas de valor
                for i, v in enumerate(top10_vals):
                    offset = 0.01 * (max(top10_vals) - min(top10_vals)) if (max(top10_vals) - min(top10_vals)) != 0 else 0.01
                    ax_coef.text(v + (offset if v >= 0 else -offset),
                                 i, f"{v:.3f}",
                                 va="center",
                                 ha="left" if v >= 0 else "right",
                                 fontsize=8.5, color="#e0e0ff")
                fig_coef.tight_layout()
                st.pyplot(fig_coef)


    # ─────────────── Panel de configuración Regresión Logística ───────────────
    def config_lr(prefix="lr"):
        section_header("🔵", "Configuración — Regresión Logística")
        col1, col2 = st.columns(2)
        with col1:
            ds_choice = st.selectbox(
                "Dataset a usar",
                ["Dataset Procesado"] + (["Dataset PCA"] if df_pca is not None else []),
                key=f"{prefix}_ds",
            )
        with col2:
            test_size = st.slider("% para prueba", 10, 40, 20, key=f"{prefix}_test")

        ds = df_proc if "Procesado" in ds_choice else df_pca
        all_num = ds.select_dtypes(include=np.number).columns.tolist()
        all_cols = ds.columns.tolist()

        col3, col4 = st.columns(2)
        with col3:
            dep_var = st.selectbox("Variable dependiente (Y)", all_cols, key=f"{prefix}_dep")
        with col4:
            indep_vars = st.multiselect(
                "Variables independientes (X)",
                [c for c in all_num if c != dep_var],
                default=[c for c in all_num if c != dep_var][:min(5, len(all_num))],
                key=f"{prefix}_indep",
            )

        col5, col6 = st.columns(2)
        with col5:
            c_reg = st.number_input("Regularización C (mayor = menos regularización)", 0.01, 100.0, 1.0, key=f"{prefix}_c")
        with col6:
            max_iter = st.number_input("Max iteraciones", 100, 5000, 1000, key=f"{prefix}_iter")

        return ds, dep_var, indep_vars, test_size / 100, c_reg, max_iter


    # ─────────────── Panel de configuración KNN ───────────────
    def config_knn(prefix="knn"):
        section_header("🟣", "Configuración — KNN")
        col1, col2 = st.columns(2)
        with col1:
            ds_choice = st.selectbox(
                "Dataset a usar",
                ["Dataset Procesado"] + (["Dataset PCA"] if df_pca is not None else []),
                key=f"{prefix}_ds",
            )
        with col2:
            test_size = st.slider("% para prueba", 10, 40, 20, key=f"{prefix}_test")

        ds = df_proc if "Procesado" in ds_choice else df_pca
        all_num = ds.select_dtypes(include=np.number).columns.tolist()
        all_cols = ds.columns.tolist()

        col3, col4 = st.columns(2)
        with col3:
            target_var = st.selectbox("Variable clase (Y)", all_cols, key=f"{prefix}_target")
        with col4:
            feat_vars = st.multiselect(
                "Features (X)",
                [c for c in all_num if c != target_var],
                default=[c for c in all_num if c != target_var][:min(5, len(all_num))],
                key=f"{prefix}_feats",
            )

        col5, col6, col7 = st.columns(3)
        with col5:
            k = st.number_input("Número de vecinos K", 1, 50, 5, key=f"{prefix}_k")
        with col6:
            weights = st.selectbox("Pesos", ["uniform", "distance"], key=f"{prefix}_w")
        with col7:
            metric = st.selectbox("Métrica distancia", ["euclidean", "manhattan", "minkowski"], key=f"{prefix}_m")

        return ds, target_var, feat_vars, test_size / 100, int(k), weights, metric


    # ─────────────── Ejecución según selección ───────────────
    if "Regresión Logística" in algo or "Ambos" in algo:
        ds_lr, dep_lr, indep_lr, ts_lr, c_lr, iter_lr = config_lr("lr")

        if dep_lr and dep_lr in ds_lr.columns:
            y_preview = ds_lr[dep_lr].dropna()
            st.markdown(
                f'<div class="info-box">📌 Variable Y: <b>{dep_lr}</b> | '
                f'Tipo: <code>{y_preview.dtype}</code> | '
                f'Clases únicas: <b>{y_preview.nunique()}</b></div>',
                unsafe_allow_html=True,
            )

        run_lr = st.button("🚀 Entrenar Regresión Logística", type="primary", key="btn_lr")
        if run_lr:
            if not indep_lr:
                st.error("Selecciona al menos una variable independiente.")
            elif not check_y_discreta(ds_lr[dep_lr], dep_lr):
                pass  # el error ya fue mostrado por check_y_discreta
            else:
                try:
                    df_model = ds_lr[indep_lr + [dep_lr]].dropna()
                    X = df_model[indep_lr].values
                    y = df_model[dep_lr].values

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=ts_lr, random_state=42, stratify=y
                    )
                    sc = StandardScaler()
                    X_train = sc.fit_transform(X_train)
                    X_test = sc.transform(X_test)

                    model_lr = LogisticRegression(C=c_lr, max_iter=int(iter_lr), random_state=42)
                    results_lr = train_and_evaluate(X_train, X_test, y_train, y_test, model_lr, "Regresión Logística", feature_names=indep_lr)
                    st.session_state["results_lr"] = results_lr
                    show_results(results_lr)
                except Exception as e:
                    st.error(f"Error: {e}")

    if "KNN" in algo or "Ambos" in algo:
        st.markdown("---")
        ds_knn, target_knn, feat_knn, ts_knn, k_knn, w_knn, m_knn = config_knn("knn")

        if target_knn and target_knn in ds_knn.columns:
            y_prev_knn = ds_knn[target_knn].dropna()
            st.markdown(
                f'<div class="info-box">📌 Variable Y: <b>{target_knn}</b> | '
                f'Tipo: <code>{y_prev_knn.dtype}</code> | '
                f'Clases únicas: <b>{y_prev_knn.nunique()}</b></div>',
                unsafe_allow_html=True,
            )

        # Elbow plot para elegir K
        if st.checkbox("📈 Mostrar Elbow Plot (selección de K)", key="elbow_cb"):
            if feat_knn and target_knn and check_y_discreta(ds_knn[target_knn], target_knn):
                df_model_e = ds_knn[feat_knn + [target_knn]].dropna()
                X_e = df_model_e[feat_knn].values
                y_e = df_model_e[target_knn].values
                if len(np.unique(y_e)) >= 2:
                    X_tr_e, X_te_e, y_tr_e, y_te_e = train_test_split(X_e, y_e, test_size=ts_knn, random_state=42)
                    sc_e = StandardScaler()
                    X_tr_e = sc_e.fit_transform(X_tr_e)
                    X_te_e = sc_e.transform(X_te_e)
                    k_range = range(1, min(31, len(X_tr_e)))
                    accs = [accuracy_score(y_te_e, KNeighborsClassifier(n_neighbors=ki).fit(X_tr_e, y_tr_e).predict(X_te_e)) for ki in k_range]
                    plot_style()
                    fig_elbow, ax_elbow = plt.subplots(figsize=(8, 4))
                    ax_elbow.plot(list(k_range), accs, marker="o", color="#7c3aed", lw=2.5)
                    ax_elbow.set_xlabel("K")
                    ax_elbow.set_ylabel("Accuracy")
                    ax_elbow.set_title("Elbow Plot — Selección de K")
                    ax_elbow.set_xticks(list(k_range))
                    fig_elbow.tight_layout()
                    st.pyplot(fig_elbow)

        run_knn = st.button("🚀 Entrenar KNN", type="primary", key="btn_knn")
        if run_knn:
            if not feat_knn:
                st.error("Selecciona al menos una feature.")
            elif not check_y_discreta(ds_knn[target_knn], target_knn):
                pass  # error ya mostrado
            else:
                try:
                    df_model_k = ds_knn[feat_knn + [target_knn]].dropna()
                    X_k = df_model_k[feat_knn].values
                    y_k = df_model_k[target_knn].values

                    X_tr_k, X_te_k, y_tr_k, y_te_k = train_test_split(
                        X_k, y_k, test_size=ts_knn, random_state=42, stratify=y_k
                    )
                    sc_k = StandardScaler()
                    X_tr_k = sc_k.fit_transform(X_tr_k)
                    X_te_k = sc_k.transform(X_te_k)

                    model_knn = KNeighborsClassifier(n_neighbors=k_knn, weights=w_knn, metric=m_knn)
                    results_knn = train_and_evaluate(X_tr_k, X_te_k, y_tr_k, y_te_k, model_knn, f"KNN (K={k_knn})")
                    st.session_state["results_knn"] = results_knn
                    show_results(results_knn)
                except Exception as e:
                    st.error(f"Error: {e}")

    # ─────────────── Comparación ───────────────
    r_lr = st.session_state.get("results_lr")
    r_knn = st.session_state.get("results_knn")

    if r_lr and r_knn:
        section_header("⚡", "Comparación de Modelos")

        comp_df = pd.DataFrame([
            {
                "Modelo": r_lr["name"],
                "Accuracy (%)": round(r_lr["accuracy"] * 100, 2),
                "F1 Macro (%)": round(r_lr["report"].get("macro avg", {}).get("f1-score", 0) * 100, 2),
                "AUC": round(r_lr["auc"], 4) if r_lr["auc"] else "N/A",
                "Muestras entrenamiento": r_lr["X_train_shape"][0],
            },
            {
                "Modelo": r_knn["name"],
                "Accuracy (%)": round(r_knn["accuracy"] * 100, 2),
                "F1 Macro (%)": round(r_knn["report"].get("macro avg", {}).get("f1-score", 0) * 100, 2),
                "AUC": round(r_knn["auc"], 4) if r_knn["auc"] else "N/A",
                "Muestras entrenamiento": r_knn["X_train_shape"][0],
            },
        ])
        st.dataframe(comp_df, use_container_width=True)

        plot_style()
        fig_comp, ax_comp = plt.subplots(figsize=(8, 4))
        metrics = ["Accuracy (%)", "F1 Macro (%)"]
        x = np.arange(len(metrics))
        w = 0.35
        v_lr = [comp_df.loc[0, m] for m in metrics]
        v_knn = [comp_df.loc[1, m] for m in metrics]
        ax_comp.bar(x - w / 2, v_lr, w, label=r_lr["name"], color="#4f46e5", edgecolor="#0f0f1a")
        ax_comp.bar(x + w / 2, v_knn, w, label=r_knn["name"], color="#7c3aed", edgecolor="#0f0f1a")
        ax_comp.set_xticks(x)
        ax_comp.set_xticklabels(metrics)
        ax_comp.set_ylim(0, 110)
        ax_comp.set_title("Comparación de Métricas")
        ax_comp.legend()
        for i, (vl, vk) in enumerate(zip(v_lr, v_knn)):
            ax_comp.text(i - w / 2, vl + 1, f"{vl:.1f}", ha="center", fontsize=9, color="#e0e0ff")
            ax_comp.text(i + w / 2, vk + 1, f"{vk:.1f}", ha="center", fontsize=9, color="#e0e0ff")
        fig_comp.tight_layout()
        st.pyplot(fig_comp)

        best = r_lr["name"] if r_lr["accuracy"] >= r_knn["accuracy"] else r_knn["name"]
        st.markdown(f'<div class="success-box">🏆 Mejor modelo por Accuracy: <b>{best}</b></div>', unsafe_allow_html=True)