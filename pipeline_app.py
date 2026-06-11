"""
ML Pipeline App — Streamlit
Exploración · Encoding · Preprocesamiento · Modelos ML (LR · KNN · RF) · Comparación · Ensamble
"""

import os as _os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, precision_recall_curve, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
)

# ─────────────────────────────────────────────────────────────────────────────
# DATASETS PREDEFINIDOS — ajusta las rutas si tus CSVs están en otra carpeta
# ─────────────────────────────────────────────────────────────────────────────
DATASET_PATHS = {
    "HR Attrition — Limpio (Sin Derivadas)": "HR-Employee-Attrition-Limpio-Sin-Derivadas.csv",
    "HR Attrition — Original (IBM/Kaggle)":  "WA_Fn-UseC_-HR-Employee-Attrition.csv",
}

# ─────────────────────────────────────────────────────────────────────────────
# PÁGINA
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="ML Pipeline Studio", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
h1,h2,h3{font-family:'Space Mono',monospace!important;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0f0f1a 0%,#1a1a2e 100%);border-right:1px solid #2a2a4a;}
[data-testid="stSidebar"] *{color:#e0e0ff!important;}
.metric-card{background:linear-gradient(135deg,#1e1e3a 0%,#252545 100%);border:1px solid #3a3a6a;border-radius:12px;padding:1rem 1.2rem;text-align:center;margin-bottom:0.4rem;}
.metric-card .label{font-size:0.72rem;letter-spacing:.1em;text-transform:uppercase;color:#8888cc;margin-bottom:.25rem;}
.metric-card .value{font-family:'Space Mono',monospace;font-size:1.6rem;font-weight:700;color:#a78bfa;}
.section-header{background:linear-gradient(90deg,#4f46e5 0%,#7c3aed 100%);border-radius:10px;padding:.6rem 1.1rem;margin:1.2rem 0 .8rem;font-family:'Space Mono',monospace;font-size:.95rem;font-weight:700;color:white!important;letter-spacing:.03em;}
.info-table{border-collapse:collapse;width:100%;}
.info-table th{background:#2a2a4a;color:#a78bfa;padding:7px 11px;text-align:left;font-size:.78rem;text-transform:uppercase;letter-spacing:.08em;}
.info-table td{padding:6px 11px;border-bottom:1px solid #2a2a4a;font-size:.88rem;}
.info-table tr:hover td{background:#1a1a3a;}
.badge-num{background:#1d4ed8;color:white;padding:2px 8px;border-radius:20px;font-size:.73rem;}
.badge-cat{background:#7c3aed;color:white;padding:2px 8px;border-radius:20px;font-size:.73rem;}
.badge-bool{background:#0f766e;color:white;padding:2px 8px;border-radius:20px;font-size:.73rem;}
.badge-date{background:#b45309;color:white;padding:2px 8px;border-radius:20px;font-size:.73rem;}
.info-box{background:#1e2a4a;border-left:4px solid #4f46e5;border-radius:0 8px 8px 0;padding:.75rem 1rem;margin:.4rem 0;font-size:.88rem;color:#c0c0f0;}
.success-box{background:#1a3a2a;border-left:4px solid #10b981;border-radius:0 8px 8px 0;padding:.75rem 1rem;margin:.4rem 0;font-size:.88rem;color:#a0f0c0;}
.warn-box{background:#3a2a10;border-left:4px solid #f59e0b;border-radius:0 8px 8px 0;padding:.75rem 1rem;margin:.4rem 0;font-size:.88rem;color:#fde68a;}
div[data-testid="stDataFrame"]{border-radius:10px;overflow:hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS GLOBALES
# ─────────────────────────────────────────────────────────────────────────────
def classify_type(dtype):
    if pd.api.types.is_bool_dtype(dtype): return "Booleana","badge-bool"
    if pd.api.types.is_datetime64_any_dtype(dtype): return "Fecha","badge-date"
    if pd.api.types.is_numeric_dtype(dtype): return "Numérica","badge-num"
    return "Categórica","badge-cat"

def metric_card(label, value):
    return f'<div class="metric-card"><div class="label">{label}</div><div class="value">{value}</div></div>'

def section_header(icon, text):
    st.markdown(f'<div class="section-header">{icon} &nbsp; {text}</div>', unsafe_allow_html=True)

def plot_style():
    plt.rcParams.update({
        "figure.facecolor":"#0f0f1a","axes.facecolor":"#1a1a2e","axes.edgecolor":"#3a3a6a",
        "axes.labelcolor":"#c0c0f0","xtick.color":"#8888cc","ytick.color":"#8888cc",
        "text.color":"#e0e0ff","grid.color":"#2a2a4a","grid.linestyle":"--","font.family":"monospace",
    })

PALETTE = ["#7c3aed","#4f46e5","#0ea5e9","#10b981","#f59e0b","#ef4444","#ec4899","#8b5cf6","#06b6d4","#84cc16"]

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for k,v in [("df_original",None),("df_processed",None),("scaler_fitted",None),
            ("results_lr",None),("results_knn",None),("results_rf",None)]:
    if k not in st.session_state: st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 ML Pipeline Studio")
    st.markdown("---")
    page = st.radio("",
        ["📁 1. Carga y Exploración","📐 2. Encoding Categórico","⚙️ 3. Preprocesamiento","🤖 4. Modelos ML"],
        label_visibility="collapsed")
    st.markdown("---")
    if st.session_state.df_original is not None:
        df_o = st.session_state.df_original
        st.markdown(f"**Dataset cargado**\n🗂️ `{df_o.shape[0]}` filas × `{df_o.shape[1]}` cols")
    for key,label in [("results_lr","🔵 LR"),("results_knn","🟣 KNN"),("results_rf","🌲 RF")]:
        if st.session_state.get(key): st.markdown(f"✅ {label} entrenado")
    st.markdown("---")
    st.markdown("<small style='opacity:.5'>ML Pipeline Studio v2.1<br>LR · KNN · Random Forest</small>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — CARGA Y EXPLORACIÓN
# ═════════════════════════════════════════════════════════════════════════════
if page == "📁 1. Carga y Exploración":
    st.title("📁 Carga y Exploración de Datos")

    section_header("📂", "Selecciona el Dataset")

    opciones = list(DATASET_PATHS.keys()) + ["⬆️ Subir archivo propio"]
    fuente = st.radio(
        "Fuente de datos",
        opciones,
        horizontal=False,
        label_visibility="collapsed",
    )

    c1, c2 = st.columns(2)
    sep = c1.selectbox("Separador", [",", ";", "|", "\t"])
    enc = c2.selectbox("Encoding", ["utf-8", "latin-1", "cp1252"])

    data = None

    if fuente in DATASET_PATHS:
        ruta = DATASET_PATHS[fuente]
        if _os.path.exists(ruta):
            try:
                data = pd.read_csv(ruta, sep=sep, encoding=enc)
                st.session_state.df_original  = data
                st.session_state.df_processed = data.copy()
                st.markdown(
                    f'<div class="success-box">✅ Dataset <b>{fuente}</b> cargado '
                    f'desde <code>{ruta}</code></div>',
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"Error al leer {ruta}: {e}")
                st.stop()
        else:
            st.markdown(
                f'<div class="warn-box">⚠️ No se encontró <code>{ruta}</code> en la carpeta raíz del proyecto.<br>'
                f'Coloca el CSV en el mismo directorio que este script, '
                f'o ajusta la variable <code>DATASET_PATHS</code> al inicio del código.</div>',
                unsafe_allow_html=True,
            )
    else:
        uploaded = st.file_uploader("Sube tu archivo CSV", type=["csv"])
        if uploaded:
            try:
                data = pd.read_csv(uploaded, sep=sep, encoding=enc)
                st.session_state.df_original  = data
                st.session_state.df_processed = data.copy()
                st.markdown(
                    '<div class="success-box">✅ Archivo cargado correctamente</div>',
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()
        else:
            if st.session_state.df_original is not None:
                data = st.session_state.df_original
            else:
                st.markdown(
                    '<div class="info-box">👆 Sube un archivo CSV para comenzar.</div>',
                    unsafe_allow_html=True,
                )

    if data is not None:

        section_header("📏","Dimensiones")
        cols = st.columns(4)
        for col, lbl, val in zip(cols,["Filas","Columnas","Duplicados","Celdas Nulas"],
                                  [f"{data.shape[0]:,}",data.shape[1],data.duplicated().sum(),f"{data.isnull().sum().sum():,}"]):
            col.markdown(metric_card(lbl,val), unsafe_allow_html=True)

        section_header("🗃️","Estructura de Variables")
        rows = ""
        for col in data.columns:
            t,b = classify_type(data[col].dtype)
            n = data[col].isnull().sum(); p = n/len(data)*100
            rows += f"<tr><td><b>{col}</b></td><td><code>{data[col].dtype}</code></td><td><span class='{b}'>{t}</span></td><td>{n} <small>({p:.1f}%)</small></td><td>{data[col].nunique()}</td></tr>"
        st.markdown(f'<table class="info-table"><thead><tr><th>Columna</th><th>Dtype</th><th>Tipo</th><th>Nulos</th><th>Únicos</th></tr></thead><tbody>{rows}</tbody></table>', unsafe_allow_html=True)

        section_header("🕳️","Nulos por Columna")
        ndf = pd.DataFrame({"Columna":data.columns,"Nulos":data.isnull().sum().values,"% Nulos":(data.isnull().sum().values/len(data)*100).round(2)})
        ndf = ndf[ndf["Nulos"]>0].sort_values("% Nulos",ascending=False)
        if ndf.empty:
            st.markdown('<div class="success-box">✅ Sin valores nulos</div>', unsafe_allow_html=True)
        else:
            plot_style(); fig,ax = plt.subplots(figsize=(10,max(3,len(ndf)*.45)))
            ax.barh(ndf["Columna"],ndf["% Nulos"],color=["#7c3aed" if v<30 else "#dc2626" for v in ndf["% Nulos"]])
            ax.axvline(30,color="#f59e0b",ls="--",lw=1.2,label="30%"); ax.legend()
            ax.set_xlabel("% Nulos"); fig.tight_layout(); st.pyplot(fig)

        section_header("📊","Resumen Estadístico")
        st.dataframe(data.describe().round(4), use_container_width=True)

        section_header("📈","Distribuciones")
        num_cols = data.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            sel = st.multiselect("Columnas",num_cols,default=num_cols[:min(4,len(num_cols))])
            if sel:
                plot_style(); nc = min(3,len(sel)); nr = (len(sel)+nc-1)//nc
                fig,axes = plt.subplots(nr,nc,figsize=(5*nc,3.5*nr))
                axes = np.array(axes).flatten()
                for i,c in enumerate(sel):
                    axes[i].hist(data[c].dropna(),bins=30,color=PALETTE[i%len(PALETTE)],edgecolor="#0f0f1a",alpha=.85)
                    axes[i].set_title(c,fontsize=10)
                for j in range(len(sel),len(axes)): axes[j].set_visible(False)
                fig.tight_layout(); st.pyplot(fig)


# ═════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — ENCODING CATEGÓRICO
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📐 2. Encoding Categórico":
    st.title("📐 Encoding de Variables Categóricas")
    if st.session_state.df_processed is None:
        st.warning("⚠️ Primero carga un dataset."); st.stop()

    data = st.session_state.df_processed.copy()
    cat_cols = data.select_dtypes(exclude=np.number).columns.tolist()
    if not cat_cols:
        st.markdown('<div class="success-box">✅ No hay variables categóricas.</div>', unsafe_allow_html=True); st.stop()

    section_header("🏷️","Variables Categóricas Detectadas")
    for col in cat_cols:
        st.markdown(f"**`{col}`** — {data[col].nunique()} únicas &nbsp; `{data[col].value_counts().head(4).index.tolist()}`")

    section_header("🔧","Configurar Encoding")
    encoding_config={}; ordinal_orders={}
    for col in cat_cols:
        with st.expander(f"📌 {col}  ({data[col].nunique()} categorías)"):
            method = st.selectbox(f"Método",["No transformar","Label Encoding","One-Hot Encoding","Ordinal Encoding"],key=f"enc_{col}")
            encoding_config[col]=method
            if method=="Ordinal Encoding":
                uv = data[col].dropna().unique().tolist()
                oi = st.text_area(f"Orden (menor→mayor)",value="\n".join([str(v) for v in uv]),key=f"ord_{col}",height=110)
                ordinal_orders[col]=[v.strip() for v in oi.strip().split("\n") if v.strip()]

    if st.button("✅ Aplicar Encoding",type="primary"):
        df_enc=data.copy(); applied=[]
        for col,method in encoding_config.items():
            if method=="Label Encoding":
                df_enc[col]=LabelEncoder().fit_transform(df_enc[col].astype(str))
                applied.append(f"✔ `{col}` → Label Encoding")
            elif method=="One-Hot Encoding":
                d=pd.get_dummies(df_enc[col],prefix=col,drop_first=True,dtype=int)
                df_enc=pd.concat([df_enc.drop(columns=[col]),d],axis=1)
                applied.append(f"✔ `{col}` → One-Hot ({d.shape[1]} cols)")
            elif method=="Ordinal Encoding":
                order=ordinal_orders.get(col,[])
                if order:
                    om={str(v).strip():i for i,v in enumerate(order)}
                    df_enc[col]=df_enc[col].astype(str).str.strip().map(om)
                    un=df_enc[col].isna().sum()
                    if un>0: df_enc[col]=df_enc[col].fillna(-1); st.warning(f"⚠️ `{col}`: {un} no mapeados → -1")
                    df_enc[col]=df_enc[col].astype(float)
                    applied.append(f"✔ `{col}` → Ordinal ({len(order)} niveles)")
        st.session_state.df_processed=df_enc
        st.success("✅ Encoding aplicado")
        for m in applied: st.markdown(m)
        st.dataframe(df_enc.head(8),use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PÁGINA 3 — PREPROCESAMIENTO
# ═════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ 3. Preprocesamiento":
    st.title("⚙️ Preprocesamiento y Normalización")
    if st.session_state.df_original is None:
        st.warning("⚠️ Primero carga un dataset."); st.stop()

    data = st.session_state.df_processed.copy()
    num_cols = data.select_dtypes(include=np.number).columns.tolist()
    cat_cols = data.select_dtypes(exclude=np.number).columns.tolist()

    section_header("🕳️","Manejo de Nulos")
    if data.isnull().sum().sum()==0:
        st.markdown('<div class="success-box">✅ Sin nulos.</div>', unsafe_allow_html=True)
    else:
        c1,c2=st.columns(2)
        ni=c1.selectbox("Numéricos",["No imputar","Media","Mediana","Cero"])
        ci=c2.selectbox("Categóricos",["No imputar","Moda","'Unknown'"])
        if st.button("Aplicar imputación"):
            for c in num_cols:
                if data[c].isnull().sum()>0:
                    if ni=="Media": data[c]=data[c].fillna(data[c].mean())
                    elif ni=="Mediana": data[c]=data[c].fillna(data[c].median())
                    elif ni=="Cero": data[c]=data[c].fillna(0)
            for c in cat_cols:
                if data[c].isnull().sum()>0:
                    if ci=="Moda": data[c]=data[c].fillna(data[c].mode()[0])
                    elif ci=="'Unknown'": data[c]=data[c].fillna("Unknown")
            st.session_state.df_processed=data; st.success("✅ Imputación aplicada."); st.rerun()

    section_header("🔁","Duplicados")
    dupes=data.duplicated().sum()
    st.markdown(f'<div class="info-box">🔁 Duplicados encontrados: <b>{dupes}</b></div>', unsafe_allow_html=True)
    if dupes>0 and st.button("Eliminar duplicados"):
        data=data.drop_duplicates(); st.session_state.df_processed=data; st.success(f"✅ {dupes} eliminados."); st.rerun()

    section_header("📐","Normalización")
    if not num_cols:
        st.info("Sin columnas numéricas.")
    else:
        st.markdown('<div class="info-box">Selecciona columnas a <b>EXCLUIR</b> (ej. variable objetivo, IDs). El resto se normalizará.</div>', unsafe_allow_html=True)
        excluir=st.multiselect("🚫 Excluir de normalización",num_cols,default=[],placeholder="Elige columnas a excluir...")
        to_scale=[c for c in num_cols if c not in excluir]
        c1,c2=st.columns(2)
        c1.markdown(f"**✅ Se normalizarán ({len(to_scale)}):**\n```\n{chr(10).join(to_scale) if to_scale else 'Ninguna'}\n```")
        c2.markdown(f"**🚫 Sin cambio ({len(excluir)}):**\n```\n{chr(10).join(excluir) if excluir else 'Ninguna'}\n```")
        method=st.radio("Método",["StandardScaler (Z-score)","MinMaxScaler (0–1)","RobustScaler (mediana–IQR)"],horizontal=True)
        st.markdown("| Método | Fórmula | Cuándo |\n|--------|---------|--------|\n| StandardScaler | `(x-μ)/σ` | Datos normales |\n| MinMaxScaler | `(x-min)/(max-min)` | Sin outliers |\n| RobustScaler | `(x-med)/IQR` | Con outliers |")
        if st.button("✅ Aplicar Normalización",type="primary"):
            if not to_scale: st.warning("⚠️ Ninguna columna para normalizar.")
            else:
                sc = StandardScaler() if "Standard" in method else (MinMaxScaler() if "MinMax" in method else RobustScaler())
                data[to_scale]=sc.fit_transform(data[to_scale])
                st.session_state.df_processed=data; st.session_state.scaler_fitted=sc
                st.success(f"✅ {len(to_scale)} columnas normalizadas. {len(excluir)} excluidas.")
                plot_style(); pv=to_scale[:min(3,len(to_scale))]
                orig=st.session_state.df_original[pv]; norm=data[pv]
                fig,axes=plt.subplots(2,len(pv),figsize=(5*len(pv),6))
                if len(pv)==1: axes=np.array(axes).reshape(2,1)
                for i,col in enumerate(pv):
                    axes[0,i].hist(orig[col].dropna(),bins=25,color="#4f46e5",edgecolor="#0f0f1a",alpha=.85); axes[0,i].set_title(f"{col}\n(original)",fontsize=9)
                    axes[1,i].hist(norm[col].dropna(),bins=25,color="#10b981",edgecolor="#0f0f1a",alpha=.85); axes[1,i].set_title(f"{col}\n(normalizado)",fontsize=9)
                fig.tight_layout(); st.pyplot(fig)

    st.markdown("---"); section_header("📋","Dataset Actual")
    st.dataframe(data.head(20),use_container_width=True)
    st.markdown(f"**Shape:** `{data.shape[0]} × {data.shape[1]}`")


# ═════════════════════════════════════════════════════════════════════════════
# PÁGINA 4 — MODELOS ML
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🤖 4. Modelos ML":
    st.title("🤖 Modelos de Machine Learning")
    if st.session_state.df_processed is None:
        st.warning("⚠️ Completa las etapas anteriores."); st.stop()

    df_proc = st.session_state.df_processed

    # ─── Helpers internos ───
    def check_y_discreta(series, nombre):
        if not np.issubdtype(np.array(series).dtype, np.number): return True
        ur = len(series.dropna().unique()) / max(len(series.dropna()),1)
        if ur>0.05 and np.issubdtype(series.dtype, np.floating):
            st.error(f"❌ **{nombre}** es continua ({series.nunique()} únicos, `{series.dtype}`). "
                     f"Usa una variable con clases discretas, o excluye esta de la normalización.")
            return False
        return True

    def compute_metrics(y_test, y_pred, y_prob, model_name, is_binary):
        res = {
            "name": model_name,
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall":    recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1":        f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "cm":        confusion_matrix(y_test, y_pred),
            "report":    classification_report(y_test, y_pred, output_dict=True, zero_division=0),
            "y_pred": y_pred, "y_test": y_test, "y_prob": y_prob, "is_binary": is_binary,
            "auc": None, "logloss": None,
        }
        if y_prob is not None:
            try:
                if is_binary:
                    res["auc"]     = roc_auc_score(y_test, y_prob)
                    res["logloss"] = log_loss(y_test, y_prob)
                else:
                    res["auc"] = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
            except: pass
        return res

    def compute_metrics_regression(y_test, y_pred, model_name):
        """Métricas para modelos de regresión."""
        mse  = mean_squared_error(y_test, y_pred)
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2   = r2_score(y_test, y_pred)
        return {
            "name": model_name, "mse": mse, "mae": mae, "rmse": rmse, "r2": r2,
            "y_pred": y_pred, "y_test": y_test, "is_regression": True,
        }

    def train_model(model, X_tr, X_te, y_tr, y_te, name, feat_names=None):
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        is_binary = len(np.unique(y_tr)) == 2
        try:
            y_prob = model.predict_proba(X_te)[:, 1] if is_binary else model.predict_proba(X_te)
        except:
            y_prob = None
        res = compute_metrics(y_te, y_pred, y_prob, name, is_binary)
        res["model"] = model
        res["feature_names"] = feat_names or []
        res["X_train_shape"] = X_tr.shape
        return res

    def show_metrics_cards(res, threshold=0.5):
        yt = np.array(res["y_test"])
        has_prob = res["y_prob"] is not None and np.ndim(res["y_prob"]) == 1
        is_bin   = res["is_binary"] and has_prob

        yp = (res["y_prob"] >= threshold).astype(int) if is_bin else res["y_pred"]

        acc  = accuracy_score(yt, yp)
        prec = precision_score(yt, yp, average="weighted", zero_division=0)
        rec  = recall_score(yt, yp, average="weighted", zero_division=0)
        f1v  = f1_score(yt, yp, average="weighted", zero_division=0)
        cm   = confusion_matrix(yt, yp)
        rep  = classification_report(yt, yp, output_dict=True, zero_division=0)

        if is_bin:
            try:    auc_val = roc_auc_score(yt, res["y_prob"])
            except: auc_val = None
            try:    ll_val  = log_loss(yt, res["y_prob"])
            except: ll_val  = None
        else:
            auc_val = res["auc"]
            ll_val  = res.get("logloss")

        auc_text = f"{auc_val:.3f}" if auc_val is not None else "N/A"
        ll_text  = f"{ll_val:.4f}"  if ll_val  is not None else "N/A"

        thr_note = f" · threshold={threshold:.2f}" if is_bin else ""
        st.markdown(
            f'<div class="info-box">📐 Métricas en <b>weighted average</b>{thr_note} '
            f'— coinciden con la fila <i>weighted avg</i> del reporte y con la Matriz de Confusión.</div>',
            unsafe_allow_html=True
        )

        cols = st.columns(6)
        for c, lbl, val in zip(cols,
            ["Accuracy", "Precision (W)", "Recall (W)", "F1-Score (W)", "ROC-AUC", "Log Loss"],
            [f"{acc*100:.1f}%", f"{prec*100:.1f}%", f"{rec*100:.1f}%", f"{f1v*100:.1f}%", auc_text, ll_text]):
            c.markdown(metric_card(lbl, val), unsafe_allow_html=True)

        ca, cb = st.columns(2)
        with ca:
            plot_style(); fig, ax = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay(cm).plot(ax=ax, colorbar=False, cmap="Blues")
            tl = f" (thr={threshold:.2f})" if is_bin else ""
            ax.set_title(f"Matriz de Confusión{tl}"); fig.tight_layout(); st.pyplot(fig)
        with cb:
            rdf = pd.DataFrame(rep).T.round(3).drop(index=["accuracy"], errors="ignore")
            st.markdown("**Reporte de Clasificación** *(las tarjetas usan fila* `weighted avg`*)*")
            st.dataframe(rdf, use_container_width=True)

        if is_bin and auc_val is not None:
            fpr, tpr, thr_arr = roc_curve(yt, res["y_prob"])
            idx = np.argmin(np.abs(thr_arr - threshold))
            plot_style(); fig2, ax2 = plt.subplots(figsize=(7, 4))
            ax2.plot(fpr, tpr, color="#7c3aed", lw=2.5, label=f"AUC={auc_val:.3f}")
            ax2.plot([0,1],[0,1], color="#555577", ls="--", lw=1.5, label="Random")
            ax2.scatter(fpr[idx], tpr[idx], s=120, zorder=5, color="#f59e0b",
                        edgecolors="#0f0f1a", label=f"thr={threshold:.2f}")
            ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR")
            ax2.set_title("Curva ROC"); ax2.legend()
            fig2.tight_layout(); st.pyplot(fig2)

        if is_bin and "Logística" in res["name"] and res["y_prob"] is not None:
            section_header("📉", "Curva Precision-Recall")
            pr, rc, _ = precision_recall_curve(yt, res["y_prob"])
            plot_style(); fig3, ax3 = plt.subplots(figsize=(7, 4))
            ax3.plot(rc, pr, color="#10b981", lw=2.5)
            ax3.set_xlabel("Recall"); ax3.set_ylabel("Precision")
            ax3.set_title("Precision-Recall Curve")
            fig3.tight_layout(); st.pyplot(fig3)

        return acc, prec, rec, f1v, cm, rep

    def show_metrics_cards_regression(res):
        """Tarjetas de métricas para Random Forest Regressor."""
        st.markdown(
            '<div class="info-box">📐 Métricas de regresión — MSE, MAE, RMSE y R²</div>',
            unsafe_allow_html=True
        )
        cols = st.columns(4)
        for c, lbl, val in zip(cols,
            ["R² Score", "RMSE", "MAE", "MSE"],
            [f"{res['r2']:.4f}", f"{res['rmse']:.4f}", f"{res['mae']:.4f}", f"{res['mse']:.4f}"]):
            c.markdown(metric_card(lbl, val), unsafe_allow_html=True)

        plot_style()
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        yt = np.array(res["y_test"]); yp = np.array(res["y_pred"])
        axes[0].scatter(yt, yp, color="#7c3aed", alpha=0.5, edgecolors="#0f0f1a", s=30)
        mn, mx = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        axes[0].plot([mn, mx], [mn, mx], color="#f59e0b", ls="--", lw=1.5, label="Perfecta")
        axes[0].set_xlabel("Valores Reales"); axes[0].set_ylabel("Predicciones")
        axes[0].set_title("Real vs Predicho"); axes[0].legend()
        residuals = yt - yp
        axes[1].hist(residuals, bins=30, color="#4f46e5", edgecolor="#0f0f1a", alpha=0.85)
        axes[1].axvline(0, color="#f59e0b", ls="--", lw=1.5)
        axes[1].set_xlabel("Residuos"); axes[1].set_ylabel("Frecuencia")
        axes[1].set_title("Distribución de Residuos")
        fig.tight_layout(); st.pyplot(fig)

    def show_cv(model_cls, X, y, cv=5, key_prefix="", is_regression=False):
        section_header("🔄","Cross Validation")
        st.markdown('<div class="info-box">Validación cruzada estratificada para estimar desempeño real.</div>', unsafe_allow_html=True)
        cv_k=st.slider("Folds (k)",3,10,5,key=f"cv_{key_prefix}")
        if st.button("Ejecutar CV",key=f"cv_btn_{key_prefix}"):
            if is_regression:
                kf = KFold(n_splits=cv_k, shuffle=True, random_state=42)
                scores = cross_val_score(model_cls, X, y, cv=kf, scoring="r2", n_jobs=-1)
                label = "R² Score"
            else:
                skf = StratifiedKFold(n_splits=cv_k, shuffle=True, random_state=42)
                scores = cross_val_score(model_cls, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
                label = "Accuracy"
            st.markdown(f'<div class="success-box">CV {label}: <b>{scores.mean():.4f}</b> ± {scores.std():.4f}</div>', unsafe_allow_html=True)
            plot_style(); fig,ax=plt.subplots(figsize=(7,3))
            ax.bar(range(1,cv_k+1),scores,color=PALETTE[:cv_k],edgecolor="#0f0f1a",width=0.6)
            ax.axhline(scores.mean(),color="#f59e0b",ls="--",lw=1.5,label=f"Media={scores.mean():.3f}")
            ax.set_xlabel("Fold"); ax.set_ylabel(label); ax.set_title("CV por Fold"); ax.legend()
            fig.tight_layout(); st.pyplot(fig)

    def show_feature_importance(model, feat_names, title="Feature Importance"):
        if not hasattr(model,"feature_importances_") or not feat_names: return
        section_header("🌟",title)
        imp=model.feature_importances_
        idx=np.argsort(imp)[::-1][:15]
        names=[feat_names[i] for i in idx]; vals=imp[idx]
        plot_style(); fig,ax=plt.subplots(figsize=(9,max(4,len(names)*.45)))
        ax.barh(names[::-1],vals[::-1],color=PALETTE[0],edgecolor="#0f0f1a",height=0.65)
        ax.set_xlabel("Importancia"); ax.set_title(title)
        for i,v in enumerate(vals[::-1]):
            ax.text(v+0.001,i,f"{v:.3f}",va="center",fontsize=8.5,color="#e0e0ff")
        fig.tight_layout(); st.pyplot(fig)

    def show_top10_coef(model, feat_names):
        if not hasattr(model,"coef_") or not feat_names: return
        section_header("🏆","Top 10 Coeficientes")
        st.markdown('<div class="info-box">Mayor valor absoluto = mayor impacto. Verde=positivo, Rojo=negativo.</div>', unsafe_allow_html=True)
        ca=model.coef_
        cv=np.mean(np.abs(ca),axis=0) if ca.shape[0]>1 else np.abs(ca[0])
        cs=np.mean(ca,axis=0) if ca.shape[0]>1 else ca[0]
        idx=np.argsort(cv)[::-1][:10]
        t10n=[feat_names[i] for i in idx]; t10v=cs[idx]; t10a=cv[idx]
        st.dataframe(pd.DataFrame({"Variable":t10n,"Coeficiente":t10v.round(4),"|Coef|":t10a.round(4),
            "Dirección":["↑ Pos" if v>0 else "↓ Neg" for v in t10v]}),use_container_width=True)
        plot_style(); fig,ax=plt.subplots(figsize=(9,5))
        bc=["#10b981" if v>0 else "#ef4444" for v in t10v]
        ax.barh(np.arange(len(t10n)),t10v,color=bc,edgecolor="#0f0f1a",height=0.65)
        ax.set_yticks(np.arange(len(t10n))); ax.set_yticklabels(t10n,fontsize=10); ax.axvline(0,color="#8888cc",lw=1.2)
        rng=max(t10v)-min(t10v) if (max(t10v)-min(t10v))!=0 else 0.01
        for i,v in enumerate(t10v):
            off=0.01*rng; ax.text(v+(off if v>=0 else -off),i,f"{v:.3f}",va="center",ha="left" if v>=0 else "right",fontsize=8.5,color="#e0e0ff")
        ax.set_xlabel("Coeficiente"); ax.set_title("Top 10 Coeficientes"); fig.tight_layout(); st.pyplot(fig)

    # ─── Selector de algoritmo ───
    algo = st.radio("Algoritmo",["🔵 Regresión Logística","🟣 KNN","🌲 Random Forest","⚡ Todos (Comparar)"],horizontal=True)

    num_cols_ds = df_proc.select_dtypes(include=np.number).columns.tolist()
    all_cols_ds = df_proc.columns.tolist()

    # ══════════════════════════════
    # REGRESIÓN LOGÍSTICA
    # ══════════════════════════════
    if "Logística" in algo or "Todos" in algo:
        section_header("🔵","Regresión Logística — Configuración")
        c1,c2=st.columns(2)
        ts_lr=c1.slider("% Test",10,40,20,key="ts_lr")/100
        dep_lr=c2.selectbox("Variable Y",all_cols_ds,key="dep_lr")

        indep_lr=st.multiselect("Variables X (independientes)",
            [c for c in num_cols_ds if c!=dep_lr],
            default=[c for c in num_cols_ds if c!=dep_lr][:min(5,len(num_cols_ds))],key="indep_lr")

        c3,c4=st.columns(2)
        c_lr=c3.number_input("C (regularización)",0.01,100.0,1.0,key="c_lr")
        iter_lr=c4.number_input("Max iteraciones",100,5000,1000,key="iter_lr")

        if dep_lr in df_proc.columns:
            yp=df_proc[dep_lr].dropna()
            st.markdown(f'<div class="info-box">📌 Y: <b>{dep_lr}</b> | tipo: <code>{yp.dtype}</code> | clases: <b>{yp.nunique()}</b></div>', unsafe_allow_html=True)

        section_header("🎚️","Threshold — Regresión Logística")
        st.markdown('<div class="info-box">Mueve el slider para recalcular métricas sin re-entrenar. 0.5 = defecto.</div>', unsafe_allow_html=True)
        thr_lr=st.slider("Threshold LR",0.01,0.99,0.50,0.01,key="thr_lr",format="%.2f")

        bc1,bc2=st.columns([3,1])
        with bc1: run_lr=st.button("🚀 Entrenar Regresión Logística",type="primary",key="btn_lr")
        with bc2:
            if st.button("🗑️ Borrar",key="clr_lr"): st.session_state["results_lr"]=None; st.rerun()

        if run_lr:
            if not indep_lr: st.error("Selecciona variables X.")
            elif not check_y_discreta(df_proc[dep_lr],dep_lr): pass
            else:
                try:
                    dm=df_proc[indep_lr+[dep_lr]].dropna()
                    X=dm[indep_lr].values; y=dm[dep_lr].values
                    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=ts_lr,random_state=42,stratify=y)
                    sc=StandardScaler(); Xtr=sc.fit_transform(Xtr); Xte=sc.transform(Xte)
                    res=train_model(LogisticRegression(C=c_lr,max_iter=int(iter_lr),random_state=42),Xtr,Xte,ytr,yte,"Regresión Logística",feat_names=indep_lr)
                    st.session_state["results_lr"]=res; st.session_state["X_lr"]=(Xtr,Xte,ytr,yte); st.session_state["feats_lr"]=indep_lr
                    st.success("✅ Entrenamiento completado.")
                except Exception as e: st.error(f"Error: {e}")

        if st.session_state.get("results_lr"):
            res=st.session_state["results_lr"]
            show_metrics_cards(res,threshold=thr_lr)
            show_top10_coef(res["model"],res["feature_names"])
            if st.session_state.get("X_lr") and st.session_state.get("feats_lr"):
                Xtr,Xte,ytr,yte=st.session_state["X_lr"]
                Xall=np.vstack([Xtr,Xte]); yall=np.concatenate([ytr,yte])
                show_cv(LogisticRegression(C=c_lr,max_iter=int(iter_lr),random_state=42),Xall,yall,key_prefix="lr")

        if "Todos" in algo: st.markdown("---")

    # ══════════════════════════════
    # KNN
    # ══════════════════════════════
    if "KNN" in algo or "Todos" in algo:
        section_header("🟣","KNN — Configuración")
        c1,c2=st.columns(2)
        ts_knn=c1.slider("% Test",10,40,20,key="ts_knn")/100
        dep_knn=c2.selectbox("Variable Y",all_cols_ds,key="dep_knn")

        feat_knn=st.multiselect("Variables X",
            [c for c in num_cols_ds if c!=dep_knn],
            default=[c for c in num_cols_ds if c!=dep_knn][:min(5,len(num_cols_ds))],key="feat_knn")

        c3,c4,c5=st.columns(3)
        k_knn=int(c3.number_input("K vecinos",1,50,5,key="k_knn"))
        w_knn=c4.selectbox("Pesos",["uniform","distance"],key="w_knn")
        m_knn=c5.selectbox("Métrica",["euclidean","manhattan","minkowski"],key="m_knn")

        if dep_knn in df_proc.columns:
            yp2=df_proc[dep_knn].dropna()
            st.markdown(f'<div class="info-box">📌 Y: <b>{dep_knn}</b> | tipo: <code>{yp2.dtype}</code> | clases: <b>{yp2.nunique()}</b></div>', unsafe_allow_html=True)

        # ── ELBOW PLOT MEJORADO: Accuracy + Precision + Recall + F1 ──────────
        if st.checkbox("📈 Elbow Plot (métricas vs K)",key="elbow_cb"):
            st.markdown('<div class="info-box">Visualiza cómo evolucionan <b>Accuracy, Precision, Recall y F1-Score</b> al variar K. Útil para elegir el K óptimo.</div>', unsafe_allow_html=True)
            if feat_knn and dep_knn and check_y_discreta(df_proc[dep_knn],dep_knn):
                dm=df_proc[feat_knn+[dep_knn]].dropna()
                Xe=dm[feat_knn].values; ye=dm[dep_knn].values
                if len(np.unique(ye))>=2:
                    Xtr2,Xte2,ytr2,yte2=train_test_split(Xe,ye,test_size=ts_knn,random_state=42)
                    sc2=StandardScaler(); Xtr2=sc2.fit_transform(Xtr2); Xte2=sc2.transform(Xte2)
                    kr=range(1,min(31,len(Xtr2)))

                    # Calcular las 4 métricas para cada K
                    accs, precs, recs, f1s, errs = [], [], [], [], []
                    for ki in kr:
                        mdl = KNeighborsClassifier(n_neighbors=ki, weights=w_knn, metric=m_knn)
                        mdl.fit(Xtr2, ytr2)
                        yp_ki = mdl.predict(Xte2)
                        accs.append(accuracy_score(yte2, yp_ki))
                        precs.append(precision_score(yte2, yp_ki, average="weighted", zero_division=0))
                        recs.append(recall_score(yte2, yp_ki, average="weighted", zero_division=0))
                        f1s.append(f1_score(yte2, yp_ki, average="weighted", zero_division=0))
                        errs.append(1 - accs[-1])

                    plot_style()
                    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

                    # Gráfico superior: las 4 métricas positivas
                    axes[0].plot(list(kr), accs,  marker="o", color="#7c3aed", lw=2.5, markersize=5, label="Accuracy")
                    axes[0].plot(list(kr), precs, marker="s", color="#0ea5e9", lw=2.0, markersize=5, label="Precision (W)")
                    axes[0].plot(list(kr), recs,  marker="^", color="#10b981", lw=2.0, markersize=5, label="Recall (W)")
                    axes[0].plot(list(kr), f1s,   marker="D", color="#f59e0b", lw=2.0, markersize=5, label="F1-Score (W)")

                    # Marcar el K actual seleccionado
                    k_idx = k_knn - 1
                    if 0 <= k_idx < len(list(kr)):
                        for metric_vals, col in [(accs,"#7c3aed"),(precs,"#0ea5e9"),(recs,"#10b981"),(f1s,"#f59e0b")]:
                            axes[0].scatter(k_knn, metric_vals[k_idx], s=180, zorder=6,
                                            color=col, edgecolors="white", linewidths=1.5)
                        axes[0].axvline(k_knn, color="#ffffff", ls=":", lw=1.2, alpha=0.5, label=f"K={k_knn} seleccionado")

                    axes[0].set_ylabel("Métrica"); axes[0].set_title("Métricas de Clasificación vs K")
                    axes[0].legend(loc="lower right", fontsize=9)
                    axes[0].set_ylim(max(0, min(min(accs),min(precs),min(recs),min(f1s)) - 0.05), 1.05)
                    axes[0].set_xticks(list(kr)); axes[0].grid(True, alpha=0.3)

                    # Gráfico inferior: Error rate (elbow clásico)
                    axes[1].plot(list(kr), errs, marker="o", color="#ef4444", lw=2.5, markersize=5, label="Error Rate")
                    if 0 <= k_idx < len(list(kr)):
                        axes[1].scatter(k_knn, errs[k_idx], s=180, zorder=6,
                                        color="#ef4444", edgecolors="white", linewidths=1.5)
                        axes[1].axvline(k_knn, color="#ffffff", ls=":", lw=1.2, alpha=0.5)
                    axes[1].set_xlabel("K (número de vecinos)")
                    axes[1].set_ylabel("Error Rate")
                    axes[1].set_title("Error Rate vs K (Elbow clásico)")
                    axes[1].legend(loc="upper right", fontsize=9)
                    axes[1].set_xticks(list(kr)); axes[1].grid(True, alpha=0.3)

                    # Tabla resumen del K seleccionado
                    if 0 <= k_idx < len(list(kr)):
                        st.markdown(f"""
                        <div class="success-box">
                        📌 <b>K={k_knn} seleccionado</b> — 
                        Accuracy: <b>{accs[k_idx]*100:.1f}%</b> · 
                        Precision: <b>{precs[k_idx]*100:.1f}%</b> · 
                        Recall: <b>{recs[k_idx]*100:.1f}%</b> · 
                        F1: <b>{f1s[k_idx]*100:.1f}%</b> · 
                        Error: <b>{errs[k_idx]*100:.1f}%</b>
                        </div>""", unsafe_allow_html=True)

                    fig.tight_layout(); st.pyplot(fig)

        section_header("🎚️","Threshold — KNN")
        st.markdown('<div class="info-box">Mueve el slider para recalcular métricas sin re-entrenar.</div>', unsafe_allow_html=True)
        thr_knn=st.slider("Threshold KNN",0.01,0.99,0.50,0.01,key="thr_knn",format="%.2f")

        bc3,bc4=st.columns([3,1])
        with bc3: run_knn=st.button("🚀 Entrenar KNN",type="primary",key="btn_knn")
        with bc4:
            if st.button("🗑️ Borrar",key="clr_knn"): st.session_state["results_knn"]=None; st.rerun()

        if run_knn:
            if not feat_knn: st.error("Selecciona variables X.")
            elif not check_y_discreta(df_proc[dep_knn],dep_knn): pass
            else:
                try:
                    dm=df_proc[feat_knn+[dep_knn]].dropna()
                    X=dm[feat_knn].values; y=dm[dep_knn].values
                    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=ts_knn,random_state=42,stratify=y)
                    sc=StandardScaler(); Xtr=sc.fit_transform(Xtr); Xte=sc.transform(Xte)
                    res=train_model(KNeighborsClassifier(n_neighbors=k_knn,weights=w_knn,metric=m_knn),Xtr,Xte,ytr,yte,f"KNN (K={k_knn})",feat_names=feat_knn)
                    st.session_state["results_knn"]=res; st.session_state["X_knn"]=(Xtr,Xte,ytr,yte); st.session_state["feats_knn"]=feat_knn
                    st.success("✅ Entrenamiento completado.")
                except Exception as e: st.error(f"Error: {e}")

        if st.session_state.get("results_knn"):
            res=st.session_state["results_knn"]
            show_metrics_cards(res,threshold=thr_knn)
            if st.session_state.get("X_knn"):
                Xtr,Xte,ytr,yte=st.session_state["X_knn"]
                Xall=np.vstack([Xtr,Xte]); yall=np.concatenate([ytr,yte])
                show_cv(KNeighborsClassifier(n_neighbors=k_knn,weights=w_knn,metric=m_knn),Xall,yall,key_prefix="knn")

        if "Todos" in algo: st.markdown("---")

    # ══════════════════════════════
    # RANDOM FOREST
    # ══════════════════════════════
    if "Random Forest" in algo or "Todos" in algo:
        section_header("🌲","Random Forest — Configuración")

        # ── SELECTOR: Clasificación vs Regresión ──────────────────────────────
        st.markdown('<div class="info-box">🌳 Elige el tipo de árbol de decisión que usarán los árboles internos del bosque.</div>', unsafe_allow_html=True)
        rf_tipo = st.radio(
            "Tipo de árbol de decisión",
            ["🏷️ Clasificación (RandomForestClassifier)", "📈 Regresión (RandomForestRegressor)"],
            horizontal=True,
            key="rf_tipo",
        )
        rf_es_regresion = "Regresión" in rf_tipo

        if rf_es_regresion:
            st.markdown("""
            <div class="warn-box">
            ⚠️ <b>Modo Regresión:</b> la variable Y debe ser <b>numérica continua</b>. 
            Las métricas cambian a MSE, MAE, RMSE y R². 
            No aplica threshold, ni ROC-AUC, ni matriz de confusión.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
            ℹ️ <b>Modo Clasificación:</b> la variable Y debe tener <b>clases discretas</b>. 
            Se muestran Accuracy, Precision, Recall, F1, ROC-AUC y Matriz de Confusión.
            </div>""", unsafe_allow_html=True)

        c1,c2=st.columns(2)
        ts_rf=c1.slider("% Test",10,40,20,key="ts_rf")/100
        dep_rf=c2.selectbox("Variable Y",all_cols_ds,key="dep_rf")

        feat_rf=st.multiselect("Variables X",
            [c for c in num_cols_ds if c!=dep_rf],
            default=[c for c in num_cols_ds if c!=dep_rf][:min(8,len(num_cols_ds))],key="feat_rf")

        c3,c4,c5=st.columns(3)
        n_est=int(c3.number_input("N° árboles",10,1000,100,key="n_est"))
        max_d=c4.selectbox("Profundidad máx",["None","5","10","15","20"],key="max_d")
        min_s=int(c5.number_input("Min muestras hoja",1,20,2,key="min_s"))
        max_depth=None if max_d=="None" else int(max_d)

        if dep_rf in df_proc.columns:
            yp3=df_proc[dep_rf].dropna()
            tipo_lbl = "continua" if rf_es_regresion else f"clases: {yp3.nunique()}"
            st.markdown(f'<div class="info-box">📌 Y: <b>{dep_rf}</b> | tipo: <code>{yp3.dtype}</code> | {tipo_lbl} | modo: <b>{"Regresión" if rf_es_regresion else "Clasificación"}</b></div>', unsafe_allow_html=True)
            if feat_rf:
                st.markdown("**📉 Varianza del Dataset (Variables X antes del entrenamiento):**")
                df_var_x = df_proc[feat_rf].dropna().var().to_frame(name="Varianza")
                st.dataframe(df_var_x.T.round(6), use_container_width=True)

        # Threshold solo aplica a clasificación
        if not rf_es_regresion:
            section_header("🎚️","Threshold — Random Forest")
            thr_rf=st.slider("Threshold RF",0.01,0.99,0.50,0.01,key="thr_rf",format="%.2f")
        else:
            thr_rf = 0.5  # irrelevante para regresión

        bc5,bc6=st.columns([3,1])
        with bc5: run_rf=st.button("🚀 Entrenar Random Forest",type="primary",key="btn_rf")
        with bc6:
            if st.button("🗑️ Borrar",key="clr_rf"): st.session_state["results_rf"]=None; st.rerun()

        if run_rf:
            if not feat_rf: st.error("Selecciona variables X.")
            elif not rf_es_regresion and not check_y_discreta(df_proc[dep_rf],dep_rf): pass
            else:
                try:
                    dm=df_proc[feat_rf+[dep_rf]].dropna()
                    X=dm[feat_rf].values; y=dm[dep_rf].values

                    if rf_es_regresion:
                        # ── Regresión ──
                        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=ts_rf,random_state=42)
                        sc=StandardScaler(); Xtr=sc.fit_transform(Xtr); Xte=sc.transform(Xte)
                        model_rf = RandomForestRegressor(
                            n_estimators=n_est, max_depth=max_depth,
                            min_samples_leaf=min_s, random_state=42, n_jobs=-1
                        )
                        model_rf.fit(Xtr, ytr)
                        y_pred_rf = model_rf.predict(Xte)
                        res = compute_metrics_regression(yte, y_pred_rf, "Random Forest (Regresión)")
                        res["model"] = model_rf
                        res["feature_names"] = feat_rf
                        res["is_regression"] = True
                        st.session_state["results_rf"] = res
                        st.session_state["X_rf"] = (Xtr, Xte, ytr, yte)
                        st.session_state["feats_rf"] = feat_rf
                        st.session_state["rf_is_regression"] = True
                        st.success("✅ Entrenamiento completado (Regresión).")
                    else:
                        # ── Clasificación ──
                        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=ts_rf,random_state=42,stratify=y)
                        sc=StandardScaler(); Xtr=sc.fit_transform(Xtr); Xte=sc.transform(Xte)
                        res=train_model(
                            RandomForestClassifier(n_estimators=n_est, max_depth=max_depth,
                                                   min_samples_leaf=min_s, random_state=42, n_jobs=-1),
                            Xtr, Xte, ytr, yte, "Random Forest", feat_names=feat_rf
                        )
                        res["is_regression"] = False
                        st.session_state["results_rf"] = res
                        st.session_state["X_rf"] = (Xtr, Xte, ytr, yte)
                        st.session_state["feats_rf"] = feat_rf
                        st.session_state["rf_is_regression"] = False
                        st.success("✅ Entrenamiento completado (Clasificación).")
                except Exception as e: st.error(f"Error: {e}")

        if st.session_state.get("results_rf"):
            res = st.session_state["results_rf"]
            is_reg = st.session_state.get("rf_is_regression", False)

            if is_reg:
                show_metrics_cards_regression(res)
                show_feature_importance(res["model"], res["feature_names"], "Feature Importance (Regresión)")
                if st.session_state.get("X_rf"):
                    Xtr,Xte,ytr,yte=st.session_state["X_rf"]
                    Xall=np.vstack([Xtr,Xte]); yall=np.concatenate([ytr,yte])
                    show_cv(
                        RandomForestRegressor(n_estimators=n_est, max_depth=max_depth,
                                              min_samples_leaf=min_s, random_state=42, n_jobs=-1),
                        Xall, yall, key_prefix="rf", is_regression=True
                    )
            else:
                show_metrics_cards(res, threshold=thr_rf)
                show_feature_importance(res["model"], res["feature_names"])
                if st.session_state.get("X_rf"):
                    Xtr,Xte,ytr,yte=st.session_state["X_rf"]
                    Xall=np.vstack([Xtr,Xte]); yall=np.concatenate([ytr,yte])
                    show_cv(
                        RandomForestClassifier(n_estimators=n_est, max_depth=max_depth,
                                               min_samples_leaf=min_s, random_state=42, n_jobs=-1),
                        Xall, yall, key_prefix="rf"
                    )
                    
                    # --- Análisis de árboles individuales ---
                    model_rf = res["model"]
                    tree_data = []
                    for idx, tree in enumerate(model_rf.estimators_):
                        preds_tree = tree.predict(Xte)
                        tree_var = np.var(preds_tree)
                        tree_mean = np.mean(preds_tree)
                        tree_data.append({
                            "Árbol": f"Árbol {idx + 1}",
                            "Varianza de Predicciones": round(tree_var, 6),
                            "Predicción Promedio": round(tree_mean, 6)
                        })
                    
                    df_trees = pd.DataFrame(tree_data)
                    avg_var = df_trees["Varianza de Predicciones"].mean()
                    avg_pred = df_trees["Predicción Promedio"].mean()
                    
                    section_header("🌲", "Varianza y Predicción de los Árboles Individuales")
                    
                    # Mostrar promedios en un cuadro
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(metric_card("Promedio de la Varianza", f"{avg_var:.6f}"), unsafe_allow_html=True)
                    with c2:
                        st.markdown(metric_card("Promedio de la Predicción", f"{avg_pred:.6f}"), unsafe_allow_html=True)
                    
                    # Mostrar la tabla de todos los árboles
                    with st.expander("🔍 Ver detalles de todos los árboles"):
                        st.dataframe(df_trees, use_container_width=True)

    # ══════════════════════════════════════════════════════════
    # COMPARACIÓN DE MODELOS
    # ══════════════════════════════════════════════════════════
    avail = {k:st.session_state.get(k) for k in ["results_lr","results_knn","results_rf"] if st.session_state.get(k)}
    # Solo comparar modelos de clasificación
    avail = {k:v for k,v in avail.items() if not v.get("is_regression", False)}

    if len(avail)>=2:
        st.markdown("---")
        section_header("⚡","Comparación de Modelos")

        first_res = list(avail.values())[0]
        classes_avail = list(np.unique(np.array(first_res["y_test"])))
        classes_str = [str(c) for c in classes_avail]

        st.markdown('<div class="info-box">Elige la clase de interés para comparar métricas por clase específica, y el método de puntuación para decidir el ganador.</div>', unsafe_allow_html=True)
        cc1,cc2 = st.columns(2)
        clase_sel = cc1.selectbox("Clase de interés para comparar",classes_str,key="clase_comp")
        metodo_comp = cc2.selectbox("Método de comparación",
            ["Por moda (más victorias)","Por probabilidad media","Ponderación por F1"],key="metodo_comp")

        METRICAS = ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]
        rows_comp = []
        for key,res in avail.items():
            yt=np.array(res["y_test"]); yp=res["y_pred"]
            cls_idx=classes_str.index(clase_sel) if clase_sel in classes_str else 0
            acc_c = accuracy_score(yt,yp)
            try: prec_c=precision_score(yt,yp,labels=[classes_avail[cls_idx]],average="micro",zero_division=0)
            except: prec_c=precision_score(yt,yp,average="macro",zero_division=0)
            try: rec_c=recall_score(yt,yp,labels=[classes_avail[cls_idx]],average="micro",zero_division=0)
            except: rec_c=recall_score(yt,yp,average="macro",zero_division=0)
            try: f1_c=f1_score(yt,yp,labels=[classes_avail[cls_idx]],average="micro",zero_division=0)
            except: f1_c=f1_score(yt,yp,average="macro",zero_division=0)
            auc_c=res["auc"] if res["auc"] is not None else 0.0
            rows_comp.append({"Modelo":res["name"],"Accuracy":round(acc_c*100,2),"Precision":round(prec_c*100,2),
                "Recall":round(rec_c*100,2),"F1-Score":round(f1_c*100,2),"ROC-AUC":round(auc_c,4)})

        comp_df=pd.DataFrame(rows_comp).set_index("Modelo")
        st.dataframe(comp_df,use_container_width=True)

        plot_style(); fig_cmp,ax_cmp=plt.subplots(figsize=(11,5))
        x=np.arange(len(METRICAS)); w=0.8/max(len(rows_comp),1)
        model_names=[r["Modelo"] for r in rows_comp]
        for i,row in enumerate(rows_comp):
            vals=[row["Accuracy"],row["Precision"],row["Recall"],row["F1-Score"],row["ROC-AUC"]*100]
            ax_cmp.bar(x+i*w,vals,w,label=row["Modelo"],color=PALETTE[i],edgecolor="#0f0f1a",alpha=.9)
            for j,v in enumerate(vals):
                ax_cmp.text(x[j]+i*w,v+0.5,f"{v:.1f}",ha="center",fontsize=7.5,color="#e0e0ff")
        ax_cmp.set_xticks(x+(w*(len(rows_comp)-1)/2)); ax_cmp.set_xticklabels(METRICAS)
        ax_cmp.set_ylim(0,115); ax_cmp.set_title(f"Comparación — Clase: {clase_sel}"); ax_cmp.legend()
        fig_cmp.tight_layout(); st.pyplot(fig_cmp)

        st.markdown("---")
        section_header("🏆","Ganador por Moda (Victorias por Métrica)")
        wins={r["Modelo"]:0 for r in rows_comp}
        detail=[]
        met_cols=["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]
        for mc in met_cols:
            vals_mc={r["Modelo"]:r[mc] for r in rows_comp}
            winner=max(vals_mc,key=vals_mc.get)
            wins[winner]+=1
            detail.append({"Métrica":mc,"Ganador":winner,"Valor":vals_mc[winner]})
        detail_df=pd.DataFrame(detail); st.dataframe(detail_df,use_container_width=True)

        wins_df=pd.DataFrame(list(wins.items()),columns=["Modelo","Victorias"]).sort_values("Victorias",ascending=False)
        plot_style(); fig_w,ax_w=plt.subplots(figsize=(7,3.5))
        ax_w.bar(wins_df["Modelo"],wins_df["Victorias"],color=PALETTE[:len(wins_df)],edgecolor="#0f0f1a",width=0.5)
        for i,v in enumerate(wins_df["Victorias"]):
            ax_w.text(i,v+0.05,str(v),ha="center",fontsize=12,color="#e0e0ff",fontweight="bold")
        ax_w.set_ylabel("Victorias"); ax_w.set_title("Victorias por Modelo")
        fig_w.tight_layout(); st.pyplot(fig_w)
        best_model=wins_df.iloc[0]["Modelo"]
        st.markdown(f'<div class="success-box">🏆 Mejor modelo: <b>{best_model}</b> con <b>{wins_df.iloc[0]["Victorias"]}</b> victorias de {len(met_cols)}</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # ENSAMBLE
    # ══════════════════════════════════════════════════════════
    avail_ens = {k:st.session_state.get(k) for k in ["results_lr","results_knn","results_rf"] if st.session_state.get(k)}
    # Excluir modelos de regresión del ensamble (VotingClassifier solo funciona con clasificadores)
    avail_ens = {k:v for k,v in avail_ens.items() if not v.get("is_regression", False)}

    if len(avail_ens)>=2:
        st.markdown("---")
        section_header("🔗","Ensamble de Modelos")
        st.markdown('<div class="info-box">Combina los modelos entrenados en un clasificador ensemble. Necesitas que todos usen las <b>mismas variables X e Y</b> y el mismo conjunto de datos. Solo disponible para modelos de <b>clasificación</b>.</div>', unsafe_allow_html=True)

        modelos_disp={
            "🔵 Regresión Logística":"results_lr",
            "🟣 KNN":"results_knn",
            "🌲 Random Forest":"results_rf",
        }
        modelos_sel=st.multiselect("Modelos a incluir en el ensamble",
            [k for k,v in modelos_disp.items() if st.session_state.get(v) and not st.session_state.get(v,{}).get("is_regression",False)],
            default=[k for k,v in modelos_disp.items() if st.session_state.get(v) and not st.session_state.get(v,{}).get("is_regression",False)],key="ens_sel")

        tipo_ens=st.radio("Tipo de votación",["Hard Voting (mayoría de votos)","Soft Voting (promedio probabilidades)","Votación Ponderada"],horizontal=True,key="tipo_ens")

        pesos={}
        if "Ponderada" in tipo_ens:
            st.markdown("**Asigna peso a cada modelo (0.0 – 5.0):**")
            cp=st.columns(len(modelos_sel))
            for i,m in enumerate(modelos_sel):
                pesos[m]=cp[i].slider(m.split(" ",1)[1],0.1,5.0,1.0,0.1,key=f"peso_{i}")

        ec1,ec2=st.columns(2)
        dep_ens=ec1.selectbox("Variable Y del ensamble",all_cols_ds,key="dep_ens")
        feat_ens=ec2.multiselect("Variables X del ensamble",
            [c for c in num_cols_ds if c!=dep_ens],
            default=[c for c in num_cols_ds if c!=dep_ens][:min(5,len(num_cols_ds))],key="feat_ens")
        ts_ens=st.slider("% Test",10,40,20,key="ts_ens")/100

        run_ens=st.button("🚀 Ejecutar Ensamble",type="primary",key="btn_ens")
        if run_ens:
            if len(modelos_sel)<2: st.error("Selecciona al menos 2 modelos.")
            elif not feat_ens: st.error("Selecciona variables X.")
            elif not check_y_discreta(df_proc[dep_ens],dep_ens): pass
            else:
                try:
                    dm=df_proc[feat_ens+[dep_ens]].dropna()
                    X=dm[feat_ens].values; y=dm[dep_ens].values
                    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=ts_ens,random_state=42,stratify=y)
                    sc=StandardScaler(); Xtr=sc.fit_transform(Xtr); Xte=sc.transform(Xte)

                    name_map={"🔵 Regresión Logística":("lr",LogisticRegression(max_iter=1000,random_state=42)),
                               "🟣 KNN":("knn",KNeighborsClassifier()),
                               "🌲 Random Forest":("rf",RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1))}
                    estimators=[(name_map[m][0],name_map[m][1]) for m in modelos_sel if m in name_map]

                    if "Ponderada" in tipo_ens:
                        w_list=[pesos.get(m,1.0) for m in modelos_sel if m in name_map]
                        vc=VotingClassifier(estimators=estimators,voting="soft",weights=w_list)
                    elif "Soft" in tipo_ens:
                        vc=VotingClassifier(estimators=estimators,voting="soft")
                    else:
                        vc=VotingClassifier(estimators=estimators,voting="hard")

                    vc.fit(Xtr,ytr); y_pred_ens=vc.predict(Xte)
                    is_bin_ens=len(np.unique(ytr))==2
                    try:
                        if "hard" not in str(vc.voting).lower():
                            yprob_ens=vc.predict_proba(Xte)[:,1] if is_bin_ens else vc.predict_proba(Xte)
                        else: yprob_ens=None
                    except: yprob_ens=None

                    res_ens=compute_metrics(yte,y_pred_ens,yprob_ens if (is_bin_ens and yprob_ens is not None) else None,"Ensamble",is_bin_ens)
                    res_ens["model"]=vc; res_ens["feature_names"]=feat_ens

                    section_header("📊","Resultados del Ensamble")
                    show_metrics_cards(res_ens,threshold=0.5)

                    section_header("📊","Ensamble vs Modelos Individuales")
                    rows_e=[{"Modelo":res_ens["name"],"Accuracy":round(res_ens["accuracy"]*100,2),
                        "F1 (macro)":round(res_ens["f1"]*100,2),"ROC-AUC":round(res_ens["auc"],4) if res_ens["auc"] else "N/A"}]
                    for k2,v2 in avail_ens.items():
                        rows_e.append({"Modelo":v2["name"],"Accuracy":round(v2["accuracy"]*100,2),
                            "F1 (macro)":round(v2["f1"]*100,2),"ROC-AUC":round(v2["auc"],4) if v2["auc"] else "N/A"})
                    st.dataframe(pd.DataFrame(rows_e),use_container_width=True)

                    plot_style(); fig_e,ax_e=plt.subplots(figsize=(10,4))
                    mods_e=[r["Modelo"] for r in rows_e]; acc_e=[r["Accuracy"] for r in rows_e]; f1_e=[r["F1 (macro)"] for r in rows_e]
                    xe=np.arange(len(mods_e)); we=0.35
                    ax_e.bar(xe-we/2,acc_e,we,label="Accuracy",color="#4f46e5",edgecolor="#0f0f1a")
                    ax_e.bar(xe+we/2,f1_e,we,label="F1 Macro",color="#7c3aed",edgecolor="#0f0f1a")
                    ax_e.set_xticks(xe); ax_e.set_xticklabels(mods_e,rotation=15,ha="right")
                    ax_e.set_ylim(0,115); ax_e.set_title("Ensamble vs Individuales"); ax_e.legend()
                    for i,(a,f) in enumerate(zip(acc_e,f1_e)):
                        ax_e.text(i-we/2,a+.5,f"{a:.1f}",ha="center",fontsize=8,color="#e0e0ff")
                        ax_e.text(i+we/2,f+.5,f"{f:.1f}",ha="center",fontsize=8,color="#e0e0ff")
                    fig_e.tight_layout(); st.pyplot(fig_e)
                except Exception as e: st.error(f"Error en ensamble: {e}")