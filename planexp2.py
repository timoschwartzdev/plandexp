
# app.py
# ============================================================
# Application DOE (Design of Experiments) - Streamlit
# Import -> Modélisation -> Visualisation -> ANOVA -> Optimisation
# ============================================================





import io
import itertools
import numpy as np
import pandas as pd
import streamlit as st

from typing import Dict, List, Tuple

# Modélisation & stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Visualisation
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# INIT SESSION_STATE (fixe toutes les erreurs "attribute missing")
# ============================================================
if "X_cols" not in st.session_state:
    st.session_state.X_cols = []

if "Y_cols" not in st.session_state:
    st.session_state.Y_cols = []

if "categorical_cols" not in st.session_state:
    st.session_state.categorical_cols = []

if "example_file" not in st.session_state:
    st.session_state.example_file = None

# =========================
# CONFIG GÉNÉRALE
# =========================
st.set_page_config(
    page_title="DOE - Analyse & Optimisation",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 DOE — Analyse, ANOVA & Optimisation multi‑objectif")
st.caption("Importez vos données, ajustez un modèle, visualisez les effets, faites une ANOVA et optimisez vos réponses.")


# =========================
# OUTILS / FONCTIONS
# =========================


def read_file(file_name: str, file_bytes: bytes, header_rows: list) -> pd.DataFrame:
    import io
    import pandas as pd

    if file_name.endswith(".csv"):
        df_raw = pd.read_csv(io.BytesIO(file_bytes), header=None)
    else:
        df_raw = pd.read_excel(
            io.BytesIO(file_bytes),
            header=None,
            engine="openpyxl",
            na_values=["#N/A", "#DIV/0!", "#VALUE!", "#REF!", "#NUM!"]
            keep_errors=False
        )

    header_part = df_raw.iloc[header_rows]

    new_columns = []
    for col in range(len(header_part.columns)):
        values = header_part.iloc[:, col].astype(str).replace("nan", "").tolist()
        new_columns.append("_".join(v for v in values if v.strip()))

    df = df_raw.drop(header_rows).reset_index(drop=True)
    df.columns = new_columns

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    return df



def guess_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Devine facteurs (X) et réponses (Y) basés sur les types."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Heuristique simple : toutes les numériques potentielles -> réponses par défaut vides
    return df.columns.tolist(), numeric_cols

def build_sklearn_model(model_type: str, degree: int = 2, standardize: bool = True) -> Pipeline:
    """Construit un pipeline sklearn selon le choix."""
    steps = []
    if standardize:
        steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))
    if model_type == "Régression polynomiale":
        steps.append(("poly", PolynomialFeatures(degree=degree, include_bias=False)))
    steps.append(("linreg", LinearRegression()))
    return Pipeline(steps)

def feature_names_after_poly(X_cols: List[str], pipeline: Pipeline) -> List[str]:
    """Reconstruit les noms de features après PolynomialFeatures si présent."""
    if "poly" in dict(pipeline.named_steps):
        poly: PolynomialFeatures = pipeline.named_steps["poly"]
        return poly.get_feature_names_out(X_cols).tolist()
    return X_cols

def model_equation(pipeline: Pipeline, X_cols: List[str]) -> str:
    """Crée une équation lisible du modèle (approx. y = b0 + Σ bi*xi + ...)."""
    lin: LinearRegression = pipeline.named_steps["linreg"]
    coef = lin.coef_.ravel()
    intercept = float(lin.intercept_.ravel()[0] if np.ndim(lin.intercept_) > 0 else lin.intercept_)
    names = feature_names_after_poly(X_cols, pipeline)
    terms = [f"{coef[i]:+.4f}·{names[i]}" for i in range(len(coef))]
    eq = f"ŷ = {intercept:+.4f} " + " ".join(terms)
    return eq

def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "R²": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
    }

def main_effect_plot(df: pd.DataFrame, factor: str, response: str, bins: int = 4):
    """Trace d'effet principal (moyenne de Y par niveau/intervalle de X)."""
    if pd.api.types.is_numeric_dtype(df[factor]):
        # Binning quantiles pour lisibilité
        binned = pd.qcut(df[factor], q=bins, duplicates="drop")
        plot_df = df.groupby(binned, observed=False)[response].mean().reset_index()
        plot_df[factor] = plot_df[factor].astype(str)
    else:
        plot_df = df.groupby(factor, observed=False)[response].mean().reset_index()
    fig = px.bar(plot_df, x=factor, y=response, title=f"Effet principal — {factor} sur {response}")
    st.plotly_chart(fig, use_container_width=True)



def interaction_plot_numeric_numeric(model, factors, response, df, grid_size=30):

    x1, x2 = factors

    # 1) grille sur les deux facteurs numériques
    x1_lin = np.linspace(df[x1].min(), df[x1].max(), grid_size)
    x2_lin = np.linspace(df[x2].min(), df[x2].max(), grid_size)
    X1, X2 = np.meshgrid(x1_lin, x2_lin)

    # dataframe grille brute
    grid = pd.DataFrame({x1: X1.ravel(), x2: X2.ravel()})

    # 2) Ajouter les colonnes manquantes (catégorielles one-hot)
    # base = colonnes utilisées pendant le fit
    base_cols = model.feature_names_in_

    for col in base_cols:
        if col not in grid.columns:
            # si colonne dummy manquante → valeur neutre (0)
            grid[col] = 0

    # 3) Réordonner les colonnes dans l’ordre du modèle
    grid = grid[base_cols]

    # 4) prédiction
    yhat = model.predict(grid)
    Z = yhat.reshape(X1.shape)

    # --- tracé plotly (inchangé) ---
    surface = go.Surface(x=X1, y=X2, z=Z, colorscale="Viridis", showscale=True)
    fig3d = go.Figure(data=[surface])
    fig3d.update_layout(
        title=f"Surface de réponse prédite — {response}",
        scene=dict(xaxis_title=x1, yaxis_title=x2, zaxis_title=response),
        height=550
    )
    st.plotly_chart(fig3d, use_container_width=True)



def compute_pareto_front(points: np.ndarray, minimize: List[bool]) -> np.ndarray:
    """
    Calcule le front de Pareto (indices non-dominés).
    points: (n, m) ; minimize[i]=True si l'objectif i est à minimiser.
    """
    n = points.shape[0]
    dominated = np.zeros(n, dtype=bool)
    # Convertir minimisation en maximisation pour logique unique
    # On transforme chaque objectif en "maximiser" en multipliant par -1 si minimise
    P = points.copy()
    for j, m in enumerate(minimize):
        if m:
            P[:, j] = -P[:, j]
    for i in range(n):
        if dominated[i]:
            continue
        # i est dominé s'il existe k tel que P[k] >= P[i] sur tous critères et > sur au moins un
        mask_better_or_equal = (P >= P[i]).all(axis=1)
        mask_strict_better = (P > P[i]).any(axis=1)
        dominated |= (mask_better_or_equal & mask_strict_better)
        dominated[i] = False  # ne pas s'auto-dominer
    return np.where(~dominated)[0]


# =========================
# SIDEBAR — Contrôles globaux
# =========================
with st.sidebar:
    st.header("⚙️ Paramètres généraux")

    st.markdown("**1) Importer vos données** (CSV/XLSX)")
    # On garde ton nom de variable : upload
    upload = st.file_uploader("Déposez un fichier", type=["csv", "xlsx", "xls"])



   
# Jeu de données d'exemple si besoin
if st.button("📄 Charger un exemple synthétique"):
    # Exemple : DOE 2 facteurs numériques + 1 catégoriel, 2 réponses
    n = 120
    rng = np.random.default_rng(42)
    A = rng.uniform(-1, 1, n)             # facteur numérique
    B = rng.uniform(0, 10, n)             # facteur numérique
    C = rng.choice(["Bas", "Haut"], n)    # facteur catégoriel
    # Génération réponses avec interactions + bruit
    y1 = 3 + 1.2*A - 0.3*B + 0.8*A*B + (C == "Haut")*1.5 + rng.normal(0, 0.6, n)
    y2 = 10 - 0.9*A + 0.5*B - 0.6*A*B - (C == "Haut")*1.0 + rng.normal(0, 0.8, n)
    df_example = pd.DataFrame({"A": A, "B": B, "C": C, "Y1": y1, "Y2": y2})

    buffer = io.BytesIO()
    df_example.to_csv(buffer, index=False)
    buffer.seek(0)

    # On stocke (nom, bytes) dans la session
    st.session_state["example_file"] = {
        "name": "example.csv",
        "bytes": buffer.getvalue()
    }
    st.success("Exemple chargé (A, B, C → Y1, Y2).")


    st.divider()
    st.caption("Besoin d'aide pour l'installation ? `python -m pip install streamlit pandas numpy scikit-learn statsmodels plotly openpyxl`")




# ============================================================
# 1) IMPORT ET CONFIGURATION DES DONNÉES
# ============================================================
df: pd.DataFrame = None
file_name, file_bytes = None, None

# Récupération de la source (Upload ou Exemple)
if upload is not None:
    try:
        file_name = upload.name
        file_bytes = upload.getvalue()
    except Exception as e:
        st.error(f"Erreur lors de la récupération du fichier : {e}")
elif st.session_state.example_file:
    file_name = st.session_state.example_file["name"]
    file_bytes = st.session_state.example_file["bytes"]

# Si un fichier est détecté, on affiche les options AVANT de lire
if file_name and file_bytes:
    st.subheader("1) Configuration de l'import")
    
    col_opt, _ = st.columns([2, 1]) 

    with col_opt:
        sub_col1, sub_col2 = st.columns([3, 1])
        with sub_col1:
            has_header = st.checkbox("Le fichier possède des en-têtes", value=True)
        with sub_col2:
            if has_header:
                nb_rows = st.number_input("Lignes", min_value=1, max_value=5, value=2, step=1, label_visibility="collapsed")
            else:
                nb_rows = 0

    # 1. On prépare la valeur dynamique
    header_val = list(range(nb_rows)) if has_header else None

    try:
        # CORRECTION : On utilise file_name et file_bytes (les variables sources)
        # au lieu de forcer 'upload.name' qui est vide si on utilise l'exemple.
        df = read_file(file_name, file_bytes, header_rows=header_val)
        
        import re
        def clean_colname(name):
            name = str(name)
            name = re.sub(r"\[.*?\]", "", name); name = re.sub(r"\(.*?\)", "", name)
            name = re.sub(r"[^0-9a-zA-Z_]", "_", name); name = re.sub(r"_+", "_", name)
            return name.strip("_")

        if not has_header:
            df.columns = [f"Col_{i}" for i in range(len(df.columns))]
        else:
            df.columns = [clean_colname(c) for c in df.columns]
            st.info("✔️ Colonnes importées et nettoyées.")

        st.session_state['df'] = df

    except Exception as e:
        st.error(f"Erreur de lecture du fichier : {e}")

# Arrêt si aucun fichier n'est chargé
if df is None:
    st.info("➡️ Importez un fichier (CSV/XLSX) ou utilisez l'exemple dans la barre latérale.")
    st.stop()

# Affichage de l'aperçu
st.markdown("### Aperçu des données")
st.dataframe(df.head(), use_container_width=True)

with st.expander("🛠️ Options de nettoyage (recommandé si Excel FR)"):
    auto_coerce = st.checkbox("Convertir automatiquement les colonnes numériques au format texte (virgules → points)", value=True)
    cols_to_try = st.multiselect(
        "Colonnes à convertir en numérique (laisser vide = tentative sur toutes les colonnes candidates)",
        options=df.columns.tolist(),
        help="Utile si vos nombres sont en texte ou avec des virgules décimales."
    )

def coerce_numeric_columns(df_in: pd.DataFrame, only_cols=None) -> pd.DataFrame:
    df_out = df_in.copy()
    candidates = list(only_cols) if only_cols else df_out.columns.tolist()
    for c in candidates:
        # On tente uniquement si la colonne n'est pas déjà numérique
        if not pd.api.types.is_numeric_dtype(df_out[c]):
            # remplace virgule décimale par point, retire espaces fines, etc.
            series = (
                df_out[c]
                .astype(str)
                .str.replace("\u202f", "", regex=False)  # espace fine (Excel FR)
                .str.replace(" ", "", regex=False)
                .str.replace(",", ".", regex=False)
            )
            coerced = pd.to_numeric(series, errors="coerce")
            # si on a converti au moins quelques valeurs, on garde
            if coerced.notna().sum() >= max(3, int(0.3 * len(coerced))):
                df_out[c] = coerced
    return df_out

if auto_coerce:
    df = coerce_numeric_columns(df, only_cols=cols_to_try if cols_to_try else None)


if df is None:
    st.info("➡️ Importez un fichier (CSV/XLSX) ou utilisez l'exemple dans la barre latérale.")
    st.stop()

st.subheader("1) Import & sélection des variables")
st.dataframe(df.head(), use_container_width=True)


all_cols = df.columns.tolist()
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Heuristiques pour Y par défaut :
#  - noms qui commencent par 'y' (insensible à la casse)
#  - sinon les dernières numériques
y_name_guess = [c for c in all_cols if c.lower().strip().startswith("y")]
default_y = [c for c in y_name_guess if c in numeric_cols] or (numeric_cols[-2:] if len(numeric_cols) >= 2 else numeric_cols[-1:] if numeric_cols else [])

# X par défaut = tout sauf Y par défaut
default_x = [c for c in all_cols if c not in default_y]

left, right = st.columns(2)

with left:
    X_cols = st.multiselect(
        "Facteurs (X)",
        options=all_cols,
        default=st.session_state.get("X_cols", default_x),
        key="X_cols",
        help="Colonnes d'entrée (facteurs)."
    )

# Pour les réponses, on propose en priorité les numériques non déjà dans X,
# mais ON AUTORISE aussi de sélectionner des colonnes non-numériques (puis conversion).
y_options_prior = [c for c in all_cols if c not in st.session_state.X_cols]
# Si possible, on place en tête les colonnes numériques
y_options = sorted(y_options_prior, key=lambda c: (c not in numeric_cols, c))

with right:
    Y_cols = st.multiselect(
        "Réponses (Y)",
        options=y_options,
        default=[c for c in st.session_state.get("Y_cols", default_y) if c in y_options] or (default_y if set(default_y).issubset(y_options) else []),
        key="Y_cols",
        help="Choisissez au moins une réponse. Si elle n'est pas numérique, l'appli tentera de la convertir."
    )

# Conversion de sécurité pour Y sélectionnées (si elles ne sont pas encore numériques)
for yc in st.session_state.Y_cols:
    if not pd.api.types.is_numeric_dtype(df[yc]):
        try:
            df[yc] = pd.to_numeric(
                df[yc].astype(str).str.replace("\u202f", "", regex=False).str.replace(" ", "", regex=False).str.replace(",", ".", regex=False),
                errors="coerce"
            )
        except Exception:
            pass  # on laissera l'étape de modélisation signaler le problème si besoin

# Garde-fous
if not st.session_state.X_cols:
    st.warning("Sélectionnez au moins un facteur (X).")
    st.stop()
if not st.session_state.Y_cols:
    st.warning("Sélectionnez au moins une réponse (Y).")
    st.stop()





# ============================================================
# INIT SESSION_STATE (évite "has no attribute" & valeurs vides)
# ============================================================
if "X_cols" not in st.session_state:
    st.session_state.X_cols = []
if "Y_cols" not in st.session_state:
    st.session_state.Y_cols = []
if "categorical_cols" not in st.session_state:
    st.session_state.categorical_cols = []
if "example_file" not in st.session_state:
    st.session_state.example_file = None





st.subheader("2) Modélisation & ANOVA")

# Si jamais X ou Y sont vides (ne devrait pas arriver grâce au bloc précédent)
if not st.session_state.X_cols or not st.session_state.Y_cols:
    st.warning("Sélectionnez au moins 1 facteur (X) et 1 réponse (Y).")
    st.stop()

# Choix du modèle sklearn
cols = st.columns([1, 1, 1, 1])
with cols[0]:
    model_type = st.selectbox("Type de modèle (sklearn)", ["Régression linéaire", "Régression polynomiale"])
with cols[1]:
    degree = st.slider("Degré (si polynomial)", min_value=2, max_value=4, value=2, step=1)
with cols[2]:
    standardize = st.checkbox("Standardiser les facteurs", value=True)
with cols[3]:
    add_interactions = st.checkbox("Inclure interactions (ANOVA/formule)", value=True, help="Uniquement côté statsmodels.")

st.markdown("### 🎯 Modélisation spécifique")

# On vérifie d'abord si la liste n'est pas vide pour éviter une erreur d'index
if not st.session_state.Y_cols:
    st.warning("⚠️ Aucune réponse (Y) sélectionnée plus haut. Veuillez en choisir au moins une dans l'étape 1.")
    st.stop()

# Un seul selectbox avec une clé unique
y_target = st.selectbox(
    "Quelle réponse souhaitez-vous analyser précisément ?",
    options=st.session_state.Y_cols,
    index=0,
    key="y_target_final",
    help="Le modèle mathématique et l'ANOVA seront calculés pour cette variable."
)

# Sécurité supplémentaire pour les facteurs
if not st.session_state.X_cols:
    st.warning("⚠️ Sélectionnez au moins un facteur (X) dans l'étape 1.")
    st.stop()



# Préparation X, y
X = df[st.session_state.X_cols].copy()
y = df[y_target].values.reshape(-1, 1)

# Encodage simple pour facteurs catégoriels côté sklearn : one-hot
X_sklearn = pd.get_dummies(X, drop_first=True)
feature_base_names = X_sklearn.columns.tolist()

# Modèle sklearn
pipe = build_sklearn_model(model_type, degree=degree, standardize=standardize)
pipe.fit(X_sklearn, y.ravel())
yhat = pipe.predict(X_sklearn).ravel()

# Affichage des métriques
m = metrics(y_true=y.ravel(), y_pred=yhat)
left, right = st.columns([1, 2])
with left:
    st.metric("R²", f"{m['R²']:.4f}")
    st.metric("RMSE", f"{m['RMSE']:.4f}")
    st.metric("MAE", f"{m['MAE']:.4f}")
with right:
    st.markdown("**Équation (approx.)**")
    st.code(model_equation(pipe, feature_base_names), language="text")

# Importances (permutation) pour classer les facteurs
try:
    imp = permutation_importance(pipe, X_sklearn, y.ravel(), n_repeats=10, random_state=0)
    importances = pd.DataFrame({
        "Facteur": feature_base_names,
        "Importance (moy.)": imp.importances_mean,
        "Std": imp.importances_std
    }).sort_values("Importance (moy.)", ascending=False)
    st.markdown("**Importance des facteurs (Permutation Importance)**")
    st.dataframe(importances, use_container_width=True)
    fig_imp = px.bar(importances, x="Facteur", y="Importance (moy.)", error_y="Std", title="Classement des facteurs")
    st.plotly_chart(fig_imp, use_container_width=True)
except Exception as e:
    st.info(f"Permutation importance non disponible : {e}")

st.divider()

# =========================
# ANOVA avec statsmodels
# =========================
st.markdown("### 📈 ANOVA (statsmodels)")

# Construction formule : Y ~ (X1 + X2 + ...)*(si interactions)
def build_formula(y_col: str, X_cols: List[str], categorical: List[str], add_inter: bool) -> str:
    def term(c: str) -> str:
        return f"C({c})" if c in categorical else c
    base = " + ".join([term(c) for c in X_cols])
    if add_inter and len(X_cols) > 1:
        # Terme d'interaction globale : (X1 + X2 + ...)^2
        # En formule patsy: (a + b + c) ** 2 = main effects + pairwise interactions
        inter = " + ".join([term(c) for c in X_cols])
        return f"{y_col} ~ ({inter})**2"
    else:
        return f"{y_col} ~ {base}"

anova_tabs = st.tabs([f"ANOVA — {y}" for y in st.session_state.Y_cols])
for tab, y_col in zip(anova_tabs, st.session_state.Y_cols):
    with tab:
        formula = build_formula(y_col, st.session_state.X_cols, st.session_state.categorical_cols, add_interactions)
        st.code(f"Formule: {formula}", language="text")
        try:
            model = ols(formula, data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            st.dataframe(anova_table, use_container_width=True)
            st.markdown("**Résumé du modèle (statsmodels)**")
            st.text(model.summary())
        except Exception as e:
            st.error(f"Échec ANOVA pour {y_col} : {e}")


# =========================
# 3) VISUALISATIONS
# =========================
st.subheader("3) Visualisations — Effets & Interactions")

vcols = st.columns([1, 1, 1])
with vcols[0]:
    resp_viz = st.selectbox("Réponse à visualiser", options=st.session_state.Y_cols, index=0)
with vcols[1]:
    factor_main = st.selectbox("Effet principal — facteur", options=st.session_state.X_cols)
with vcols[2]:
    bins = st.slider("Nombre d'intervalles (si facteur numérique)", 3, 10, 5)

# Effet principal
main_effect_plot(df, factor=factor_main, response=resp_viz, bins=bins)

# Interactions
st.markdown("#### Interactions")
if len(st.session_state.X_cols) >= 2:
    f1, f2 = st.multiselect("Choisir 2 facteurs pour interaction (surface si num-num)", options=st.session_state.X_cols, default=st.session_state.X_cols[:2])
    if len(f1) > 0 and len(f2) > 0 and f1 != f2:
        # Cas num-num : surface via modèle sklearn
        if pd.api.types.is_numeric_dtype(df[f1]) and pd.api.types.is_numeric_dtype(df[f2]):
            interaction_plot_numeric_numeric(pipe, [f1, f2], resp_viz, df)
        else:
            # Interaction catégoriel/numérique : lignes
            tmp = df[[f1, f2, resp_viz]].copy()
            # Assurer que l'un est num et l'autre cat
            num = f1 if pd.api.types.is_numeric_dtype(df[f1]) else f2
            cat = f2 if num == f1 else f1
            gdf = tmp.groupby([cat, pd.qcut(tmp[num], q=6, duplicates="drop")], observed=False)[resp_viz].mean().reset_index()
            gdf[num] = gdf[num].astype(str)
            fig = px.line(gdf, x=num, y=resp_viz, color=cat, markers=True,
                          title=f"Interaction {cat} × {num} sur {resp_viz}")
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Sélectionnez au moins 2 facteurs pour explorer les interactions.")


# =========================
# 4) OPTIMISATION MULTI-OBJECTIF
# =========================
st.subheader("4) Optimisation multi‑objectif (pondération)")

# Choix des réponses et objectifs
if len(st.session_state.Y_cols) >= 2:
    opt_cols = st.multiselect("Réponses à optimiser (≥ 2)", options=st.session_state.Y_cols, default=st.session_state.Y_cols[:2])
else:
    opt_cols = st.multiselect("Réponses à optimiser (≥ 2)", options=st.session_state.Y_cols, default=st.session_state.Y_cols)

if len(opt_cols) < 2:
    st.info("Choisissez au moins deux réponses pour l'optimisation multi‑objectif.")
else:
    # Ajuster un modèle pour chaque réponse (sur la même base X_sklearn)
    models_by_y: Dict[str, Pipeline] = {}
    for yc in opt_cols:
        yi = df[yc].values.reshape(-1, 1)
        m_i = build_sklearn_model(model_type, degree=degree, standardize=standardize)
        m_i.fit(X_sklearn, yi.ravel())
        models_by_y[yc] = m_i

    # Objectifs (max/min/target) + poids
    goals = {}
    weights = {}
    targets = {}
    st.markdown("**Objectifs & poids**")
    for yc in opt_cols:
        cols_goal = st.columns([1, 1, 1])
        with cols_goal[0]:
            g = st.selectbox(f"Objectif pour {yc}", ["Maximiser", "Minimiser", "Cibler une valeur"], key=f"goal_{yc}")
        with cols_goal[1]:
            w = st.slider(f"Poids {yc}", 0.0, 1.0, 1.0/len(opt_cols), 0.05, key=f"w_{yc}")
        with cols_goal[2]:
            t = st.number_input(f"Cible {yc} (si ciblage)", value=float(df[yc].median()), key=f"t_{yc}")
        goals[yc] = g
        weights[yc] = w
        targets[yc] = t

    # Contraintes/ranges sur les facteurs numériques
    st.markdown("**Domaines de recherche (grille) pour les facteurs numériques**")
    num_factors = [c for c in st.session_state.X_cols if pd.api.types.is_numeric_dtype(df[c])]
    ranges = {}
    grid_res = st.slider("Résolution de la grille (par facteur numérique)", 10, 60, 25, 5)
    for nf in num_factors:
        c1, c2 = st.columns(2)
        with c1:
            vmin = st.number_input(f"{nf} min", value=float(df[nf].min()))
        with c2:
            vmax = st.number_input(f"{nf} max", value=float(df[nf].max()))
        if vmax <= vmin:
            st.warning(f"Plage invalide pour {nf}.")
        ranges[nf] = (vmin, vmax)

    # Modalités fixes pour les catégorielles (on optimise sur une modalité à la fois)
    cat_factors = [c for c in st.session_state.X_cols if c not in num_factors]
    cat_choices = {}
    for cf in cat_factors:
        cat_choices[cf] = st.selectbox(f"Modalité pour {cf}", options=sorted(df[cf].dropna().unique().tolist()))

    if st.button("🚀 Lancer l'optimisation"):
        # Grille de recherche
        grids = [np.linspace(ranges[nf][0], ranges[nf][1], grid_res) for nf in num_factors]
        mesh = list(itertools.product(*grids)) if grids else [()]
        candidates = []
        for point in mesh:
            row = {}
            for i, nf in enumerate(num_factors):
                row[nf] = point[i]
            for cf in cat_factors:
                row[cf] = cat_choices[cf]
            candidates.append(row)
        cand_df = pd.DataFrame(candidates) if candidates else pd.DataFrame(columns=st.session_state.X_cols)

        if cand_df.empty:
            st.error("Aucun candidat généré (vérifiez vos plages).")
        else:
            # Préparer design candidat pour sklearn (one hot)
            cand_X = pd.get_dummies(cand_df[st.session_state.X_cols], drop_first=True)
            # Align columns with training matrix
            cand_X = cand_X.reindex(columns=X_sklearn.columns, fill_value=0.0)

            # Prédire toutes les réponses
            preds = {}
            for yc in opt_cols:
                preds[yc] = models_by_y[yc].predict(cand_X).ravel()
            P = pd.DataFrame(preds)

            # Normalisation et score multi‑objectif (0..1)
            norm = pd.DataFrame(index=P.index)
            for yc in opt_cols:
                if goals[yc] == "Cibler une valeur":
                    # Distance à la cible → plus proche = mieux
                    dist = np.abs(P[yc] - targets[yc])
                    # inverser et normaliser
                    # on normalise par le max pour être dans [0,1] (si max=0 -> tout égal)
                    dmax = dist.max()
                    score = 1.0 - (dist / dmax if dmax > 0 else 0.0)
                    norm[yc] = score.fillna(0.0)
                else:
                    # Max / Min → MinMax scaling vers [0,1]
                    vmin, vmax = P[yc].min(), P[yc].max()
                    if vmax == vmin:
                        norm[yc] = 0.5
                    else:
                        if goals[yc] == "Maximiser":
                            norm[yc] = (P[yc] - vmin) / (vmax - vmin)
                        else:  # Minimiser
                            norm[yc] = (vmax - P[yc]) / (vmax - vmin)

            # Score pondéré
            wsum = sum(weights.values()) or 1.0
            score = sum(weights[yc] * norm[yc] for yc in opt_cols) / wsum

            best_idx = int(score.idxmax())
            best_point = cand_df.loc[best_idx, st.session_state.X_cols]
            best_preds = P.loc[best_idx, opt_cols]

            st.success("✅ Meilleure solution trouvée (pondération).")
            st.write("**Facteurs optimaux :**")
            st.json(best_point.to_dict())
            st.write("**Réponses prédites :**")
            st.json(best_preds.to_dict())

            # Affichage Pareto (si 2 réponses)
            if len(opt_cols) == 2:
                # Minimisation flags pour calcul Pareto
                minimize_flags = [goals[opt_cols[0]] == "Minimiser", goals[opt_cols[1]] == "Minimiser"]
                pareto_idx = compute_pareto_front(P[opt_cols].values, minimize=minimize_flags)
                P["pareto"] = False
                P.loc[pareto_idx, "pareto"] = True
                figp = px.scatter(
                    P, x=opt_cols[0], y=opt_cols[1], color="pareto",
                    title="Front de Pareto (prédictions sur grille)",
                    labels={opt_cols[0]: opt_cols[0], opt_cols[1]: opt_cols[1]}
                )
                # Marquer la meilleure solution pondérée
                figp.add_trace(go.Scatter(
                    x=[best_preds[opt_cols[0]]], y=[best_preds[opt_cols[1]]],
                    mode="markers", marker=dict(size=14, color="gold", line=dict(color="black", width=1.5)),
                    name="Meilleur compromis (pondéré)"
                ))
                st.plotly_chart(figp, use_container_width=True)


# =========================
# 5) CONCLUSION / AIDE
# =========================
with st.expander("📚 Conseils d'utilisation & rappels"):
    st.markdown("""
- **Import** : CSV/XLSX avec colonnes facteurs (X) et réponses (Y).
- **Modélisation** : choisissez régression linéaire ou polynomiale (degré 2-4), avec standardisation.  
  L'équation affichée correspond au modèle sklearn (avec éventuels termes polynomiaux).
- **Importance des facteurs** : le *Permutation Importance* donne un classement robuste des facteurs influents.
- **ANOVA** : via `statsmodels` (Type II). Cochez vos variables catégorielles pour utiliser `C(var)` et, si souhaité, ajoutez des interactions.
- **Visualisations** : 
  - Effets principaux (moyennes par niveau/quantiles).
  - Interactions : surface/contours pour 2 facteurs numériques ; lignes pour cas mixte.
- **Optimisation multi‑objectif** :
  - Définissez objectifs (max/min/cible) et poids.
  - La recherche se fait par grille sur les facteurs **numériques** (les catégoriels sont fixés).
  - Affiche un meilleur compromis pondéré + (si 2 réponses) le front de Pareto.

ℹ️ **Astuces** :
- Si vos facteurs sont codés (-1 / +1), conservez ce codage (c'est idéal pour l'analyse).
- Pour Excel, assurez-vous d'avoir `openpyxl` installé.
- Lancez toujours l'appli avec : `python -m streamlit run app.py`.
    """)

