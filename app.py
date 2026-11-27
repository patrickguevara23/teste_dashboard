"""
Dashboard de Vigilância Epidemiológica – Dengue
-----------------------------------------------
Casos confirmados de dengue x indicadores climáticos (granularidade mensal).

Execução:
    streamlit run app.py
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# =============================================================================
# CONSTANTES
# =============================================================================
ARQUIVO_PADRAO = "dengue-dataset.csv"

COL_DATA = "data"
COL_CASOS = "casos-confirmados"
COL_CHUVA = "chuva"
COL_TMED = "temperatura-media"
COL_TMIN = "temperatura-minima"
COL_TMAX = "temperatura-maxima"

COLUNAS_OBRIGATORIAS = [COL_DATA, COL_CASOS, COL_CHUVA, COL_TMED, COL_TMIN, COL_TMAX]
COLUNAS_FLOAT = [COL_CHUVA, COL_TMED, COL_TMIN, COL_TMAX]

# Nomes dos meses em PT-BR via dicionário fixo (sem locale.setlocale)
MESES_ABREV = {
    1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez",
}
MESES_NOME = {
    1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril", 5: "Maio", 6: "Junho",
    7: "Julho", 8: "Agosto", 9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro",
}

# Cores (casos = laranja, chuva = azul, linhas auxiliares = cinza)
COR_CASOS = "#f47230"
COR_CHUVA = "#257de0"
COR_AUXILIAR = "#666666"

NA_TEXTO = "N/A"


# =============================================================================
# FORMATAÇÃO PT-BR
# =============================================================================
def _troca_separadores(texto: str) -> str:
    """Converte '1,234.5' (padrão US) em '1.234,5' (padrão PT-BR)."""
    return texto.replace(",", "#").replace(".", ",").replace("#", ".")


def fmt_int(valor) -> str:
    """Inteiro com separador de milhar: 1130 -> '1.130'."""
    if valor is None or pd.isna(valor):
        return NA_TEXTO
    return _troca_separadores(f"{int(round(float(valor))):,}")


def fmt_dec(valor, casas: int = 1) -> str:
    """Decimal PT-BR: 376.666 -> '376,7'."""
    if valor is None or pd.isna(valor):
        return NA_TEXTO
    return _troca_separadores(f"{float(valor):,.{casas}f}")


def fmt_pct_sinal(valor, casas: int = 1) -> str:
    """Percentual com sinal: 12.34 -> '+12,3%' (evita '-0,0%')."""
    if valor is None or pd.isna(valor):
        return NA_TEXTO
    arred = round(float(valor), casas)
    if arred == 0:
        return f"{fmt_dec(0, casas)}%"
    return f"{'+' if arred > 0 else '-'}{fmt_dec(abs(arred), casas)}%"


def fmt_int_sinal(valor) -> str:
    """Inteiro com sinal: 150 -> '+150'."""
    v = int(round(float(valor)))
    if v == 0:
        return "0"
    return f"{'+' if v > 0 else '-'}{fmt_int(abs(v))}"


# =============================================================================
# CARGA E TRATAMENTO
# =============================================================================
@st.cache_data(show_spinner="Carregando e tratando os dados...")
def load_data(conteudo: bytes) -> tuple[pd.DataFrame | None, dict]:
    """
    Lê os bytes do CSV, valida o schema e aplica a sanitização.

    Retorna (df, relatorio). As mensagens (st.info / st.warning / st.error) são
    exibidas fora desta função, para não ficarem presas ao cache.
    """
    relatorio = {
        "colunas_faltantes": [],
        "linhas_data_invalida": 0,
        "duplicatas_exatas": 0,
        "meses_conflitantes": [],
        "valores_invalidos": {},
        "erro_leitura": None,
    }

    # Leitura bruta como texto: a conversão de tipos é feita de forma controlada abaixo.
    # utf-8-sig lê UTF-8 normal e também tolera o BOM que o Excel costuma gravar.
    try:
        df = pd.read_csv(
            io.BytesIO(conteudo), sep=",", encoding="utf-8-sig",
            dtype=str, keep_default_na=True,
        )
    except Exception as exc:  # arquivo vazio, encoding inválido etc.
        relatorio["erro_leitura"] = str(exc)
        return None, relatorio

    # Sanitização dos nomes das colunas
    df.columns = df.columns.str.strip()

    # Validação do schema (colunas extras são ignoradas)
    faltantes = [c for c in COLUNAS_OBRIGATORIAS if c not in df.columns]
    if faltantes:
        relatorio["colunas_faltantes"] = faltantes
        return None, relatorio
    df = df[COLUNAS_OBRIGATORIAS].copy()

    # --- data: converte, descarta inválidas e normaliza para o 1º dia do mês
    datas = pd.to_datetime(df[COL_DATA].str.strip(), format="%Y-%m-%d", errors="coerce")
    relatorio["linhas_data_invalida"] = int(datas.isna().sum())
    df[COL_DATA] = datas
    df = df.loc[df[COL_DATA].notna()].copy()
    df[COL_DATA] = df[COL_DATA].dt.to_period("M").dt.to_timestamp()

    # --- casos-confirmados: Int64 (inteiro que aceita vazio)
    texto = df[COL_CASOS].str.strip().replace("", np.nan)
    numeros = pd.to_numeric(texto, errors="coerce")
    nao_inteiros = numeros.notna() & (numeros % 1 != 0)
    invalidos = int(((texto.notna() & numeros.isna()) | nao_inteiros).sum())
    numeros = numeros.mask(nao_inteiros)  # valor não inteiro vira vazio (não arredonda)
    df[COL_CASOS] = numeros.astype("Int64")
    if invalidos:
        relatorio["valores_invalidos"][COL_CASOS] = invalidos

    # --- colunas float (separador decimal ponto); sem conversão de unidade
    for col in COLUNAS_FLOAT:
        texto = df[col].str.strip().replace("", np.nan)
        numeros = pd.to_numeric(texto, errors="coerce")
        invalidos = int((texto.notna() & numeros.isna()).sum())
        df[col] = numeros.astype("float64")
        if invalidos:
            relatorio["valores_invalidos"][col] = invalidos

    # --- duplicatas exatas (após a normalização da data)
    antes = len(df)
    df = df.drop_duplicates(keep="first")
    relatorio["duplicatas_exatas"] = antes - len(df)

    # --- mesmo mês com valores diferentes: mantém a última ocorrência
    conflito = df[COL_DATA].duplicated(keep=False)
    if conflito.any():
        meses = sorted(df.loc[conflito, COL_DATA].unique())
        relatorio["meses_conflitantes"] = [pd.Timestamp(m).strftime("%m/%Y") for m in meses]
        df = df.drop_duplicates(subset=COL_DATA, keep="last")

    # --- colunas derivadas (somente para filtros e gráficos)
    df = df.sort_values(COL_DATA).reset_index(drop=True)
    df["ano"] = df[COL_DATA].dt.year.astype(int)
    df["mes"] = df[COL_DATA].dt.month.astype(int)
    df["nome_mes"] = df["mes"].map(MESES_ABREV)
    df["ano_mes"] = df[COL_DATA].dt.strftime("%m/%Y")

    return df, relatorio


def exibir_relatorio_carga(relatorio: dict) -> None:
    """Mostra as mensagens de qualidade de dados geradas na carga."""
    if relatorio["linhas_data_invalida"]:
        st.info(
            f"{fmt_int(relatorio['linhas_data_invalida'])} linha(s) descartada(s) "
            "por data inválida (formato esperado: AAAA-MM-DD)."
        )
    if relatorio["duplicatas_exatas"]:
        st.info(f"{fmt_int(relatorio['duplicatas_exatas'])} linha(s) duplicada(s) removida(s).")
    if relatorio["meses_conflitantes"]:
        st.warning(
            "Mais de uma linha com valores diferentes para o mesmo mês. "
            "Foi mantida a última ocorrência nos meses: "
            + ", ".join(relatorio["meses_conflitantes"])
        )
    if relatorio["valores_invalidos"]:
        detalhes = "; ".join(f"`{c}`: {fmt_int(n)}" for c, n in relatorio["valores_invalidos"].items())
        st.warning(f"Valores não numéricos (ou não inteiros em casos) tratados como vazio — {detalhes}.")


# =============================================================================
# FILTROS
# =============================================================================
def render_sidebar_filtros(df: pd.DataFrame) -> tuple[list[int], list[int], tuple[int, int] | None]:
    """Desenha os filtros na sidebar e devolve as seleções."""
    st.sidebar.header("Filtros")

    anos_disp = sorted(df["ano"].unique().tolist())
    anos = st.sidebar.multiselect("Ano", options=anos_disp, placeholder="Todos os anos")

    meses_disp = sorted(df["mes"].unique().tolist())  # ordem cronológica Jan → Dez
    meses = st.sidebar.multiselect(
        "Mês", options=meses_disp, format_func=lambda m: MESES_NOME[m], placeholder="Todos os meses",
    )

    faixa = None
    casos_validos = df[COL_CASOS].dropna()
    if casos_validos.empty:
        st.sidebar.caption("Sem casos preenchidos na base: filtro de faixa indisponível.")
    else:
        vmin, vmax = int(casos_validos.min()), int(casos_validos.max())
        if vmin == vmax:
            st.sidebar.caption(f"Todos os meses têm {fmt_int(vmin)} casos: filtro de faixa indisponível.")
        else:
            faixa = st.sidebar.slider(
                "Faixa de casos confirmados",
                min_value=vmin, max_value=vmax, value=(vmin, vmax),
                help="Afeta apenas a tabela de detalhamento e o gráfico de dispersão.",
            )
    return anos, meses, faixa


def apply_filters(
    df: pd.DataFrame, anos: list[int], meses: list[int], faixa: tuple[int, int] | None,
) -> dict[str, pd.DataFrame]:
    """
    Gera os recortes usados no dashboard:
      - 'periodo': Ano x Mês  -> KPIs, série temporal, Pareto
      - 'anos'   : só Ano     -> Média Anual (ignora o filtro de mês)
      - 'detalhe': Ano x Mês x faixa de casos -> tabela e dispersão
    Multiselect vazio = todos os valores.
    """
    mask_ano = df["ano"].isin(anos) if anos else pd.Series(True, index=df.index)
    mask_mes = df["mes"].isin(meses) if meses else pd.Series(True, index=df.index)

    df_anos = df.loc[mask_ano]
    df_periodo = df.loc[mask_ano & mask_mes]

    df_detalhe = df_periodo
    if faixa is not None:
        vmin_base, vmax_base = int(df[COL_CASOS].min()), int(df[COL_CASOS].max())
        # Com o slider na faixa total, meses sem casos continuam visíveis na tabela.
        if (faixa[0], faixa[1]) != (vmin_base, vmax_base):
            casos = df_periodo[COL_CASOS]
            df_detalhe = df_periodo.loc[casos.notna() & casos.between(faixa[0], faixa[1]).fillna(False)]

    return {"periodo": df_periodo, "anos": df_anos, "detalhe": df_detalhe}


# =============================================================================
# KPIs
# =============================================================================
def calculate_kpis(df: pd.DataFrame, recortes: dict, anos: list[int], meses: list[int]) -> dict:
    """Calcula os 4 KPIs conforme as regras de negócio. Valores None = 'N/A'."""
    df_periodo = recortes["periodo"]
    casos = df_periodo[COL_CASOS]

    # Total de Casos e Média Mensal (nulos não entram em soma nem em contagem)
    meses_preenchidos = int(casos.notna().sum())
    total = int(casos.sum()) if meses_preenchidos else None
    media_mensal = total / meses_preenchidos if meses_preenchidos else None

    # Média Anual (Ano Completo): ignora o filtro de mês; só anos com 12 meses preenchidos
    df_anos = recortes["anos"]
    por_ano = df_anos.groupby("ano")[COL_CASOS].agg(meses_ok="count", soma="sum")
    completos = por_ano.loc[por_ano["meses_ok"] == 12, "soma"]
    media_anual = float(completos.astype(float).mean()) if not completos.empty else None

    # Variação YoY
    yoy = {"pct": None, "delta": None, "motivo": None, "ano": None}
    anos_considerados = anos if anos else sorted(df["ano"].unique().tolist())
    if len(anos_considerados) != 1:
        yoy["motivo"] = "Selecione exatamente 1 ano para comparar."
    else:
        ano_atual = int(anos_considerados[0])
        ano_ant = ano_atual - 1
        yoy["ano"] = ano_atual
        if not (df["ano"] == ano_ant).any():
            yoy["motivo"] = f"{ano_ant} não existe na base."
        else:
            meses_sel = meses if meses else list(range(1, 13))
            atual = df.loc[(df["ano"] == ano_atual) & df["mes"].isin(meses_sel) & df[COL_CASOS].notna()]
            anter = df.loc[(df["ano"] == ano_ant) & df["mes"].isin(meses_sel) & df[COL_CASOS].notna()]
            comuns = sorted(set(atual["mes"]) & set(anter["mes"]))
            if not comuns:
                yoy["motivo"] = f"Sem meses preenchidos em comum entre {ano_atual} e {ano_ant}."
            else:
                tot_atual = int(atual.loc[atual["mes"].isin(comuns), COL_CASOS].sum())
                tot_ant = int(anter.loc[anter["mes"].isin(comuns), COL_CASOS].sum())
                if tot_ant == 0:
                    yoy["motivo"] = f"Total de {ano_ant} nos meses comparados é zero."
                else:
                    yoy["pct"] = (tot_atual - tot_ant) / tot_ant * 100
                    yoy["delta"] = tot_atual - tot_ant
                    yoy["motivo"] = (
                        f"{ano_atual} x {ano_ant} em {len(comuns)} mês(es) comum(ns): "
                        + ", ".join(MESES_ABREV[m] for m in comuns)
                    )

    return {
        "total": total,
        "media_mensal": media_mensal,
        "media_anual": media_anual,
        "anos_completos": completos.index.tolist(),
        "yoy": yoy,
    }


def render_kpis(kpis: dict) -> None:
    """Nível 1 – Visão Executiva."""
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total de Casos Confirmados", fmt_int(kpis["total"]))
    c2.metric("Média Mensal de Casos", fmt_dec(kpis["media_mensal"]))

    ajuda_anual = (
        "Anos completos considerados: " + ", ".join(map(str, kpis["anos_completos"]))
        if kpis["anos_completos"] else "Nenhum ano selecionado tem os 12 meses preenchidos."
    )
    c3.metric("Média Anual (Ano Completo)", fmt_dec(kpis["media_anual"]), help=ajuda_anual)

    yoy = kpis["yoy"]
    if yoy["pct"] is None:
        c4.metric("Variação YoY (%)", NA_TEXTO, delta=None, help=yoy["motivo"])
    else:
        delta_txt = f"{fmt_int_sinal(yoy['delta'])} casos"
        c4.metric(
            "Variação YoY (%)",
            fmt_pct_sinal(yoy["pct"]),
            delta=delta_txt,
            delta_color="inverse" if yoy["delta"] != 0 else "off",  # alta = vermelho; queda = verde
            help=yoy["motivo"],
        )


# =============================================================================
# GRÁFICOS
# =============================================================================
def _layout_base(fig: go.Figure, titulo: str) -> go.Figure:
    """Layout comum: separadores PT-BR, grade discreta, legenda no topo."""
    fig.update_layout(
        title=titulo,
        separators=",.",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=70, b=10),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False)
    return fig


def _grade_mensal(df_periodo: pd.DataFrame) -> pd.DataFrame:
    """
    Reindexa a série para uma grade mensal (anos do intervalo x meses selecionados).
    Meses sem linha na base viram NaN -> aparecem como lacuna, sem interpolação.
    """
    meses = sorted(df_periodo["mes"].unique())
    anos = range(df_periodo["ano"].min(), df_periodo["ano"].max() + 1)
    grade = pd.DatetimeIndex([pd.Timestamp(a, m, 1) for a in anos for m in meses])
    grade = grade[(grade >= df_periodo[COL_DATA].min()) & (grade <= df_periodo[COL_DATA].max())]
    serie = df_periodo.set_index(COL_DATA)[[COL_CASOS, COL_CHUVA]].reindex(grade)
    serie.index.name = COL_DATA
    return serie.reset_index()


def grafico_serie_temporal(df_periodo: pd.DataFrame) -> go.Figure:
    serie = _grade_mensal(df_periodo)
    serie["rotulo"] = serie[COL_DATA].dt.strftime("%m/%Y")
    casos = serie[COL_CASOS].astype("float64")  # <NA> -> NaN, para o Plotly abrir lacuna

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=serie[COL_DATA], y=serie[COL_CHUVA], name="Chuva (mm)",
            marker_color=COR_CHUVA, opacity=0.45, customdata=serie["rotulo"],
            hovertemplate="Chuva: %{y:,.1f} mm<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=serie[COL_DATA], y=casos, name="Casos confirmados", mode="lines+markers",
            line=dict(color=COR_CASOS, width=2), marker=dict(size=5),
            connectgaps=False,  # mês vazio = lacuna
            hovertemplate="Casos: %{y:,.0f}<extra></extra>",
        ),
        secondary_y=False,
    )
    _layout_base(fig, "Evolução mensal: casos confirmados x chuva")
    fig.update_xaxes(tickformat="%m/%Y", hoverformat="%m/%Y", title_text="Ano-Mês")
    fig.update_yaxes(title_text="Casos confirmados", rangemode="tozero", secondary_y=False)
    fig.update_yaxes(title_text="Chuva (mm)", rangemode="tozero", showgrid=False, secondary_y=True)
    return fig


def dados_pareto(df_periodo: pd.DataFrame) -> pd.DataFrame:
    """Média de casos por mês do calendário, ordenada desc., com % acumulado."""
    media = (
        df_periodo.groupby("mes")[COL_CASOS]
        .apply(lambda s: s.astype("float64").mean())  # NaN ignorado na média
        .dropna()
        .sort_values(ascending=False, kind="mergesort")
        .rename("media_casos")
        .reset_index()
    )
    media["nome_mes"] = media["mes"].map(MESES_ABREV)
    soma = media["media_casos"].sum()
    media["pct_acumulado"] = media["media_casos"].cumsum() / soma * 100 if soma else np.nan
    return media


def grafico_pareto(pareto: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=pareto["nome_mes"], y=pareto["media_casos"], name="Média de casos",
            marker_color=COR_CASOS,
            text=[fmt_dec(v) if v % 1 else fmt_int(v) for v in pareto["media_casos"]],
            textposition="outside", cliponaxis=False,
            hovertemplate="Média de casos: %{y:,.1f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=pareto["nome_mes"], y=pareto["pct_acumulado"], name="% acumulado",
            mode="lines+markers", line=dict(color=COR_AUXILIAR, width=2), marker=dict(size=8),
            hovertemplate="Acumulado: %{y:,.1f}%<extra></extra>",
        ),
        secondary_y=True,
    )
    _layout_base(fig, "Sazonalidade: ranking dos meses por média de casos (Pareto)")
    fig.update_xaxes(type="category", title_text="Mês")
    fig.update_yaxes(title_text="Média de casos confirmados", rangemode="tozero", secondary_y=False)
    fig.update_yaxes(
        title_text="% acumulado", range=[0, 105], ticksuffix="%", showgrid=False, secondary_y=True,
    )
    return fig


def grafico_dispersao(df_detalhe: pd.DataFrame) -> go.Figure | None:
    """Temperatura média x casos, bolha = chuva, com tendência linear (numpy.polyfit)."""
    pts = df_detalhe.dropna(subset=[COL_TMED, COL_CASOS, COL_CHUVA]).copy()
    if pts.empty:
        return None
    x = pts[COL_TMED].to_numpy(dtype=float)
    y = pts[COL_CASOS].astype("float64").to_numpy()
    chuva = pts[COL_CHUVA].to_numpy(dtype=float)

    # Tamanho da bolha proporcional à chuva (área); chuva negativa não gera bolha negativa
    tamanho = np.clip(chuva, 0, None)
    maior = tamanho.max()
    sizeref = 2.0 * maior / (40.0 ** 2) if maior > 0 else 1.0

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x, y=y, mode="markers", name="Meses",
            marker=dict(
                size=tamanho, sizemode="area", sizeref=sizeref, sizemin=4,
                color=COR_CASOS, opacity=0.6, line=dict(width=1, color="white"),
            ),
            customdata=pts[["ano_mes", COL_CHUVA]].to_numpy(dtype=object),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>Temperatura média: %{x:,.1f} °C"
                "<br>Casos: %{y:,.0f}<br>Chuva: %{customdata[1]:,.1f} mm<extra></extra>"
            ),
        )
    )

    # Linha de tendência: precisa de ao menos 2 temperaturas distintas
    if np.unique(x).size >= 2:
        a, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        fig.add_trace(
            go.Scatter(
                x=xs, y=a * xs + b, mode="lines", name="Tendência linear",
                line=dict(color=COR_AUXILIAR, width=2, dash="dash"), hoverinfo="skip",
            )
        )

    _layout_base(fig, "Correlação climática: temperatura média x casos (bolha = chuva)")
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title_text="Temperatura média (°C)", showgrid=True, gridcolor="rgba(128,128,128,0.18)")
    fig.update_yaxes(title_text="Casos confirmados")
    return fig


def render_charts(recortes: dict) -> None:
    """Nível 2 – Análise Exploratória."""
    df_periodo, df_detalhe = recortes["periodo"], recortes["detalhe"]

    st.subheader("Análise exploratória")
    if df_periodo.empty:
        st.warning("Nenhum dado para a combinação de Ano e Mês selecionada. Ajuste os filtros.")
        return

    st.plotly_chart(grafico_serie_temporal(df_periodo), width="stretch")

    col_a, col_b = st.columns(2)
    with col_a:
        pareto = dados_pareto(df_periodo)
        if pareto.empty:
            st.warning("Sem casos preenchidos no recorte para montar o ranking.")
        else:
            st.plotly_chart(grafico_pareto(pareto), width="stretch")
    with col_b:
        fig = grafico_dispersao(df_detalhe)
        if fig is None:
            st.warning(
                "Sem meses com temperatura média, casos e chuva preenchidos "
                "no recorte (inclui a faixa de casos)."
            )
        else:
            st.plotly_chart(fig, width="stretch")
            st.caption("Considera também o filtro de faixa de casos. Meses com dados vazios ficam fora.")


# =============================================================================
# DETALHAMENTO
# =============================================================================
ROTULOS_TABELA = {
    COL_DATA: "Mês/Ano",
    COL_CASOS: "Casos confirmados",
    COL_CHUVA: "Chuva (mm)",
    COL_TMED: "Temp. média (°C)",
    COL_TMIN: "Temp. mínima (°C)",
    COL_TMAX: "Temp. máxima (°C)",
}


def csv_exportacao(df_detalhe: pd.DataFrame) -> bytes:
    """CSV para Excel PT-BR: sep=';', decimal=',', utf-8-sig, data em MM/AAAA."""
    saida = df_detalhe[COLUNAS_OBRIGATORIAS].copy()
    saida[COL_DATA] = saida[COL_DATA].dt.strftime("%m/%Y")
    return saida.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")


def render_tabela(df_detalhe: pd.DataFrame) -> None:
    """Nível 3 – Detalhamento granular."""
    st.subheader("Detalhamento")
    if df_detalhe.empty:
        st.warning("Nenhum mês atende aos filtros atuais (Ano, Mês e faixa de casos).")
        return

    # Colunas continuam numéricas/datetime (ordenação correta); o Styler só formata a exibição.
    tabela = df_detalhe[COLUNAS_OBRIGATORIAS].rename(columns=ROTULOS_TABELA)
    estilo = tabela.style.format(
        {
            ROTULOS_TABELA[COL_DATA]: lambda d: d.strftime("%m/%Y"),
            ROTULOS_TABELA[COL_CASOS]: lambda v: "" if pd.isna(v) else fmt_int(v),
            **{ROTULOS_TABELA[c]: (lambda v: "" if pd.isna(v) else fmt_dec(v)) for c in COLUNAS_FLOAT},
        }
    )
    st.dataframe(estilo, width="stretch", hide_index=True)
    st.caption(f"{fmt_int(len(df_detalhe))} mês(es) no recorte.")

    st.download_button(
        "Baixar recorte em CSV",
        data=csv_exportacao(df_detalhe),
        file_name="dengue-recorte.csv",
        mime="text/csv",
    )


# =============================================================================
# APLICAÇÃO
# =============================================================================
def obter_conteudo_csv() -> bytes | None:
    """Usa o upload, se houver; senão, o dengue-dataset.csv da pasta do script."""
    st.sidebar.header("Fonte de dados")
    upload = st.sidebar.file_uploader(
        "Enviar novo CSV (mesmo schema)", type=["csv"],
        help="Colunas: " + ", ".join(COLUNAS_OBRIGATORIAS),
    )
    if upload is not None:
        st.sidebar.caption(f"Usando: {upload.name}")
        return upload.getvalue()  # bytes, para o cache funcionar corretamente

    caminho = Path(__file__).parent / ARQUIVO_PADRAO
    if caminho.exists():
        st.sidebar.caption(f"Usando: {ARQUIVO_PADRAO}")
        return caminho.read_bytes()
    return None


def main() -> None:
    st.set_page_config(page_title="Vigilância Epidemiológica – Dengue", layout="wide")
    st.title("Vigilância Epidemiológica – Dengue")
    st.caption("Casos confirmados e indicadores climáticos · granularidade mensal")

    conteudo = obter_conteudo_csv()
    if conteudo is None:
        st.info(f"Arquivo `{ARQUIVO_PADRAO}` não encontrado na pasta do app. Envie um CSV pela barra lateral.")
        st.stop()

    df, relatorio = load_data(conteudo)

    if relatorio["erro_leitura"]:
        st.error(f"Não foi possível ler o CSV: {relatorio['erro_leitura']}")
        st.stop()
    if relatorio["colunas_faltantes"]:
        st.error("Colunas obrigatórias ausentes no CSV: " + ", ".join(f"`{c}`" for c in relatorio["colunas_faltantes"]))
        st.stop()

    exibir_relatorio_carga(relatorio)

    if df.empty:
        st.warning("O CSV não tem nenhuma linha com data válida.")
        st.stop()

    anos, meses, faixa = render_sidebar_filtros(df)
    recortes = apply_filters(df, anos, meses, faixa)

    # Nível 1
    st.subheader("Visão executiva")
    kpis = calculate_kpis(df, recortes, anos, meses)
    render_kpis(kpis)
    if recortes["periodo"].empty:
        st.warning("Nenhum mês corresponde aos filtros de Ano e Mês selecionados.")
    st.divider()

    # Nível 2
    render_charts(recortes)
    st.divider()

    # Nível 3
    render_tabela(recortes["detalhe"])


if __name__ == "__main__":
    main()
