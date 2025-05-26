# ==============================================
# IMPORTAÇÕES
# ==============================================
import os
import re
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import timedelta
from ta.momentum import RSIIndicator, StochasticOscillator
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import KMeans
from tqdm import tqdm
from sklearn.metrics import silhouette_samples, silhouette_score
import statistics
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from yellowbrick.cluster import SilhouetteVisualizer

# ==============================================
# CONSTANTES
# ==============================================
INTERVALLIMITS = {
    "1h": 730,
    "1d": None,
    "5d": None,
    "1wk": None,
    "1mo": None,
    "3mo": None,
}


# ==============================================
# FUNÇÕES DE UTILIDADE
# ==============================================
def cleanTicker(ticker):
    """Remove caracteres não alfanuméricos do ticker"""
    return re.sub(r"\W+", "", ticker)


def clean(data):
    """Limpa dados substituindo infinitos e NANs por zero"""
    data.replace([np.inf, -np.inf], 0, inplace=True)
    data.fillna(0, inplace=True)
    return data


# ==============================================
# FUNÇÕES PARA OBTENÇÃO E PROCESSAMENTO DE DADOS
# ==============================================
@st.cache_data
def getData(ticker, inicio, fim, intervalo="1d"):
    """Obtém dados do Yahoo Finance ou de arquivo local se disponível"""
    cleanedTicker = cleanTicker(ticker)
    file = f"datasets/{cleanedTicker.lower()}-{intervalo}.csv"

    if intervalo not in INTERVALLIMITS:
        print(f"Intervalo '{intervalo}' não suportado. Use um dos seguintes: {', '.join(INTERVALLIMITS.keys())}.")
        return pd.DataFrame()

    if os.path.exists(file):
        data = pd.read_csv(file, index_col="Date", parse_dates=True)
    else:
        maxDays = INTERVALLIMITS.get(intervalo)
        data = pd.DataFrame()

        try:
            startDate = pd.to_datetime(inicio)
            endDate = pd.to_datetime(fim)
            daysDiff = (endDate - startDate).days

            if maxDays is not None and daysDiff > maxDays:
                adjustedEndDate = startDate + timedelta(days=maxDays)
                print(
                    f"A data final {endDate.strftime('%Y-%m-%d')} excede o máximo permitido para o intervalo '{intervalo}'. "
                    f"Ajustando para {adjustedEndDate.strftime('%Y-%m-%d')}.")
                endDate = adjustedEndDate

            data = yf.download(
                ticker,
                start=inicio,
                end=endDate.strftime("%Y-%m-%d"),
                interval=intervalo,
            )[["Close", "High", "Low", "Open", "Volume"]]

            data.index.name = "Date"
            data.columns = data.columns.droplevel(1)
            data.index = pd.to_datetime(data.index)
            # data.to_csv(file, index=True)

        except Exception as e:
            print(f"Erro ao baixar os dados: {e}")

    return data


def getCandleBodyToWickRatio(openPrice, closePrice, highPrice, lowPrice):
    """Calcula a relação entre o corpo e o pavio do candle"""
    bodySize = abs(closePrice - openPrice)
    upperWick = highPrice - max(openPrice, closePrice)
    lowerWick = min(openPrice, closePrice) - lowPrice
    totalWick = upperWick + lowerWick

    if bodySize == 0:  # Evita divisão por zero
        return float("inf")

    wickToBodyRatio = (totalWick / bodySize) * 100
    return wickToBodyRatio.round(2)


# ==============================================
# FUNÇÕES PARA DETECÇÃO DE OUTLIERS
# ==============================================
def ISOFModel(estimators, contamination):
    """Cria modelo IsolationForest para detecção de outliers"""
    return IsolationForest(random_state=42, n_estimators=estimators, contamination=contamination)


def plotOutliers(data, isClose=True):
    """Plota gráfico de outliers para preço ou volume"""
    fig = go.Figure()

    if isClose:
        y_col = "ReturnClose"
        outlier_col = "OutlierClose"
        title = "Detecte movimentos anormais nos preços que fogem do padrão histórico (pontos vermelhos)."
        color = "red"
    else:
        y_col = "ReturnVolume"
        outlier_col = "OutlierVolume"
        title = "Detecte movimentos anormais no volume que fogem do padrão histórico (pontos verdes)."
        color = "green"

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data[y_col],
            mode="lines",
            name="Retorno Percentual",
            line={"color": "blue"},
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data[data[outlier_col] == -1].index,
            y=data[data[outlier_col] == -1][y_col],
            mode="markers",
            name="Outliers (-1)",
            marker={"color": color, "size": 16},
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Data de negociação",
        yaxis_title="Retorno Percentual",
        height=600,
        width=1200,
        xaxis=dict(type="date", tickformat="%b %Y"),
        yaxis=dict(tickformat=".2%"),

    )

    st.plotly_chart(fig)


# ==============================================
# FUNÇÕES PARA CLUSTERIZAÇÃO ( A.M NÃO SUPERVISIONADO )
# ==============================================
def assignKmeansClusters(data, clusters):
    """Aplica K-means e atribui clusters aos dados"""
    kmeans = KMeans(
        n_clusters=clusters,
        init="random",
        n_init=10,
        max_iter=300,
        tol=0.0001,
        verbose=0,
        random_state=42,
    )

    rotulosFeatures = kmeans.fit_predict(data)
    data["Cluster"] = rotulosFeatures

    return data, kmeans


def calculateSilhouette(data, ranges):
    """Calcula o coeficiente de silhueta para diferentes valores de K"""
    silhouettes = []
    individualSilhouettes = []

    # Filtra valores de k válidos
    validRanges = [k for k in ranges if k < len(data)]

    if not validRanges:
        print("Nenhum valor de k é válido para o número de amostras fornecido.")
        return None, None, None

    for k in tqdm(list(ranges)):
        if k >= len(data):
            print(f"Pulando K={k} pois o número de clusters é maior ou igual ao número de amostras ({len(data)}).")
            continue

        kmeans = KMeans(
            init="random",
            n_clusters=k,
            n_init=10,
            max_iter=300,
            tol=0.0001,
            random_state=42,
        )

        try:
            clusterIDs = kmeans.fit_predict(data)
            silhouettes.append(silhouette_score(data, clusterIDs))
            individualSilhouettes.append(silhouette_samples(data, clusterIDs))
        except ValueError as e:
            print(f"Erro ao processar K={k}: {e}")
            continue

    if silhouettes:
        silhouetteData = pd.DataFrame({"Coeficiente de Silhueta": silhouettes}, index=list(ranges)[:len(silhouettes)])
        bestK = silhouetteData.idxmax()["Coeficiente de Silhueta"]
    else:
        silhouetteData = pd.DataFrame()
        bestK = None
        kmeans = None

    return bestK, silhouetteData, kmeans


def filterCombinations(data, ranges, include_outliers=True):
    """Filtra combinações de dados para análise de clusters"""
    # Definir condições de filtro
    if include_outliers:
        condOutlier = pd.Series(True, index=data.index)
    else:
        condOutlier = (data["OutlierClose"] != -1) & (data["OutlierVolume"] != -1)

    condRSI = (data["RSI"] > 70) | ((data["RSI"] < 30) & (data["RSI"] > 0))
    condSTO = (data["%K"] > 80) | ((data["%K"] < 20) & (data["%K"] > 0))

    filters = [
        data[(condOutlier) & (condRSI) & (condSTO)],
        data[(condOutlier) & (condRSI)],
        data[(condOutlier) & (condSTO)],
        data[(condOutlier)],
    ]

    bestSilhouette = -1
    bestK = None
    bestModel = None
    bestData = None
    outData = None

    for filteredData in filters:
        if filteredData.empty:
            print("Filtro ignorado: dados vazios após o filtro.")
            continue

        filteredData = filteredData[filteredData.columns.intersection(["CloseScaler"])]
        if filteredData.shape[0] < 2:
            continue

        if filteredData.shape[0] < min(ranges):
            print(f"Filtro ignorado: apenas {filteredData.shape[0]} amostras.")
            continue

        try:
            adjustedRanges = range(min(ranges), min(filteredData.shape[0], max(ranges)) + 1)
            k, silhouetteData, model = calculateSilhouette(filteredData, adjustedRanges)

            if silhouetteData is None:
                continue

            maxSilhouette = silhouetteData["Coeficiente de Silhueta"].max()

            if maxSilhouette > bestSilhouette:
                bestSilhouette = maxSilhouette
                bestK = k
                bestModel = model
                bestData = silhouetteData
                outData = filteredData

        except Exception as e:
            print(f"Erro ao processar filtro com {filteredData.shape[0]} amostras: {e}")

    return bestK, bestData, bestModel, outData


def calculateAverageDistance(data, rounds=2):
    """Calcula a distância média entre preços consecutivos"""
    diff = data["Close"].diff().abs()
    return diff.mean().round(rounds)


def getClustersMode(data, rounds=2):
    """Calcula a moda dos clusters"""
    uniqueModes = set()
    modeByCluster = {}

    for cluster, group in data.groupby("Cluster")["CloseScaler"]:
        uniqueValues = np.round(group.unique(), rounds)
        uniqueValues.sort()

        while len(uniqueValues) > 0:
            modeValue = statistics.mode(uniqueValues)
            if modeValue not in uniqueModes:
                uniqueModes.add(modeValue)
                modeByCluster[cluster] = modeValue
                break
            else:
                uniqueValues = uniqueValues[uniqueValues != modeValue]

    return pd.Series(modeByCluster)


# ==============================================
# FUNÇÕES PARA VISUALIZAÇÃO
# ==============================================
def plotSilhouettes(data):
    """Plota gráfico de coeficientes de silhueta"""
    data = data.reset_index()
    data.columns = ["Quantidade de Clusters", "Silhueta Média"]

    fig = go.Figure(
        data=[
            go.Bar(
                x=data["Quantidade de Clusters"],
                y=data["Silhueta Média"],
                text=data["Silhueta Média"].round(3),
                textposition="auto",
                marker=dict(color=data["Silhueta Média"], colorscale="Viridis"),
            )
        ]
    )

    fig.update_layout(
        title="Coeficiente de Silhueta",
        title_x=0.5,
        xaxis_title="Quantidade de Clusters",
        yaxis_title="Silhueta Média",
        height=500,
        width=700,
        template="plotly_white",
        xaxis=dict(
            tickmode="array",
            tickvals=data["Quantidade de Clusters"],
            ticktext=data["Quantidade de Clusters"],
        ),
    )

    st.plotly_chart(fig)


def calculateClusteringMetrics(data, labels, kmeans):
    """Calcula métricas de avaliação de clusters"""
    inertia = kmeans.inertia_
    daviesBouldin = davies_bouldin_score(data, labels)
    calinskiHarabasz = calinski_harabasz_score(data, labels)

    return {
        "Inertia (WCSS)": round(inertia, 2),
        "Davies-Bouldin Index": round(daviesBouldin, 2),
        "Calinski-Harabasz Index": round(calinskiHarabasz, 2),
    }


def plotClusters(data, labels):
    """Plota clusters em 2D usando PCA"""
    if not isinstance(data, pd.DataFrame):
        raise ValueError("O 'data' deve ser um pandas DataFrame.")
    if not isinstance(labels, np.ndarray):
        raise ValueError("Os 'labels' devem ser um array NumPy.")

    pca = PCA(n_components=2)
    dataPCA = pca.fit_transform(data)
    unique_labels = np.unique(labels)
    centroids = np.array([dataPCA[labels == label].mean(axis=0) for label in unique_labels])

    grafico = px.scatter(
        x=dataPCA[:, 0],
        y=dataPCA[:, 1],
        color=labels.astype(str),
        size=[10] * data.shape[0],
        labels={"x": "X", "y": "Y", "color": "Cluster"},
        template="plotly_dark",
    )

    for i, centroid in enumerate(centroids):
        grafico.add_scatter(
            x=[centroid[0]],
            y=[centroid[1]],
            mode="markers+text",
            marker=dict(size=15, color="pink", symbol="x"),
            text=[f"C-{i}"],
            textposition="top center",
            name=f"Centróide {i}",
            showlegend=True,
        )

    grafico.update_layout(
        title="Visualização dos Clusters - Explore os grupos identificados em 2 dimensões, com centróides marcados (X)",
        width=900,
        height=600,
    )

    st.plotly_chart(grafico)


def plotClusters3D(data, labels):
    """Plota clusters em 3D usando PCA"""
    if not isinstance(data, pd.DataFrame):
        raise ValueError("O 'data' deve ser um pandas DataFrame.")
    if not isinstance(labels, np.ndarray):
        raise ValueError("Os 'labels' devem ser um array NumPy.")

    pca = PCA(n_components=3)
    dataPCA = pca.fit_transform(data)
    unique_labels = np.unique(labels)
    centroids = np.array([dataPCA[labels == label].mean(axis=0) for label in unique_labels])

    grafico = px.scatter_3d(
        x=dataPCA[:, 0],
        y=dataPCA[:, 1],
        z=dataPCA[:, 2],
        color=labels.astype(str),
        size=[10] * data.shape[0],
        labels={"x": "X", "y": "Y", "z": "Z", "color": "Cluster"},
        template="plotly_dark",
    )

    for i, centroid in enumerate(centroids):
        grafico.add_scatter3d(
            x=[centroid[0]],
            y=[centroid[1]],
            z=[centroid[2]],
            mode="markers+text",
            marker=dict(size=8, color="white", symbol="x"),
            text=[f"C-{i}"],
            textposition="top center",
            name=f"Centróide {i}",
            showlegend=True,
        )

    grafico.update_layout(
        title="Visualização dos Clusters em 3D - Analise os clusters em 3 dimensões para entender melhor suas relações espaciais.",
        width=900,
        height=600,
    )

    st.plotly_chart(grafico)


def plotSupportResistance(data, modeByCluster, countByCluster, TICKER):
    """Plota linhas de suporte e resistência"""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data.index, y=data["Close"], mode="lines", name="Preço de Fechamento"
        )
    )

    for i, x in enumerate(modeByCluster):
        fig.add_trace(
            go.Scatter(
                x=[data.index.min(), data.index.max()],
                y=[x, x],
                mode="lines",
                name=f"Suporte/Resistência {x} ({countByCluster[i]})",
                opacity=0.8,
                line=(dict(color="orange", dash="solid")),
            )
        )

        fig.add_annotation(
            x=data.index.min(),
            y=x,
            text=f"{x:.2f}",
            showarrow=False,
            font=dict(color="black", size=12),
            align="center",
            bgcolor="orange",
            opacity=0.9,
        )

    fig.update_layout(
        title="Suporte e Resistência - Principais faixas de preço onde o ativo pode encontrar dificuldade para subir (resistência) ou cair (suporte).",

        xaxis_title="Data",
        yaxis_title="Preço de Fechamento",
        width=1400,
        height=600,
        template="plotly_white",
    )

    st.plotly_chart(fig)


from io import StringIO

def baixar_csv(data):
    """Converte DataFrame para CSV em memória"""
    csv = data.to_csv(index=True, encoding='utf-8')
    return csv

# ==============================================
# CONFIGURAÇÃO DO STREAMLIT
# ==============================================
def main():
    st.title("Análise de Dados Financeiros com Streamlit")
    from datetime import date

    with st.sidebar:
        st.header("Ativos Financeiros")
        tickers_famosos = [
            "BTC-USD",  # Bitcoin
            "ETH-USD",  # Ethereum
            "VALE3.SA",  # Vale S.A.
            "PETR4.SA",  # Petrobras
            "AAPL",  # Apple
            "MSFT",  # Microsoft
            "GOOGL",  # Alphabet (Google)
            "AMZN",  # Amazon
            "TSLA",  # Tesla
            "META",  # Meta (Facebook)
            "NFLX",  # Netflix
            "NVDA",  # Nvidia
            "BRK-B",  # Berkshire Hathaway
            "JNJ",  # Johnson & Johnson
            "ITUB4.SA",  # Itaú Unibanco

        ]

        ticker = st.selectbox(
            "Digite ou selecione o ticker do ativo:",
            options=tickers_famosos,
            index=0,
            help="Você pode escolher um dos principais ativos ou digitar um ticker diferente"
        )

        start_date = st.date_input("Data de início", value=date(2024, 1, 1),
                                   help="Digite ou escolha a data inicial do ativo escolhido")
        end_date = st.date_input("Data de fim", value=date(2030, 1, 1),
                                 help="Digite ou escolha a data final do ativo escolhido")
        interval = st.selectbox("Intervalo de tempo:", list(INTERVALLIMITS.keys()),
                                index=list(INTERVALLIMITS.keys()).index("1d"), help='Selecione o intervalo entre os dados (1h, 1d, 5d, etc.). Intervalos menores como 1h mostram mais detalhes, mas têm limite máximo de dias. Intervalos diários ou maiores permitem análises de longo prazo.')


        window_rsi = st.slider("Dias para cálculo dos indicadores",
                               min_value=5, max_value=50, value=14, step=1,help='Determine o período de cálculo para RSI e osciladores (5-50). Valores menores reagem rápido a mudanças, mas são mais voláteis. 14 é o padrão do mercado para análises de médio prazo.')

        # Opções de inclusão de outliers
        st.subheader("Opções de Outliers")
        number_contamination = st.number_input(
            "Digite o valor para a sensibilidade dos outliers:",
            min_value=0.01,
            max_value=0.5,
            value=0.05,
            step=0.01,
            format="%.2f",
            help='Defina quão sensível será a detecção de valores atípicos (0.01-0.50). Valores mais altos capturam mais outliers, mas podem incluir variações normais. 0.05 (5%) é um bom equilíbrio para a maioria dos casos.'
        )
        st.subheader("Incluir Outliers no cálculo final")
        include_outliers_close = st.checkbox(
            "Incluir outliers de preço (Close)",
            value=False,
            help="Marque esta opção para incluir valores atípicos de preço na análise"
        )
        include_outliers_volume = st.checkbox(
            "Incluir outliers de volume",
            value=False,
            help="Marque esta opção para incluir valores atípicos de volume na análise"
        )

        # Opção original mantida para compatibilidade
        include_outliers = st.checkbox(
            "Incluir todos os outliers nos cálculos finais",
            value=False,
            help="Marque esta opção para incluir TODOS os valores atípicos na análise"
        )
        st.subheader("Baixar dados no formato CSV")


    # Coleta e processamento de dados
    data = getData(ticker, start_date, end_date, interval)
    # Botão de download - adicione esta parte
    csv = baixar_csv(data)

    st.sidebar.download_button(
        label="📥 Baixar dados como CSV",
        data=csv,
        file_name=f'dados_{ticker}_{start_date}_{end_date}.csv',
        mime='text/csv',
        help='Clique para baixar todos os dados processados em formato CSV'
    )

    if not data.empty:

        # Cálculo de métricas financeiras
        data["ReturnClose"] = data["Close"].pct_change()
        volPctChange = data["Volume"].pct_change()
        volPctChange.replace([np.inf, -np.inf], np.nan, inplace=True)
        volPctChange.fillna(0, inplace=True)
        data["ReturnVolume"] = volPctChange
        data["CloseStd"] = data["Close"].rolling(window=window_rsi).std()
        data["VolumeStd"] = data["Volume"].rolling(window=window_rsi).std()

        # Indicadores técnicos
        rsi = RSIIndicator(data["Close"], window=window_rsi)
        data["RSI"] = rsi.rsi()
        stoch = StochasticOscillator(data["High"], data["Low"], data["Close"])
        data["%K"] = stoch.stoch()

        # Limpeza e normalização
        data = clean(data)
        data["WickPercent"] = data.apply(
            lambda x: getCandleBodyToWickRatio(x["Open"], x["Close"], x["High"], x["Low"]),
            axis=1,
        )
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(subset=["WickPercent"], inplace=True)

        scaler = MinMaxScaler(feature_range=(0, 100))
        columns = list(["Close", "Volume", "ReturnClose", "ReturnVolume"])
        dataCols = data[["Close", "RSI", "%K", "WickPercent"]]
        dataScaler = scaler.fit_transform(data[columns])
        columns = columns[1:]
        columns.insert(0, "CloseScaler")
        dataNorm = pd.DataFrame(dataScaler, columns=columns, index=data.index)
        data = pd.concat([dataNorm, dataCols], axis=1)

        # Detecção de outliers
        isof = IsolationForest(n_estimators=100, contamination=number_contamination)
        data["OutlierClose"] = isof.fit_predict(data[["ReturnClose"]])
        data["OutlierVolume"] = isof.fit_predict(data[["ReturnVolume"]])

        # Exibição dos dados
        #st.write("Dados coletados e processados:")
        #st.write(data.head())

        # Visualização de outliers
        plotOutliers(data, isClose=True)

        plotOutliers(data, isClose=False)

        # Análise de clusters - modificação para usar as novas opções
        dataFilter = data

        # Determinar quais outliers incluir
        if include_outliers:  # Se a opção original estiver marcada, inclui tudo
            include_outliers_close = True
            include_outliers_volume = True

        # Criar condições de filtro
        cond_close = pd.Series(True, index=data.index) if include_outliers_close else (data["OutlierClose"] != -1)
        cond_volume = pd.Series(True, index=data.index) if include_outliers_volume else (data["OutlierVolume"] != -1)

        ranges = range(4, 11)
        bestK, silhouettes, kmeans, dataOutlier = filterCombinations(dataFilter, ranges, include_outliers=(
                    include_outliers_close or include_outliers_volume))

        if silhouettes is not None and not silhouettes.empty:
            st.subheader("Análise do Coeficiente de Silhueta")
            st.write("Melhor K encontrado:", bestK)
            st.write(
                "Avalie quão bem definidos estão os grupos/clusters (valores mais altos indicam grupos mais distintos).")
            plotSilhouettes(silhouettes)

        # Visualização de clusters
        if kmeans and dataOutlier is not None:
            st.write(f"\nMelhor valor de K = {bestK}")
            st.write(
                f"Veja como cada ponto está alinhado com seu cluster (áreas largas e uniformes indicam bons agrupamentos).")

            fig, ax = plt.subplots(figsize=(6, 2))
            visualizer = SilhouetteVisualizer(kmeans, colors="paired", ax=ax)
            visualizer.fit(dataOutlier)
            ax.tick_params(axis="both", labelsize=6)
            ax.set_title("Silhouette Plot", fontsize=7)
            ax.set_xlabel("Coeficiente de Silhueta", fontsize=7)
            ax.set_ylabel("Clusters", fontsize=7)
            st.pyplot(fig)

            dataOutlier = data.drop(columns=["Close", "OutlierClose", "OutlierVolume"])
            dataWithClusters, kmeans = assignKmeansClusters(data=dataOutlier, clusters=bestK)
            labels = kmeans.labels_

            plotClusters(dataWithClusters, labels)
            plotClusters3D(dataWithClusters, labels)

            dataWithoutClusters = dataWithClusters.drop(columns=["Cluster"])
            metrics = calculateClusteringMetrics(dataWithoutClusters, labels, kmeans)

            modeByCluster = getClustersMode(dataWithClusters)
            countByCluster = []
            rounds = 2

            for x in modeByCluster:
                countByCluster.append(dataOutlier["CloseScaler"].round(rounds).value_counts()[x])

            # Desnormalizar dados para plotar S/R
            modeByClusterReshaped = np.array([[value] for value in modeByCluster])
            n_features = scaler.min_.shape[0]
            modeByClusterPadded = np.zeros((modeByClusterReshaped.shape[0], n_features))
            modeByClusterPadded[:, 0] = modeByClusterReshaped[:, 0]
            modeByClusterDesnormalized = scaler.inverse_transform(modeByClusterPadded)
            modeByClusterDesnormalized = [x[0].round(rounds) for x in modeByClusterDesnormalized]

            plotSupportResistance(
                data=data,
                modeByCluster=modeByClusterDesnormalized,
                countByCluster=countByCluster,
                TICKER=ticker
            )
    else:
        st.write("Erro ao coletar dados. Verifique o ticker e as datas inseridas.")


if __name__ == "__main__":
    main()
