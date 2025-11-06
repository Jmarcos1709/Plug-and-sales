import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# --- Configurações da Página ---
st.set_page_config(
    page_title="Dashboard de Performance de Vendas",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Funções de Pré-processamento ---

def load_and_preprocess_data(uploaded_file):
    """Carrega e pré-processa o arquivo de dados."""
    
    # Tenta ler o arquivo como CSV. O separador será inferido.
    try:
        # Tenta ler com o separador mais comum (vírgula)
        df = pd.read_csv(uploaded_file, header=None, sep=',')
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        return None

    # Renomear as colunas de interesse
    # Coluna 3 (índice 2) -> Gasto Total de Tráfego
    # Coluna 22 (índice 21) -> Total de Atendimentos
    # Coluna 24 (índice 23) -> Número de Vendas
    
    # Mapeamento de colunas (índice baseado em 0)
    col_map = {
        2: 'Gasto_Trafego',
        21: 'Total_Atendimentos',
        23: 'Numero_Vendas'
    }
    
    # Verifica se as colunas existem no DataFrame
    for index in col_map.keys():
        if index not in df.columns:
            st.error(f"Erro: A coluna esperada no índice {index+1} não foi encontrada no arquivo.")
            return None

    # Seleciona e renomeia as colunas
    df = df.rename(columns=col_map)
    df = df[list(col_map.values())]

    # Função para limpar e converter valores monetários/numéricos
    def clean_numeric(series):
        # Remove "R$", pontos e substitui vírgula por ponto para conversão
        series = series.astype(str).str.replace('R$', '', regex=False).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
        return pd.to_numeric(series, errors='coerce')

    # Limpeza e conversão
    df['Gasto_Trafego'] = clean_numeric(df['Gasto_Trafego'])
    df['Total_Atendimentos'] = pd.to_numeric(df['Total_Atendimentos'], errors='coerce')
    df['Numero_Vendas'] = pd.to_numeric(df['Numero_Vendas'], errors='coerce')

    # Remove linhas com valores nulos após a conversão
    df.dropna(inplace=True)
    
    # Garante que as colunas de contagem sejam inteiras
    df['Total_Atendimentos'] = df['Total_Atendimentos'].astype(int)
    df['Numero_Vendas'] = df['Numero_Vendas'].astype(int)

    return df

# --- Funções de Cálculo de Métricas ---

def calculate_kpis(df):
    """Calcula os KPIs principais."""
    total_traffic_cost = df['Gasto_Trafego'].sum()
    total_leads = df['Total_Atendimentos'].sum()
    total_sales = df['Numero_Vendas'].sum()
    
    # Custo por Atendimento (CPA)
    cpa = total_traffic_cost / total_leads if total_leads > 0 else 0
    
    # Taxa de Conversão (Lead para Venda)
    conversion_rate = (total_sales / total_leads) * 100 if total_leads > 0 else 0
    
    # Custo por Aquisição (CPL) - Custo por Venda
    cost_per_acquisition = total_traffic_cost / total_sales if total_sales > 0 else 0
    
    return {
        'Gasto Total': total_traffic_cost,
        'Total Atendimentos': total_leads,
        'Total Vendas': total_sales,
        'Custo por Atendimento (CPA)': cpa,
        'Taxa de Conversão (%)': conversion_rate,
        'Custo por Aquisição (CPA)': cost_per_acquisition
    }

# --- Funções de Visualização (CORRIGIDAS PARA MATPLOTLIB/SEABORN) ---

def plot_correlation(df):
    """Cria um gráfico de dispersão para analisar a correlação."""
    
    st.subheader("Análise de Correlação: Atendimentos vs. Vendas")
    
    # Calcula a correlação de Pearson
    correlation = df['Total_Atendimentos'].corr(df['Numero_Vendas'])
    st.info(f"Coeficiente de Correlação (Pearson) entre Atendimentos e Vendas: **{correlation:.2f}**")
    
    # --- CÓDIGO MATPLOTLIB/SEABORN ---
    
    # 1. Criar a "figura" e os "eixos" (o canvas) do Matplotlib
    fig, ax = plt.subplots()
    
    # 2. Usar o Seaborn para desenhar o gráfico de regressão (scatter + linha)
    sns.regplot(
        data=df,
        x='Total_Atendimentos',
        y='Numero_Vendas',
        ax=ax,  # Informa ao Seaborn para desenhar neste "canvas"
        scatter_kws={'alpha': 0.6, 'edgecolors': 'DarkSlateGrey'}, # Estilo
        line_kws={'color': 'red'} # Estilo da linha
    )
    
    # 3. Definir títulos e legendas no Matplotlib
    ax.set_title('Relação entre Total de Atendimentos e Número de Vendas')
    ax.set_xlabel('Total de Atendimentos (Leads)')
    ax.set_ylabel('Número de Vendas')
    
    # 4. Usar o comando CERTO do Streamlit para Matplotlib
    st.pyplot(fig, use_container_width=True)

def plot_traffic_vs_sales(df):
    """Cria um gráfico de dispersão para Gasto de Tráfego vs. Vendas."""
    
    st.subheader("Análise de Eficiência: Gasto de Tráfego vs. Vendas")
    
    # Agrupa os dados para melhor visualização se houver muitas linhas
    df_agg = df.groupby('Gasto_Trafego').agg(
        Total_Vendas=('Numero_Vendas', 'sum'),
        Total_Atendimentos=('Total_Atendimentos', 'sum')
    ).reset_index()
    
    # --- CÓDIGO MATPLOTLIB/SEABORN ---
    
    # 1. Criar o canvas
    fig, ax = plt.subplots()
    
    # 2. Usar o Seaborn para desenhar o gráfico de dispersão (bolhas)
    sns.scatterplot(
        data=df_agg,
        x='Gasto_Trafego',
        y='Total_Vendas',
        size='Total_Atendimentos', # O tamanho do ponto representa o volume de atendimentos
        sizes=(50, 1000),         # Define o range do tamanho
        alpha=0.7,
        edgecolors='DarkSlateGrey',
        ax=ax
    )
    
    # 3. Definir títulos e legendas
    ax.set_title('Gasto de Tráfego vs. Vendas (Tamanho do Ponto = Atendimentos)')
    ax.set_xlabel('Gasto de Tráfego (R$)')
    ax.set_ylabel('Total de Vendas')
    ax.legend(title='Total Atendimentos') # Adiciona legenda para o tamanho
    
    # 4. Usar o comando CERTO do Streamlit para Matplotlib
    st.pyplot(fig, use_container_width=True)

# --- Layout do Dashboard ---

def main():
    st.title("📈 Dashboard de Performance de Vendas e Tráfego")
    st.markdown("Análise de métricas de marketing e vendas: Gasto de Tráfego, Atendimentos e Vendas.")

    # Sidebar para upload de arquivo
    st.sidebar.header("Upload de Dados")
    uploaded_file = st.sidebar.file_uploader("Carregue seu arquivo CSV/TXT", type=["csv", "txt"])

    if uploaded_file is not None:
        # Para o Streamlit ler o arquivo, precisamos de um objeto de arquivo
        # O arquivo do usuário foi salvo como /home/ubuntu/upload/dadospreenchidos.txt
        # Vamos simular o upload para o ambiente de teste
        if st.session_state.get('test_mode', False):
            with open("/home/ubuntu/upload/dadospreenchidos.txt", "r") as f:
                uploaded_file = StringIO(f.read())
        
        df = load_and_preprocess_data(uploaded_file)

        if df is not None and not df.empty:
            st.success(f"Dados carregados com sucesso! {len(df)} registros válidos encontrados.")
            
            # 1. KPIs
            kpis = calculate_kpis(df)
            st.header("Métricas Chave de Performance (KPIs)")
            
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            col1.metric("Gasto Total", f"R$ {kpis['Gasto Total']:,.2f}")
            col2.metric("Total Atendimentos", f"{kpis['Total Atendimentos']:,}")
            col3.metric("Total Vendas", f"{kpis['Total Vendas']:,}")
            col4.metric("Custo por Atendimento (CPA)", f"R$ {kpis['Custo por Atendimento (CPA)']:.2f}")
            col5.metric("Taxa de Conversão", f"{kpis['Taxa de Conversão (%)']:.2f}%")
            col6.metric("Custo por Aquisição (CPA)", f"R$ {kpis['Custo por Aquisição (CPA)']:.2f}")

            st.markdown("---")

            # 2. Análise de Correlação
            plot_correlation(df)
            
            st.markdown("---")

            # 3. Análise de Eficiência
            plot_traffic_vs_sales(df)
            
            st.markdown("---")
            
            # 4. Visualização dos Dados Brutos (Opcional)
            if st.checkbox("Mostrar Dados Brutos"):
                st.dataframe(df)

        elif df is not None and df.empty:
            st.warning("O DataFrame resultante está vazio após o pré-processamento. Verifique se as colunas 3, 22 e 24 contêm dados válidos.")
    else:
        st.info("Por favor, carregue um arquivo de dados para começar a análise.")

if __name__ == "__main__":
    # Define o modo de teste para carregar o arquivo fornecido pelo usuário
    st.session_state['test_mode'] = True
    main()
    st.session_state['test_mode'] = False # Desativa o modo de teste após a execução inicial