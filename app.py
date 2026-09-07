Python
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- Seiten-Konfiguration ---
st.set_page_config(page_title="Risk & Optimization Cockpit", layout="wide")
st.title("📈 Portfolio Risk & Optimization Cockpit")
st.markdown("Dieses Dashboard lädt historische Daten via Yahoo Finance, analysiert das Risiko und simuliert die Efficient Frontier nach Markowitz.")

# --- Sidebar (Eingaben) ---
st.sidebar.header("⚙️ Portfolio Parameter")

# Ticker Eingabe (Komma-getrennt)
ticker_input = st.sidebar.text_input("Ticker Symbole (Komma-getrennt)", value="AAPL, MSFT, SPY, ^VIX")
tickers = [t.strip().upper() for t in ticker_input.split(",")]

# Datumseingabe
start_date = st.sidebar.date_input("Startdatum", datetime.now() - timedelta(days=3*365))
end_date = st.sidebar.date_input("Enddatum", datetime.now())
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", value=2.0, step=0.1) / 100

load_data = st.sidebar.button("🚀 Daten laden & Analysieren")

# --- Hauptlogik ---
if load_data:
    if not tickers or tickers == [""]:
        st.warning("Bitte gib mindestens einen Ticker ein.")
    else:
        with st.spinner('Lade Daten von Yahoo Finance...'):
            try:
                # Daten herunterladen
                data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
                
                # Bei nur einem Ticker gibt yfinance eine Series zurück, wir brauchen ein DataFrame
                if isinstance(data, pd.Series):
                    data = data.to_frame(name=tickers[0])
                
                # Fehlende Werte entfernen
                data = data.dropna()
                
                if data.empty:
                    st.error("Keine Daten gefunden. Bitte überprüfe die Ticker und den Zeitraum.")
                    st.stop()
                    
                st.success("Daten erfolgreich geladen!")
                
                # --- Berechnungen ---
                daily_returns = data.pct_change().dropna()
                normalized_prices = (data / data.iloc[0]) * 100
                
                # Metriken (252 Handelstage)
                ann_returns = daily_returns.mean() * 252
                ann_volatility = daily_returns.std() * np.sqrt(252)
                sharpe_ratios = (ann_returns - risk_free_rate) / ann_volatility
                
                # Maximum Drawdown
                cum_returns = (1 + daily_returns).cumprod()
                rolling_max = cum_returns.cummax()
                drawdowns = (cum_returns / rolling_max) - 1
                max_drawdowns = drawdowns.min()
                
                # --- UI Tabs ---
                tab1, tab2, tab3 = st.tabs(["📊 Historie & Metriken", "⚠️ Risk Cockpit", "🎯 Optimierung & Simulation"])
                
                # --- Tab 1: Historie & Metriken ---
                with tab1:
                    st.subheader("Historische Performance (Normalisiert auf 100)")
                    fig_perf = px.line(normalized_prices, x=normalized_prices.index, y=normalized_prices.columns,
                                       labels={'value': 'Performance', 'variable': 'Asset', 'Date': 'Datum'})
                    st.plotly_chart(fig_perf, use_container_width=True)
                    
                    st.subheader("Key Performance Indicators (KPIs)")
                    metrics_df = pd.DataFrame({
                        'Ann. Rendite (%)': ann_returns * 100,
                        'Ann. Volatilität (%)': ann_volatility * 100,
                        'Sharpe Ratio': sharpe_ratios,
                        'Max Drawdown (%)': max_drawdowns * 100
                    }).round(2)
                    st.dataframe(metrics_df.style.highlight_max(subset=['Sharpe Ratio', 'Ann. Rendite (%)'], color='lightgreen')
                                               .highlight_min(subset=['Max Drawdown (%)', 'Ann. Volatilität (%)'], color='lightcoral'))

                # --- Tab 2: Risk Cockpit ---
                with tab2:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Korrelationsmatrix")
                        corr_matrix = daily_returns.corr()
                        fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", 
                                             color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
                        st.plotly_chart(fig_corr, use_container_width=True)
                        
                    with col2:
                        st.subheader("Annualisierte Kovarianzmatrix")
                        cov_matrix = daily_returns.cov() * 252
                        st.dataframe(cov_matrix.round(4), use_container_width=True)

                # --- Tab 3: Optimierung (Markowitz) ---
                with tab3:
                    if len(tickers) < 2:
                        st.info("Für eine Portfolio-Optimierung werden mindestens 2 Assets benötigt.")
                    else:
                        st.subheader("Monte-Carlo-Simulation & Efficient Frontier")
                        num_portfolios = 5000
                        
                        # Arrays für Ergebnisse
                        results = np.zeros((3, num_portfolios))
                        weights_record = []
                        
                        progress_bar = st.progress(0)
                        
                        for i in range(num_portfolios):
                            # Zufällige Gewichtung
                            weights = np.random.random(len(tickers))
                            weights /= np.sum(weights)
                            weights_record.append(weights)
                            
                            # Portfolio Metriken
                            p_return = np.sum(weights * ann_returns)
                            p_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                            
                            results[0,i] = p_return
                            results[1,i] = p_std_dev
                            results[2,i] = (p_return - risk_free_rate) / p_std_dev # Sharpe
                            
                            if i % 1000 == 0:
                                progress_bar.progress(i / num_portfolios)
                        
                        progress_bar.progress(1.0)
                        
                        # Optimale Portfolios identifizieren
                        idx_max_sharpe = np.argmax(results[2])
                        idx_min_vol = np.argmin(results[1])
                        
                        # Plotly Scatter Plot
                        fig_ef = go.Figure()
                        
                        # Alle simulierten Portfolios
                        fig_ef.add_trace(go.Scatter(
                            x=results[1,:], y=results[0,:],
                            mode='markers',
                            marker=dict(
                                size=5,
                                color=results[2,:], # Einfärben nach Sharpe
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(title='Sharpe Ratio')
                            ),
                            name='Simulierte Portfolios',
                            hoverinfo='text',
                            text=[f"Rendite: {r:.2%}<br>Vola: {v:.2%}<br>Sharpe: {s:.2f}" 
                                  for r, v, s in zip(results[0,:], results[1,:], results[2,:])]
                        ))
                        
                        # Max Sharpe Point
                        fig_ef.add_trace(go.Scatter(
                            x=[results[1, idx_max_sharpe]], y=[results[0, idx_max_sharpe]],
                            mode='markers+text',
                            marker=dict(color='red', size=15, symbol='star'),
                            name='Max Sharpe Ratio',
                            text=['Max Sharpe'], textposition="top center"
                        ))
                        
                        # Min Volatility Point
                        fig_ef.add_trace(go.Scatter(
                            x=[results[1, idx_min_vol]], y=[results[0, idx_min_vol]],
                            mode='markers+text',
                            marker=dict(color='orange', size=12, symbol='star'),
                            name='Min Volatility',
                            text=['Min Volatility'], textposition="bottom center"
                        ))
                        
                        fig_ef.update_layout(title='Efficient Frontier',
                                             xaxis_title='Volatilität (Risiko)',
                                             yaxis_title='Erwartete Rendite',
                                             showlegend=True)
                        
                        st.plotly_chart(fig_ef, use_container_width=True)
                        
                        # Tabelle für optimale Gewichtungen
                        st.subheader("Optimale Asset Allocation")
                        alloc_df = pd.DataFrame({
                            'Asset': tickers,
                            'Max Sharpe (%)': np.array(weights_record[idx_max_sharpe]) * 100,
                            'Min Volatility (%)': np.array(weights_record[idx_min_vol]) * 100
                        }).set_index('Asset').round(2)
                        
                        st.dataframe(alloc_df.style.background_gradient(cmap='Blues'))
                        
            except Exception as e:
                st.error(f"Es gab einen Fehler beim Verarbeiten der Daten: {e}")

st.markdown("---")
st.markdown("💡 *Erstellt im Rahmen des AI-Kurses. Nutzt yfinance für historische Marktdaten.*")
