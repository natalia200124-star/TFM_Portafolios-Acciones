import streamlit as st 
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

st.title("Optimización de Portafolios – Modelo de Markowitz")

tickers_input = st.text_input(
    "Ingrese los tickers separados por comas (ej: AAPL,MSFT,GOOGL):"
)

if st.button("Ejecutar optimización"):

    tickers = [t.strip().upper() for t in tickers_input.split(",")]

    try:
        # ============================================================
        # DESCARGA DE PRECIOS CLOSE DESDE 2024
        # ============================================================
        data = yf.download(tickers, start="2024-01-01")["Close"]

        if isinstance(data.columns, pd.MultiIndex):
            data = data.droplevel(0, axis=1)

        st.subheader("Precios de cierre (2024)")
        st.dataframe(data.head())

        # ================= Base Metrics =================
        returns = data.pct_change().dropna()
        mean_returns = returns.mean()
        cov_matrix = returns.cov()

        # ============================================================
        # FUNCIONES DE OPTIMIZACIÓN
        # ============================================================
        def negative_sharpe(weights):
            port_ret = np.sum(weights * mean_returns)
            port_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            return -(port_ret / port_vol)

        def volatility(weights):
            return np.sqrt(weights.T @ cov_matrix @ weights)

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = tuple((0, 1) for _ in tickers)
        initial_weights = np.ones(len(tickers)) / len(tickers)

        # ============================================================
        # OPTIMIZACIÓN SHARPE MÁXIMO
        # ============================================================
        result_sharpe = minimize(
            negative_sharpe, initial_weights, method="SLSQP",
            bounds=bounds, constraints=constraints
        )
        weights_sharpe = result_sharpe.x
        ret_sharpe = np.sum(weights_sharpe * mean_returns)
        vol_sharpe = np.sqrt(weights_sharpe.T @ cov_matrix @ weights_sharpe)
        sharpe_ratio = ret_sharpe / vol_sharpe

        # ============================================================
        # OPTIMIZACIÓN VOLATILIDAD MÍNIMA
        # ============================================================
        result_minvol = minimize(
            volatility, initial_weights, method="SLSQP",
            bounds=bounds, constraints=constraints
        )
        weights_minvol = result_minvol.x
        ret_minvol = np.sum(weights_minvol * mean_returns)
        vol_minvol = np.sqrt(weights_minvol.T @ cov_matrix @ weights_minvol)
        sharpe_minvol = ret_minvol / vol_minvol

        # ============================================================
        # PORTAFOLIO DE PESOS IGUALES
        # ============================================================
        weights_equal = np.array([1/len(tickers)] * len(tickers))
        ret_equal = np.sum(weights_equal * mean_returns)
        vol_equal = np.sqrt(weights_equal.T @ cov_matrix @ weights_equal)
        sharpe_equal = ret_equal / vol_equal

        # ============================================================
        # RENDIMIENTOS ACUMULADOS
        # ============================================================
        cumulative_returns_assets = (1 + returns).cumprod()

        daily_port = returns.dot(weights_sharpe)
        cumulative_portfolio = (1 + daily_port).cumprod()

        # ============================================================
        # TABLA DE PRECIOS PRINCIPALES (UNA SOLA TABLA)
        # ============================================================
        st.subheader("Precios principales de cierre (2025)")

        key_prices = data.loc[data.index.year.isin([2024, 2025])]
        key_prices = key_prices.tail(10)   # Los últimos precios representativos

        st.dataframe(key_prices)

        # ============================================================
        # EXPORTACIÓN CSV COMPLETA
        # ============================================================
        export_df = pd.DataFrame({"Fecha": data.index})

        for t in tickers:
            export_df[f"Close_{t}"] = data[t].values
            export_df[f"Retorno_{t}"] = returns[t].reindex(export_df["Fecha"]).values
            export_df[f"Acumulado_{t}"] = cumulative_returns_assets[t].reindex(export_df["Fecha"]).values

        export_df["Retorno_Portafolio"] = daily_port.reindex(export_df["Fecha"]).values
        export_df["Acumulado_Portafolio"] = cumulative_portfolio.reindex(export_df["Fecha"]).values

        export_df.to_csv("resultados_portafolio.csv", index=False)

        st.success("Archivo CSV generado correctamente.")

        # ============================================================
        # RESULTADOS DE LOS PORTAFOLIOS
        # ============================================================
        st.subheader("Comparación de los tres portafolios")

        df_port = pd.DataFrame({
            "Portafolio": ["Sharpe Máximo", "Mínima Volatilidad", "Pesos Iguales"],
            "Retorno Diario Esperado": [ret_sharpe, ret_minvol, ret_equal],
            "Volatilidad": [vol_sharpe, vol_minvol, vol_equal],
            "Sharpe Ratio": [sharpe_ratio, sharpe_minvol, sharpe_equal]
        })

        st.dataframe(df_port)

        # ============================================================
        # GRÁFICO DE LÍNEAS POR TICKER (PRECIOS)
        # ============================================================
        st.subheader("Tendencia de precios desde 2024")
        st.line_chart(data)

        # ============================================================
        # PESOS DEL PORTAFOLIO SHARPE (CON PORCENTAJE)
        # ============================================================
        st.subheader("Pesos óptimos (Portafolio Sharpe Máximo)")

        df_sharpe = pd.DataFrame({
            "Ticker": tickers,
            "Peso Decimal": np.round(weights_sharpe, 5),
            "Peso (%)": np.round(weights_sharpe * 100, 2)
        })

        st.dataframe(df_sharpe)

        # Gráfico de barras
        fig, ax = plt.subplots()
        ax.barh(df_sharpe["Ticker"], df_sharpe["Peso Decimal"])
        ax.set_xlabel("Peso")
        st.pyplot(fig)

        # ============================================================
        # INTERPRETACIÓN AUTOMÁTICA DE LOS PESOS
        # ============================================================
        st.subheader("¿Qué significan estos pesos y cómo invertirlos?")

        st.write("""
        Los **pesos óptimos** representan cuánto debería invertirse en cada acción
        para lograr el mejor equilibrio entre riesgo y retorno.  
        Por ejemplo:
        - Un peso del **40%** significa que por cada 100 dólares invertidos,
          40 deberían ir a esa acción.
        - Si un activo tiene un peso muy bajo, el modelo considera que aporta
          más riesgo que beneficio.
        
        **Para un inversor principiante**, esta tabla funciona como una guía directa:
        solo debe distribuir su dinero siguiendo estos porcentajes, lo cual 
        reduce riesgos y mejora la probabilidad de obtener retornos consistentes.
        """)

        # ============================================================
        # RENDIMIENTO ACUMULADO POR ACCIÓN
        # ============================================================
        st.subheader("Rendimiento acumulado por acción")
        st.line_chart(cumulative_returns_assets)

        # ============================================================
        # RENDIMIENTO ACUMULADO DEL PORTAFOLIO SHARPE
        # ============================================================
        st.subheader("Rendimiento acumulado del portafolio (Sharpe)")
        st.line_chart(cumulative_portfolio)

        # ============================================================
        # INTERPRETACIÓN AUTOMÁTICA JUSTIFICADA
        # ============================================================
        st.subheader("Interpretación automática del mejor portafolio")

        best = max(
            [
                ("Sharpe Máximo", sharpe_ratio),
                ("Mínima Volatilidad", sharpe_minvol),
                ("Pesos Iguales", sharpe_equal)
            ],
            key=lambda x: x[1]
        )[0]

        if best == "Sharpe Máximo":
            st.write("""
            El portafolio recomendado es **Sharpe Máximo**, porque logra el mejor
            equilibrio entre riesgo y retorno.  
            Para una persona sin experiencia en bolsa, este portafolio ofrece la
            forma más eficiente de invertir:  
            **obtiene más retorno por cada unidad de riesgo**.
            """)
        elif best == "Mínima Volatilidad":
            st.write("""
            Se recomienda el portafolio de **Mínima Volatilidad**, ideal para quienes
            buscan seguridad y estabilidad.  
            Minimiza las variaciones y protege mejor el capital.
            """)
        else:
            st.write("""
            El portafolio sugerido es **Pesos Iguales**, una estrategia muy sencilla
            para quienes desean invertir sin complicarse.  
            Al distribuir el dinero de manera uniforme, se reduce el impacto de un solo
            activo sobre el total del portafolio.
            """)

        st.success(f"El mejor portafolio según los datos es: **{best}**")

    except Exception as e:
        st.error(f"Error: {e}")

