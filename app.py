import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from datetime import datetime, timedelta
import warnings
import base64
import io
warnings.filterwarnings('ignore')

# Configurazione pagina
st.set_page_config(
    page_title="Trading System Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titolo principale
st.title("📊 SISTEMA TRADING CON STOP LOSS")
st.markdown("**Analisi completa di trend e performance su S&P 500**")

# ========== SIDEBAR: PARAMETRI CONFIGURABILI ==========
st.sidebar.header("🔧 Configurazione Sistema")

# Parametri principali
ticker = st.sidebar.selectbox(
    "📈 Ticker da analizzare",
    ["^GSPC", "^IXIC", "^DJI", "^RUT", ^STXX50E", "^GDAXI", "FTSEMIB.MI", "^VIX", ""N225", "^HSI", "AAPL", "MSFT", "GOOGL", "TSLA"],
    index=0,
    help="Seleziona l'asset da analizzare"
)

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.sidebar.date_input(
        "📅 Data inizio",
        value=datetime(2013, 1, 1),
        min_value=datetime(2010, 1, 1),
        max_value=datetime.now()
    )
with col2:
    end_date = st.sidebar.date_input(
        "📅 Data fine",
        value=datetime.now(),
        min_value=start_date,
        max_value=datetime.now()
    )

# Parametri Trading System
st.sidebar.subheader("⚙️ Parametri Trading")
stop_loss_pct = st.sidebar.slider(
    "🛑 Stop Loss (%)",
    min_value=1.0,
    max_value=10.0,
    value=3.0,
    step=0.5,
    help="Percentuale di stop loss (es: 3.0 = -3%)"
)

# Parametri Rilevamento Trend
st.sidebar.subheader("📊 Parametri Trend")
trend_window_days = st.sidebar.slider(
    "📈 Finestra Trend (giorni)",
    min_value=10,
    max_value=120,
    value=20,
    step=5,
    help="Giorni per calcolare il trend"
)

min_trend_duration = st.sidebar.slider(
    "⏱️ Durata Minima Trend (giorni)",
    min_value=5,
    max_value=30,
    value=9,
    step=1,
    help="Durata minima del trend"
)

col3, col4 = st.sidebar.columns(2)
with col3:
    threshold_positive = st.sidebar.number_input(
        "📈 Soglia Rialzista",
        min_value=0.1,
        max_value=1.0,
        value=0.2,
        step=0.1,
        help="Soglia per trend rialzista"
    )
with col4:
    threshold_negative = st.sidebar.number_input(
        "📉 Soglia Ribassista",
        min_value=-1.0,
        max_value=-0.1,
        value=-0.2,
        step=0.1,
        help="Soglia per trend ribassista"
    )

# Pulsante per avviare l'analisi
run_analysis = st.sidebar.button("🚀 AVVIA ANALISI", type="primary", use_container_width=True)

# ========== FUNZIONI CORE (identiche allo script) ==========
@st.cache_data
def download_data(ticker, start_date, end_date):
    """Download dati con cache per velocità"""
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    return data

def detect_trend_changes(data, window_days=90, min_duration_days=60, threshold_pos=0.5, threshold_neg=-0.5):
    """Rileva i punti dove iniziano nuovi trend che dureranno mesi"""
    trends = []
    slopes = []
    dates = []
    
    # Calcola pendenze su finestra mobile
    for i in range(window_days, len(data)):
        window_data = data.iloc[i-window_days:i+1]
        x = np.arange(len(window_data))
        y = window_data['Close'].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        slope_normalized = slope * 100
        slopes.append(slope_normalized)
        dates.append(window_data.index[-1])
    
    slopes = np.array(slopes)
    
    # Rileva cambi di direzione significativi
    trend_changes = []
    current_trend = None
    trend_start_idx = 0
    threshold_positive = threshold_pos
    threshold_negative = threshold_neg
    
    for i in range(1, len(slopes)):
        curr_slope = slopes[i]
        
        if curr_slope > threshold_positive:
            new_trend = "RIALZISTA"
        elif curr_slope < threshold_negative:
            new_trend = "RIBASSISTA"
        else:
            new_trend = "LATERALE"
        
        if current_trend != new_trend and current_trend is not None:
            duration = i - trend_start_idx
            min_duration_check = min_duration_days // 5
            
            if duration >= min_duration_check:
                trend_slopes = slopes[trend_start_idx:i]
                trend_dates_range = dates[trend_start_idx:i]
                
                if len(trend_dates_range) > 0:
                    avg_slope = np.mean(trend_slopes)
                    max_slope = np.max(trend_slopes)
                    min_slope = np.min(trend_slopes)
                    
                    trend_changes.append({
                        'start_date': trend_dates_range[0],
                        'end_date': trend_dates_range[-1],
                        'trend_type': current_trend,
                        'avg_slope': avg_slope,
                        'max_slope': max_slope,
                        'min_slope': min_slope,
                        'duration_days': (trend_dates_range[-1] - trend_dates_range[0]).days,
                        'start_price': data.loc[trend_dates_range[0], 'Close'],
                        'end_price': data.loc[trend_dates_range[-1], 'Close']
                    })
        
        if current_trend != new_trend:
            current_trend = new_trend
            trend_start_idx = i
    
    # Aggiungi l'ultimo trend se abbastanza lungo
    min_duration_check = min_duration_days // 5
    if len(slopes) - trend_start_idx >= min_duration_check:
        trend_slopes = slopes[trend_start_idx:]
        trend_dates_range = dates[trend_start_idx:]
        
        if len(trend_dates_range) > 0:
            avg_slope = np.mean(trend_slopes)
            max_slope = np.max(trend_slopes)
            min_slope = np.min(trend_slopes)
            
            trend_changes.append({
                'start_date': trend_dates_range[0],
                'end_date': trend_dates_range[-1],
                'trend_type': current_trend,
                'avg_slope': avg_slope,
                'max_slope': max_slope,
                'min_slope': min_slope,
                'duration_days': (trend_dates_range[-1] - trend_dates_range[0]).days,
                'start_price': data.loc[trend_dates_range[0], 'Close'],
                'end_price': data.loc[trend_dates_range[-1], 'Close'],
                'is_current': True
            })
    
    return trend_changes, slopes, dates

def create_trading_system_with_stoploss(data, trends, stop_loss_pct=3.0):
    """Simula un trading system con stop loss"""
    trading_data = data.copy()
    trading_data['Signal'] = 0
    trading_data['Position'] = 0
    trading_data['Trade_Return'] = 0.0
    trading_data['Cumulative_Return'] = 1.0
    trading_data['Equity'] = 100000.0
    trading_data['Stop_Loss_Level'] = 0.0
    
    # Genera segnali di trading
    for trend in trends:
        start_date = trend['start_date']
        trend_type = trend['trend_type']
        
        try:
            signal_date_idx = trading_data.index.get_loc(start_date) + 1
            if signal_date_idx < len(trading_data):
                if trend_type == 'RIALZISTA':
                    trading_data.iloc[signal_date_idx:, trading_data.columns.get_loc('Signal')] = 1
                elif trend_type == 'RIBASSISTA':
                    trading_data.iloc[signal_date_idx:, trading_data.columns.get_loc('Signal')] = -1
                else:
                    trading_data.iloc[signal_date_idx:, trading_data.columns.get_loc('Signal')] = 0
        except (KeyError, IndexError):
            continue
    
    # Simula il trading con stop loss
    trades = []
    current_position = 0
    entry_price = 0
    entry_date = None
    capital = 100000.0
    shares = 0
    stop_loss_level = 0
    stop_loss_hit = False
    
    for i in range(1, len(trading_data)):
        date = trading_data.index[i]
        signal = trading_data.iloc[i]['Signal']
        prev_signal = trading_data.iloc[i-1]['Signal']
        open_price = trading_data.iloc[i]['Open']
        close_price = trading_data.iloc[i]['Close']
        low_price = trading_data.iloc[i]['Low']
        
        # CONTROLLO STOP LOSS se in posizione
        if current_position == 1 and not stop_loss_hit:
            if low_price <= stop_loss_level:
                exit_price = stop_loss_level
                exit_date = date
                trade_return = (exit_price - entry_price) / entry_price
                capital_after = capital * (1 + trade_return)
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return': trade_return,
                    'return_pct': trade_return * 100,
                    'capital_before': capital,
                    'capital_after': capital_after,
                    'duration_days': (exit_date - entry_date).days,
                    'exit_reason': 'STOP_LOSS'
                })
                
                capital = capital_after
                current_position = 0
                shares = 0
                stop_loss_hit = True
                continue
        
        # Rileva cambio di segnale
        if signal != prev_signal:
            # Chiudi posizione esistente
            if current_position == 1 and not stop_loss_hit:
                exit_price = open_price
                exit_date = date
                trade_return = (exit_price - entry_price) / entry_price
                capital_after = capital * (1 + trade_return)
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return': trade_return,
                    'return_pct': trade_return * 100,
                    'capital_before': capital,
                    'capital_after': capital_after,
                    'duration_days': (exit_date - entry_date).days,
                    'exit_reason': 'SIGNAL_CHANGE'
                })
                
                capital = capital_after
                current_position = 0
                shares = 0
            
            if signal != prev_signal:
                stop_loss_hit = False
            
            # Apri nuova posizione
            if signal == 1 and not stop_loss_hit:
                current_position = 1
                entry_price = open_price
                entry_date = date
                shares = capital / entry_price
                stop_loss_level = entry_price * (1 - stop_loss_pct / 100)
                trading_data.iloc[i, trading_data.columns.get_loc('Stop_Loss_Level')] = stop_loss_level
        
        # Aggiorna equity
        if current_position == 1:
            current_equity = shares * close_price
            trading_data.iloc[i, trading_data.columns.get_loc('Equity')] = current_equity
            trading_data.iloc[i, trading_data.columns.get_loc('Position')] = 1
            trading_data.iloc[i, trading_data.columns.get_loc('Stop_Loss_Level')] = stop_loss_level
        else:
            trading_data.iloc[i, trading_data.columns.get_loc('Equity')] = capital
            trading_data.iloc[i, trading_data.columns.get_loc('Position')] = 0
    
    # Chiudi eventuale posizione finale
    if current_position == 1:
        exit_price = trading_data.iloc[-1]['Close']
        exit_date = trading_data.index[-1]
        trade_return = (exit_price - entry_price) / entry_price
        capital_after = capital * (1 + trade_return)
        
        trades.append({
            'entry_date': entry_date,
            'exit_date': exit_date,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'return': trade_return,
            'return_pct': trade_return * 100,
            'capital_before': capital,
            'capital_after': capital_after,
            'duration_days': (exit_date - entry_date).days,
            'is_open': True,
            'exit_reason': 'STILL_OPEN'
        })
        
        final_equity = shares * exit_price
        trading_data.iloc[-1, trading_data.columns.get_loc('Equity')] = final_equity
    
    return trading_data, trades

def calculate_trading_statistics_with_stoploss(trades, trading_data):
    """Calcola le statistiche del trading system con CAGR CORRETTO"""
    if not trades:
        return None
    
    # CALCOLO PERIODO E CAGR CORRETTO
    start_date = trading_data.index[0]
    end_date = trading_data.index[-1]
    
    # Rimuovi timezone se presente
    if hasattr(start_date, 'tz') and start_date.tz is not None:
        start_date = start_date.tz_localize(None)
    if hasattr(end_date, 'tz') and end_date.tz is not None:
        end_date = end_date.tz_localize(None)
    
    # Calcola il periodo in anni (preciso)
    period_days = (end_date - start_date).days
    years = period_days / 365.25
    
    # Statistiche di base
    num_trades = len(trades)
    winning_trades = [t for t in trades if t['return'] > 0]
    losing_trades = [t for t in trades if t['return'] < 0]
    stop_loss_trades = [t for t in trades if t.get('exit_reason') == 'STOP_LOSS']
    
    num_winning = len(winning_trades)
    num_losing = len(losing_trades)
    num_stop_loss = len(stop_loss_trades)
    win_rate = (num_winning / num_trades) * 100 if num_trades > 0 else 0
    stop_loss_rate = (num_stop_loss / num_trades) * 100 if num_trades > 0 else 0
    
    # Rendimenti
    returns = [t['return'] for t in trades]
    total_return = (trading_data['Equity'].iloc[-1] / trading_data['Equity'].iloc[0] - 1) * 100
    avg_return = np.mean(returns) * 100
    std_return = np.std(returns) * 100
    
    # CALCOLO CAGR CORRETTO
    initial_value = trading_data['Equity'].iloc[0]
    final_value = trading_data['Equity'].iloc[-1]
    
    if initial_value > 0 and years > 0:
        cagr = ((final_value / initial_value) ** (1 / years)) - 1
        cagr_pct = cagr * 100
    else:
        cagr_pct = 0
    
    # Best/Worst trades
    best_trade = max(returns) * 100 if returns else 0
    worst_trade = min(returns) * 100 if returns else 0
    
    # Durata media
    avg_duration = np.mean([t['duration_days'] for t in trades]) if trades else 0
    
    # Calcola drawdown
    equity_series = trading_data['Equity']
    running_max = equity_series.expanding().max()
    drawdown = (equity_series - running_max) / running_max * 100
    max_drawdown = drawdown.min()
    
    # Sharpe ratio (approssimato)
    if std_return > 0:
        sharpe_ratio = (avg_return * np.sqrt(252)) / (std_return * np.sqrt(252))
    else:
        sharpe_ratio = 0
    
    stats = {
        'num_trades': num_trades,
        'num_winning': num_winning,
        'num_losing': num_losing,
        'num_stop_loss': num_stop_loss,
        'win_rate': win_rate,
        'stop_loss_rate': stop_loss_rate,
        'total_return': total_return,
        'avg_return': avg_return,
        'std_return': std_return,
        'best_trade': best_trade,
        'worst_trade': worst_trade,
        'avg_duration': avg_duration,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'final_equity': final_value,
        'initial_equity': initial_value,
        'cagr_annual': cagr_pct,
        'analysis_years': years,
        'analysis_days': period_days,
        'start_date': start_date,
        'end_date': end_date
    }
    
    return stats, drawdown

def create_trend_plot(data, trends, ticker):
    """Crea grafico trend per Streamlit"""
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # ANDAMENTO SOTTOSTANTE
    ax.plot(data.index, data['Close'], color='#1E1E1E', linewidth=2, 
            label=f'{ticker} - Prezzo', alpha=0.8)
    
    # COLORI PER TIPI DI TREND
    colors = {
        'RIALZISTA': '#00AA00',
        'RIBASSISTA': '#FF0000',
        'LATERALE': '#4682B4'
    }
    
    # MARCA I PUNTI DI CAMBIO TREND
    for i, trend in enumerate(trends):
        start_date = trend['start_date']
        start_price = trend['start_price']
        trend_type = trend['trend_type']
        
        marker = '^' if trend_type == 'RIALZISTA' else 'v' if trend_type == 'RIBASSISTA' else 'o'
        marker_color = colors[trend_type]
        
        ax.scatter([start_date], [start_price], 
                  color=marker_color, s=150, marker=marker, 
                  zorder=10, edgecolors='white', linewidth=2)
        
        duration_months = trend['duration_days'] // 30
        price_change = ((trend['end_price'] - trend['start_price']) / trend['start_price']) * 100
        
        ax.annotate(f'{trend_type}\n{duration_months}m\n{price_change:+.1f}%', 
                   xy=(start_date, start_price), 
                   xytext=(0, 40 if trend_type == 'RIALZISTA' else -40), 
                   textcoords='offset points',
                   ha='center', va='bottom' if trend_type == 'RIALZISTA' else 'top',
                   fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=marker_color, alpha=0.8),
                   color='white', zorder=12)
    
    # Prezzo attuale
    current_price = data['Close'].iloc[-1]
    ax.axhline(y=current_price, color='#FF6600', linewidth=2, 
              linestyle='--', alpha=0.8, 
              label=f'Prezzo attuale: ${current_price:.2f}')
    
    # Legenda
    for trend_type, color in colors.items():
        ax.plot([], [], color=color, linewidth=3, label=f'Trend {trend_type}', alpha=0.8)
    
    ax.set_title(f'🎯 IDENTIFICAZIONE TREND - {ticker}', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Data', fontsize=12, fontweight='bold')
    ax.set_ylabel('Prezzo ($)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    
    # Date formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig

def create_performance_plot(trading_data, trades, stats, drawdown_series, ticker, stop_loss_pct):
    """Crea grafico performance per Streamlit"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    # GRAFICO 1: PREZZO + SEGNALI DI TRADING
    position_data = trading_data[trading_data['Position'] == 1]
    no_position_data = trading_data[trading_data['Position'] == 0]
    
    if not no_position_data.empty:
        ax1.plot(no_position_data.index, no_position_data['Close'], 
                color='#FFD700', linewidth=2, alpha=0.8, label='Nessuna posizione')
    
    if not position_data.empty:
        ax1.plot(position_data.index, position_data['Close'], 
                color='#228B22', linewidth=2, alpha=0.9, label='Posizione Long')
    
    stop_loss_data = trading_data[trading_data['Stop_Loss_Level'] > 0]
    if not stop_loss_data.empty:
        ax1.plot(stop_loss_data.index, stop_loss_data['Stop_Loss_Level'], 
                color='red', linewidth=1.5, linestyle='--', alpha=0.7, 
                label=f'Stop Loss Level (-{stop_loss_pct}%)')
    
    # Marca entrate e uscite
    for trade in trades:
        ax1.scatter([trade['entry_date']], [trade['entry_price']], 
                   color='green', s=100, marker='^', zorder=10,
                   edgecolors='white', linewidth=1)
        
        if not trade.get('is_open', False):
            if trade.get('exit_reason') == 'STOP_LOSS':
                ax1.scatter([trade['exit_date']], [trade['exit_price']], 
                           color='red', s=120, marker='X', zorder=10,
                           edgecolors='white', linewidth=1)
            else:
                ax1.scatter([trade['exit_date']], [trade['exit_price']], 
                           color='red', s=100, marker='v', zorder=10,
                           edgecolors='white', linewidth=1)
    
    ax1.set_title(f'📊 TRADING SYSTEM - {ticker}', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Prezzo ($)', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    
    # GRAFICO 2: EQUITY LINE
    initial_equity = trading_data['Equity'].iloc[0]
    ax2.plot(trading_data.index, trading_data['Equity'], color='#0066CC', linewidth=2)
    ax2.axhline(y=initial_equity, color='gray', linestyle='--', alpha=0.5)
    
    ax2.fill_between(trading_data.index, trading_data['Equity'], initial_equity,
                     where=(trading_data['Equity'] >= initial_equity),
                     color='green', alpha=0.2, interpolate=True)
    ax2.fill_between(trading_data.index, trading_data['Equity'], initial_equity,
                     where=(trading_data['Equity'] < initial_equity),
                     color='red', alpha=0.2, interpolate=True)
    
    ax2.set_title('💰 EQUITY LINE', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Capitale ($)', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # GRAFICO 3: DRAWDOWN
    ax3.fill_between(trading_data.index, drawdown_series, 0, color='red', alpha=0.5)
    ax3.plot(trading_data.index, drawdown_series, color='darkred', linewidth=2)
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    
    ax3.set_title('📉 DRAWDOWN', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Drawdown (%)', fontsize=10)
    ax3.set_xlabel('Data', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Formato date
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

# ========== ESECUZIONE DELL'ANALISI ==========
if run_analysis:
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Download dati
        status_text.text("📥 Download dati...")
        progress_bar.progress(10)
        stock_data = download_data(ticker, start_date, end_date)
        
        if len(stock_data) < 100:
            st.error("❌ Dati insufficienti per l'analisi. Seleziona un periodo più lungo.")
            st.stop()
        
        # Step 2: Rilevamento trend
        status_text.text("🔍 Rilevamento trend...")
        progress_bar.progress(30)
        trends, slopes_history, dates_history = detect_trend_changes(
            stock_data, 
            window_days=trend_window_days, 
            min_duration_days=min_trend_duration,
            threshold_pos=threshold_positive,
            threshold_neg=threshold_negative
        )
        
        # Step 3: Trading system
        status_text.text("💼 Simulazione trading...")
        progress_bar.progress(60)
        trading_data, trades = create_trading_system_with_stoploss(
            stock_data, trends, stop_loss_pct=stop_loss_pct
        )
        
        # Step 4: Statistiche
        status_text.text("📊 Calcolo statistiche...")
        progress_bar.progress(80)
        stats, drawdown_series = calculate_trading_statistics_with_stoploss(trades, trading_data)
        
        # Step 5: Completamento
        status_text.text("✅ Analisi completata!")
        progress_bar.progress(100)
        
        # Pausa per mostrare il completamento
        import time
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        # ========== RISULTATI ==========
        st.success("🎉 Analisi completata con successo!")
        
        # Riepilogo parametri
        st.subheader("📋 Parametri utilizzati")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📈 Ticker", ticker)
            st.metric("🛑 Stop Loss", f"-{stop_loss_pct}%")
        with col2:
            st.metric("📅 Periodo", f"{(stock_data.index[-1] - stock_data.index[0]).days} giorni")
            st.metric("📊 Trend Window", f"{trend_window_days} giorni")
        with col3:
            st.metric("⏱️ Min Duration", f"{min_trend_duration} giorni")
            st.metric("📈 Soglia +", f"{threshold_positive}")
        with col4:
            st.metric("📉 Soglia -", f"{threshold_negative}")
            st.metric("🔍 Trend trovati", len(trends))
        
        # Performance principale
        st.subheader("🎯 Performance Generale")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "💰 Capitale Finale", 
                f"${stats['final_equity']:,.0f}",
                f"${stats['final_equity'] - stats['initial_equity']:,.0f}"
            )
        with col2:
            st.metric(
                "📊 Rendimento Totale", 
                f"{stats['total_return']:+.1f}%"
            )
        with col3:
            st.metric(
                "🎯 CAGR", 
                f"{stats['cagr_annual']:+.2f}%",
                f"{stats['cagr_annual'] - 10:+.1f}% vs S&P500"
            )
        with col4:
            st.metric(
                "📉 Max Drawdown", 
                f"{stats['max_drawdown']:.1f}%"
            )
        with col5:
            st.metric(
                "📊 Sharpe Ratio", 
                f"{stats['sharpe_ratio']:.2f}"
            )
        
        # Statistiche operazioni
        st.subheader("🎲 Statistiche Operazioni")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Totale Operazioni", stats['num_trades'])
            st.metric("⏱️ Durata Media", f"{stats['avg_duration']:.0f} giorni")
        with col2:
            st.metric("✅ Operazioni Vincenti", f"{stats['num_winning']} ({stats['win_rate']:.1f}%)")
            st.metric("🚀 Miglior Trade", f"{stats['best_trade']:+.1f}%")
        with col3:
            st.metric("❌ Operazioni Perdenti", f"{stats['num_losing']} ({100-stats['win_rate']:.1f}%)")
            st.metric("💥 Peggior Trade", f"{stats['worst_trade']:+.1f}%")
        with col4:
            st.metric("🛑 Stop Loss Attivate", f"{stats['num_stop_loss']} ({stats['stop_loss_rate']:.1f}%)")
            st.metric("📊 Rendimento Medio", f"{stats['avg_return']:+.2f}%")
        
        # Grafici
        st.subheader("📈 Grafico Identificazione Trend")
        trend_fig = create_trend_plot(stock_data, trends, ticker)
        st.pyplot(trend_fig)
        
        st.subheader("💹 Grafico Performance Trading")
        perf_fig = create_performance_plot(trading_data, trades, stats, drawdown_series, ticker, stop_loss_pct)
        st.pyplot(perf_fig)
        
        # Tabella operazioni
        st.subheader("📋 Dettaglio Operazioni")
        if trades:
            trades_df = pd.DataFrame([
                {
                    'Data Entrata': trade['entry_date'].strftime('%Y-%m-%d'),
                    'Prezzo Entrata': f"${trade['entry_price']:.2f}",
                    'Data Uscita': trade['exit_date'].strftime('%Y-%m-%d'),
                    'Prezzo Uscita': f"${trade['exit_price']:.2f}",
                    'Rendimento': f"{trade['return_pct']:+.1f}%",
                    'Durata': f"{trade['duration_days']} giorni",
                    'Motivo': 'Stop Loss' if trade.get('exit_reason') == 'STOP_LOSS' else 'Fine Trend'
                }
                for trade in trades
            ])
            st.dataframe(trades_df, use_container_width=True)
        
        # Verifica matematica
        st.subheader("🔍 Verifica Matematica CAGR")
        calculated_final = stats['initial_equity'] * ((1 + stats['cagr_annual']/100) ** stats['analysis_years'])
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🧮 Capitale Calcolato", f"${calculated_final:,.0f}")
        with col2:
            st.metric("💰 Capitale Reale", f"${stats['final_equity']:,.0f}")
        with col3:
            st.metric("✅ Differenza", f"${abs(calculated_final - stats['final_equity']):,.0f}")
        
        if abs(calculated_final - stats['final_equity']) < 1000:
            st.success("✅ CAGR matematicamente corretto!")
        else:
            st.warning("⚠️ Possibile errore nel calcolo CAGR")
            
    except Exception as e:
        st.error(f"❌ Errore durante l'analisi: {str(e)}")
        progress_bar.empty()
        status_text.empty()

else:
    # Schermata iniziale
    st.info("👈 Configura i parametri nella sidebar e clicca 'AVVIA ANALISI' per iniziare")
    
    # Istruzioni
    st.subheader("📖 Come usare questa app")
    st.markdown("""
    **1. Configurazione (Sidebar):**
    - Seleziona il ticker da analizzare (S&P 500, NASDAQ, singole azioni)
    - Imposta le date di inizio e fine analisi
    - Configura la percentuale di stop loss (1-10%)
    - Regola i parametri di rilevamento trend
    
    **2. Avvio Analisi:**
    - Clicca il pulsante "🚀 AVVIA ANALISI"
    - Attendi il completamento (download dati + calcoli)
    
    **3. Risultati:**
    - Performance generale del sistema
    - Grafici interattivi di trend e trading
    - Statistiche dettagliate delle operazioni
    - Verifica matematica del CAGR
    """)
    
    st.subheader("ℹ️ Informazioni sul Sistema")
    st.markdown("""
    **🎯 Strategia:**
    - Acquisto 1 giorno dopo segnale rialzista
    - Stop loss automatico alla percentuale impostata
    - Vendita 1 giorno dopo segnale ribassista
    - Nessuna posizione durante trend ribassisti
    
    **📊 Metriche Calcolate:**
    - CAGR (Compound Annual Growth Rate) corretto
    - Maximum Drawdown
    - Sharpe Ratio
    - Win Rate e analisi stop loss
    """)

# Footer
st.markdown("---")
st.markdown("**📱 App ottimizzata per mobile | 🖥️ Funziona anche su desktop**")
