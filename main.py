import argparse
import numpy as np
import pandas as pd
from collections import deque
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

INITIAL_CASH     = 100_000
CONTRACT_MULT    = 50
FAST_MA          = 50
SLOW_MA          = 200
LOOKBACK         = 300
MOMENTUM_LB      = 126
STOP_LOSS_PCT    = 0.025
TRAIL_STOP_PCT   = 0.030
POSITION_PCT     = 0.95
REBALANCE_DAYS   = 5


def fetch_data(ticker="ES=F", start="2022-01-01", end="2024-12-31"):
    import yfinance as yf
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No data returned for {ticker}.")
    raw = raw.reset_index()
    raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
    raw = raw.rename(columns={"Datetime": "Date", "datetime": "Date"})
    raw["Date"] = pd.to_datetime(raw["Date"])
    df = raw[["Date", "Open", "High", "Low", "Close", "Volume"]].dropna()
    return df.sort_values("Date").reset_index(drop=True)


def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains[-period:].mean()
    avg_loss = losses[-period:].mean()
    if avg_loss == 0:
        return 100.0
    return 100 - (100 / (1 + avg_gain / avg_loss))


def calculate_signals(prices):
    p = np.array(prices)
    cur = p[-1]
    fast_ma   = p[-FAST_MA:].mean()
    slow_ma   = p[-SLOW_MA:].mean()
    momentum  = (cur - p[-MOMENTUM_LB]) / p[-MOMENTUM_LB]
    ma_slope  = (fast_ma - slow_ma) / slow_ma
    volatility = np.diff(p[-21:]).std() / p[-21:-1].mean()
    rsi = calculate_rsi(p[-30:], 14)

    go_long = (
        cur > fast_ma and
        cur > slow_ma and
        fast_ma > slow_ma and
        momentum > 0 and
        ma_slope > 0.01 and
        rsi < 75
    )
    go_short = not go_long and (
        cur < fast_ma or
        cur < slow_ma or
        fast_ma < slow_ma or
        momentum < -0.05 or
        volatility > 0.02
    )
    return {
        "go_long": go_long,
        "go_short": go_short,
        "neutral": not go_long and not go_short,
        "fast_ma": fast_ma, "slow_ma": slow_ma,
        "momentum": momentum, "volatility": volatility, "rsi": rsi,
    }


def run_backtest(df):
    prices         = deque(maxlen=LOOKBACK)
    nav            = float(INITIAL_CASH)
    in_position    = False
    entry_price    = None
    highest_price  = None
    last_rebalance = None

    equity_curve  = []
    trades        = []
    daily_returns = []

    for _, row in df.iterrows():
        price = float(row["Close"])
        date  = row["Date"]
        prices.append(price)

        equity_curve.append({"date": date, "nav": nav, "price": price})

        if len(prices) < 2:
            daily_returns.append(0.0)
            continue

        price_ret = (price - float(list(prices)[-2])) / float(list(prices)[-2])
        if in_position:
            daily_returns.append(price_ret * POSITION_PCT)
            nav *= (1 + price_ret * POSITION_PCT)
        else:
            daily_returns.append(0.0)

        if len(prices) < SLOW_MA:
            continue

        if in_position and highest_price is not None:
            highest_price = max(highest_price, price)

        stopped = False
        if in_position and entry_price is not None:
            pnl_pct            = (price - entry_price) / entry_price
            drawdown_from_high = (highest_price - price) / highest_price

            if pnl_pct <= -STOP_LOSS_PCT:
                trades.append({"entry": entry_price, "exit": price, "pnl_pct": pnl_pct, "reason": "stop_loss"})
                in_position = False; entry_price = None; highest_price = None
                last_rebalance = date; stopped = True

            elif drawdown_from_high >= TRAIL_STOP_PCT:
                trades.append({"entry": entry_price, "exit": price, "pnl_pct": pnl_pct, "reason": "trail_stop"})
                in_position = False; entry_price = None; highest_price = None
                last_rebalance = date; stopped = True

        if stopped:
            continue

        if last_rebalance is not None and (date - last_rebalance).days < REBALANCE_DAYS:
            continue

        sig = calculate_signals(list(prices))

        if sig["go_long"] and not in_position:
            in_position   = True
            entry_price   = price
            highest_price = price
            last_rebalance = date

        elif (sig["go_short"] or sig["neutral"]) and in_position:
            pnl_pct = (price - entry_price) / entry_price
            trades.append({"entry": entry_price, "exit": price, "pnl_pct": pnl_pct, "reason": "signal_exit"})
            in_position = False; entry_price = None; highest_price = None
            last_rebalance = date

    equity_curve_df = pd.DataFrame(equity_curve)
    equity_curve_df["nav"] = INITIAL_CASH * (1 + np.array(daily_returns)).cumprod()

    return (
        equity_curve_df,
        pd.DataFrame(trades) if trades else pd.DataFrame(columns=["entry","exit","pnl_pct","reason"]),
        np.array(daily_returns),
    )


def max_drawdown(nav):
    peak = np.maximum.accumulate(nav)
    return ((nav - peak) / peak).min()


def sharpe(ret, rf=0.05/252):
    excess = ret - rf
    return np.sqrt(252) * excess.mean() / (excess.std() + 1e-10)


def sortino(ret, rf=0.05/252):
    excess   = ret - rf
    downside = excess[excess < 0]
    dstd     = np.sqrt(np.mean(downside**2) + 1e-10)
    return np.sqrt(252) * excess.mean() / dstd


def calmar(ret, nav):
    ann = (1 + ret.mean()) ** 252 - 1
    mdd = abs(max_drawdown(nav))
    return ann / mdd if mdd != 0 else 0


def win_rate(trades_df):
    if trades_df.empty:
        return 0.0
    return (trades_df["pnl_pct"] > 0).mean()


def profit_factor(trades_df):
    if trades_df.empty:
        return 0.0
    wins   = trades_df.loc[trades_df["pnl_pct"] > 0, "pnl_pct"].sum()
    losses = trades_df.loc[trades_df["pnl_pct"] < 0, "pnl_pct"].abs().sum()
    return wins / losses if losses != 0 else float("inf")


def monte_carlo(nav_end, ann_return, ann_vol, n_sims=10_000, horizon=252):
    mu_dt  = (ann_return - 0.5 * ann_vol**2) / 252
    sig_dt = ann_vol / np.sqrt(252)
    z      = np.random.normal(0, 1, (n_sims, horizon))
    return nav_end * np.exp(np.cumsum(mu_dt + sig_dt * z, axis=1)[:, -1])


def fit_t(returns):
    df_t, loc, scale = stats.t.fit(returns)
    ll_t    = np.sum(stats.t.logpdf(returns, df_t, loc, scale))
    ll_norm = np.sum(stats.norm.logpdf(returns, returns.mean(), returns.std()))
    return {
        "df": df_t, "loc": loc, "scale": scale,
        "ll_t": ll_t, "ll_norm": ll_norm,
        "aic_t": 2*3 - 2*ll_t,
        "aic_norm": 2*2 - 2*ll_norm,
    }


def heston_rn_prob(S0, v0, r=0.05, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
                   T=1.0, n_sims=50_000):
    n_steps = 252
    dt = T / n_steps
    S  = np.full(n_sims, float(S0))
    v  = np.full(n_sims, float(v0))
    for _ in range(n_steps):
        dw1 = np.random.normal(0, np.sqrt(dt), n_sims)
        dw2 = rho * dw1 + np.sqrt(1 - rho**2) * np.random.normal(0, np.sqrt(dt), n_sims)
        v   = np.maximum(v + kappa * (theta - v) * dt + xi * np.sqrt(np.maximum(v, 0)) * dw2, 0)
        S   = S * np.exp((r - 0.5 * v) * dt + np.sqrt(np.maximum(v, 0)) * dw1)
    return float(np.mean(S > S0))


def print_results(equity_df, trades_df, daily_returns):
    nav = equity_df["nav"].values
    ret = daily_returns

    total_ret = (nav[-1] - nav[0]) / nav[0]
    ann_ret   = (nav[-1] / nav[0]) ** (252 / len(nav)) - 1
    ann_vol   = ret.std() * np.sqrt(252)
    mdd       = max_drawdown(nav)
    sr        = sharpe(ret)
    so        = sortino(ret)
    cal       = calmar(ret, nav)
    wr        = win_rate(trades_df)
    pf        = profit_factor(trades_df)

    active_ret = ret[ret != 0]
    if len(active_ret) == 0:
        active_ret = ret
    log_ret = np.log(1 + active_ret)
    t       = fit_t(active_ret)

    mc             = monte_carlo(nav[-1], ann_ret, ann_vol)
    mc_prob_profit = np.mean(mc > nav[-1])

    rn_prob = heston_rn_prob(equity_df["price"].values[-1], v0=ann_vol**2)

    print(f"\nPeriod          : {equity_df['date'].iloc[0].date()} -> {equity_df['date'].iloc[-1].date()}")
    print(f"Start NAV       : ${nav[0]:,.0f}")
    print(f"End NAV         : ${nav[-1]:,.0f}")
    print(f"Total Return    : {total_ret:.2%}")
    print(f"Ann. Return     : {ann_ret:.2%}")
    print(f"Ann. Volatility : {ann_vol:.2%}")
    print(f"Max Drawdown    : {mdd:.2%}")
    print(f"Sharpe          : {sr:.4f}")
    print(f"Sortino         : {so:.4f}")
    print(f"Calmar          : {cal:.4f}")
    print(f"Win Rate        : {wr:.2%}")
    print(f"Profit Factor   : {pf:.4f}")
    print(f"Total Trades    : {len(trades_df)}")

    print(f"\nLog Return Mean : {log_ret.mean()*100:.4f}%")
    print(f"Log Return Std  : {log_ret.std()*100:.4f}%")
    print(f"Skewness        : {stats.skew(log_ret):.4f}")
    print(f"Excess Kurtosis : {stats.kurtosis(log_ret):.4f}")

    print(f"\nT-dist df       : {t['df']:.2f}")
    print(f"T-dist loc      : {t['loc']*100:.4f}%")
    print(f"T-dist scale    : {t['scale']*100:.4f}%")
    print(f"LL (t)          : {t['ll_t']:.2f}")
    print(f"LL (norm)       : {t['ll_norm']:.2f}")
    print(f"AIC (t)         : {t['aic_t']:.2f}")
    print(f"AIC (norm)      : {t['aic_norm']:.2f}")

    print(f"\nMC P5           : ${np.percentile(mc, 5):,.0f}")
    print(f"MC P25          : ${np.percentile(mc, 25):,.0f}")
    print(f"MC P50          : ${np.percentile(mc, 50):,.0f}")
    print(f"MC P75          : ${np.percentile(mc, 75):,.0f}")
    print(f"MC P95          : ${np.percentile(mc, 95):,.0f}")
    print(f"MC Mean         : ${mc.mean():,.0f}")
    print(f"MC P(profit)    : {mc_prob_profit:.2%}")

    print(f"\nHeston P(up)    : {rn_prob:.2%}")
    print(f"Heston P(down)  : {1-rn_prob:.2%}")

    if not trades_df.empty:
        print("\nTrade breakdown:")
        print(trades_df.groupby("reason")["pnl_pct"].describe().round(4).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    print("Fetching ES=F data from Yahoo Finance (2022-01-01 to 2024-12-31)...")
    df = fetch_data()
    print(f"Loaded {len(df)} trading days")

    equity_df, trades_df, daily_returns = run_backtest(df)
    print_results(equity_df, trades_df, daily_returns)