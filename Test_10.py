import requests
import os
import yfinance as yf
import pandas as pd
import ta

from datetime import datetime
from FinMind.data import DataLoader


# =========================
# 讀取 Token
# =========================
def load_token():
    # 優先從環境變數讀取（GitHub Actions）
    token = os.environ.get("FINMIND_TOKEN")
    if token:
        return token
    # 本地開發時從檔案讀取
    try:
        with open("finmind_token.txt", "r") as f:
            return f.read().strip()
    except:
        print("❌ Token 讀取失敗")
        return None


# =========================
# FinMind 初始化
# =========================
api = DataLoader()

token = load_token()
if token:
    api.login_by_token(api_token=token)
else:
    print("⚠ 未登入 FinMind")


# =========================
# 股票清單（從檔案讀取）
# =========================
def load_watchlist():
    try:
        with open("watchlist.txt", "r") as f:
            return [line.strip() for line in f if line.strip()]
    except:
        print("❌ watchlist.txt 讀取失敗，使用空清單")
        return []

WATCHLIST = load_watchlist()


# =========================
# 持股（從檔案讀取）
# =========================
def load_holdings():
    try:
        with open("holdings.txt", "r") as f:
            return [line.strip() for line in f if line.strip()]
    except:
        print("❌ holdings.txt 讀取失敗，使用空清單")
        return []

HOLDINGS = load_holdings()


# =========================
# 股票名稱（從檔案讀取）
# =========================
def load_stock_names():
    try:
        names = {}
        with open("stock_names.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "," in line:
                    symbol, name = line.split(",", 1)
                    names[symbol.strip()] = name.strip()
        return names
    except:
        print("❌ stock_names.txt 讀取失敗，使用空字典")
        return {}

STOCK_NAMES = load_stock_names()


# =========================
# 抓資料
# =========================
def get_stock_data(symbol):

    try:
        # 台股
        if ".TW" in symbol:

            stock_id = symbol.replace(".TW", "")

            df = api.taiwan_stock_daily(
                stock_id=stock_id,
                start_date="2024-01-01",
                end_date=datetime.today().strftime("%Y-%m-%d")
            )

            if df.empty:
                return None

            df = df.rename(columns={
                "max": "High",
                "min": "Low",
                "open": "Open",
                "close": "Close",
                "Trading_Volume": "Volume"
            })

            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

            df = df[["Open","High","Low","Close","Volume"]]
            df = df.dropna()

            return df

        # 美股
        else:
            df = yf.download(symbol, period="6mo", auto_adjust=False)

            if df.empty:
                return None

            if hasattr(df.columns, "levels"):
                df.columns = df.columns.get_level_values(0)

            df = df.dropna()
            return df

    except Exception as e:
        print(symbol, "Error:", e)
        return None


# =========================
# 技術指標
# =========================
def add_indicators(df):

    close = df["Close"].astype(float)

    df["MA20"] = close.rolling(20).mean()
    df["VOL_MA20"] = df["Volume"].rolling(20).mean()

    macd = ta.trend.MACD(
        close,
        n_slow=60,
        n_fast=5,
        n_sign=20
    )

    df["MACD"] = macd.macd()
    df["MACD_SIGNAL"] = macd.macd_signal()

    df = df.dropna()

    return df



# =========================
# 籌碼面評分（台股限定）
# =========================
def get_institutional_score(symbol, consecutive_days=3):
    """
    外資連續買超 N 天 +1 分
    投信連續買超 N 天 +1 分
    """
    signals = []
    score = 0

    if ".TW" not in symbol:
        return signals, score

    try:
        stock_id = symbol.replace(".TW", "")

        df = api.taiwan_stock_institutional_investors(
            stock_id=stock_id,
            start_date="2024-01-01",
            end_date=datetime.today().strftime("%Y-%m-%d")
        )

        if df is None or df.empty:
            return signals, score

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        # 外資（Foreign_Investor）
        foreign = df[df["name"] == "Foreign_Investor"][["date", "buy", "sell"]].copy()
        foreign["net"] = foreign["buy"] - foreign["sell"]
        foreign = foreign.tail(consecutive_days)
        if len(foreign) >= consecutive_days and (foreign["net"] > 0).all():
            signals.append(f"外資連續買超{consecutive_days}天")
            score += 1

        # 投信（Investment_Trust）
        trust = df[df["name"] == "Investment_Trust"][["date", "buy", "sell"]].copy()
        trust["net"] = trust["buy"] - trust["sell"]
        trust = trust.tail(consecutive_days)
        if len(trust) >= consecutive_days and (trust["net"] > 0).all():
            signals.append(f"投信連續買超{consecutive_days}天")
            score += 1

    except Exception as e:
        print(f"{symbol} 籌碼資料錯誤: {e}")

    return signals, score


# =========================
# 技術策略（起漲版）
# =========================
def check_signal(df):

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    signals = []
    score = 0

    close_now = latest["Close"]
    close_prev = prev["Close"]

    ma20 = latest["MA20"]
    ma20_prev = prev["MA20"]

    macd_now = latest["MACD"]
    macd_prev = prev["MACD"]
    macd_signal_now = latest["MACD_SIGNAL"]
    macd_signal_prev = prev["MACD_SIGNAL"]

    # 突破MA20
    if close_now >= ma20 and close_prev < ma20_prev:
        signals.append("突破MA20")
        score += 3

    # MACD黃金交叉
    if macd_now > macd_signal_now and macd_prev <= macd_signal_prev:
        signals.append("MACD黃金交叉")
        score += 3

    # 動能
    if macd_now > macd_prev:
        signals.append("MACD動能增強")
        score += 1

    # 價格上漲
    if close_now > close_prev:
        signals.append("價格上漲")
        score += 1

    return signals, score


# =========================
# AI 評語
# =========================
def get_ai_comment(score, strength):

    if score >= 5 and strength < 8:
        return "🔥 剛突破且動能強，低風險起漲點"

    elif strength > 15:
        return "⚠ 漲幅過大，避免追高"

    elif score >= 4:
        return "👀 趨勢轉強，可觀察"

    else:
        return "⚠ 尚未形成趨勢"


# =========================
# 持股分析
# =========================
def check_holdings(df):

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    alerts = []

    close_now = latest["Close"]
    ma20 = latest["MA20"]

    macd_now = latest["MACD"]
    macd_prev = prev["MACD"]

    if close_now < ma20:
        alerts.append("⚠ 跌破MA20")
    else:
        alerts.append("✓ 站穩MA20")

    if macd_now < macd_prev:
        alerts.append("⚠ MACD動能下降")
    else:
        alerts.append("✓ MACD動能正常")

    return alerts


# =========================
# LINE 發送
# =========================
def send_line_message(message):
    # 優先從環境變數讀取（GitHub Actions）
    token = os.environ.get("LINE_CHANNEL_TOKEN")
    user_id = os.environ.get("LINE_USER_ID")

    # 本地開發時從檔案讀取
    if not token or not user_id:
        try:
            with open("line_channel_token.txt", "r") as f:
                token = f.read().strip()
            with open("line_user_id.txt", "r") as f:
                user_id = f.read().strip()
        except:
            print("❌ LINE 設定檔讀取失敗")
            return

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {
        "to": user_id,
        "messages": [{"type": "text", "text": message}]
    }

    response = requests.post(
        "https://api.line.me/v2/bot/message/push",
        headers=headers,
        json=payload
    )

    if response.status_code == 200:
        print("✅ LINE 發送成功")
    else:
        print(f"❌ 發送失敗：{response.status_code} {response.text}")


def send_line_long(message, max_len=4900):
    """自動分段發送長訊息"""
    chunks = [message[i:i+max_len] for i in range(0, len(message), max_len)]
    for chunk in chunks:
        send_line_message(chunk)


# =========================
# 主程式
# =========================
def run():

    results = []
    holding_results = []

    # ========= 選股 =========
    for stock in WATCHLIST:

        df = get_stock_data(stock)
        if df is None or len(df) < 60:
            continue

        df = add_indicators(df)

        signals, score = check_signal(df)

        # 籌碼面（台股限定）
        chip_signals, chip_score = get_institutional_score(stock)
        signals += chip_signals
        score += chip_score

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        price = float(latest["Close"])
        prev_price = float(prev["Close"])

        change_pct = (price - prev_price) / prev_price * 100

        ma20 = latest["MA20"]
        strength = (price - ma20) / ma20 * 100

        volume_now = latest["Volume"]
        volume_ma20 = latest["VOL_MA20"]

        ai_comment = get_ai_comment(score, strength)

        best = False
        if score >= 4 and strength <= 15 and volume_now > volume_ma20:
            best = True

        results.append({
            "stock": stock,
            "score": score,
            "signals": signals,
            "price": price,
            "change": change_pct,
            "strength": strength,
            "best": best,
            "comment": ai_comment
        })

    # ========= 持股（先算完） =========
    for stock in HOLDINGS:

        df = get_stock_data(stock)

        if df is None or len(df) < 60:
            continue

        df = add_indicators(df)

        alerts = check_holdings(df)

        holding_results.append({
            "stock": stock,
            "alerts": alerts
        })

    # ========= 排序 =========
    results.sort(key=lambda x: (x["score"], x["strength"]), reverse=True)

    # ========= 終端機輸出 =========
    print("\n📊【選股結果】\n")

    for r in results[:10]:

        name = STOCK_NAMES.get(r["stock"], "")

        print(f"{r['stock']} {name} 💰{r['price']:.2f} ({r['change']:+.2f}%) ⭐{r['score']} 📈{r['strength']:+.2f}%")

        for s in r["signals"]:
            print("  ✓", s)

        print("  🧠", r["comment"])

        if r["best"]:
            print("  🔥 最佳進場")

        print()

    print("\n📌【持股分析】\n")

    for h in holding_results:

        name = STOCK_NAMES.get(h["stock"], "")

        print(f"{h['stock']} {name}")

        for a in h["alerts"]:
            print(" ", a)

        print()

    # ========= 組合 LINE 訊息 =========
    now_str = datetime.today().strftime("%Y/%m/%d %H:%M")
    msg = f"📊【選股結果】{now_str}\n"

    for r in results[:10]:
        name = STOCK_NAMES.get(r["stock"], "")
        msg += f"\n{r['stock']} {name}\n"
        msg += f"💰{r['price']:.2f} ({r['change']:+.2f}%) ⭐{r['score']} 📈{r['strength']:+.2f}%\n"
        for s in r["signals"]:
            msg += f"  ✓ {s}\n"
        msg += f"  🧠 {r['comment']}\n"
        if r["best"]:
            msg += "  🔥 最佳進場\n"

    msg += "\n📌【持股分析】\n"
    for h in holding_results:
        name = STOCK_NAMES.get(h["stock"], "")
        msg += f"\n{h['stock']} {name}\n"
        for a in h["alerts"]:
            msg += f"  {a}\n"

    send_line_long(msg)


# =========================
# 執行
# =========================
if __name__ == "__main__":
    run()
