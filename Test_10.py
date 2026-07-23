import mplfinance as mpf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import requests
import os
import sys
import json
import yfinance as yf
import pandas as pd
import ta

from datetime import datetime, timezone, timedelta
import xml.etree.ElementTree as ET
from FinMind.data import DataLoader


# =========================
# 讀取 Token
# =========================
def load_token():
    token = os.environ.get("FINMIND_TOKEN")
    if token:
        return token
    try:
        with open("finmind_token.txt", "r") as f:
            return f.read().strip()
    except:
        print("❌ Token 讀取失敗")
        return None


# =========================
# 台灣時區
# =========================
TW_TZ = timezone(timedelta(hours=8))


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
        if ".TW" in symbol:
            stock_id = symbol.replace(".TW", "")
            df = api.taiwan_stock_daily(
                stock_id=stock_id,
                start_date="2024-01-01",
                end_date=datetime.now(TW_TZ).strftime("%Y-%m-%d")
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
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            df = df.dropna()
            return df
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
        window_slow=60,
        window_fast=5,
        window_sign=20
    )
    df["MACD"] = macd.macd()
    df["MACD_SIGNAL"] = macd.macd_signal()
    df = df.dropna()
    return df


# =========================
# 籌碼面評分（台股限定）
# =========================
def get_institutional_score(symbol, consecutive_days=3):
    signals = []
    score = 0
    if ".TW" not in symbol:
        return signals, score
    try:
        stock_id = symbol.replace(".TW", "")
        df = api.taiwan_stock_institutional_investors(
            stock_id=stock_id,
            start_date="2024-01-01",
            end_date=datetime.now(TW_TZ).strftime("%Y-%m-%d")
        )
        if df is None or df.empty:
            return signals, score
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        # 外資
        foreign = df[df["name"] == "Foreign_Investor"][["date", "buy", "sell"]].copy()
        foreign["net"] = foreign["buy"] - foreign["sell"]
        foreign = foreign.tail(consecutive_days)
        if len(foreign) >= consecutive_days:
            if (foreign["net"] > 0).all():
                signals.append(f"外資連續買超{consecutive_days}天")
                score += 1
            elif (foreign["net"] < 0).all():
                signals.append(f"外資連續賣超{consecutive_days}天")
                score -= 1

        # 投信
        trust = df[df["name"] == "Investment_Trust"][["date", "buy", "sell"]].copy()
        trust["net"] = trust["buy"] - trust["sell"]
        trust = trust.tail(consecutive_days)
        if len(trust) >= consecutive_days:
            if (trust["net"] > 0).all():
                signals.append(f"投信連續買超{consecutive_days}天")
                score += 1
            elif (trust["net"] < 0).all():
                signals.append(f"投信連續賣超{consecutive_days}天")
                score -= 1

    except Exception as e:
        print(f"{symbol} 籌碼資料錯誤: {e}")
    return signals, score


# =========================
# 技術策略（多空版）
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

    # 多頭訊號
    if close_now >= ma20 and close_prev < ma20_prev:
        signals.append("突破MA20")
        score += 3

    if macd_now > macd_signal_now and macd_prev <= macd_signal_prev:
        signals.append("MACD黃金交叉")
        score += 3

    if macd_now > macd_prev:
        signals.append("MACD動能增強")
        score += 1

    if close_now > close_prev:
        signals.append("價格上漲")
        score += 1

    # 空頭訊號
    if close_now < ma20 and close_prev >= ma20_prev:
        signals.append("跌破MA20")
        score -= 3

    if macd_now < macd_signal_now and macd_prev >= macd_signal_prev:
        signals.append("MACD死亡交叉")
        score -= 3

    if macd_now < macd_prev:
        signals.append("MACD動能減弱")
        score -= 1

    if close_now < close_prev:
        signals.append("價格下跌")
        score -= 1

    return signals, score


# =========================
# AI 評語
# =========================
def get_ai_comment(score, strength):
    if score >= 5 and strength < 8:
        return "🔥 剛突破且動能強，低風險起漲點"
    elif score <= -5 and strength > -8:
        return "🩸 剛跌破且動能弱，注意風險"
    elif strength > 15:
        return "⚠ 漲幅過大，避免追高"
    elif strength < -15:
        return "⚠ 跌幅過大，避免追空"
    elif score >= 4:
        return "👀 趨勢轉強，可觀察"
    elif score <= -4:
        return "👀 趨勢轉弱，可觀察"
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
# LINE 憑證
# =========================
def get_line_credentials():
    token = os.environ.get("LINE_CHANNEL_TOKEN")
    user_id = os.environ.get("LINE_USER_ID")
    if not token or not user_id:
        try:
            with open("line_channel_token.txt", "r") as f:
                token = f.read().strip()
            with open("line_user_id.txt", "r") as f:
                user_id = f.read().strip()
        except:
            print("❌ LINE 設定檔讀取失敗")
            return None, None
    return token, user_id


def send_line_message(message):
    token, user_id = get_line_credentials()
    if not token or not user_id:
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
    chunks = [message[i:i+max_len] for i in range(0, len(message), max_len)]
    for chunk in chunks:
        send_line_message(chunk)


# =========================
# 生成股票圖表
# =========================
def generate_chart(symbol, df):
    try:
        os.makedirs("charts", exist_ok=True)

        # 取足夠數據計算 MACD(5,60,20)，慢線需要至少 120 天
        df_full = df.copy()
        df_full.index = pd.to_datetime(df_full.index)
        df_full = df_full[["Open", "High", "Low", "Close", "Volume"]].astype(float)

        # 用全部數據計算指標
        ma20_full = df_full["Close"].rolling(20).mean()
        macd_obj = ta.trend.MACD(
            df_full["Close"],
            window_fast=5,
            window_slow=60,
            window_sign=20
        )
        macd_full = macd_obj.macd()
        signal_full = macd_obj.macd_signal()
        histogram_full = macd_obj.macd_diff()

        # 只取最近 60 天顯示
        df_chart = df_full.tail(60)
        ma20 = ma20_full.tail(60)
        macd = macd_full.tail(60)
        signal = signal_full.tail(60)
        histogram = histogram_full.tail(60)

        apds = [
            mpf.make_addplot(ma20, color='orange', width=1.5, label='MA20'),
            mpf.make_addplot(macd, panel=2, color='blue', width=1.2, label='MACD'),
            mpf.make_addplot(signal, panel=2, color='red', width=1.2, label='Signal'),
            mpf.make_addplot(histogram, panel=2, type='bar', color='gray', alpha=0.5),
        ]

        name = STOCK_NAMES.get(symbol, symbol)
        filename = f"charts/{symbol.replace('.', '_')}.png"

        mpf.plot(
            df_chart,
            type='candle',
            style='charles',
            title=f"{symbol} {name}",
            ylabel='Price',
            ylabel_lower='Volume',
            volume=True,
            addplot=apds,
            panel_ratios=(3, 1, 2),
            figsize=(12, 8),
            savefig=filename
        )

        plt.close('all')
        print(f"✅ 圖表已生成：{filename}")
        return filename

    except Exception as e:
        print(f"❌ 圖表生成失敗 {symbol}：{e}")
        return None


# =========================
# 建立多頭選股 Bubble
# =========================
def make_long_bubble(r, repo):
    symbol = r["stock"]
    name = STOCK_NAMES.get(symbol, symbol)
    today = datetime.now(TW_TZ).strftime("%Y%m%d")
    chart_url = f"https://github.com/{repo}/raw/refs/heads/main/charts/{symbol.replace('.', '_')}.png?v={today}"
    signals_text = "\n".join([f"✓ {s}" for s in r["signals"]])
    best_text = "\n🔥 最佳進場" if r["best"] else ""

    # 抓個股新聞
    news_count = 6 if r["score"] >= 6 else 4
    stock_news = fetch_stock_news(symbol, news_count)

    # 基本資訊
    body_contents = [
        {
            "type": "text",
            "text": f"💰 {r['price']:.2f} ({r['change']:+.2f}%)",
            "size": "md",
            "weight": "bold"
        },
        {
            "type": "text",
            "text": f"⭐ 評分：{r['score']}",
            "size": "sm",
            "color": "#555555",
            "margin": "sm"
        },
        {
            "type": "text",
            "text": signals_text + best_text,
            "size": "sm",
            "color": "#333333",
            "margin": "md",
            "wrap": True
        },
        {
            "type": "text",
            "text": r["comment"],
            "size": "sm",
            "color": "#27ae60",
            "margin": "sm",
            "wrap": True
        }
    ]

    # 個股新聞
    if stock_news:
        body_contents.append({
            "type": "separator",
            "margin": "md"
        })
        body_contents.append({
            "type": "text",
            "text": "📰 個股新聞",
            "weight": "bold",
            "size": "sm",
            "margin": "md",
            "color": "#333333"
        })
        for i, news in enumerate(stock_news):
            body_contents.append({
                "type": "text",
                "text": f"{i+1}. {news['title']}",
                "size": "xs",
                "color": "#555555",
                "wrap": True,
                "maxLines": 2,
                "margin": "sm",
                "action": {
                    "type": "uri",
                    "uri": news["link"]
                }
            })

    return {
        "type": "bubble",
        "size": "mega",
        "header": {
            "type": "box",
            "layout": "vertical",
            "contents": [{
                "type": "text",
                "text": f"📈 {symbol} {name}",
                "weight": "bold",
                "size": "lg",
                "color": "#ffffff"
            }],
            "backgroundColor": "#c0392b"
        },
        "hero": {
            "type": "image",
            "url": chart_url,
            "size": "full",
            "aspectRatio": "3:2",
            "aspectMode": "fit"
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "contents": body_contents
        }
    }


# =========================
# 建立空頭選股 Bubble
# =========================
def make_short_bubble(r, repo):
    symbol = r["stock"]
    name = STOCK_NAMES.get(symbol, symbol)
    today = datetime.now(TW_TZ).strftime("%Y%m%d")
    chart_url = f"https://github.com/{repo}/raw/refs/heads/main/charts/{symbol.replace('.', '_')}.png?v={today}"
    signals_text = "\n".join([f"✗ {s}" for s in r["signals"]])

    # 抓個股新聞
    news_count = 6 if r["score"] <= -6 else 4
    stock_news = fetch_stock_news(symbol, news_count)

    # 基本資訊
    body_contents = [
        {
            "type": "text",
            "text": f"💰 {r['price']:.2f} ({r['change']:+.2f}%)",
            "size": "md",
            "weight": "bold"
        },
        {
            "type": "text",
            "text": f"⭐ 評分：{r['score']}",
            "size": "sm",
            "color": "#555555",
            "margin": "sm"
        },
        {
            "type": "text",
            "text": signals_text,
            "size": "sm",
            "color": "#333333",
            "margin": "md",
            "wrap": True
        },
        {
            "type": "text",
            "text": r["comment"],
            "size": "sm",
            "color": "#e74c3c",
            "margin": "sm",
            "wrap": True
        }
    ]

    # 個股新聞
    if stock_news:
        body_contents.append({
            "type": "separator",
            "margin": "md"
        })
        body_contents.append({
            "type": "text",
            "text": "📰 個股新聞",
            "weight": "bold",
            "size": "sm",
            "margin": "md",
            "color": "#333333"
        })
        for i, news in enumerate(stock_news):
            body_contents.append({
                "type": "text",
                "text": f"{i+1}. {news['title']}",
                "size": "xs",
                "color": "#555555",
                "wrap": True,
                "maxLines": 2,
                "margin": "sm",
                "action": {
                    "type": "uri",
                    "uri": news["link"]
                }
            })

    return {
        "type": "bubble",
        "size": "mega",
        "header": {
            "type": "box",
            "layout": "vertical",
            "contents": [{
                "type": "text",
                "text": f"📉 {symbol} {name}",
                "weight": "bold",
                "size": "lg",
                "color": "#ffffff"
            }],
            "backgroundColor": "#27ae60"
        },
        "hero": {
            "type": "image",
            "url": chart_url,
            "size": "full",
            "aspectRatio": "3:2",
            "aspectMode": "fit"
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "contents": body_contents
        }
    }


# =========================
# 建立持股 Bubble
# =========================
def make_holding_bubble(h, repo):
    symbol = h["stock"]
    name = STOCK_NAMES.get(symbol, symbol)
    today = datetime.now(TW_TZ).strftime("%Y%m%d")
    chart_url = f"https://github.com/{repo}/raw/refs/heads/main/charts/{symbol.replace('.', '_')}.png?v={today}"
    alerts_text = "\n".join(h["alerts"])

    return {
        "type": "bubble",
        "size": "mega",
        "header": {
            "type": "box",
            "layout": "vertical",
            "contents": [{
                "type": "text",
                "text": f"📌 {symbol} {name}",
                "weight": "bold",
                "size": "lg",
                "color": "#ffffff"
            }],
            "backgroundColor": "#f39c12"
        },
        "hero": {
            "type": "image",
            "url": chart_url,
            "size": "full",
            "aspectRatio": "3:2",
            "aspectMode": "fit"
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "contents": [
                {
                    "type": "text",
                    "text": f"💰 {h['price']:.2f} ({h['change']:+.2f}%)",
                    "size": "md",
                    "weight": "bold"
                },
                {
                    "type": "text",
                    "text": "📊 持股分析",
                    "weight": "bold",
                    "size": "sm",
                    "margin": "md"
                },
                {
                    "type": "text",
                    "text": alerts_text,
                    "size": "sm",
                    "color": "#333333",
                    "margin": "sm",
                    "wrap": True
                }
            ]
        }
    }



# =========================
# 抓取財經新聞 RSS
# =========================
def fetch_news(rss_url, count=10):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(rss_url, headers=headers, timeout=10)
        root = ET.fromstring(response.content)
        items = root.findall(".//item")
        news = []
        for item in items[:count]:
            title = item.findtext("title", "").strip()
            link = item.findtext("link", "").strip()
            if title and link:
                news.append({"title": title, "link": link})
        return news
    except Exception as e:
        print(f"❌ RSS 抓取失敗 {rss_url}：{e}")
        return []


# =========================
# 抓取個股新聞 RSS
# =========================
def fetch_stock_news(symbol, count=6):
    try:
        if ".TW" in symbol:
            url = f"https://tw.stock.yahoo.com/rss?s={symbol}"
        else:
            url = f"https://finance.yahoo.com/rss/headline?s={symbol}"
        return fetch_news(url, count)
    except Exception as e:
        print(f"❌ 個股新聞抓取失敗 {symbol}：{e}")
        return []


# =========================
# 建立新聞 Bubble
# =========================
def make_news_bubble(title, news_list, bg_color):
    contents = []
    for i, news in enumerate(news_list):
        # 标题显示两行约 50 字
        label = f"{i+1}. {news['title']}"
        contents.append({
            "type": "box",
            "layout": "vertical",
            "contents": [{
                "type": "text",
                "text": label,
                "size": "sm",
                "color": "#333333",
                "wrap": True,
                "maxLines": 2,
                "action": {
                    "type": "uri",
                    "uri": news["link"]
                }
            }],
            "margin": "sm",
            "paddingBottom": "sm"
        })
        # 分隔线（最后一则不加）
        if i < len(news_list) - 1:
            contents.append({
                "type": "separator",
                "margin": "sm"
            })

    return {
        "type": "bubble",
        "size": "mega",
        "header": {
            "type": "box",
            "layout": "vertical",
            "contents": [{
                "type": "text",
                "text": title,
                "weight": "bold",
                "size": "lg",
                "color": "#ffffff"
            }],
            "backgroundColor": bg_color
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "contents": contents,
            "paddingAll": "md"
        }
    }


# =========================
# 建立個股新聞 Bubble
# =========================
def make_stock_news_bubble(symbol, news_list, bg_color):
    name = STOCK_NAMES.get(symbol, symbol)
    title = f"📰 {symbol} {name}"
    return make_news_bubble(title, news_list, bg_color)


# =========================
# 發送 LINE Flex
# =========================
def send_flex_carousel(bubbles, alt_text, token, user_id):
    if not bubbles:
        return
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    for i in range(0, len(bubbles), 12):
        chunk = bubbles[i:i+12]
        carousel = {"type": "carousel", "contents": chunk}
        payload = {
            "to": user_id,
            "messages": [{"type": "flex", "altText": alt_text, "contents": carousel}]
        }
        response = requests.post(
            "https://api.line.me/v2/bot/message/push",
            headers=headers,
            json=payload
        )
        if response.status_code == 200:
            print(f"✅ LINE Flex 發送成功（{len(chunk)} 支）")
        else:
            print(f"❌ LINE Flex 發送失敗：{response.status_code} {response.text}")


# =========================
# 主程式（分析 + 生成圖表）
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

    # ========= 持股 =========
    for stock in HOLDINGS:
        df = get_stock_data(stock)
        if df is None or len(df) < 60:
            continue
        df = add_indicators(df)
        alerts = check_holdings(df)

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        price = float(latest["Close"])
        prev_price = float(prev["Close"])
        change_pct = (price - prev_price) / prev_price * 100

        holding_results.append({
            "stock": stock,
            "alerts": alerts,
            "price": price,
            "change": change_pct
        })

    # ========= 排序 =========
    results.sort(key=lambda x: x["score"], reverse=True)

    # ========= 終端機輸出 =========
    print("\n📊【選股結果】\n")
    for r in results:
        if r["score"] >= 3 or r["score"] <= -3:
            name = STOCK_NAMES.get(r["stock"], "")
            direction = "📈" if r["score"] >= 3 else "📉"
            print(f"{direction} {r['stock']} {name} 💰{r['price']:.2f} ({r['change']:+.2f}%) ⭐{r['score']}")
            for s in r["signals"]:
                print("  •", s)
            print("  🧠", r["comment"])
            if r["best"]:
                print("  🔥 最佳進場")
            print()

    print("\n📌【持股分析】\n")
    for h in holding_results:
        name = STOCK_NAMES.get(h["stock"], "")
        print(f"{h['stock']} {name} 💰{h['price']:.2f} ({h['change']:+.2f}%)")
        for a in h["alerts"]:
            print(" ", a)
        print()

    # ========= 生成圖表 =========
    # 多頭 score >= 3，空頭 score <= -3，所有持股
    chart_symbols = set()
    for r in results:
        if r["score"] >= 3 or r["score"] <= -3:
            chart_symbols.add(r["stock"])
    for h in holding_results:
        chart_symbols.add(h["stock"])

    for symbol in chart_symbols:
        df = get_stock_data(symbol)
        if df is not None and len(df) >= 60:
            generate_chart(symbol, df)

    # ========= 儲存結果到 JSON =========
    now_str = datetime.now(TW_TZ).strftime("%Y/%m/%d %H:%M")
    output = {
        "now_str": now_str,
        "results": results,
        "holding_results": holding_results
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print("✅ 結果已儲存到 results.json")


# =========================
# 發送 LINE 通知
# =========================
def notify():
    try:
        with open("results.json", "r", encoding="utf-8") as f:
            output = json.load(f)
    except:
        print("❌ results.json 讀取失敗")
        return

    now_str = output["now_str"]
    results = output["results"]
    holding_results = output["holding_results"]

    repo = os.environ.get("GITHUB_REPO", "")
    token, user_id = get_line_credentials()
    if not token or not user_id:
        return

    # ========= 財經新聞 =========
    # 國際財經（多來源備援）
    intl_news = fetch_news("https://tw.news.yahoo.com/rss/finance", 10)
    if not intl_news:
        intl_news = fetch_news("https://tw.stock.yahoo.com/rss?category=intl-markets", 10)
    # 台灣財經（經濟日報）
    tw_news = fetch_news("https://money.udn.com/rssfeed/news/1001/5591?ch=money", 10)
    if not tw_news:
        tw_news = fetch_news("https://www.ctee.com.tw/feed", 10)

    news_bubbles = []
    if intl_news:
        news_bubbles.append(make_news_bubble("🌍 國際財經頭條", intl_news, "#1a3a5c"))
    if tw_news:
        news_bubbles.append(make_news_bubble("🇹🇼 台灣財經頭條", tw_news, "#1a5c2a"))

    if news_bubbles:
        send_flex_carousel(news_bubbles, f"📰 今日財經頭條 {now_str}", token, user_id)

    # ========= 多頭選股（score >= 3，分數高的在左） =========
    long_results = sorted([r for r in results if r["score"] >= 3], key=lambda x: x["score"], reverse=True)
    long_bubbles = [make_long_bubble(r, repo) for r in long_results]
    if long_bubbles:
        send_flex_carousel(long_bubbles, f"📈 多頭選股 {now_str}", token, user_id)
    else:
        send_line_message(f"📈 多頭選股 {now_str}\n今日無符合條件股票")

    # ========= 空頭選股（score <= -3，分數低的在左） =========
    short_results = sorted([r for r in results if r["score"] <= -3], key=lambda x: x["score"])
    short_bubbles = [make_short_bubble(r, repo) for r in short_results]
    if short_bubbles:
        send_flex_carousel(short_bubbles, f"📉 空頭選股 {now_str}", token, user_id)
    else:
        send_line_message(f"📉 空頭選股 {now_str}\n今日無符合條件股票")

    # ========= 持股分析 =========
    holding_bubbles = [make_holding_bubble(h, repo) for h in holding_results]
    if holding_bubbles:
        send_flex_carousel(holding_bubbles, f"📌 持股分析 {now_str}", token, user_id)


# =========================
# 執行
# =========================
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    if mode == "analyze":
        run()
    elif mode == "notify":
        notify()
