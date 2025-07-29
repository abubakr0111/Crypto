from flask import Flask
from threading import Thread
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, MessageHandler, ContextTypes, filters
import logging
import os
import json
import csv
import requests
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import re
from datetime import datetime
from io import BytesIO
import telegram
import asyncio

# 🔐 Токен и ID админа
TOKEN = '7313454103:AAGJEMNktJUABdhxky-BT8eTSAJ3TpcLdvA'
ADMIN_ID = 664563521

# Константы
TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d']
user_state = {}
user_list = set()

# Логгирование
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# Главное меню
main_menu = ReplyKeyboardMarkup(
    [[KeyboardButton("📈 Начать прогноз"), KeyboardButton("📖 Инструкция")]],
    resize_keyboard=True)

# Экранирование Markdown
def escape_markdown(text):
    return re.sub(r'([_*()~`>#+=|{}.!\-])', r'\\1', text)

# Старт
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_list.add(chat_id)
    user_state[chat_id] = {}
    await update.message.reply_text(escape_markdown(
        "Привет! Я бот для прогнозов криптовалют 🔮\n\n"
        "Нажми «📈 Начать прогноз», чтобы ввести пару, или «📖 Инструкция»"),
        reply_markup=main_menu, parse_mode="MarkdownV2")

# Инструкция
async def instruction(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(escape_markdown(
        "📖 Как пользоваться ботом:\n"
        "1. Нажми «📈 Начать прогноз»\n"
        "2. Введи пару (например: BTCUSDT)\n"
        "3. Выбери таймфрейм\n"
        "4. Получишь график + прогноз\n\n"
        "🛠️ Также есть админ-панель /admin"),
        parse_mode="MarkdownV2")

# Обработка сообщений
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = update.message.text.strip().upper()

    if text == "📈 НАЧАТЬ ПРОГНОЗ":
        user_state[chat_id] = {}
        await update.message.reply_text(
            escape_markdown("Введите торговую пару (например: BTCUSDT)"),
            parse_mode="MarkdownV2")
    elif text == "📖 ИНСТРУКЦИЯ":
        await instruction(update, context)
    elif chat_id in user_state and 'pair' not in user_state[chat_id]:
        user_state[chat_id]['pair'] = text
        await send_timeframe_buttons(update, context)
    else:
        await update.message.reply_text(
            escape_markdown("Пожалуйста, выберите действие через меню."),
            reply_markup=main_menu, parse_mode="MarkdownV2")

# Кнопки таймфрейма
async def send_timeframe_buttons(update: Update, context: ContextTypes.DEFAULT_TYPE):
    buttons = [[InlineKeyboardButton(tf, callback_data=f"tf:{tf}")] for tf in TIMEFRAMES]
    buttons.append([InlineKeyboardButton("⬅️ Назад", callback_data="back")])
    await update.message.reply_text("Выберите таймфрейм:", reply_markup=InlineKeyboardMarkup(buttons))

# Обработка кнопок
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat.id
    data = query.data

    if data.startswith("tf:"):
        tf = data.split(":")[1]
        pair = user_state[chat_id].get("pair")
        if pair:
            await query.edit_message_text("⏳ Получаю данные...")
            candles = get_futures_candles(pair, tf)
            if candles is not None:
                add_indicators(candles)
                fig = plot_candlestick(candles)
                buf = BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                plt.close(fig)

                prediction = escape_markdown(predict_trend(candles))
                support, resistance = get_support_resistance(candles)
                tp = calc_tp(candles)
                sl = calc_sl(candles)

                caption = (f"<b>{pair} — {tf}</b>\n"
                           f"📈 Прогноз:\n{prediction}\n"
                           f"🔻 Поддержка: {support:.2f}\n"
                           f"🔺 Сопротивление: {resistance:.2f}\n"
                           f"🎯 Take Profit: {tp}\n"
                           f"🛑 Stop Loss: {sl}")

                await context.bot.send_photo(chat_id=chat_id,
                                             photo=buf,
                                             filename="chart.png",
                                             caption=caption,
                                             parse_mode="HTML")

                save_forecast({
                    "user": chat_id,
                    "pair": pair,
                    "timeframe": tf,
                    "prediction": prediction,
                    "timestamp": datetime.now().isoformat()
                })
            else:
                await query.edit_message_text("❌ Не удалось получить данные для пары. Убедитесь, что она существует.")
    elif data == "back":
        user_state[chat_id].pop("pair", None)
        await context.bot.send_message(chat_id, "Введите новую торговую пару:")
    elif data == "admin_back":
        await query.edit_message_text("↩️ Вы вышли из админ-панели.")

# Получение свечей с Binance Futures
def get_futures_candles(symbol: str, interval: str, limit: int = 100):
    try:
        url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
        response = requests.get(url)
        if response.status_code != 200:
            return None
        data = response.json()
        if not data or isinstance(data, dict):
            return None
        ohlc = [{
            'Date': datetime.fromtimestamp(c[0] / 1000),
            'Open': float(c[1]),
            'High': float(c[2]),
            'Low': float(c[3]),
            'Close': float(c[4]),
            'Volume': float(c[5])
        } for c in data]
        return pd.DataFrame(ohlc).set_index('Date')
    except Exception as e:
        logging.error(f"Ошибка загрузки фьючерсных свечей: {e}")
        return None

# Индикаторы
def add_indicators(df):
    df['EMA20'] = df['Close'].ewm(span=20).mean()
    df['EMA50'] = df['Close'].ewm(span=50).mean()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['MACD'] = df['Close'].ewm(12).mean() - df['Close'].ewm(26).mean()

# Поддержка и сопротивление
def get_support_resistance(df):
    highs = df['High'].rolling(10).max()
    lows = df['Low'].rolling(10).min()
    return lows.iloc[-1], highs.iloc[-1]

# График
def plot_candlestick(df):
    support, resistance = get_support_resistance(df)
    apds = [
        mpf.make_addplot(df['EMA20'], color='blue'),
        mpf.make_addplot(df['EMA50'], color='purple'),
        mpf.make_addplot([support] * len(df), color='green', linestyle='--'),
        mpf.make_addplot([resistance] * len(df), color='red', linestyle='--')
    ]
    fig, _ = mpf.plot(df, type='candle', style='charles', addplot=apds, volume=True, returnfig=True)
    return fig

# Прогноз
def predict_trend(df):
    close = df['Close'].iloc[-1]
    ema20 = df['EMA20'].iloc[-1]
    ema50 = df['EMA50'].iloc[-1]
    rsi = df['RSI'].iloc[-1]
    macd = df['MACD'].iloc[-1]
    support, resistance = get_support_resistance(df)

    signals = []
    if close > resistance:
        signals.append("Пробой сопротивления 🚀")
    elif close < support:
        signals.append("Пробой поддержки 📉")
    if ema20 > ema50:
        signals.append("EMA20 выше EMA50 (бычий сигнал)")
    else:
        signals.append("EMA20 ниже EMA50 (медвежий сигнал)")
    if rsi > 70:
        signals.append("Перекупленность (RSI > 70)")
    elif rsi < 30:
        signals.append("Перепроданность (RSI < 30)")
    if macd > 0:
        signals.append("MACD положительный")
    else:
        signals.append("MACD отрицательный")
    return '\n'.join(signals)

# TP / SL
def calc_tp(df):
    return round(df['Close'].iloc[-1] * 1.03, 2)

def calc_sl(df):
    return round(df['Close'].iloc[-1] * 0.97, 2)

# Сохранение истории
def save_forecast(entry):
    data = []
    if os.path.exists("forecast_history.json"):
        with open("forecast_history.json", "r") as f:
            data = json.load(f)
    data.append(entry)
    with open("forecast_history.json", "w") as f:
        json.dump(data, f, indent=2)

# Админ-панель
async def admin_panel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id != ADMIN_ID:
        if update.message:
            await update.message.reply_text("❌ Нет доступа", parse_mode="HTML")
        elif update.callback_query:
            await update.callback_query.answer("❌ Нет доступа", show_alert=True)
        return

    text = (f"<b>🛠️ Админ-панель</b>\n"
            f"👥 Пользователей: <b>{len(user_list)}</b>\n\n"
            "📥 <b>/set_photo</b> — загрузить новое фото\n"
            "📢 <b>/broadcast &lt;текст&gt;</b> — отправка всем\n"
            "🗑 <b>/clear_users</b> — очистить список\n"
            "🕓 <b>/history</b> — история прогнозов (файл)")

    keyboard = [[InlineKeyboardButton("⬅️ Назад", callback_data="admin_back")]]
    try:
        if update.message:
            await update.message.reply_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))
        elif update.callback_query:
            await update.callback_query.edit_message_text(text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))
    except Exception as e:
        print(f"Ошибка при отправке admin_panel: {e}")

# История
async def send_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id != ADMIN_ID:
        return
    if not os.path.exists("forecast_history.json"):
        await update.message.reply_text("История пуста.")
        return
    with open("forecast_history.json", "r") as f:
        data = json.load(f)
    filename = "forecast_history.csv"
    with open(filename, "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["user", "pair", "timeframe", "prediction", "timestamp"])
        writer.writeheader()
        writer.writerows(data)
    await context.bot.send_document(chat_id=update.effective_chat.id, document=open(filename, "rb"))

# Фото от админа
async def set_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id != ADMIN_ID:
        return
    await update.message.reply_text("Отправьте новое фото.", parse_mode="HTML")

async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id != ADMIN_ID:
        return
    photo_file = await update.message.photo[-1].get_file()
    await photo_file.download_to_drive("admin_uploaded_image.jpg")
    await update.message.reply_text("✅ Фото обновлено!", parse_mode="HTML")

# Flask keep-alive
flask_app = Flask('')

@flask_app.route('/')
def home():
    return "Бот работает!"

def keep_alive():
    Thread(target=lambda: flask_app.run(host='0.0.0.0', port=8080)).start()

# Основной запуск
async def main():
    await telegram.Bot(token=TOKEN).delete_webhook()
    application = ApplicationBuilder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("admin", admin_panel))
    application.add_handler(CommandHandler("history", send_history))
    application.add_handler(CommandHandler("set_photo", set_photo))
    application.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    application.add_handler(CallbackQueryHandler(button_handler))
    keep_alive()
    print("🤖 Бот запущен!")
    await application.run_polling()

if __name__ == '__main__':
    try:
        import nest_asyncio
        nest_asyncio.apply()
        asyncio.get_event_loop().run_until_complete(main())
    except RuntimeError:
        print("Event loop is already running.")
