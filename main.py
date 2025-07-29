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
import asyncio
import logging
import tempfile
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters
)

# 🔐 Получаем токен из переменной окружения
TOKEN = os.environ.get("BOT_TOKEN")
ADMIN_ID = 664563521

TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d']
user_state = {}
user_list = set()

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Меню
main_menu = ReplyKeyboardMarkup(
    [[KeyboardButton("📈 Начать прогноз"), KeyboardButton("📖 Инструкция")]],
    resize_keyboard=True
)

def escape_markdown(text):
    return re.sub(r'([_*()~`>#+=|{}.!\-])', r'\\1', text)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_list.add(chat_id)
    user_state[chat_id] = {}
    await update.message.reply_text(
        escape_markdown(
            "Привет! Я бот для прогнозов криптовалют 🔮\n\n"
            "Нажми «📈 Начать прогноз», чтобы ввести пару, или «📖 Инструкция»"
        ),
        reply_markup=main_menu,
        parse_mode="MarkdownV2"
    )

async def instruction(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        escape_markdown(
            "📖 Как пользоваться ботом:\n"
            "1. Нажми «📈 Начать прогноз»\n"
            "2. Введи пару (например: BTCUSDT)\n"
            "3. Выбери таймфрейм\n"
            "4. Получишь график + прогноз\n\n"
            "🛠️ Также есть админ-панель /admin"
        ),
        parse_mode="MarkdownV2"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = update.message.text.strip().upper()

    if text == "📈 НАЧАТЬ ПРОГНОЗ":
        user_state[chat_id] = {}
        await update.message.reply_text(
            escape_markdown("Введите торговую пару (например: BTCUSDT)"),
            parse_mode="MarkdownV2"
        )
    elif text == "📖 ИНСТРУКЦИЯ":
        await instruction(update, context)
    elif chat_id in user_state and 'pair' not in user_state[chat_id]:
        user_state[chat_id]['pair'] = text
        await send_timeframe_buttons(update, context)
    else:
        await update.message.reply_text(
            escape_markdown("Пожалуйста, выберите действие через меню."),
            reply_markup=main_menu,
            parse_mode="MarkdownV2"
        )

async def send_timeframe_buttons(update: Update, context: ContextTypes.DEFAULT_TYPE):
    buttons = [[InlineKeyboardButton(tf, callback_data=f"tf:{tf}")] for tf in TIMEFRAMES]
    buttons.append([InlineKeyboardButton("⬅️ Назад", callback_data="back")])
    await update.message.reply_text(
        "Выберите таймфрейм:",
        reply_markup=InlineKeyboardMarkup(buttons)
    )

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
            try:
                candles = get_futures_candles(pair, tf)
                if candles is not None and not candles.empty:
                    add_indicators(candles)
                    
                    # Генерация графика
                    fig = plot_candlestick(candles)
                    buf = BytesIO()
                    fig.savefig(buf, format='png', dpi=100)
                    buf.seek(0)
                    plt.close(fig)
                    
                    prediction = escape_markdown(predict_trend(candles))
                    support, resistance = get_support_resistance(candles)
                    tp = calc_tp(candles)
                    sl = calc_sl(candles)

                    caption = (
                        f"<b>{pair} — {tf}</b>\n"
                        f"📈 Прогноз:\n{prediction}\n"
                        f"🔻 Поддержка: {support:.2f}\n"
                        f"🔺 Сопротивление: {resistance:.2f}\n"
                        f"🎯 Take Profit: {tp}\n"
                        f"🛑 Stop Loss: {sl}"
                    )

                    await context.bot.send_photo(
                        chat_id=chat_id,
                        photo=buf,
                        caption=caption,
                        parse_mode="HTML"
                    )

                    save_forecast({
                        "user": chat_id,
                        "pair": pair,
                        "timeframe": tf,
                        "prediction": prediction,
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    await query.edit_message_text("❌ Не удалось получить данные для пары. Убедитесь, что она существует.")
            except Exception as e:
                logger.error(f"Error generating chart: {e}")
                await query.edit_message_text("❌ Произошла ошибка при генерации графика.")
    elif data == "back":
        user_state[chat_id].pop("pair", None)
        await context.bot.send_message(chat_id, "Введите новую торговую пару:")
    elif data == "admin_back":
        await query.edit_message_text("↩️ Вы вышли из админ-панели.")

def get_futures_candles(symbol: str, interval: str, limit: int = 100):
    try:
        url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
        response = requests.get(url, timeout=10)
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
        logger.error(f"Ошибка загрузки фьючерсных свечей: {e}")
        return None

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

def get_support_resistance(df):
    highs = df['High'].rolling(10).max()
    lows = df['Low'].rolling(10).min()
    return lows.iloc[-1], highs.iloc[-1]

def plot_candlestick(df):
    plt.ioff()  # Отключаем интерактивный режим
    plt.switch_backend('Agg')  # Используем non-GUI бэкенд
    
    support, resistance = get_support_resistance(df)
    apds = [
        mpf.make_addplot(df['EMA20'], color='blue'),
        mpf.make_addplot(df['EMA50'], color='purple'),
        mpf.make_addplot([support] * len(df), color='green', linestyle='--'),
        mpf.make_addplot([resistance] * len(df), color='red', linestyle='--')
    ]
    
    fig, _ = mpf.plot(
        df,
        type='candle',
        style='charles',
        addplot=apds,
        volume=True,
        returnfig=True,
        figsize=(10, 8)
    )
    
    return fig

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

def calc_tp(df):
    return round(df['Close'].iloc[-1] * 1.03, 2)

def calc_sl(df):
    return round(df['Close'].iloc[-1] * 0.97, 2)

def save_forecast(entry):
    try:
        history_path = os.path.join(os.getcwd(), 'forecast_history.json')
        data = []
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                data = json.load(f)
        data.append(entry)
        with open(history_path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving forecast: {e}")

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
        logger.error(f"Ошибка при отправке admin_panel: {e}")

async def send_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id != ADMIN_ID:
        return
    
    history_path = os.path.join(os.getcwd(), 'forecast_history.json')
    if not os.path.exists(history_path):
        await update.message.reply_text("История пуста.")
        return
    
    try:
        with open(history_path, "r") as f:
            data = json.load(f)
        
        filename = os.path.join(os.getcwd(), 'forecast_history.csv')
        with open(filename, "w", newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["user", "pair", "timeframe", "prediction", "timestamp"])
            writer.writeheader()
            writer.writerows(data)
        
        with open(filename, "rb") as file:
            await context.bot.send_document(
                chat_id=update.effective_chat.id,
                document=file,
                filename="forecast_history.csv"
            )
    except Exception as e:
        logger.error(f"Error sending history: {e}")
        await update.message.reply_text("❌ Ошибка при формировании истории.")

async def set_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id != ADMIN_ID:
        return
    await update.message.reply_text("Отправьте новое фото.", parse_mode="HTML")

async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id != ADMIN_ID:
        return
    
    try:
        photo_file = await update.message.photo[-1].get_file()
        photo_path = os.path.join(os.getcwd(), 'admin_uploaded_image.jpg')
        await photo_file.download_to_drive(photo_path)
        await update.message.reply_text("✅ Фото обновлено!", parse_mode="HTML")
    except Exception as e:
        logger.error(f"Error saving photo: {e}")
        await update.message.reply_text("❌ Ошибка при сохранении фото.", parse_mode="HTML")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Ошибка: {context.error}", exc_info=True)
    if update and update.effective_chat:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Произошла ошибка. Пожалуйста, попробуйте позже."
        )

# ... (весь ваш предыдущий код остается без изменений до функции main)

async def main():
    # Создаем приложение
    application = ApplicationBuilder().token(TOKEN).build()
    
    # Добавляем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("admin", admin_panel))
    application.add_handler(CommandHandler("history", send_history))
    application.add_handler(CommandHandler("set_photo", set_photo))
    application.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    application.add_handler(CallbackQueryHandler(button_handler))
    application.add_error_handler(error_handler)
    
    # Удаляем существующий webhook
    await application.bot.delete_webhook()
    
    # Запускаем webhook
    await application.run_webhook(
        listen="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),  # Render использует порт из переменной окружения
        url_path=TOKEN,
        webhook_url=f"https://{os.environ.get('RENDER_EXTERNAL_HOSTNAME')}/{TOKEN}",
        secret_token=os.environ.get('WEBHOOK_SECRET', 'YOUR_SECRET_TOKEN')
    )

if __name__ == '__main__':
    # Просто запускаем main() без asyncio.run()
    application = ApplicationBuilder().token(TOKEN).build()
    
    # Добавляем обработчики (дублируем, так как это теперь синхронный код)
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("admin", admin_panel))
    application.add_handler(CommandHandler("history", send_history))
    application.add_handler(CommandHandler("set_photo", set_photo))
    application.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    application.add_handler(CallbackQueryHandler(button_handler))
    application.add_error_handler(error_handler)
    
    # Для Render лучше использовать polling, если у вас нет домена с HTTPS
    application.run_polling()

if __name__ == '__main__':
    application = ApplicationBuilder().token(TOKEN).build()
    
    # Добавьте все обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("admin", admin_panel))
    # ... остальные обработчики ...
    
    application.run_polling()
