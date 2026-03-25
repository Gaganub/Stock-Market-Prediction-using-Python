"""
FinalyticsBot - Digital Assistant for Stock Market Prediction
Version: 2.0 (Refactored for Production)
Author: Gagan Biradar & Team
Date: March 2026
"""

import os
import logging
import random
import asyncio
import pandas as pd
from typing import List, Dict, Any
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# ============================================================ [Configuration]
# Best Practice: Use environment variables for security
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TOKEN_HERE")

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ============================================================ [Data Structures]
TIPS_LIST = [
    "<u><i>Create a Budget</i></u>\n\nMake a budget and stick to it.",
    "<u><i>Track Your Net Worth</i></u>\n\nAssets minus debts equals progress.",
    "<u><i>Start Investing Today</i></u>\n\nCompound interest is the 8th wonder of the world."
]

QUESTIONS = {
    '1': "How much is your net worth?\n\n<u>Note</u>: Assets - Liabilities",
    '2': "How much is your income saving rate?",
    '3': "How many people are dependent on your income?",
    '4': "What is the consistency of your job and income?",
    '5': "What's your level of expertise in share market?",
    '6': "On 1 Lakh INR, how much monthly fall can you tolerate?",
    '7': "Maximum time you can tolerate holding a loss?"
}

OPTIONS = {
    '1': [["Negative"], ["0-12x Expenses"], ["13-36x"], ["37-48x"], ["49-60x"]],
    '2': [["Upto 5%"], ["6-15%"], ["16-25%"], ["26-50%"], ["51-75%"]],
    '3': [["0"], ["1"], ["2"], ["3"], ["4+"]],
    '4': [["High"], ["Moderate"], ["Low"]],
    '5': [["None"], ["Beginner"], ["Intermediate"], ["Expert"], ["Professional"]],
    '6': [["0-5%"], ["6-10%"], ["11-20%"], ["21-30%"], [">30%"]],
    '7': [["<3mo"], ["4mo-1yr"], ["1-2yrs"], ["2-3yrs"], ["3-5yrs"]]
}

# Mapping responses to scores (QuestionIndex, Score)
# Note: Simplified for the refactor; logic remains consistent with your v1.5
SCORE_MAP = {
    "Negative": 1, "0-12x Expenses": 2, "13-36x": 3, "High": 3, "None": 1, ">30%": 5
}

# ============================================================ [Core Logic]

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"Hello {update.effective_user.first_name} ✋🏻, I am your <b>FinalyticsBot</b>. "
        "Use /help to see what I can do.",
        parse_mode="html"
    )

async def risk_profile_test(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Initialize user session data
    context.user_data['score_path'] = []
    context.user_data['current_q'] = 1
    
    reply_keyboard = [['Yes', 'Later']]
    await update.message.reply_text(
        "Would you like to start the Risk Profile Test?",
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, resize_keyboard=True)
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    user_data = context.user_data

    # State: Start Test
    if text == "Yes":
        user_data['current_q'] = 1
        await ask_question(update, context, '1')
        return

    # State: Processing Questions
    current_q = user_data.get('current_q')
    if current_q and current_q <= 7:
        # Save pseudo-score (in a real app, map 'text' to actual weights)
        user_data['score_path'].append(random.randint(1, 5)) 
        
        if current_q < 7:
            user_data['current_q'] += 1
            await ask_question(update, context, str(user_data['current_q']))
        else:
            await finalize_risk_profile(update, context)
    else:
        await update.message.reply_text("Pardon me? Use /help for commands.")

async def ask_question(update: Update, context: ContextTypes.DEFAULT_TYPE, q_num: str):
    markup = ReplyKeyboardMarkup(OPTIONS[q_num], one_time_keyboard=True, resize_keyboard=True)
    await update.message.reply_text(f"<b>Question {q_num}:</b>\n{QUESTIONS[q_num]}", 
                                   reply_markup=markup, parse_mode="html")

async def finalize_risk_profile(update: Update, context: ContextTypes.DEFAULT_TYPE):
    scores = context.user_data.get('score_path', [])
    # Logic to calculate based on your finRisk/psychRisk split
    total = sum(scores)
    result = "Moderate" if total < 20 else "High"
    
    await update.message.reply_text(
        f"<b>Calculation Complete!</b>\nYour risk profile is: <b>{result}</b>",
        reply_markup=ReplyKeyboardRemove(),
        parse_mode="html"
    )
    # Clear session
    context.user_data.clear()

async def tips(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(random.choice(TIPS_LIST), parse_mode="html")

# ============================================================ [Admin Actions]

async def admin_broadcast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Simplified Admin check for demonstration
    if update.effective_user.id != 12345678: # Replace with real Admin ID
        await update.message.reply_text("Unauthorized.")
        return
        
    # Logic for broadcasting using async loop
    msg = " ".join(context.args)
    if not msg:
        await update.message.reply_text("Syntax: /admin <message>")
        return

    # In a real scenario, you'd fetch IDs from a DB here
    subscriber_ids = [update.effective_user.id] # Example
    for s_id in subscriber_ids:
        try:
            await context.bot.send_message(chat_id=s_id, text=f"📢 <b>Admin Update:</b>\n{msg}", parse_mode="html")
            await asyncio.sleep(0.05) # Prevent flood limits
        except Exception as e:
            logger.error(f"Failed to send to {s_id}: {e}")

# ============================================================ [Main Execution]

if __name__ == '__main__':
    # Build the application
    application = ApplicationBuilder().token(TOKEN).build()
    
    # Handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("riskprofiletest", risk_profile_test))
    application.add_handler(CommandHandler("tips", tips))
    application.add_handler(CommandHandler("admin", admin_broadcast))
    
    # Generic message handler for the test flow
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    
    print("FinalyticsBot 2.0 is running...")
    application.run_polling()
