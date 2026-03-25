import os
import logging
import random
import asyncio
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# ============================================================ [Setup]
# Use an environment variable or paste your token here
TOKEN = "1704757799:AAGJRzgiQP-m4YINSAfWrRsYbcikFtTJryo"

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================ [Data]
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

# ============================================================ [Functions]

async def show_typing(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Helper to show the 'typing...' status."""
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await show_typing(update, context)
    name = update.effective_user.first_name
    await update.message.reply_text(
        f"Hello {name} ✋🏻, I'm <b>FinalyticsBot</b>. I can help with your risk profile and finance tips.\n\nTry /riskprofiletest or /tips",
        parse_mode="html"
    )

async def tips(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await show_typing(update, context)
    await update.message.reply_text(random.choice(TIPS_LIST), parse_mode="html")

async def risk_profile_test(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Reset user session
    context.user_data['score'] = 0
    context.user_data['q_idx'] = 1
    
    reply_keyboard = [['Yes', 'Later']]
    await update.message.reply_text(
        f"Hi {update.effective_user.first_name}, ready to start the risk test?",
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, resize_keyboard=True)
    )

async def handle_responses(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    user_data = context.user_data

    # User clicked 'Yes' to start
    if text == "Yes":
        user_data['q_idx'] = 1
        await ask_next_question(update, context)
        return
    
    # User clicked 'Later'
    if text == "Later":
        await update.message.reply_text("No problem! Take your time. 😊", reply_markup=ReplyKeyboardRemove())
        return

    # Check if we are currently in a test flow
    q_idx = user_data.get('q_idx')
    if q_idx and q_idx <= 7:
        # Simple score logic: Add 1-5 points based on selection (mock logic)
        user_data['score'] = user_data.get('score', 0) + random.randint(1, 5)
        
        if q_idx < 7:
            user_data['q_idx'] += 1
            await ask_next_question(update, context)
        else:
            await show_result(update, context)
    else:
        await update.message.reply_text("I'm not sure what you mean. Try /help.")

async def ask_next_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q_num = str(context.user_data['q_idx'])
    markup = ReplyKeyboardMarkup(OPTIONS[q_num], one_time_keyboard=True, resize_keyboard=True)
    await update.message.reply_text(
        f"<b>Question {q_num}:</b>\n{QUESTIONS[q_num]}", 
        reply_markup=markup, 
        parse_mode="html"
    )

async def show_result(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await show_typing(update, context)
    score = context.user_data.get('score', 0)
    
    # Simple logic for the result
    if score > 25:
        risk = "High"
    elif score > 15:
        risk = "Moderate"
    else:
        risk = "Low"
        
    await update.message.reply_text(
        f"<b>Test Complete!</b>\nYour profile: <b>{risk} Risk</b>\n\nCheck /stocks for suggestions.",
        reply_markup=ReplyKeyboardRemove(),
        parse_mode="html"
    )
    context.user_data.clear() # Clear session data

# ============================================================ [Main]

if __name__ == '__main__':
    # Initialize the Application
    app = ApplicationBuilder().token(TOKEN).build()
    
    # Register Handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("tips", tips))
    app.add_handler(CommandHandler("riskprofiletest", risk_profile_test))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_responses))
    
    print("FinalyticsBot v1.6 is online...")
    app.run_polling()
