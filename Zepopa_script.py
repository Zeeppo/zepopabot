import logging
import os
import random
import re
from collections import deque, defaultdict

# Third-party libraries
import openai # We still use this library!
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (Application, CommandHandler, ContextTypes,
                          MessageHandler, filters)

# --- Configuration ---
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
# <<< NEW: OpenRouter Configuration >>>
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# Specify the model you want to use from OpenRouter's list
OPENROUTER_MODEL_NAME = os.getenv("OPENROUTER_MODEL_NAME", "openai/gpt-3.5-turbo") # Default example
# Optional but recommended headers for OpenRouter
OPENROUTER_REFERRER = os.getenv("OPENROUTER_REFERRER", "") # e.g., https://your-app-url.com
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "Telegram Context Bot")

BOT_USERNAME = os.getenv("TELEGRAM_BOT_USERNAME") # e.g., "@MyCoolBot"
RESPONSE_PROBABILITY = float(os.getenv("RESPONSE_PROBABILITY", 0.1))

# --- Validation ---
if not TELEGRAM_BOT_TOKEN: raise ValueError("Missing TELEGRAM_BOT_TOKEN")
if not OPENROUTER_API_KEY: raise ValueError("Missing OPENROUTER_API_KEY")
if not BOT_USERNAME: raise ValueError("Missing TELEGRAM_BOT_USERNAME")
if not (0 <= RESPONSE_PROBABILITY <= 1): raise ValueError("RESPONSE_PROBABILITY must be between 0 and 1.")

# --- Configure Logging ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# --- Configure API Client (OpenRouter) ---
# Construct headers
default_headers = {"Content-Type": "application/json"}
if OPENROUTER_REFERRER:
    default_headers["HTTP-Referer"] = OPENROUTER_REFERRER
if OPENROUTER_APP_NAME:
    default_headers["X-Title"] = OPENROUTER_APP_NAME

# Initialize the async OpenAI client pointing to OpenRouter
# NOTE: Use openai library version >= 1.0.0 for this syntax
try:
    llm_client = openai.AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1", # OpenRouter API endpoint
        api_key=OPENROUTER_API_KEY,
        default_headers=default_headers,
    )
    logger.info(f"OpenRouter client configured for model: {OPENROUTER_MODEL_NAME}")
except Exception as e:
     logger.error(f"Failed to initialize OpenRouter client: {e}")
     raise

# --- Constants ---
MAX_HISTORY_LENGTH = 15
SYSTEM_PROMPT = """
You are a helpful and conversational AI assistant participating in a Telegram group chat.
Analyze the provided message history to understand the ongoing context, the topic of discussion, and the general tone (e.g., friendly, serious, humorous).
Respond to the last message naturally, appropriately, and in a way that fits the conversation's flow and tone. Keep your responses concise unless asked for detail.
Do not explicitly mention that you are analyzing tone or context unless relevant to the query. Avoid starting responses with 'Okay', 'Sure', or similar acknowledgements unless directly answering a question.
If asked about your capabilities, mention you can understand conversation history. Старайся отвечать на русском. Разговаривай как норм чел, шути иногда. Ты еще самый главный фанат российского блогера, инфобизнесмена и коуча, известного своими противоречивыми взглядами и высказыванями Арсена Маркаряна(Даже главнее Паши).

***IMPORTANT FORMATTING INSTRUCTIONS:***
- Use Telegram's MarkdownV2 formatting for all responses where applicable.
- **For code blocks:** Use triple backticks. Specify the language if known (e.g., ```python\n# your code here\n```). For plain code or unknown languages, just use ```\ncode here\n```.
- **For inline code:** Use single backticks, like `my_variable`.
- **Escape:** Remember to escape MarkdownV2 special characters (`_`, `*`, `[`, `]`, `(`, `)`, `~`, `` ` ``, `>`, `#`, `+`, `-`, `=`, `|`, `{`, `}`, `.`, `!`) with a preceding backslash (`\\`) if they appear in *regular text* (outside code blocks/spans). For example, write `1\\. Example` instead of `1. Example`. This is crucial for the message to render correctly.
"""

# --- In-Memory Message Storage ---
chat_histories = defaultdict(lambda: deque(maxlen=MAX_HISTORY_LENGTH))

# --- Helper Functions ---

async def get_llm_response(chat_id: int) -> str | None:
    """
    Gets a response from the configured LLM via OpenRouter based on chat history.
    """
    history = chat_histories[chat_id]
    if not history:
        logger.warning(f"Attempted to get LLM response for chat {chat_id} with empty history.")
        return None

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg_data in history:
        sender_id, text, is_bot = msg_data
        role = "assistant" if is_bot else "user"
        content = f"{sender_id}: {text}"
        messages.append({"role": role, "content": content})

    logger.info(f"Sending request to OpenRouter ({OPENROUTER_MODEL_NAME}) for chat {chat_id} with {len(messages)} messages.")
    # logger.debug(f"OpenRouter messages for chat {chat_id}: {messages}")

    try:
        # Use the initialized async client
        response = await llm_client.chat.completions.create(
            model=OPENROUTER_MODEL_NAME, # Use the configured OpenRouter model
            messages=messages,
            temperature=0.75,
            max_tokens=1000,
            top_p=1.0,
            frequency_penalty=0.1,
            presence_penalty=0.1
            # Note: Some OpenRouter models might support different/additional parameters
        )
        ai_response = response.choices[0].message.content.strip()
        logger.info(f"Received OpenRouter response for chat {chat_id}: {ai_response}")
        # Optional: Log token usage if needed (check response object structure)
        # logger.info(f"OpenRouter Usage: {response.usage}")
        return ai_response
    except openai.APIConnectionError as e:
        logger.error(f"OpenRouter API connection error for chat {chat_id}: {e}")
        return "Sorry, I couldn't connect to the AI service."
    except openai.RateLimitError as e:
        logger.error(f"OpenRouter Rate limit exceeded for chat {chat_id}: {e}")
        return "I'm experiencing high demand right now, please try again shortly."
    except openai.APIStatusError as e:
        logger.error(f"OpenRouter API status error for chat {chat_id}: Status={e.status_code} Response={e.response}")
        return f"Sorry, there was an issue with the AI service (Code: {e.status_code})."
    except Exception as e:
        logger.error(f"Generic OpenRouter API error for chat {chat_id}: {e}")
        return "Hmm, I'm having a little trouble thinking right now. Please try again later."


# --- Telegram Bot Handlers ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends explanation on /start."""
    await update.message.reply_text(f"хуй {RESPONSE_PROBABILITY*100:.0f}%")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles incoming messages, stores them, and responds if mentioned OR randomly."""
    message = update.message
    chat_id = message.chat_id
    text = message.text
    user = message.from_user
    bot_id = context.bot.id

    if not text or not user: return
    if not (message.chat.type == 'group' or message.chat.type == 'supergroup'): return
    if user.id == bot_id: return # Ignore self

    sender_name = user.username or user.first_name
    message_data = (sender_name, text, False)
    chat_histories[chat_id].append(message_data)
    logger.info(f"Stored message in chat {chat_id}: {sender_name}: {text}")

    # --- Trigger Conditions ---
    bot_mentioned = False
    bot_username_in_context = context.bot.username
    if bot_username_in_context in text:
         bot_mentioned = True
    elif message.entities:
        for entity in message.entities:
            if entity.type == 'mention':
                mentioned_user = text[entity.offset:entity.offset + entity.length]
                if mentioned_user == bot_username_in_context:
                    bot_mentioned = True
                    break

    should_respond_randomly = random.random() < RESPONSE_PROBABILITY

    # --- Decide whether to respond ---
    if bot_mentioned or should_respond_randomly:
        trigger_reason = "mention" if bot_mentioned else "random chance"
        logger.info(f"Triggering response in chat {chat_id} via OpenRouter due to: {trigger_reason}.")

        await context.bot.send_chat_action(chat_id=chat_id, action='typing')
        # <<< Use the new LLM function >>>
        ai_response = await get_llm_response(chat_id)

        if ai_response:
            bot_display_name = BOT_USERNAME.lstrip('@')
            bot_response_data = (bot_display_name, ai_response, True)
            chat_histories[chat_id].append(bot_response_data)
            logger.info(f"Stored bot response in chat {chat_id}: {bot_display_name}: {ai_response}")
            await message.reply_text(ai_response)
            try:
                await message.reply_text(
                    text=ai_response,
                    parse_mode=ParseMode.MARKDOWN_V2
                )
                logger.info(f"Sent response with MarkdownV2 formatting to chat {chat_id}.")

            except BadRequest as e:
                # Check if the error is likely due to parsing
                # Common error messages include "Can't parse entities" or specific character errors
                if "parse" in str(e).lower() or "entity" in str(e).lower():
                    logger.warning(
                        f"Failed to send message with MarkdownV2 formatting to chat {chat_id} due to parsing error: {e}. Sending as plain text.")
                    # Fallback: Send as plain text
                    try:
                        await message.reply_text(text=ai_response)
                    except Exception as fallback_e:
                        logger.error(f"Failed to send fallback plain text message to chat {chat_id}: {fallback_e}")
                else:
                    # Different type of BadRequest, re-raise or handle differently if needed
                    logger.error(
                        f"BadRequest error when sending message to chat {chat_id}, not necessarily parsing: {e}")
                    # Optionally, send a generic error message?
                    # await message.reply_text("Sorry, I couldn't send my response correctly.")

            except Exception as e:
                # Catch other potential sending errors
                logger.error(f"Unexpected error when sending message to chat {chat_id}: {e}")
                # Optionally, send a generic error message?
                # await message.reply_text("Sorry, an error occurred while sending my response.")

            else:
                # LLM failed, error already logged by get_llm_response.
                # Send a fallback only if explicitly mentioned.
                if bot_mentioned:
                    fallback_message = "Sorry, I couldn't process that request right now."
                    # Consider using the error message returned by get_llm_response if it's user-friendly
                    # ai_error_response = await get_llm_response(chat_id) # This would call it again, bad idea
                    # Maybe get_llm_response should return a tuple (success: bool, content: str)
                    await message.reply_text(fallback_message)
                pass  # Silent failure on random trigger is usually better
            else:
                logger.debug(f"No mention and random chance ({RESPONSE_PROBABILITY * 100:.0f}%) not met in chat {chat_id}.")
    else:
         logger.debug(f"No mention and random chance ({RESPONSE_PROBABILITY*100:.0f}%) not met in chat {chat_id}.")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log Errors caused by Updates."""
    logger.error(f"Update {update} caused error {context.error}", exc_info=context.error)

# --- Main Bot Execution ---

def main() -> None:
    """Start the bot."""
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND & (filters.ChatType.GROUPS),
        handle_message
    ))
    application.add_error_handler(error_handler)

    logger.info(f"Starting bot polling... Bot username: {BOT_USERNAME}, Response probability: {RESPONSE_PROBABILITY*100:.0f}%")
    logger.info(f"Using OpenRouter model: {OPENROUTER_MODEL_NAME}")
    if OPENROUTER_REFERRER: logger.info(f"OpenRouter Referer: {OPENROUTER_REFERRER}")
    if OPENROUTER_APP_NAME: logger.info(f"OpenRouter App Name: {OPENROUTER_APP_NAME}")

    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()