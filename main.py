# main.py
import os, json, uuid, html, logging
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import JSONResponse
import httpx
from tenacity import retry, wait_exponential, stop_after_attempt
import openai

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from dao import engine_from_env, init_db, upsert_user, get_user_by_tid, add_mistake, MistakeEvent
from srs import schedule_initial

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
WEBHOOK_SUFFIX = os.getenv("WEBHOOK_SECRET_SUFFIX", "dev")
WEBHOOK_HEADER_SECRET = os.getenv("WEBHOOK_SECRET_HEADER", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL","gpt-5-nano")
DB_URL = os.getenv("DB_URL","sqlite:///./voice_tutor.db")
DEFAULT_LANG = os.getenv("DEFAULT_TARGET_LANGUAGE","en")
DEFAULT_CEFR = os.getenv("DEFAULT_CEFR_LEVEL","B2")
OPENAI_WHISPER_MODEL = os.getenv("OPENAI_WHISPER_MODEL","whisper-1")

TTS_PROVIDER = os.getenv("TTS_PROVIDER","azure")
AZURE_TTS_KEY = os.getenv("AZURE_TTS_KEY","")
AZURE_TTS_REGION = os.getenv("AZURE_TTS_REGION","")
V_EN = os.getenv("AZURE_TTS_VOICE_EN","en-US-JennyNeural")
V_FR = os.getenv("AZURE_TTS_VOICE_FR","fr-FR-DeniseNeural")
V_KK = os.getenv("AZURE_TTS_VOICE_KK","kk-KZ-AigulNeural")

ADMIN_BEARER = os.getenv("ADMIN_BEARER","dev")

app = FastAPI()
engine = engine_from_env(DB_URL)
init_db(engine)

print("🚀 FastAPI server starting up...")
print(f"📊 Database URL: {DB_URL}")
print(f"🤖 Bot token configured: {'✅' if BOT_TOKEN else '❌'}")
print(f"🔐 Webhook suffix: {WEBHOOK_SUFFIX}")
logger.info("FastAPI application initialized")

TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"
FILE_API = f"https://api.telegram.org/file/bot{BOT_TOKEN}"

def assert_admin(auth: str):
    if not auth or not auth.startswith("Bearer ") or auth.split(" ",1)[1] != ADMIN_BEARER:
        raise HTTPException(status_code=401, detail="unauthorized")

@app.get("/healthz")
def health():
    return {"ok": True, "time": datetime.now().isoformat()}

# ---------- External calls ----------
@retry(wait=wait_exponential(multiplier=1, max=8), stop=stop_after_attempt(2))
async def tg_get_file_path(file_id: str) -> str:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{TELEGRAM_API}/getFile", params={"file_id": file_id})
        r.raise_for_status()
        return r.json()["result"]["file_path"]

@retry(wait=wait_exponential(multiplier=1, max=8), stop=stop_after_attempt(2))
async def tg_download(file_path: str) -> bytes:
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.get(f"{FILE_API}/{file_path}")
        r.raise_for_status()
        return r.content

@retry(wait=wait_exponential(multiplier=1, max=8), stop=stop_after_attempt(2))
async def tg_send_voice(chat_id: int, audio_bytes: bytes, caption: str):
    print(f"🎵 Sending voice message: {len(audio_bytes)} bytes to chat {chat_id}")
    async with httpx.AsyncClient(timeout=60) as client:
        files = {'voice': ('reply.ogg', audio_bytes, 'audio/ogg')}
        data = {'chat_id': chat_id, 'caption': caption[:1024]}
        r = await client.post(f"{TELEGRAM_API}/sendVoice", data=data, files=files)
        r.raise_for_status()
        print(f"✅ Voice message sent successfully (status: {r.status_code})")

@retry(wait=wait_exponential(multiplier=1, max=8), stop=stop_after_attempt(2))
async def tg_send_message(chat_id: int, text: str, reply_markup=None, parse_mode="Markdown"):
    """Send a text message with optional inline keyboard"""
    print(f"💬 Sending message to chat {chat_id}: {text[:50]}...")
    async with httpx.AsyncClient(timeout=30) as client:
        data = {
            "chat_id": chat_id,
            "text": text[:4096],  # Telegram message limit
            "parse_mode": parse_mode
        }
        if reply_markup:
            data["reply_markup"] = json.dumps(reply_markup)
        
        r = await client.post(f"{TELEGRAM_API}/sendMessage", json=data)
        r.raise_for_status()
        print(f"✅ Message sent successfully (status: {r.status_code})")

@retry(wait=wait_exponential(multiplier=1, max=8), stop=stop_after_attempt(2))
async def tg_answer_callback_query(callback_query_id: str, text: str = "", show_alert: bool = False):
    """Answer a callback query (button press)"""
    print(f"📞 Answering callback query: {callback_query_id}")
    async with httpx.AsyncClient(timeout=30) as client:
        data = {
            "callback_query_id": callback_query_id,
            "text": text[:200],  # Callback answer text limit
            "show_alert": show_alert
        }
        r = await client.post(f"{TELEGRAM_API}/answerCallbackQuery", json=data)
        r.raise_for_status()
        print(f"✅ Callback query answered successfully")

def create_inline_keyboard(buttons):
    """
    Create an inline keyboard markup.
    buttons: List of lists of dicts with 'text' and 'callback_data' keys
    Example: [[{"text": "Button 1", "callback_data": "btn1"}]]
    """
    return {
        "inline_keyboard": buttons
    }

def create_reply_keyboard(buttons, resize_keyboard=True, one_time_keyboard=False):
    """
    Create a reply keyboard markup (buttons under input field).
    buttons: List of lists of strings or dicts with 'text' key
    Example: [["🏠 Start", "⚙️ Settings"], ["🆘 Help"]]
    """
    return {
        "keyboard": buttons,
        "resize_keyboard": resize_keyboard,
        "one_time_keyboard": one_time_keyboard
    }

def remove_reply_keyboard():
    """Remove the reply keyboard"""
    return {"remove_keyboard": True}

# --- STT (OpenAI Whisper endpoint) ---
@retry(wait=wait_exponential(multiplier=1, max=8), stop=stop_after_attempt(2))
def stt_openai(file_bytes: bytes, lang_hint: str|None):
    client  = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    # Create a file-like object with proper name for OpenAI API
    from io import BytesIO
    audio_file = BytesIO(file_bytes)
    audio_file.name = "voice.oga"  # OpenAI needs to know the format
    
    transcript = client.audio.transcriptions.create(
        model=OPENAI_WHISPER_MODEL,
        file=audio_file,
        language=lang_hint,
        response_format="text"
    )
    return transcript

# --- LLM evaluation (strict JSON) ---
EVAL_SYS = """You are a meticulous speaking coach. Return STRICT JSON only. No prose.
Scores are 0-5. Use neutral/international accent guidance. Reflect uncertainty in "confidence".
"""

def eval_user_prompt(transcript: str, target_language: str, cefr_level: str) -> list:
    logger.info(f"Generating evaluation prompt for {transcript} {target_language} {cefr_level}")
    print(f"Generating evaluation prompt for {transcript} {target_language} {cefr_level}")
    schema_hint = {
      "fluency": 0, "pronunciation": 0, "grammar": 0, "lexis": 0, "pragmatics": 0,
      "notes": ["..." ],
      "errors": [
        {"category":"pronunciation","subcategory":"/θ/→/s/","evidence":"...",
         "fix":"...", "ipa":"...", "drill_type":"minimal_pairs"}
      ],
      "confidence": 0.8,
      "suggested_prompt_for_next_turn": "..."
    }
    return [
      {"role":"system","content":EVAL_SYS + f"\nTarget={target_language} {cefr_level}."},
      {"role":"user","content": transcript},
      {"role":"assistant","content": json.dumps(schema_hint)}
    ]

@retry(wait=wait_exponential(multiplier=1, max=8), stop=stop_after_attempt(2))
async def llm_eval_openai(transcript: str, target_language: str, cefr_level: str, json_mode: bool = False) -> dict:
    try:
        client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
        response = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=eval_user_prompt(transcript, target_language, cefr_level),
            # temperature=0.2,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        if not content or content.strip() == "":
            print(f"Warning: Empty response from model {LLM_MODEL}")
            # Return a fallback JSON response for empty content
            return {"error": "empty_response", "message": "Model returned empty content"}
        
        # Parse the JSON response
        return json.loads(content)

    except Exception as e:
        print(f"OpenAI API error: {e}")
        raise e

# --- Azure TTS to OGG/OPUS ---
@retry(wait=wait_exponential(multiplier=1, max=8), stop=stop_after_attempt(2))
async def tts_azure(text: str, lang_code: str) -> bytes:
    voice_map = {"en": V_EN, "fr": V_FR, "kk": V_KK}
    voice = voice_map.get(lang_code, V_EN)
    ssml = f"<speak version='1.0' xml:lang='{lang_code}'><voice name='{voice}'>{html.escape(text)}</voice></speak>"
    url = f"https://{AZURE_TTS_REGION}.tts.speech.microsoft.com/cognitiveservices/v1"
    headers = {
      "Ocp-Apim-Subscription-Key": AZURE_TTS_KEY,
      "Content-Type": "application/ssml+xml",
      "X-Microsoft-OutputFormat": "ogg-24khz-16bit-mono-opus",
      "User-Agent": "VoiceTutor/1.0"
    }
    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.post(url, headers=headers, content=ssml.encode("utf-8"))
        r.raise_for_status()
        return r.content

# ---------- Callback Query Handler ----------
async def handle_callback_query(callback_query):
    """Handle button presses (callback queries)"""
    print("🔘 Processing callback query")
    
    query_id = callback_query["id"]
    data = callback_query["data"]
    user = callback_query["from"]
    message = callback_query.get("message")
    
    if not message:
        await tg_answer_callback_query(query_id, "Error: No message found")
        return {"ok": True}
    
    chat_id = message["chat"]["id"]
    tg_id = user["id"]
    
    print(f"👤 Button pressed by: {user.get('first_name', 'Unknown')} (ID: {tg_id})")
    print(f"📊 Callback data: {data}")
    
    # Get or create user
    db_user = get_user_by_tid(engine, tg_id) or upsert_user(engine, tg_id, {
        "display_name": user.get("first_name", "User"),
        "target_language": DEFAULT_LANG, 
        "cefr_level": DEFAULT_CEFR
    })
    
    # Handle different button actions
    if data.startswith("lang_"):
        # Language selection
        new_lang = data.split("_", 1)[1]
        upsert_user(engine, tg_id, {"target_language": new_lang})
        await tg_answer_callback_query(query_id, f"Language set to {new_lang.upper()}")
        
        # Send confirmation message
        response_text = f"✅ Language updated to *{new_lang.upper()}*\n\nCurrent settings:\n• Language: {new_lang}\n• Level: {db_user.cefr_level}"
        await tg_send_message(chat_id, response_text)
        
    elif data.startswith("level_"):
        # CEFR level selection
        new_level = data.split("_", 1)[1]
        upsert_user(engine, tg_id, {"cefr_level": new_level})
        await tg_answer_callback_query(query_id, f"Level set to {new_level}")
        
        # Send confirmation message
        response_text = f"✅ Level updated to *{new_level}*\n\nCurrent settings:\n• Language: {db_user.target_language}\n• Level: {new_level}"
        await tg_send_message(chat_id, response_text)
        
    elif data == "settings":
        # Show settings menu
        await tg_answer_callback_query(query_id, "Settings menu")
        
        # Create settings keyboard
        settings_keyboard = create_inline_keyboard([
            [
                {"text": "🇺🇸 English", "callback_data": "lang_en"},
                {"text": "🇫🇷 French", "callback_data": "lang_fr"},
                {"text": "🇰🇿 Kazakh", "callback_data": "lang_kk"}
            ],
            [
                {"text": "A1 Beginner", "callback_data": "level_A1"},
                {"text": "A2 Elementary", "callback_data": "level_A2"}
            ],
            [
                {"text": "B1 Intermediate", "callback_data": "level_B1"},
                {"text": "B2 Upper-Int", "callback_data": "level_B2"}
            ],
            [
                {"text": "C1 Advanced", "callback_data": "level_C1"},
                {"text": "C2 Proficient", "callback_data": "level_C2"}
            ],
            [{"text": "❌ Close", "callback_data": "close_settings"}]
        ])
        
        settings_text = (f"⚙️ *Settings Menu*\n\n"
                        f"Current language: *{db_user.target_language.upper()}*\n"
                        f"Current level: *{db_user.cefr_level}*\n\n"
                        f"Choose a new language or level:")
        
        await tg_send_message(chat_id, settings_text, settings_keyboard)
        
    elif data == "close_settings":
        # Close settings menu
        await tg_answer_callback_query(query_id, "Settings closed")
        await tg_send_message(chat_id, "Settings menu closed. Send me a voice message to continue practicing!")
        
    elif data == "help":
        # Show help
        await tg_answer_callback_query(query_id, "Help information")
        help_text = (
            "🆘 *Help & Instructions*\n\n"
            "1. Send me a voice message (up to 90 seconds)\n"
            "2. I'll analyze your pronunciation, grammar, and fluency\n"
            "3. You'll receive feedback and suggestions\n\n"
            "🔧 Use /start to see the main menu\n"
            "⚙️ Click Settings to change language/level\n"
            "💬 You can also type commands:\n"
            "• `lang en` - Set language to English\n"
            "• `level B2` - Set CEFR level to B2"
        )
        await tg_send_message(chat_id, help_text)
        
    else:
        # Unknown button
        await tg_answer_callback_query(query_id, "Unknown action")
        print(f"⚠️  Unknown callback data: {data}")
    
    return {"ok": True}

# ---------- Admin ----------
@app.post("/admin/users")
async def admin_users(request: Request, authorization: str | None = Header(None)):
    assert_admin(authorization or "")
    body = await request.json()
    tg_id = int(body["telegram_id"])
    defaults = {k:v for k,v in body.items() if k!="telegram_id"}
    u = upsert_user(engine, tg_id, defaults)
    return {"ok": True, "user_id": u.user_id}

# ---------- Telegram webhook ----------
@app.post(f"/webhook/{WEBHOOK_SUFFIX}")
async def telegram_webhook(req: Request, x_telegram_bot_api_secret_token: str | None = Header(None)):
    print("\n" + "="*50)
    print("📨 Incoming webhook request")
    logger.info("Received webhook request")
    
    # Header check
    if WEBHOOK_HEADER_SECRET and x_telegram_bot_api_secret_token != WEBHOOK_HEADER_SECRET:
        print("❌ Secret token validation failed")
        raise HTTPException(status_code=401, detail="bad secret")

    update = await req.json()
    print(f"📝 Update received: {json.dumps(update, indent=2)}")
    
    # Handle callback queries (button presses)
    callback_query = update.get("callback_query")
    if callback_query:
        print("🔘 Callback query received (button press)")
        return await handle_callback_query(callback_query)
    
    msg = update.get("message") or update.get("edited_message")
    if not msg: 
        print("⚠️  No message in update, ignoring")
        return {"ok": True}

    chat_id = msg["chat"]["id"]
    from_user = msg["from"]
    tg_id = from_user["id"]
    print(f"👤 User: {from_user.get('first_name', 'Unknown')} (ID: {tg_id})")
    print(f"💬 Chat ID: {chat_id}")
    
    user = get_user_by_tid(engine, tg_id) or upsert_user(engine, tg_id, {
        "display_name": from_user.get("first_name","User"),
        "target_language": DEFAULT_LANG, "cefr_level": DEFAULT_CEFR
    })
    print(f"🔍 User profile: lang={user.target_language}, level={user.cefr_level}")

    # text commands
    if "text" in msg:
        txt = msg["text"].strip()
        print(f"💬 Text message: '{txt}'")
        
        if txt.startswith("/start"):
            print("🚀 Processing /start command")
            hello = (f"🎤 *Welcome to Voice Tutor!*\n\n"
                    f"Send me a *voice note* (≤90s) and I'll help you practice speaking!\n\n"
                    f"📊 *Current Settings:*\n"
                    f"• Language: *{user.target_language.upper()}*\n"
                    f"• Level: *{user.cefr_level}*\n\n"
                    f"Use the buttons below or the menu buttons under the input field:")
            
            # Create inline keyboard with main menu buttons
            inline_keyboard = create_inline_keyboard([
                [
                    {"text": "⚙️ Settings", "callback_data": "settings"},
                    {"text": "🆘 Help", "callback_data": "help"}
                ]
            ])
            
            # Create reply keyboard (buttons under input field)
            reply_keyboard = create_reply_keyboard([
                ["🏠 Start", "⚙️ Settings"],
                ["🆘 Help", "🎤 Practice"]
            ])
            
            # Send message with inline keyboard first
            await tg_send_message(chat_id, hello, inline_keyboard)
            
            # Send a follow-up message with reply keyboard
            await tg_send_message(
                chat_id, 
                "👆 You can also use the menu buttons below:", 
                reply_keyboard
            )
            
            print("✅ /start response sent with both inline and reply keyboards")
            return {"ok": True}
        
        # Handle reply keyboard button presses
        if txt in ["🏠 Start", "/start"]:
            # Same as /start command but shorter response since keyboard is already shown
            hello = (f"🎤 *Voice Tutor Ready!*\n\n"
                    f"📊 *Current Settings:*\n"
                    f"• Language: *{user.target_language.upper()}*\n"
                    f"• Level: *{user.cefr_level}*\n\n"
                    f"Send me a voice message to begin practicing!")
            
            await tg_send_message(chat_id, hello)
            return {"ok": True}
        
        if txt == "⚙️ Settings":
            # Show settings menu (same as callback)
            settings_keyboard = create_inline_keyboard([
                [
                    {"text": "🇺🇸 English", "callback_data": "lang_en"},
                    {"text": "🇫🇷 French", "callback_data": "lang_fr"},
                    {"text": "🇰🇿 Kazakh", "callback_data": "lang_kk"}
                ],
                [
                    {"text": "A1 Beginner", "callback_data": "level_A1"},
                    {"text": "A2 Elementary", "callback_data": "level_A2"}
                ],
                [
                    {"text": "B1 Intermediate", "callback_data": "level_B1"},
                    {"text": "B2 Upper-Int", "callback_data": "level_B2"}
                ],
                [
                    {"text": "C1 Advanced", "callback_data": "level_C1"},
                    {"text": "C2 Proficient", "callback_data": "level_C2"}
                ],
                [{"text": "❌ Close", "callback_data": "close_settings"}]
            ])
            
            settings_text = (f"⚙️ *Settings Menu*\n\n"
                            f"Current language: *{user.target_language.upper()}*\n"
                            f"Current level: *{user.cefr_level}*\n\n"
                            f"Choose a new language or level:")
            
            await tg_send_message(chat_id, settings_text, settings_keyboard)
            return {"ok": True}
        
        if txt == "🆘 Help":
            # Show help (same as callback)
            help_text = (
                "🆘 *Help & Instructions*\n\n"
                "1. Send me a voice message (up to 90 seconds)\n"
                "2. I'll analyze your pronunciation, grammar, and fluency\n"
                "3. You'll receive feedback and suggestions\n\n"
                "🔧 Use 🏠 Start to see the main menu\n"
                "⚙️ Click Settings to change language/level\n"
                "💬 You can also type commands:\n"
                "• `lang en` - Set language to English\n"
                "• `level B2` - Set CEFR level to B2"
            )
            await tg_send_message(chat_id, help_text)
            return {"ok": True}
        
        if txt == "🎤 Practice":
            # Encourage voice message
            practice_text = (
                "🎤 *Ready to Practice!*\n\n"
                "Send me a voice message (up to 90 seconds) and I'll help you improve your speaking skills!\n\n"
                f"📊 Practicing in: *{user.target_language.upper()}* (Level: *{user.cefr_level}*)"
            )
            await tg_send_message(chat_id, practice_text)
            return {"ok": True}
        
        # Original text commands
        if txt.startswith("lang "):
            new_lang = txt.split(" ",1)[1].strip().lower()
            print(f"🌍 Updating language to: {new_lang}")
            upsert_user(engine, tg_id, {"target_language": new_lang})
            await tg_send_message(chat_id, f"✅ Language updated to *{new_lang.upper()}*")
            return {"ok": True}
        if txt.startswith("level "):
            new_level = txt.split(" ",1)[1].strip().upper()
            print(f"📊 Updating level to: {new_level}")
            upsert_user(engine, tg_id, {"cefr_level": new_level})
            await tg_send_message(chat_id, f"✅ Level updated to *{new_level}*")
            return {"ok": True}

    # voice handler
    voice = msg.get("voice")
    if not voice:  # ignore non-voice
        print("⚠️  No voice message, ignoring")
        return {"ok": True}

    print(f"🎤 Voice message detected: {voice.get('duration', 0)}s, file_id: {voice['file_id']}")
    
    # download voice
    file_id = voice["file_id"]
    print(f"⬇️  Downloading voice file: {file_id}")
    file_path = await tg_get_file_path(file_id)
    print(f"📂 File path: {file_path}")
    audio_bytes = await tg_download(file_path)
    print(f"✅ Downloaded {len(audio_bytes)} bytes")

    # STT
    print(f"🗣️  Starting speech-to-text for language: {user.target_language}")
    transcript = stt_openai(audio_bytes, user.target_language)
    print(f"📝 Transcript: '{transcript}'")

    # LLM eval
    print(f"🤖 Starting LLM evaluation (level: {user.cefr_level})")
    eval_json = await llm_eval_openai(transcript, user.target_language, user.cefr_level)
    print(f"📊 Evaluation result: {json.dumps(eval_json, indent=2)}")

    # Persist errors as MistakeEvents (flat schedule)
    now = datetime.now()
    errors = eval_json.get("errors", [])
    print(f"💾 Saving {len(errors)} mistake events to database")
    
    for i, e in enumerate(errors):
        next_due, interval, easiness, repetitions = schedule_initial(now)
        evt = MistakeEvent(
            id=f"evt_{uuid.uuid4().hex[:8]}",
            user_id=user.user_id,
            timestamp=now.isoformat()+"Z",
            modality="speaking",
            target_language=user.target_language,
            cefr_tag=user.cefr_level,
            category=e.get("category","misc"),
            subcategory=e.get("subcategory"),
            source_prompt=eval_json.get("suggested_prompt_for_next_turn"),
            user_attempt=transcript,
            correction=e.get("fix"),
            explanation=e.get("fix"),
            ipa=e.get("ipa"),
            severity=3,
            confidence=eval_json.get("confidence", 0.5),
            times_seen=1, times_correct=0,
            last_seen=now.isoformat()+"Z",
            next_due=next_due.isoformat()+"Z",
            interval=interval, easiness=easiness, repetitions=repetitions
        )
        add_mistake(engine, evt)
        print(f"  📌 Error {i+1}: {e.get('category')} - {e.get('subcategory', 'N/A')}")

    # Build concise caption (end-of-turn summary)
    caption = (
        f"Fluency {eval_json.get('fluency',0)}/5, Pron {eval_json.get('pronunciation',0)}/5, "
        f"Grammar {eval_json.get('grammar',0)}/5, Lexis {eval_json.get('lexis',0)}/5.\n"
        f"Tip: {(eval_json.get('notes') or [''])[0]}\n"
        f"Next: {eval_json.get('suggested_prompt_for_next_turn','')}"
    )
    print(f"📋 Response caption: '{caption}'")

    # TTS reply (OGG/OPUS voice message)
    print(f"🔊 Generating TTS response for language: {user.target_language}")
    tts_audio = await tts_azure(caption, user.target_language)
    print(f"✅ Generated TTS audio: {len(tts_audio)} bytes")
    
    print(f"📤 Sending voice response to chat {chat_id}")
    await tg_send_voice(chat_id, tts_audio, caption)
    
    # Send a follow-up message with action buttons
    follow_up_text = "🎯 *What would you like to do next?*"
    follow_up_inline_keyboard = create_inline_keyboard([
        [
            {"text": "⚙️ Settings", "callback_data": "settings"},
            {"text": "🆘 Help", "callback_data": "help"}
        ]
    ])
    
    # Also maintain the reply keyboard for convenience
    follow_up_reply_keyboard = create_reply_keyboard([
        ["🏠 Start", "⚙️ Settings"],
        ["🆘 Help", "🎤 Practice"]
    ])
    
    await tg_send_message(chat_id, follow_up_text, follow_up_inline_keyboard)
    await tg_send_message(chat_id, "Use the menu buttons below for quick access:", follow_up_reply_keyboard)
    print("🎉 Response sent successfully!")
    print("="*50 + "\n")
    return {"ok": True}
