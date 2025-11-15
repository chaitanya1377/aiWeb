from flask import Flask, render_template, request, jsonify
from openai import OpenAI                # optional, not used right now
from google import genai
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

# You can keep this if you plan to use OpenAI later
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


@app.route("/")
def homePage():
    return render_template("index.html")


@app.route("/select")
def selectPage():
    return render_template("select.html")


@app.route("/custom")
def customPage():
    return render_template("custom.html")


@app.route("/code")
def codePage():
    return render_template("code.html")


@app.route("/research")
def researchPage():
    return render_template("research.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message", "").strip()
    provider = request.json.get("provider", "Gemini")
    style = request.json.get("style", "Default")

    if not user_msg:
        return jsonify({"reply": "⚠️ Please enter a message first."})

    # We only support the unified "Gemini" provider from frontend
    if provider != "Gemini":
        return jsonify({"reply": "❌ Invalid provider"}), 400

    # ---- Style prompts shared by Gemini & Groq ----
    style_prompts = {
        "ChatGPT": "Act like ChatGPT. Clear, structured, friendly, step-by-step when needed.",
        "DeepSeek": "Act like DeepSeek R1. First write 'Reasoning:' with detailed steps, then 'Answer:'.",
        "Perplexity": "Act like Perplexity. Use factual bullet points with mini-citations like (NASA, 2023).",
        "Claude": "Act like Claude. Warm, thoughtful, very explanatory and gentle.",
        "GitHub Copilot": "Act like GitHub Copilot. Output code first, minimal explanation.",
        "Gemini": "Respond normally as Gemini.",
        "Default": "Respond normally."
    }

    prefix = style_prompts.get(style, "Respond normally.")

    # -------------- 1–3: GEMINI FALLBACK CHAIN --------------

    gemini_models = [
        "gemini-2.5-flash",  # fastest & smartest, but sometimes overloaded
        "gemini-2.0-flash",  # very stable
        "gemini-1.5-flash"   # older but rock solid
    ]

    for model_name in gemini_models:
        try:
            response = gemini_client.models.generate_content(
                model=model_name,
                contents=f"{prefix}\n\nUser: {user_msg}"
            )
            # Success → return immediately
            return jsonify({"reply": response.text})

        except Exception as e:
            # If this model fails, just try the next one.
            # (You can log e if you want)
            continue

    # -------------- 4–6: GROQ FALLBACK CHAIN --------------

    groq_models = [
        "llama-3.1-8b-instant",     # stable production 8B
        "llama-3.3-70b-versatile",  # higher quality, production
        "mixtral-8x7b-32768"        # solid multi-expert model
    ]

    for model_name in groq_models:
        try:
            groq_response = groq_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": prefix},
                    {"role": "user", "content": user_msg}
                ]
            )

            reply = groq_response.choices[0].message["content"]
            return jsonify({"reply": reply})

        except Exception as e:
            # Try next Groq model
            continue

    # -------------- If EVERYTHING fails (very rare) --------------

    return jsonify({
        "reply": "⚠️ All AI models are temporarily busy. Please try again in a few seconds."
    })


if __name__ == "__main__":
    app.run(debug=True)
