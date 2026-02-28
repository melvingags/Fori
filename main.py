import os
import json
import logging
from flask import Flask, request, jsonify, send_file
import google.generativeai as genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(model_name="gemini-2.5-flash")
        logger.info("Gemini loaded for LEX.")
    except Exception as e:
        logger.error(f"Gemini failed: {e}")

LEGAL_SYSTEM_PROMPT = """You are LEX, an AI-powered global legal navigator. You provide clear, accurate legal information to people worldwide.

Your role:
- Answer legal questions based on the laws of the specified country
- Cite specific laws, acts, sections and articles where possible
- Explain legal concepts in plain, simple language anyone can understand
- Cover criminal law, civil law, constitutional rights, employment, property, family law
- Always mention relevant penalties, sentencing ranges, or legal consequences when asked
- Provide practical next steps the person can take
- Be empathetic — people asking legal questions are often in difficult situations

IMPORTANT RULES:
- Always specify which country's law you are applying
- Always end with: "This is legal information, not legal advice. Consult a qualified lawyer for your specific situation."
- If you don't know the specific law, say so honestly and suggest where to find it
- Never make up laws or citations
- Keep answers clear, structured and practical
- Use numbered lists and clear headings for readability"""

@app.route('/')
def index():
    static_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'index.html')
    if os.path.exists(static_path):
        return send_file(static_path)
    return "<h1>LEX is online</h1><p>Static files not found. Check your repo structure.</p>", 200

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question', '').strip()
    country = data.get('country', 'Nigeria')
    category = data.get('category', 'General Law')

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    if not model:
        return jsonify({'error': 'AI engine offline. Please try again later.'}), 503

    try:
        prompt = f"""Country: {country}
Legal Category: {category}
User Question: {question}

Please provide a clear, accurate legal answer based on the laws of {country}.
Structure your answer with clear sections and cite specific laws where possible."""

        chat = model.start_chat(history=[
            {"role": "user", "parts": [LEGAL_SYSTEM_PROMPT]},
            {"role": "model", "parts": ["Understood. I am LEX, your AI legal navigator. I will provide clear, accurate legal information based on real laws, always specifying the jurisdiction and recommending professional legal advice for specific situations."]}
        ])

        response = chat.send_message(prompt)
        answer = response.text

        return jsonify({
            'answer': answer,
            'country': country,
            'category': category
        })

    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({'error': f'Unable to process your question: {str(e)[:100]}'}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'online', 'ai': 'online' if model else 'offline'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
