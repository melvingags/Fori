import os
import json
import logging
import datetime
from collections import defaultdict
from flask import Flask, request, jsonify, send_file
import google.generativeai as genai
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(model_name="gemini-2.5-flash")
        logger.info("LEX AI Engine online.")
    except Exception as e:
        logger.error(f"Gemini failed: {e}")

usage_tracker = defaultdict(lambda: {"count": 0, "date": str(datetime.date.today())})

FREE_PROMPT = """You are LEX, an AI legal information assistant. You provide clear, simple legal information to everyday people.

PERSONALITY: Formal but simple and clear. You speak plainly so anyone can understand — no legal jargon unless explained.

YOUR CAPABILITIES:
- Answer legal questions based on the laws of the specified country
- Cite specific laws, acts, and sections where possible
- Explain what the law means in simple terms
- Provide basic next steps the person can take

RULES:
- Keep answers clear and structured with numbered points
- Always name the specific law or act you are referencing
- Always end with: "This is legal information only, not legal advice. For your specific situation, consult a qualified lawyer."
- Do NOT draft documents (premium feature)
- Do NOT provide deep case analysis (premium feature)
- Be helpful but keep responses concise"""

PREMIUM_PROMPT = """You are LEX Premium, an advanced AI legal advisor. You provide friendly, thorough legal guidance with warmth and care.

PERSONALITY: Like a knowledgeable friend who happens to be a lawyer. Warm, approachable, detailed, and genuinely helpful. You make people feel supported and informed.

YOUR CAPABILITIES:
- Answer any legal question with detailed analysis
- Cite specific laws, acts, sections, and relevant case law
- Draft legal documents (letters, notices, complaints, simple contracts)
- Analyze the user's legal position and recommend strategy
- Search for current legal developments when needed
- Explain complex legal concepts with relatable examples
- Predict likely outcomes based on the facts given

DOCUMENT DRAFTING: When asked to draft a document, produce a complete, properly formatted legal document ready to use.

RULES:
- Be warm, supportive and thorough
- Use clear headings and numbered points
- Always cite the specific law
- Always end with: "This analysis is provided as legal information. For court proceedings or binding legal matters, I recommend consulting a qualified lawyer."
- Go above and beyond — give the user everything they need"""

BUSINESS_PROMPT = """You are LEX Business, an elite AI legal counsel system designed for businesses and legal professionals.

PERSONALITY: Like a senior partner at a top law firm. Precise, authoritative, thorough, and highly professional. You speak the language of lawyers and business leaders.

YOUR CAPABILITIES:
- Provide comprehensive legal analysis with full citations
- Draft any legal document: contracts, NDAs, employment agreements, demand letters, corporate resolutions
- Conduct multi-jurisdictional legal analysis
- Predict case outcomes with probability assessments based on precedent
- Identify legal risks and compliance gaps
- Provide strategic legal recommendations
- Analyze contracts and identify problematic clauses
- Research current laws, regulations and recent case law

DOCUMENT DRAFTING: Produce complete, court-ready legal documents with proper headings, clauses, and legal language.

CASE ANALYSIS FORMAT:
1. Legal Issues Identified
2. Applicable Laws and Regulations
3. Precedent Cases (if relevant)
4. Legal Position Assessment
5. Risk Analysis
6. Strategic Recommendations
7. Probability of Success (%)

RULES:
- Be precise, thorough and authoritative
- Always cite specific statutes with section numbers
- Use proper legal terminology
- Structure responses professionally with clear headings
- Always end with: "This analysis constitutes legal information prepared by LEX Business AI. It does not establish an attorney-client relationship. Verify all citations before use in legal proceedings."
- Deliver elite-level legal intelligence"""

def search_legal_info(query, country):
    if not TAVILY_API_KEY:
        return None
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": f"{query} {country} law legal",
                "search_depth": "advanced",
                "max_results": 5,
                "include_domains": [
                    "legislation.gov.uk", "law.cornell.edu", "nigerialii.org",
                    "worldlii.org", "austlii.edu.au", "bailii.org",
                    "justice.gov", "supremecourt.gov.ng", "lawphil.net"
                ]
            },
            timeout=10
        )
        if response.status_code == 200:
            results = response.json().get("results", [])
            if results:
                search_summary = "\n\nCURRENT LEGAL RESEARCH FOUND:\n"
                for r in results[:3]:
                    search_summary += f"- {r.get('title', '')}: {r.get('content', '')[:300]}\n"
                return search_summary
    except Exception as e:
        logger.error(f"Search error: {e}")
    return None

def detect_query_type(question):
    q = question.lower()
    if any(word in q for word in ["draft", "write", "create", "prepare", "template", "letter", "contract", "agreement", "notice", "document"]):
        return "document"
    elif any(word in q for word in ["analyze", "analyse", "chance", "likely", "outcome", "win", "lose", "case", "predict", "probability"]):
        return "analysis"
    elif any(word in q for word in ["current", "latest", "recent", "new law", "updated", "2024", "2025", "2026", "today"]):
        return "research"
    else:
        return "question"

def check_rate_limit(ip, tier):
    if tier != "free":
        return True
    today = str(datetime.date.today())
    user = usage_tracker[ip]
    if user["date"] != today:
        usage_tracker[ip] = {"count": 0, "date": today}
    if usage_tracker[ip]["count"] >= 5:
        return False
    usage_tracker[ip]["count"] += 1
    return True

def get_remaining_questions(ip):
    today = str(datetime.date.today())
    user = usage_tracker[ip]
    if user["date"] != today:
        return 5
    return max(0, 5 - user["count"])

@app.route('/')
def index():
    static_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'index.html')
    if os.path.exists(static_path):
        return send_file(static_path)
    return "<h1>LEX AI is online</h1>", 200

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question', '').strip()
    country = data.get('country', 'Nigeria')
    category = data.get('category', 'General Law')
    tier = data.get('tier', 'free').lower()

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    if not model:
        return jsonify({'error': 'LEX AI engine is offline. Please try again later.'}), 503

    ip = request.remote_addr
    if not check_rate_limit(ip, tier):
        return jsonify({
            'error': 'You have reached your 5 free questions for today. Upgrade to LEX Premium for unlimited access.',
            'limit_reached': True
        }), 429

    try:
        if tier == "business":
            system_prompt = BUSINESS_PROMPT
        elif tier == "premium":
            system_prompt = PREMIUM_PROMPT
        else:
            system_prompt = FREE_PROMPT

        query_type = detect_query_type(question)

        search_context = ""
        if query_type == "research" or tier in ["premium", "business"]:
            search_result = search_legal_info(question, country)
            if search_result:
                search_context = search_result

        type_instructions = {
            "document": "\n\nTASK TYPE: DOCUMENT DRAFTING\nThe user wants you to draft a legal document. Produce a complete, properly formatted document.",
            "analysis": "\n\nTASK TYPE: CASE ANALYSIS\nAnalyze this legal situation thoroughly. Assess the legal position, risks, and likely outcomes.",
            "research": "\n\nTASK TYPE: LEGAL RESEARCH\nResearch current laws and recent developments on this topic.",
            "question": "\n\nTASK TYPE: LEGAL QUESTION\nAnswer this legal question clearly and thoroughly."
        }

        prompt = f"""Country/Jurisdiction: {country}
Legal Category: {category}
Query Type: {query_type.upper()}
User Tier: {tier.upper()}

User Request:
{question}
{search_context}
{type_instructions.get(query_type, '')}

Provide a comprehensive response based on the laws of {country}."""

        tier_intro = {
            "business": "elite-level legal counsel",
            "premium": "thorough, friendly legal guidance",
            "free": "clear, simple legal information"
        }

        chat = model.start_chat(history=[
            {"role": "user", "parts": [system_prompt]},
            {"role": "model", "parts": [f"Understood. I am LEX {'Business' if tier == 'business' else 'Premium' if tier == 'premium' else ''}. Ready to provide {tier_intro[tier]} based on the laws of {country}."]}
        ])

        response = chat.send_message(prompt)
        answer = response.text

        result = {
            'answer': answer,
            'country': country,
            'category': category,
            'tier': tier,
            'query_type': query_type,
            'web_search_used': bool(search_context),
        }

        if tier == "free":
            result['remaining_questions'] = get_remaining_questions(ip)

        return jsonify(result)

    except Exception as e:
        logger.error(f"LEX AI Error: {e}")
        return jsonify({'error': f'LEX encountered an error: {str(e)[:150]}'}), 500

@app.route('/draft', methods=['POST'])
def draft_document():
    data = request.json
    doc_type = data.get('doc_type', '')
    details = data.get('details', '')
    country = data.get('country', 'Nigeria')
    tier = data.get('tier', 'free')

    if tier == 'free':
        return jsonify({'error': 'Document drafting is a Premium feature. Upgrade to access this capability.'}), 403

    if not model:
        return jsonify({'error': 'LEX AI engine is offline.'}), 503

    try:
        system_prompt = BUSINESS_PROMPT if tier == 'business' else PREMIUM_PROMPT

        prompt = f"""DOCUMENT DRAFTING REQUEST
Country: {country}
Document Type: {doc_type}
Details: {details}

Draft a complete, properly formatted {doc_type} legally appropriate for {country}.
Include all necessary clauses, proper headings, signature blocks, and legal language."""

        chat = model.start_chat(history=[
            {"role": "user", "parts": [system_prompt]},
            {"role": "model", "parts": ["Understood. I will draft a complete legal document."]}
        ])

        response = chat.send_message(prompt)
        return jsonify({'document': response.text, 'doc_type': doc_type, 'country': country})

    except Exception as e:
        logger.error(f"Draft error: {e}")
        return jsonify({'error': str(e)[:150]}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'online',
        'ai': 'online' if model else 'offline',
        'web_search': 'online' if TAVILY_API_KEY else 'offline',
        'tiers': ['free', 'premium', 'business'],
        'version': '2.0'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
