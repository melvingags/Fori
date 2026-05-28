import os
import json
import logging
import datetime
import io
import hmac
import hashlib
from collections import defaultdict
from flask import Flask, request, jsonify, send_file, Response
import google.generativeai as genai
import requests

try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    from docx import Document as DocxDocument
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

# ── LEMON SQUEEZY CONFIG ──
LS_API_KEY = os.environ.get("LEMONSQUEEZY_API_KEY")
LS_STORE_ID = os.environ.get("LEMONSQUEEZY_STORE_ID", "314031")
LS_WEBHOOK_SECRET = os.environ.get("LEMONSQUEEZY_WEBHOOK_SECRET", "vaxjfkoll65lexsecret")

LS_VARIANT_IDS = {
    "premium_monthly":  "1396669",
    "premium_yearly":   "1396678",
    "lawyer_monthly":   "1396689",
    "lawyer_yearly":    "1396693",
    "verified_badge":   "1396698",
    "boost_7day":       "1396701",
    "boost_30day":      "1396704",
}

LS_VARIANT_MAP = {
    "1396669": {"plan": "premium",   "role": "individual", "billing": "monthly"},
    "1396678": {"plan": "premium",   "role": "individual", "billing": "yearly"},
    "1396689": {"plan": "premium",   "role": "lawyer",     "billing": "monthly"},
    "1396693": {"plan": "premium",   "role": "lawyer",     "billing": "yearly"},
    "1396698": {"plan": "badge",     "role": "lawyer",     "billing": "one_time"},
    "1396701": {"plan": "boost_7",   "role": "lawyer",     "billing": "one_time"},
    "1396704": {"plan": "boost_30",  "role": "lawyer",     "billing": "one_time"},
}

model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(model_name="gemini-2.5-flash")
        logger.info("LEX AI Engine online.")
    except Exception as e:
        logger.error(f"Gemini failed: {e}")

# ── GROQ FALLBACK (silent, instant switch when Gemini fails) ──
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

def _groq_messages(prompt, system_prompt=None, ack=None, history=None):
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    if ack:
        msgs.append({"role": "assistant", "content": ack})
    if history:
        for m in history:
            r = "user" if m.get("role") == "user" else "assistant"
            msgs.append({"role": r, "content": m.get("content", "")})
    msgs.append({"role": "user", "content": prompt})
    return msgs

def _groq_generate(prompt, system_prompt=None, ack=None, history=None):
    resp = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={"model": GROQ_MODEL, "max_tokens": 8000,
              "messages": _groq_messages(prompt, system_prompt, ack, history)},
        timeout=90,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

def _groq_stream(prompt, system_prompt=None, ack=None, history=None):
    resp = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={"model": GROQ_MODEL, "max_tokens": 8000, "stream": True,
              "messages": _groq_messages(prompt, system_prompt, ack, history)},
        timeout=90,
        stream=True,
    )
    resp.raise_for_status()
    for raw in resp.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                delta = json.loads(payload)["choices"][0]["delta"].get("content", "")
            except Exception:
                continue
            if delta:
                yield delta

def _gemini_history(system_prompt, ack, history):
    gem = [
        {"role": "user", "parts": [system_prompt]},
        {"role": "model", "parts": [ack if ack else "Understood."]},
    ]
    if history:
        for m in history:
            role = "user" if m.get("role") == "user" else "model"
            gem.append({"role": role, "parts": [m.get("content", "")]})
    return gem

def generate_answer(prompt, system_prompt=None, ack=None, history=None):
    """Try Gemini; on ANY failure, silently fall back to Groq. Returns full text."""
    if model:
        try:
            if system_prompt is None:
                return model.generate_content(prompt).text
            chat = model.start_chat(history=_gemini_history(system_prompt, ack, history))
            return chat.send_message(prompt).text
        except Exception as e:
            logger.warning(f"Gemini failed -> Groq fallback: {str(e)[:120]}")
    return _groq_generate(prompt, system_prompt, ack, history)

def stream_answer(prompt, system_prompt=None, ack=None, history=None):
    """Generator. Try Gemini streaming; if it fails before output, silently switch to Groq."""
    if model:
        emitted = False
        try:
            chat = model.start_chat(history=_gemini_history(system_prompt, ack, history))
            for chunk in chat.send_message(prompt, stream=True):
                if chunk.text:
                    emitted = True
                    yield chunk.text
            return
        except Exception as e:
            logger.warning(f"Gemini stream failed -> Groq fallback: {str(e)[:120]}")
            if emitted:
                return
    for delta in _groq_stream(prompt, system_prompt, ack, history):
        yield delta

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

LEGAL_DOMAINS = [
    "worldlii.org", "law.cornell.edu", "legal-dictionary.thefreedictionary.com",
    "lawteacher.net", "legislation.gov.uk", "bailii.org",
    "nigerialii.org", "supremecourt.gov.ng", "lawnigeria.com",
    "ghanalii.org", "kenyalaw.org", "saflii.org", "lawlibrary.org.za",
    "justice.gov", "supremecourt.gov", "law.justia.com", "canlii.org",
    "austlii.edu.au", "lawphil.net", "singaporelegaladvice.com",
    "eur-lex.europa.eu", "echr.coe.int",
]

SOURCE_TYPE_KEYWORDS = {
    "case law": ["v.", "vs.", "appeal", "court", "judgment", "ruling", "decision", "held"],
    "legislation": ["act", "law", "statute", "code", "regulation", "section", "article", "ordinance"],
    "regulation": ["regulation", "rule", "order", "directive", "circular", "gazette"],
    "commentary": ["guide", "overview", "explained", "analysis", "what is", "how to"],
}

def classify_source(title, content):
    text = (title + " " + content).lower()
    for stype, keywords in SOURCE_TYPE_KEYWORDS.items():
        if any(k in text for k in keywords):
            return stype
    return "legal reference"

def optimise_legal_query(question, country, model_instance):
    if not model_instance:
        return f"{question} {country} law"
    try:
        prompt = f"""You are a legal research assistant. Convert the following user question into an optimised legal search query for finding case law, legislation and legal precedents.

User question: {question}
Country: {country}

Rules:
- Include the country name
- Include relevant legal terminology
- Include relevant act/law names if you know them
- Keep it under 15 words
- Return ONLY the search query, nothing else"""
        optimised = generate_answer(prompt).strip().replace('"', '').replace("'", '')
        return optimised if optimised else f"{question} {country} law"
    except:
        return f"{question} {country} law legal"

def search_legal_info(query, country, optimise=False):
    if not TAVILY_API_KEY:
        return None, []
    try:
        search_query = optimise_legal_query(query, country, model) if optimise else f"{query} {country} law legal"
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": search_query,
                "search_depth": "advanced",
                "max_results": 7,
                "include_domains": LEGAL_DOMAINS,
                "include_answer": True,
            },
            timeout=12
        )
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            if not results:
                return None, []
            country_lower = country.lower()
            scored = []
            for r in results:
                score = r.get('score', 0)
                title = r.get('title', '')
                content = r.get('content', '')
                url = r.get('url', '')
                if country_lower in (title + content + url).lower():
                    score += 0.3
                scored.append({'title': title, 'content': content, 'url': url, 'score': score, 'type': classify_source(title, content)})
            scored.sort(key=lambda x: x['score'], reverse=True)
            top = scored[:5]
            search_summary = "\n\nLEGAL RESEARCH SOURCES FOUND:\n"
            for r in top:
                search_summary += f"\n[{r['type'].upper()}] {r['title']}\n{r['content'][:400]}\n"
            sources = [{'title': r['title'], 'url': r['url'], 'type': r['type'], 'snippet': r['content'][:200]} for r in top]
            return search_summary, sources
    except Exception as e:
        logger.error(f"Search error: {e}")
    return None, []

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
        return send_file(static_path, mimetype='text/html')
    return "<h1>FORI AI is online</h1>", 200

@app.route('/auth.html')
def auth():
    static_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'auth.html')
    if os.path.exists(static_path):
        return send_file(static_path, mimetype='text/html')
    return "Not found", 404

@app.route('/dashboard.html')
def dashboard():
    static_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard.html')
    if os.path.exists(static_path):
        return send_file(static_path, mimetype='text/html')
    return "Not found", 404

@app.route('/dashboard-lawyer.html')
def dashboard_lawyer():
    static_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard-lawyer.html')
    if os.path.exists(static_path):
        return send_file(static_path, mimetype='text/html')
    return "Not found", 404

@app.route('/admin.html')
def admin():
    static_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'admin.html')
    if os.path.exists(static_path):
        return send_file(static_path, mimetype='text/html')
    return "Not found", 404


# ── MEMORY SYSTEM ──
def build_memory_context(memories):
    """Build a memory context string from saved memories to inject into prompts."""
    if not memories:
        return ""
    lines = ["\n\n=== USER MEMORY CONTEXT ==="]
    lines.append("The following is important context from this user's previous interactions with LEX.")
    lines.append("Use this to provide personalised, contextually aware responses:\n")
    for m in memories[:10]:
        mtype = m.get('type', 'interaction')
        summary = m.get('summary', '')
        date = m.get('date', '')
        if summary:
            lines.append(f"[{mtype.upper()} — {date}]: {summary}")
    lines.append("\n=== END MEMORY CONTEXT ===")
    lines.append("When relevant, reference this history naturally. If the user asks about a previous case or document, refer to it directly.")
    return "\n".join(lines)


@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question', '').strip()
    country = data.get('country', 'Nigeria')
    category = data.get('category', 'General Law')
    tier = data.get('tier', 'free').lower()
    memories = data.get('memories', [])

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
        # FIX 1: memory_context now defined before use
        memory_context = build_memory_context(memories)

        if tier == "business":
            system_prompt = BUSINESS_PROMPT
        elif tier == "premium":
            system_prompt = PREMIUM_PROMPT
        else:
            system_prompt = FREE_PROMPT

        query_type = detect_query_type(question)

        search_context = ""
        search_sources = []
        if query_type == "research" or tier in ["premium", "business"]:
            optimise = tier in ["premium", "business"]
            search_context, search_sources = search_legal_info(question, country, optimise=optimise)
            if not search_context:
                search_context = ""

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
{memory_context}

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

        ack = f"Understood. I am LEX {'Business' if tier == 'business' else 'Premium' if tier == 'premium' else ''}. Ready to provide {tier_intro[tier]} based on the laws of {country}."
        answer = generate_answer(prompt, system_prompt=system_prompt, ack=ack)

        result = {
            'answer': answer,
            'country': country,
            'category': category,
            'tier': tier,
            'query_type': query_type,
            'web_search_used': bool(search_context),
            'sources': search_sources,
        }

        if tier == "free":
            result['remaining_questions'] = get_remaining_questions(ip)

        return jsonify(result)

    except Exception as e:
        logger.error(f"LEX AI Error: {e}")
        return jsonify({'error': f'LEX encountered an error: {str(e)[:150]}'}), 500


# Alias so frontend /api/ask calls resolve correctly
@app.route('/api/ask', methods=['POST', 'OPTIONS'])
def api_ask():
    if request.method == 'OPTIONS':
        return '', 204
    return ask()


@app.route('/ask-stream', methods=['POST'])
def ask_stream():
    data = request.json
    question = data.get('question', '').strip()
    country = data.get('country', 'Nigeria')
    category = data.get('category', 'General Law')
    tier = data.get('tier', 'free').lower()
    history = data.get('history', [])
    memories = data.get('memories', [])
    memory_context = build_memory_context(memories)

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    if not model:
        return jsonify({'error': 'LEX AI engine is offline. Please try again later.'}), 503

    ip = request.remote_addr
    if not check_rate_limit(ip, tier):
        def limit_gen():
            yield f"data: {json.dumps({'error': 'You have reached your 5 free questions for today. Upgrade to LEX Premium for unlimited access.', 'limit_reached': True})}\n\n"
        return Response(limit_gen(), mimetype='text/event-stream')

    try:
        if tier == "business":
            system_prompt = BUSINESS_PROMPT
        elif tier == "premium":
            system_prompt = PREMIUM_PROMPT
        else:
            system_prompt = FREE_PROMPT

        query_type = detect_query_type(question)

        search_context = ""
        search_sources = []
        if query_type == "research" or tier in ["premium", "business"]:
            optimise = tier in ["premium", "business"]
            search_context, search_sources = search_legal_info(question, country, optimise=optimise)
            if not search_context:
                search_context = ""

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
{memory_context}

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

        memory_intro = f" I have access to this user's previous interactions and will reference them naturally when relevant." if memories else ""
        ack = f"Understood. I am LEX {'Business' if tier == 'business' else 'Premium' if tier == 'premium' else ''}. Ready to provide {tier_intro[tier]} based on the laws of {country}.{memory_intro}"

        def generate():
            try:
                full_answer = ""
                for piece in stream_answer(prompt, system_prompt=system_prompt, ack=ack, history=history):
                    full_answer += piece
                    yield f"data: {json.dumps({'chunk': piece})}\n\n"
                remaining = get_remaining_questions(ip) if tier == "free" else None
                yield f"data: {json.dumps({'done': True, 'answer': full_answer, 'query_type': query_type, 'web_search_used': bool(search_context), 'sources': search_sources, 'remaining_questions': remaining})}\n\n"
                if tier == "free":
                    usage_tracker[ip]["count"] = max(0, usage_tracker[ip]["count"] - 1)
            except Exception as e:
                logger.error(f"Stream error: {e}")
                yield f"data: {json.dumps({'error': f'LEX encountered an error: {str(e)[:150]}'})}\n\n"

        return Response(generate(), mimetype='text/event-stream', headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        })

    except Exception as e:
        logger.error(f"LEX Stream Setup Error: {e}")
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

        ack = "Understood. I will draft a complete legal document."
        document = generate_answer(prompt, system_prompt=system_prompt, ack=ack)
        return jsonify({'document': document, 'doc_type': doc_type, 'country': country})

    except Exception as e:
        logger.error(f"Draft error: {e}")
        return jsonify({'error': str(e)[:150]}), 500


def extract_text_from_file(file):
    filename = file.filename.lower()
    content = file.read()

    if filename.endswith('.pdf') and PDF_SUPPORT:
        text = []
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text.append(t)
        return '\n'.join(text)

    elif filename.endswith('.docx') and DOCX_SUPPORT:
        doc = DocxDocument(io.BytesIO(content))
        return '\n'.join([p.text for p in doc.paragraphs if p.text.strip()])

    elif filename.endswith('.txt'):
        return content.decode('utf-8', errors='ignore')

    else:
        return content.decode('utf-8', errors='ignore')


CONTRACT_REVIEW_PROMPT = """You are LEX Contract Analyst, an expert legal AI specialising in contract review across all jurisdictions worldwide.

Analyse the contract below and provide a thorough review structured exactly as follows:

## OVERVIEW
Brief summary of what this contract is and its key purpose.

## PARTIES & KEY TERMS
List the parties involved and the main commercial terms (duration, value, governing law etc).

## CLAUSE ANALYSIS
For each significant clause, provide:
- Clause name
- Risk level: 🔴 HIGH / 🟡 MEDIUM / 🟢 LOW
- What it says (plain English)
- Any concerns or issues

## RED FLAGS 🚨
List any clauses that are unusual, one-sided, potentially unenforceable, or that the user should urgently review with a lawyer.

## MISSING CLAUSES ⚠️
List important clauses that are absent from this contract but should be present.

## RECOMMENDATIONS
Specific actionable advice for the user before signing.

## OVERALL RISK RATING
Give an overall contract risk rating: LOW / MEDIUM / HIGH / VERY HIGH with a brief explanation.

Be thorough, clear and use plain English throughout. Flag anything that could harm the user."""

REDLINE_PROMPT = """You are LEX Redline Expert, a specialist in contract redlining and negotiation across all jurisdictions.

Review the contract below and produce a detailed redline report structured as follows:

## REDLINE REPORT

For each clause that needs attention, provide:

### [Clause Name/Number]
**Issue:** What is wrong or risky about this clause
**Risk Level:** 🔴 HIGH / 🟡 MEDIUM / 🟢 LOW
**Original Language:**
> [Quote the problematic text]

**Suggested Revision:**
[Provide improved alternative wording]

**Why:** Brief explanation of why this change protects the user better

---

## NEGOTIATION PRIORITIES
Rank the top 5 changes the user should push for, in order of importance.

## CLAUSES TO REJECT
List any clauses that should be removed entirely and why.

## CLAUSES TO ADD
List new clauses the user should request be added to protect their interests.

Be specific with revised language. Write as if advising a client preparing for contract negotiation."""


@app.route('/review-stream', methods=['POST'])
def review_stream():
    if not model:
        return jsonify({'error': 'LEX AI engine is offline.'}), 503

    file = request.files.get('file')
    country = request.form.get('country', 'Nigeria')
    tier = request.form.get('tier', 'free')
    memories = json.loads(request.form.get('memories', '[]'))
    memory_context = build_memory_context(memories)

    if not file:
        return jsonify({'error': 'No file uploaded.'}), 400

    try:
        text = extract_text_from_file(file)
        if not text or len(text.strip()) < 50:
            return jsonify({'error': 'Could not extract text from file. Please ensure it is a readable PDF, Word or text document.'}), 400

        if len(text) > 15000:
            text = text[:15000] + '\n\n[Document truncated for analysis — first 15,000 characters reviewed]'

        prompt = f"""Country/Jurisdiction: {country}
{memory_context}

CONTRACT TEXT:
{text}

{CONTRACT_REVIEW_PROMPT}"""

        def generate():
            try:
                ack = "Understood. I am LEX Contract Analyst. I will provide a thorough contract review."
                sys_p = BUSINESS_PROMPT if tier == 'business' else PREMIUM_PROMPT
                full = ""
                for piece in stream_answer(prompt, system_prompt=sys_p, ack=ack):
                    full += piece
                    yield f"data: {json.dumps({'chunk': piece})}\n\n"
                yield f"data: {json.dumps({'done': True, 'full': full})}\n\n"
            except Exception as e:
                logger.error(f"Review stream error: {e}")
                yield f"data: {json.dumps({'error': str(e)[:150]})}\n\n"

        return Response(generate(), mimetype='text/event-stream', headers={
            'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'
        })

    except Exception as e:
        logger.error(f"Review setup error: {e}")
        return jsonify({'error': str(e)[:150]}), 500


@app.route('/redline-stream', methods=['POST'])
def redline_stream():
    if not model:
        return jsonify({'error': 'LEX AI engine is offline.'}), 503

    file = request.files.get('file')
    country = request.form.get('country', 'Nigeria')
    tier = request.form.get('tier', 'free')

    if not file:
        return jsonify({'error': 'No file uploaded.'}), 400

    try:
        text = extract_text_from_file(file)
        if not text or len(text.strip()) < 50:
            return jsonify({'error': 'Could not extract text from file. Please ensure it is a readable PDF, Word or text document.'}), 400

        if len(text) > 15000:
            text = text[:15000] + '\n\n[Document truncated — first 15,000 characters reviewed]'

        prompt = f"""Country/Jurisdiction: {country}

CONTRACT TEXT:
{text}

{REDLINE_PROMPT}"""

        def generate():
            try:
                ack = "Understood. I am LEX Redline Expert. I will produce a detailed redline report with specific revision suggestions."
                sys_p = BUSINESS_PROMPT if tier == 'business' else PREMIUM_PROMPT
                full = ""
                for piece in stream_answer(prompt, system_prompt=sys_p, ack=ack):
                    full += piece
                    yield f"data: {json.dumps({'chunk': piece})}\n\n"
                yield f"data: {json.dumps({'done': True, 'full': full})}\n\n"
            except Exception as e:
                logger.error(f"Redline stream error: {e}")
                yield f"data: {json.dumps({'error': str(e)[:150]})}\n\n"

        return Response(generate(), mimetype='text/event-stream', headers={
            'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'
        })

    except Exception as e:
        logger.error(f"Redline setup error: {e}")
        return jsonify({'error': str(e)[:150]}), 500


CLAUSE_LIBRARY_PROMPT = """You are LEX Clause Library, an expert in legal drafting across all jurisdictions worldwide.

The user is looking for a legal clause. Provide the following structured response:

## CLAUSE: [Clause Name]
**Category:** [e.g. Employment / NDA / Lease / Service Agreement / etc.]
**Jurisdiction:** [Country/Region or General]
**Risk Level:** 🟢 LOW / 🟡 MEDIUM / 🔴 HIGH (for the party using it)

---

## PLAIN ENGLISH EXPLANATION
What this clause means in simple terms and why it matters.

## STANDARD CLAUSE TEXT
```
[Provide clean, professional standard clause text ready to use]
```

## VARIATIONS
Provide 2–3 alternative versions:
1. **[Variation name e.g. "Stronger / More Protective"]**
```
[Variation text]
```
2. **[Variation name e.g. "Balanced / Mutual"]**
```
[Variation text]
```

## NEGOTIATION NOTES
Key points to watch for when this clause is used against you, and what to push for.

## JURISDICTION NOTES
Any important differences for the specified jurisdiction compared to international standard.

Be thorough, practical and provide clause text that is ready to copy and use."""

OUTCOME_PROMPT = """You are LEX Case Analyst, an expert in legal case assessment and outcome prediction across all jurisdictions.

Analyse the case described and provide a structured assessment:

## CASE SUMMARY
Brief restatement of the key facts and legal issue.

## LEGAL FRAMEWORK
What area of law applies and the key legal principles relevant to this case.

## OUTCOME PREDICTION

### Likelihood of Success
Provide a probability range (e.g. 60–75%) with confidence level and clear reasoning.

### Favourable Factors ✅
List specific facts that support a positive outcome.

### Unfavourable Factors ❌
List specific facts that work against a positive outcome.

### Key Uncertainties ⚠️
What unknowns could significantly change the outcome.

## POSSIBLE OUTCOMES
Describe 3 realistic scenarios:
1. **Best Case** — what it looks like and what needs to go right
2. **Most Likely Case** — the realistic middle ground
3. **Worst Case** — what it looks like if things go wrong

## FINANCIAL ASSESSMENT
Rough estimate of potential costs, damages or compensation involved if applicable.

## RECOMMENDED NEXT STEPS
Specific, actionable advice ranked by priority.

## URGENCY FLAG
🔴 URGENT / 🟡 TIME SENSITIVE / 🟢 NO IMMEDIATE DEADLINE — with explanation of any deadlines or limitation periods that apply.

Be honest about uncertainty. Never guarantee outcomes. Always recommend consulting a qualified lawyer for formal legal advice."""


@app.route('/clause-stream', methods=['POST'])
def clause_stream():
    if not model:
        return jsonify({'error': 'LEX AI engine is offline.'}), 503

    data = request.json
    query = data.get('query', '').strip()
    country = data.get('country', 'Nigeria')
    tier = data.get('tier', 'free')
    memories = data.get('memories', [])
    memory_context = build_memory_context(memories)

    if not query:
        return jsonify({'error': 'Please describe the clause you need.'}), 400

    prompt = f"""Jurisdiction: {country}
{memory_context}
Clause Request: {query}

{CLAUSE_LIBRARY_PROMPT}"""

    def generate():
        try:
            ack = "Understood. I am LEX Clause Library. I will provide comprehensive clause information including standard text, variations and negotiation notes."
            sys_p = BUSINESS_PROMPT if tier == 'business' else PREMIUM_PROMPT
            full = ""
            for piece in stream_answer(prompt, system_prompt=sys_p, ack=ack):
                full += piece
                yield f"data: {json.dumps({'chunk': piece})}\n\n"
            yield f"data: {json.dumps({'done': True, 'full': full})}\n\n"
        except Exception as e:
            logger.error(f"Clause stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)[:150]})}\n\n"

    return Response(generate(), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'
    })


@app.route('/outcome-stream', methods=['POST'])
def outcome_stream():
    if not model:
        return jsonify({'error': 'LEX AI engine is offline.'}), 503

    data = request.json
    facts = data.get('facts', '').strip()
    country = data.get('country', 'Nigeria')
    area = data.get('area', 'General')
    tier = data.get('tier', 'free')
    memories = data.get('memories', [])
    memory_context = build_memory_context(memories)

    if not facts or len(facts) < 30:
        return jsonify({'error': 'Please provide more detail about your case for an accurate assessment.'}), 400

    prompt = f"""Jurisdiction: {country}
Area of Law: {area}
{memory_context}

CASE FACTS:
{facts}

{OUTCOME_PROMPT}"""

    def generate():
        try:
            ack = "Understood. I am LEX Case Analyst. I will provide a thorough case assessment with outcome predictions and recommendations."
            sys_p = BUSINESS_PROMPT if tier == 'business' else PREMIUM_PROMPT
            full = ""
            for piece in stream_answer(prompt, system_prompt=sys_p, ack=ack):
                full += piece
                yield f"data: {json.dumps({'chunk': piece})}\n\n"
            yield f"data: {json.dumps({'done': True, 'full': full})}\n\n"
        except Exception as e:
            logger.error(f"Outcome stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)[:150]})}\n\n"

    return Response(generate(), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'
    })


COMPARE_PROMPT = """You are LEX Document Comparison Expert, a specialist in comparative legal analysis across all jurisdictions.

You have been provided with two legal documents. Conduct a thorough side-by-side comparison structured as follows:

## DOCUMENT OVERVIEW

### Document A — [Identify type, parties, date if visible]
Brief summary of Document A's purpose and key terms.

### Document B — [Identify type, parties, date if visible]
Brief summary of Document B's purpose and key terms.

---

## KEY DIFFERENCES

For each significant difference found, provide:

### [Topic/Clause Name]
| | Document A | Document B |
|---|---|---|
| **Position** | [What A says] | [What B says] |
| **Impact** | [Who this favours and why] |

Risk Level: 🔴 HIGH / 🟡 MEDIUM / 🟢 LOW difference

---

## CONFLICTING CLAUSES ⚠️
List any clauses that directly contradict each other between the two documents.

## CLAUSES IN A BUT NOT IN B
List important clauses present in Document A that are missing from Document B.

## CLAUSES IN B BUT NOT IN A
List important clauses present in Document B that are missing from Document A.

## OVERALL COMPARISON VERDICT

### Which document is more favourable?
State clearly whether Document A or Document B is more protective/favourable for the typical signing party, and why.

### Risk Summary
- Document A overall risk: LOW / MEDIUM / HIGH
- Document B overall risk: LOW / MEDIUM / HIGH

## RECOMMENDATIONS
Specific advice on which document to use, what to negotiate, and what clauses to add or remove before signing."""


@app.route('/compare-stream', methods=['POST'])
def compare_stream():
    if not model:
        return jsonify({'error': 'LEX AI engine is offline.'}), 503

    file_a = request.files.get('file_a')
    file_b = request.files.get('file_b')
    country = request.form.get('country', 'Nigeria')
    tier = request.form.get('tier', 'free')

    if not file_a or not file_b:
        return jsonify({'error': 'Please upload both documents to compare.'}), 400

    try:
        text_a = extract_text_from_file(file_a)
        text_b = extract_text_from_file(file_b)

        if not text_a or len(text_a.strip()) < 30:
            return jsonify({'error': 'Could not extract text from Document A.'}), 400
        if not text_b or len(text_b.strip()) < 30:
            return jsonify({'error': 'Could not extract text from Document B.'}), 400

        if len(text_a) > 8000:
            text_a = text_a[:8000] + '\n\n[Document A truncated — first 8,000 characters reviewed]'
        if len(text_b) > 8000:
            text_b = text_b[:8000] + '\n\n[Document B truncated — first 8,000 characters reviewed]'

        prompt = f"""Jurisdiction: {country}

===== DOCUMENT A =====
{text_a}

===== DOCUMENT B =====
{text_b}

{COMPARE_PROMPT}"""

        # FIX 2: compare_stream — fixed double-brace f-string bug
        def generate():
            try:
                ack = "Understood. I am LEX Document Comparison Expert. I will conduct a thorough side-by-side analysis of both documents."
                sys_p = BUSINESS_PROMPT if tier == 'business' else PREMIUM_PROMPT
                full = ""
                for piece in stream_answer(prompt, system_prompt=sys_p, ack=ack):
                    full += piece
                    yield f"data: {json.dumps({'chunk': piece})}\n\n"
                yield f"data: {json.dumps({'done': True, 'full': full})}\n\n"
            except Exception as e:
                logger.error(f"Compare stream error: {e}")
                yield f"data: {json.dumps({'error': str(e)[:150]})}\n\n"

        return Response(generate(), mimetype='text/event-stream', headers={
            'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'
        })

    except Exception as e:
        logger.error(f"Compare setup error: {e}")
        return jsonify({'error': str(e)[:150]}), 500


RESEARCH_PROMPT = """You are LEX Legal Research Engine, an expert legal researcher with access to current case law, legislation and legal commentary across all jurisdictions worldwide.

Conduct thorough legal research on the topic provided and structure your response as follows:

## RESEARCH TOPIC
Restate the research topic clearly with the relevant jurisdiction.

## APPLICABLE LEGISLATION
List the key laws, acts, codes and regulations that apply, with specific sections where relevant. Format each as:
**[Law Name]** (Year) — [Jurisdiction]
Section/Article [X]: [What it says and why it matters]

## RELEVANT CASE LAW
List the most relevant cases with brief summaries. Format each as:
**[Case Name]** [Year] — [Court]
*Held:* [What the court decided and the principle established]

## CURRENT LEGAL POSITION
A clear, authoritative statement of what the law currently says on this topic in the specified jurisdiction.

## RECENT DEVELOPMENTS
Any recent changes, amendments, new cases or upcoming reforms relevant to this topic.

## JURISDICTIONAL VARIATIONS
If relevant, note how the law differs in nearby or commonly compared jurisdictions.

## RESEARCH SUMMARY
A concise 3-5 sentence summary of the key legal position for practical use.

## SOURCES & CITATIONS
List all sources referenced with proper legal citation format."""


@app.route('/research-stream', methods=['POST'])
def research_stream():
    if not model:
        return jsonify({'error': 'LEX AI engine is offline.'}), 503

    data = request.json
    topic = data.get('topic', '').strip()
    country = data.get('country', 'Nigeria')
    tier = data.get('tier', 'free')
    memories = data.get('memories', [])
    memory_context = build_memory_context(memories)

    if not topic or len(topic) < 5:
        return jsonify({'error': 'Please enter a research topic.'}), 400

    search_context, sources = search_legal_info(topic, country, optimise=True)

    prompt = f"""Research Topic: {topic}
Jurisdiction: {country}
{memory_context}

{search_context or ''}

{RESEARCH_PROMPT}"""

    def generate():
        try:
            system = BUSINESS_PROMPT if tier == 'business' else PREMIUM_PROMPT
            ack = f"Understood. I am LEX Legal Research Engine. I will conduct thorough research on this topic for {country} jurisdiction."
            full = ""
            for piece in stream_answer(prompt, system_prompt=system, ack=ack):
                full += piece
                yield f"data: {json.dumps({'chunk': piece})}\n\n"
            yield f"data: {json.dumps({'done': True, 'full': full, 'sources': sources})}\n\n"
        except Exception as e:
            logger.error(f"Research stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)[:150]})}\n\n"

    return Response(generate(), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'
    })


@app.route('/summarise-memory', methods=['POST'])
def summarise_memory():
    if not model:
        return jsonify({'error': 'AI engine offline.'}), 503

    data = request.json
    interaction_type = data.get('type', 'question')
    content_to_summarise = data.get('content', '').strip()

    if not content_to_summarise or len(content_to_summarise) < 20:
        return jsonify({'error': 'Nothing to summarise.'}), 400

    type_instructions = {
        'question': 'Summarise the legal question asked and the key advice given. Include the legal topic, jurisdiction if mentioned, and the main conclusion.',
        'contract_review': 'Summarise the contract reviewed. Include document type, parties if visible, overall risk rating, and the top 3 red flags found.',
        'redline': 'Summarise the document redlined. Include document type, the most critical revision needed, and overall assessment.',
        'clause': 'Summarise the clause searched for and the key information retrieved.',
        'outcome': 'Summarise the case described and the predicted outcome. Include area of law, probability of success range, and urgency level.',
        'compare': 'Summarise the two documents compared and the verdict. Include document types and which was more favourable.',
    }

    instruction = type_instructions.get(interaction_type, type_instructions['question'])

    prompt = f"""You are a legal memory assistant. Your job is to create a concise 1-3 sentence memory summary of the following legal interaction.

Interaction Type: {interaction_type}
Instructions: {instruction}

Content to summarise:
{content_to_summarise[:3000]}

Rules:
- Write in past tense from LEX's perspective (e.g. "User asked about...", "Reviewed a contract...")
- Be specific — include document names, risk levels, outcomes where present
- Maximum 3 sentences
- No preamble, just the summary itself"""

    try:
        summary = generate_answer(prompt).strip()
        return jsonify({'summary': summary})
    except Exception as e:
        logger.error(f"Memory summarise error: {e}")
        return jsonify({'error': str(e)[:150]}), 500


@app.route('/create-checkout', methods=['POST'])
def create_checkout():
    data = request.json
    variant_key = data.get('variant_key', '').strip()
    user_email = data.get('email', '').strip()
    user_uid = data.get('uid', '').strip()
    custom_data = data.get('custom_data', {})
    # FIX 3: dynamic redirect_url — accept from request body, fallback to default
    redirect_url = data.get('redirect_url', 'https://lex-rvmc.onrender.com/dashboard.html?payment=success')

    if not variant_key or variant_key not in LS_VARIANT_IDS:
        return jsonify({'error': 'Invalid product selected.'}), 400

    if not LS_API_KEY:
        return jsonify({'error': 'Payment system not configured.'}), 503

    variant_id = LS_VARIANT_IDS[variant_key]

    try:
        payload = {
            "data": {
                "type": "checkouts",
                "attributes": {
                    "checkout_options": {
                        "embed": False,
                        "media": False,
                        "logo": True,
                    },
                    "checkout_data": {
                        "email": user_email,
                        "custom": {
                            "uid": user_uid,
                            "variant_key": variant_key,
                            **custom_data
                        }
                    },
                    "product_options": {
                        "redirect_url": redirect_url,
                        "receipt_link_url": "https://lex-rvmc.onrender.com/dashboard.html",
                        "receipt_thank_you_note": "Thank you for subscribing to LEX! Your account has been upgraded."
                    },
                    "expires_at": None
                },
                "relationships": {
                    "store": {
                        "data": {"type": "stores", "id": str(LS_STORE_ID)}
                    },
                    "variant": {
                        "data": {"type": "variants", "id": str(variant_id)}
                    }
                }
            }
        }

        response = requests.post(
            "https://api.lemonsqueezy.com/v1/checkouts",
            headers={
                "Authorization": f"Bearer {LS_API_KEY}",
                "Content-Type": "application/vnd.api+json",
                "Accept": "application/vnd.api+json",
            },
            json=payload,
            timeout=15
        )

        if response.status_code == 201:
            checkout_data = response.json()
            checkout_url = checkout_data['data']['attributes']['url']
            return jsonify({'checkout_url': checkout_url, 'variant_key': variant_key})
        else:
            logger.error(f"LS checkout error: {response.status_code} {response.text}")
            return jsonify({'error': 'Could not create checkout. Please try again.'}), 500

    except Exception as e:
        logger.error(f"Checkout creation error: {e}")
        return jsonify({'error': str(e)[:150]}), 500


@app.route('/webhook/lemonsqueezy', methods=['POST'])
def lemonsqueezy_webhook():
    raw_body = request.get_data()
    signature = request.headers.get('X-Signature', '')

    mac = hmac.new(LS_WEBHOOK_SECRET.encode('utf-8'), raw_body, hashlib.sha256)
    expected = mac.hexdigest()

    if not hmac.compare_digest(expected, signature):
        logger.warning("LS webhook: invalid signature")
        return jsonify({'error': 'Invalid signature'}), 401

    try:
        event_name = request.headers.get('X-Event-Name', '')
        payload = request.json
        meta = payload.get('meta', {})
        custom_data = meta.get('custom_data', {})
        user_uid = custom_data.get('uid', '')
        variant_key = custom_data.get('variant_key', '')

        attrs = payload.get('data', {}).get('attributes', {})
        variant_id = str(attrs.get('first_order_item', {}).get('variant_id', '')
                         or attrs.get('variant_id', ''))

        variant_info = LS_VARIANT_MAP.get(variant_id, {})
        plan = variant_info.get('plan', '')
        role = variant_info.get('role', 'individual')
        billing = variant_info.get('billing', '')

        logger.info(f"LS webhook: {event_name} | uid={user_uid} | variant={variant_id} | plan={plan} | role={role}")

        if not user_uid:
            logger.warning("LS webhook: no uid in custom_data")
            return jsonify({'status': 'ok', 'note': 'no uid'}), 200

        update_payload = {}

        if event_name in ('order_created', 'subscription_created'):
            if plan == 'premium':
                # FIX 4: lawyer webhook now correctly sets role for lawyer variants
                update_payload = {
                    'plan': 'premium',
                    'role': role,
                    'premiumSince': datetime.datetime.utcnow().isoformat(),
                    'billing': billing,
                    'lsVariantId': variant_id,
                    'lsSubscriptionId': attrs.get('subscription_id', '') or attrs.get('id', ''),
                }
            elif plan == 'badge':
                update_payload = {'verifiedBadge': True, 'badgePurchasedAt': datetime.datetime.utcnow().isoformat()}
            elif plan == 'boost_7':
                boost_end = (datetime.datetime.utcnow() + datetime.timedelta(days=7)).isoformat()
                update_payload = {'profileBoost': True, 'boostEndsAt': boost_end, 'boostType': '7day'}
            elif plan == 'boost_30':
                boost_end = (datetime.datetime.utcnow() + datetime.timedelta(days=30)).isoformat()
                update_payload = {'profileBoost': True, 'boostEndsAt': boost_end, 'boostType': '30day'}

        elif event_name == 'subscription_cancelled':
            update_payload = {
                'plan': 'free',
                'premiumSince': None,
                'billing': None,
                'lsSubscriptionId': None,
            }

        elif event_name == 'subscription_resumed':
            update_payload = {
                'plan': 'premium',
                'role': role,
                'billing': billing,
                'lsVariantId': variant_id,
            }

        elif event_name == 'subscription_expired':
            update_payload = {
                'plan': 'free',
                'premiumSince': None,
                'billing': None,
            }

        if update_payload and user_uid:
            try:
                import firebase_admin
                from firebase_admin import credentials, firestore as fs

                if not firebase_admin._apps:
                    cred_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
                    if cred_path:
                        cred = credentials.Certificate(cred_path)
                        firebase_admin.initialize_app(cred)
                    else:
                        firebase_admin.initialize_app()

                db = fs.client()
                db.collection('users').document(user_uid).update(update_payload)
                logger.info(f"Firestore updated for uid={user_uid}: {update_payload}")
            except Exception as fe:
                logger.error(f"Firestore update error: {fe}")

        return jsonify({'status': 'ok', 'event': event_name}), 200

    except Exception as e:
        logger.error(f"Webhook processing error: {e}")
        return jsonify({'error': str(e)[:150]}), 500


@app.route('/health')
def health():
    return jsonify({
        'status': 'online',
        'ai': 'online' if (model or GROQ_API_KEY) else 'offline',
        'ai_primary': 'gemini-2.5-flash' if model else 'offline',
        'ai_fallback': 'groq:llama-3.3-70b-versatile' if GROQ_API_KEY else 'none',
        'web_search': 'online' if TAVILY_API_KEY else 'offline',
        'tiers': ['free', 'premium', 'business'],
        'version': '2.2'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
