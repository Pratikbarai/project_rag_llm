from flask import Flask, request, render_template
import datetime
import PyPDF2
import requests
from newspaper import Article
import nextcord
from nextcord.ext import commands
from dotenv import load_dotenv
import os
from transformers import RagTokenizer, RagSequenceForGeneration
import threading
from io import BytesIO

app = Flask(__name__)

def fetch_and_extract_articles(query, page_number=1, results_per_page=10):
    articles = get_json_response(f"https://www.googleapis.com/customsearch/v1", {
        "key": "YOUR_GOOGLE_CSE_API_KEY",
        "cx": "YOUR_GOOGLE_CSE_ID",
        "q": query,
        "num": results_per_page,
        "start": (page_number - 1) * results_per_page,
    })

    for article in articles["items"]:
        if article['link'].endswith('.pdf'):
            response = requests.get(article['link'])
            response.raise_for_status()

            pdf_file = BytesIO(response.content)

            pdf_reader = PyPDF2.PdfFileReader(pdf_file)

            text = ''
            for page in pdf_reader.pages:
                text += page.extractText()

            article['text'] = text

    return articles

def get_json_response(url, params):
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

load_dotenv()
BOT_TOKEN = os.getenv('BOT_TOKEN')

intents = nextcord.Intents.default()
intents.typing = False
intents.presences = False
intents.members = True

bot = commands.Bot(command_prefix='!', intents=intents)

@app.route('/')
def home():
    return "Hello, World!"

@app.route('/process', methods=['POST'])
def process():
    date_str = request.form.get('date')
    pdf_files = request.form.getlist('pdf_files')

    try:
        date = datetime.datetime.strptime(date_str, "%d-%m-%Y").date()
        events = get_events_from_pdf(date, pdf_files)
    except ValueError:
        return "Invalid date format. Please provide the date in DD-MM-YYYY format."

    return render_template('events.html', events=events)

def get_events_from_pdf(date, pdf_files):
    events = []
    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)
        if date.strftime("%d-%m-%Y") in text:
            events.append({
                'date': date.strftime("%d-%m-%Y"),
                'text': text
            })
    return events

def extract_text_from_pdf(pdf_file):
    with open(pdf_file, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extractText()
        return text

def interpret_event(event_text):
    # Initialize RAG model and tokenizer
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
    model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq")

    # Generate question for RAG model
    question = "What is the historical context of this event?"

    # Encode inputs
    inputs = tokenizer([question], [event_text], return_tensors="pt", padding=True, truncation=True)

    # Generate outputs
    outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

    # Decode outputs
    historical_context = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    return historical_context

@app.route('/interpret_event',