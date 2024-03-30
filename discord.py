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
from transformers import RagSequenceForGeneration

app = Flask(__name__)

def fetch_and_extract_articles(query, page_number=1, results_per_page=10):
    articles = fetch_and_extract_articles(query, page_number, results_per_page)

    for article in articles:
        if article['link'].endswith('.pdf'):
            response = requests.get(article['link'])
            response.raise_for_status()

            pdf_file = BytesIO(response.content)

            pdf_reader = PyPDF2.PdfReader(pdf_file)

            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text()

            article['text'] = text

    return articles

def get_json_response(url, params):
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

intents = nextcord.Intents.default()
intents.typing = False
intents.presences = False
intents.members = True  # Enable the Server Members Intent

bot = commands.Bot(command_prefix='!', intents=intents)

@app.route('/')
def home():
    return "Hello, World!"

def extract_text_from_pdf(pdf_file):
    with open(pdf_file, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
        return text

def get_events_from_pdf(date, month, pdf_files):
    events = []
    date_str = date.strftime("%d")
    month_str = date.strftime("%m")
    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)
        if date_str in text and month_str in text:
            events.append({
                'date': date.strftime("%d-%m-%Y"),
                'text': text
            })
    return events

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

@app.route('/process', methods=['POST'])
def process():
    date_str = request.form.get('date')
    pdf_files = request.form.getlist('pdf_files')

    try:
        date = datetime.datetime.strptime(date_str, "%d-%m-%Y").date()
        month = date.month
        events = get_events_from_pdf(date, month, pdf_files)
        interpreted_events = []
        for event in events:
            historical_context = interpret_event(event['text'])
            interpreted_events.append({
                'date': event['date'],
                'text': event['text'],
                'historical_context': historical_context
            })
    except ValueError:
        return "Invalid date format. Please provide the date in DD-MM-YYYY format."

    return render_template('events.html', events=interpreted_events)

def get_news_articles(date):
    """
    Fetch news articles for a given date using Google CSE.
    """
    # Define the URL for your request
    url = "https://www.googleapis.com/customsearch/v1"

    # Format the date as a string
    date_str = date.strftime("%d-%m-%Y")

    # Define the parameters for your request
    params = {
        "q": f"date:{date_str}",
        "key": 'AIzaSyABg8fDE8wgTVDc1xK6ZYQyzPUMQjrRdu4',
        "cx": 'e4adf04a2fc0c49fd',
    }

    try:
        # Send a GET request to the URL with the parameters
        response = requests.get(url, params=params)

        # Convert the response to JSON
        response_json = response.json()

        # Check if the 'items' key is in the response
        if "items" in response_json:
            articles = response_json["items"]
        else:
            print("No 'items' key in response")
            articles = []

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while making the request: {e}")
        articles = []

    return articles

def interpret_article(article_url):
    """
    Interpret the given news article and relate it to historical context.
    """
    article = Article(article_url)
    article.download()
    article.parse()
    article.nlp()

    # Extract relevant information from the article
    title = article.title
    summary = article.summary
    text = article.text

    # Initialize RAG model and tokenizer
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
    model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq")

    # Generate question for RAG model
    question = "What is the historical context of this news article?"

    # Check if text is not None and not an empty string
    if text is not None and text.strip():
        # Encode inputs
        inputs = tokenizer([question], [text], return_tensors="pt", padding=True, truncation=True)

        # Check if inputs is not None
        if inputs is not None:
            # Generate outputs
            outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

            # Decode outputs
            historical_context = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        else:
            historical_context = "Unable to generate historical context for this article."
    else:
        historical_context = "Unable to generate historical context due to empty article text."

    return format_article(title, summary, text, historical_context)
def format_article(title, summary, text, historical_context):
    return {
        "title": title,
        "summary": summary,
        "text": text,
        "historical_context": historical_context,
    }

def upsc_current_affairs_interpreter(date):
    """
    Fetch news articles for the given date and interpret them with historical context.
    """
    articles = get_news_articles(date)
    interpreted_articles = []

    for article in articles:
        url = article["link"]
        interpreted_article = interpret_article(url)
        interpreted_articles.append(interpreted_article)

    return interpreted_articles

@app.route('/news', methods=['POST'])
def news():
    date = request.form.get('date')

    try:
        target_date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
        interpreted_articles = upsc_current_affairs_interpreter(target_date)
    except ValueError:
        return "Invalid date format. Please provide the date in YYYY-MM-DD format."

    return render_template('news.html', articles=interpreted_articles)

@bot.event
async def on_ready():
    print(f'{bot.user.name} has connected to Discord!')

@bot.event
async def on_member_join(member):
    welcome_channel = nextcord.utils.get(member.guild.text_channels, name="welcome")
    if welcome_channel:
        await welcome_channel.send(f"Welcome {member.mention} to the UPSC Current Affairs Interpreter Server! Use the `!current_affairs` command to get daily news updates and their historical context.")

@bot.command(name='current_affairs')
async def current_affairs(ctx, date: str):
    try:
        # Try to parse the date in the format DD/MM/YY
        try:
            target_date = datetime.datetime.strptime(date, "%d/%m/%y").date()
        except ValueError:
            # If that fails, try to parse the date in the format DD/MM
            target_date = datetime.datetime.strptime(date, "%d/%m").date()
            # If the year is not provided, assume the current year
            target_date = target_date.replace(year=datetime.datetime.now().year)

        interpreted_articles = upsc_current_affairs_interpreter(target_date)

        for article in interpreted_articles:
            embed = nextcord.Embed(title=article['title'], description=article['summary'])
            embed.add_field(name="Text", value=article['text'], inline=False)
            embed.add_field(name="Historical Context", value=article['historical_context'], inline=False)
            await ctx.send(embed=embed)

    except ValueError:
        await ctx.send("Invalid date format. Please provide the date in DD/MM/YY or DD/MM format.")

def run_bot():
    BOT_TOKEN = os.getenv('BOT_TOKEN')
    bot.run(BOT_TOKEN)

def run_app():
    app.run()

if __name__ == "__main__":
    # Create thread for app
    app_thread = threading.Thread(target=run_app)

    # Start app thread
    app_thread.start()

    # Run bot in main thread
    run_bot()