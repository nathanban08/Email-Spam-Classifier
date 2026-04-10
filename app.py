import base64
import os
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, redirect, request, send_from_directory
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
APP_DIR = Path(__file__).resolve().parent
CREDENTIALS_PATH = APP_DIR / 'credentials.json'
TOKEN_PATH = APP_DIR / 'token.json'
DATA_PATH = APP_DIR / 'spam_Emails_data.csv'

app = Flask(__name__)

def load_credentials():
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
        if creds and creds.valid:
            return creds
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            save_credentials(creds)
            return creds
    return None


def save_credentials(creds):
    with open(TOKEN_PATH, 'w') as token_file:
        token_file.write(creds.to_json())


def create_gmail_service(creds):
    return build('gmail', 'v1', credentials=creds)


def clean_text(text):
    return text if isinstance(text, str) else ''


def get_email_body(msg):
    payload = msg.get('payload', {})
    data = payload.get('body', {}).get('data')
    if data:
        return base64.urlsafe_b64decode(data).decode(errors='ignore')

    body_parts = []

    def walk(parts):
        for part in parts:
            mime_type = part.get('mimeType', '')
            if mime_type == 'text/plain' and part.get('body', {}).get('data'):
                decoded = base64.urlsafe_b64decode(part['body']['data']).decode(errors='ignore')
                body_parts.append(decoded)
            if part.get('parts'):
                walk(part['parts'])

    walk(payload.get('parts', []))
    return '\n'.join(body_parts)


def get_email_header(msg, name):
    headers = msg.get('payload', {}).get('headers', [])
    for header in headers:
        if header.get('name', '').lower() == name.lower():
            return header.get('value', '')
    return ''


def build_model():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=['text'])
    df = df[df['label'].isin(['Ham', 'Spam'])]

    x_train_text, _, y_train, _ = train_test_split(
        df['text'],
        df['label'].map({'Ham': 0, 'Spam': 1}),
        test_size=0.2,
        random_state=42,
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=10,
        max_df=0.95,
        stop_words='english',
    )
    x_train = vectorizer.fit_transform(x_train_text)

    model = LogisticRegression(max_iter=10000)
    model.fit(x_train, y_train)

    return vectorizer, model


vectorizer, model = build_model()


@app.route('/')
def index():
    return send_from_directory(APP_DIR, 'Website.html')


@app.route('/connect')
def connect():
    flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_PATH), SCOPES)
    creds = flow.run_local_server(port=0)
    save_credentials(creds)
    return redirect('/')


@app.route('/api/emails')
def api_emails():
    creds = load_credentials()
    if not creds:
        return jsonify({'error': 'not_authorized', 'message': 'Please connect Gmail first.'}), 401

    service = create_gmail_service(creds)
    results = service.users().messages().list(userId='me', maxResults=10).execute()
    messages = results.get('messages', [])

    if not messages:
        return jsonify({'emails': []})

    email_items = []
    bodies = []

    for msg_info in messages:
        msg = service.users().messages().get(userId='me', id=msg_info['id'], format='full').execute()
        body = get_email_body(msg) or ''
        bodies.append(clean_text(body))
        email_items.append({
            'id': msg_info['id'],
            'subject': get_email_header(msg, 'Subject') or '(No subject)',
            'sender': get_email_header(msg, 'From') or '(Unknown sender)',
            'preview': ' '.join(body.split()[:20]) + ('...' if len(body.split()) > 20 else ''),
        })

    x_input = vectorizer.transform(bodies)
    predictions = model.predict(x_input)
    probabilities = model.predict_proba(x_input)

    for index, email in enumerate(email_items):
        spam_score = float(probabilities[index][1])
        email['prediction'] = 'Spam' if predictions[index] == 1 else 'Real'
        email['confidence'] = round(spam_score if predictions[index] == 1 else (1 - spam_score), 3)

    return jsonify({'emails': email_items})


@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(force=True, silent=True) or {}
    text = clean_text(data.get('text', ''))

    if not text:
        return jsonify({'error': 'empty_text', 'message': 'Please send a non-empty email text for prediction.'}), 400

    x_input = vectorizer.transform([text])
    prediction = int(model.predict(x_input)[0])
    probability = float(model.predict_proba(x_input)[0][1])

    return jsonify({
        'prediction': 'Spam' if prediction == 1 else 'Real',
        'confidence': round(probability if prediction == 1 else 1 - probability, 3),
    })


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
