# Email Spam Detector Web App

Flask web app that connects to Gmail via OAuth, fetches recent emails, and uses a trained spam classifier to label messages as spam or real. Includes manual text prediction, Gmail inbox analysis, and a simple UI for live email filtering and confidence scoring.

This project includes a local Flask web app that connects to the Gmail API, fetches recent emails, and runs them through the spam detector trained from `spam_Emails_data.csv`.

## Files added
- `app.py` — Flask backend and Gmail API integration
- `Website.html` — interactive front-end page served by `app.py`
- `requirements.txt` — Python dependencies

## Run locally
1. Install dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```
2. Run the app:
   ```bash
   python3 app.py
   ```
3. Open the website in your browser:
   ```
   http://127.0.0.1:5000
   ```
4. Click **Connect Gmail** and authenticate with your Gmail account.
5. Use **Fetch Latest Emails** to load messages and view spam predictions.

## Notes
- OAuth credentials are loaded from `credentials.json`.
- Fresh Google authentication data is stored in `token.json`.
- Manual text prediction is available from the page.

