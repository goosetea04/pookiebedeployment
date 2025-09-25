# core/firebase_client.py
import firebase_admin
from firebase_admin import credentials, firestore
from .config import settings

def _build_cred_dict() -> dict:
    private_key = settings.FB_PRIVATE_KEY or ""
    # Fix escaped newlines
    private_key = private_key.replace("\\n", "\n")
    return {
        "type": settings.FB_ACCOUNT_TYPE,
        "project_id": settings.FB_PROJECT_ID,
        "private_key_id": settings.FB_PRIVATE_KEY_ID,
        "private_key": private_key,
        "client_email": settings.FB_CLIENT_EMAIL,
        "client_id": settings.FB_CLIENT_ID,
        "auth_uri": settings.FB_AUTH_URI,
        "token_uri": settings.FB_TOKEN_URI,
        "auth_provider_x509_cert_url": settings.FB_AUTH_CERT_URL,
        "client_x509_cert_url": settings.FB_CERT_URL,
        "universe_domain": "googleapis.com",
    }

def init_firebase():
    if not firebase_admin._apps:  # avoid double init in reload
        cred = credentials.Certificate(_build_cred_dict())
        firebase_admin.initialize_app(cred)

# Create a singleton Firestore client
def get_db():
    init_firebase()
    return firestore.client()
