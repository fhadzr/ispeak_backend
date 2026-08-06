# test_supabase.py
from dotenv import load_dotenv
load_dotenv()  # ← tambah ini, load .env dulu sebelum os.getenv

from supabase import create_client
import os

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

print(f"URL: {url}")
print(f"KEY: {key[:20]}..." if key else "KEY: None")

client = create_client(url, key)
print(client.table("documents").select("id").limit(1).execute())