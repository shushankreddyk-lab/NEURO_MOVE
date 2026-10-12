import os
import requests
import time

DATA_DIR = r"D:\eeg-minirocket-project\data\Kaya_Finger_Movements"
os.makedirs(DATA_DIR, exist_ok=True)

COLLECTION_ID = 3917698
print(f"Fetching metadata for Figshare Collection {COLLECTION_ID}...")

# 1. Get all articles (datasets) in the collection
res = requests.get(f"https://api.figshare.com/v2/collections/{COLLECTION_ID}/articles")
articles = res.json()

if not isinstance(articles, list):
    print("Error fetching collection:", articles)
    exit(1)

print(f"Found {len(articles)} datasets in the collection. Starting download...")

for article in articles:
    article_id = article['id']
    # 2. Get files for each article
    file_res = requests.get(f"https://api.figshare.com/v2/articles/{article_id}/files")
    files = file_res.json()
    
    for f in files:
        filename = f['name']
        download_url = f['download_url']
        file_path = os.path.join(DATA_DIR, filename)
        
        # Skip if already downloaded and not empty
        if os.path.exists(file_path) and os.path.getsize(file_path) > 1000000:
            print(f"Skipping {filename}, already downloaded.")
            continue
            
        print(f"Downloading {filename}...")
        
        # 3. Stream download the file with retries
        success = False
        for attempt in range(5):
            try:
                with requests.get(download_url, stream=True, timeout=30) as r:
                    r.raise_for_status()
                    with open(file_path, 'wb') as out_file:
                        for chunk in r.iter_content(chunk_size=1024*1024): 
                            if chunk:
                                out_file.write(chunk)
                success = True
                break
            except Exception as e:
                print(f"Connection error on attempt {attempt+1}: {e}. Retrying...")
                time.sleep(3)
                
        if not success:
            print(f"Failed to download {filename} after 5 attempts.")
                    
print("\nAll Kaya Finger Movement datasets processed!")
