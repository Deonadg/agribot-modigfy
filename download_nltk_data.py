import nltk

def download_nltk_resources():
    print("Downloading NLTK resources...")
    resources = ["punkt", "punkt_tab", "wordnet", "omw-1.4"]
    
    for resource in resources:
        try:
            print(f"Downloading {resource}...")
            nltk.download(resource)
            print(f"Successfully downloaded {resource}")
        except Exception as e:
            print(f"Error downloading {resource}: {str(e)}")
    
    print("Download complete!")

if __name__ == "__main__":
    download_nltk_resources()
