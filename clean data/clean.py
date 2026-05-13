import argparse
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import emoji
from nltk.corpus import stopwords
import contractions
import ftfy
from nltk import pos_tag
import pandas as pd
import importlib.resources as pkg_resources
from symspellpy import SymSpell, Verbosity
from dotenv import load_dotenv
import os

# load_dotenv(".env")

# # Get the API key
# API_KEY = os.getenv("API_KEY")

# Download required NLTK data
def ensure_nltk_data():
    resources = {
        'punkt': 'tokenizers/punkt',
        'wordnet': 'corpora/wordnet',
        'omw-1.4': 'corpora/omw-1.4',
        'averaged_perceptron_tagger_eng': 'taggers/averaged_perceptron_tagger_eng',
        'stopwords': 'corpora/stopwords'
    }
    for r, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(r)

ensure_nltk_data()
stop_words = set(stopwords.words('english'))
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
with pkg_resources.path("symspellpy", "frequency_dictionary_en_82_765.txt") as dictionary_path:
    sym_spell.load_dictionary(str(dictionary_path), term_index=0, count_index=1)


important_words = {
    "not", "no", "never",
    "but", "however",
    "very", "really",
    "more", "less"
}

lemmatizer = WordNetLemmatizer()


def extract_car_model(text):
    if not isinstance(text, str):
        return str(text)
    match = re.search(
        r'\b(Tesla|Nissan|Skoda|Rivian|Chevy|Ford|Lucid|Volkswagen|Toyota|Sentra|BMW|ID Buzz|Hyundai|Mustang|Kia|Volvo|Audi|Lexus|Jeep|Porsche|Mazda|Mercedes|Range Rover|Renault)\b',
        text, re.IGNORECASE)
    return match.group(0) if match else "Other"

def extract_topic(text):
    if not isinstance(text, str):
        return str(text)
    topics = ['battery', 'range', 'price', 'charging', 'autopilot', 'design', 'performance', 'reliability', 'fuel']
    for t in topics:
        if re.search(rf'\b{t}\b', text, re.IGNORECASE):
            return t
    return "Other"
#remove everything but !, - and ?, also expands contractions.
def clean_text_regex(text):
    text = ftfy.fix_text(text)
    text = contractions.fix(text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[:_@#$%^&*()\"\'<>/\\|~`+=;,.!?]", " ", text)
    return text

#convert emojis and emotional punctuations to text
def replace_emojis_and_punctuations(text):
    text = emoji.demojize(text)
    text = re.sub(r":([a-zA-Z0-9_+-]+):", r" \1_emoji ", text)
    text = re.sub(r"([!?]{2,})", " exclamation_strong ", text)
    text = re.sub(r"(?<!\!)\!(?!\!)", " exclamation ", text)
    text = re.sub(r"\?", " ", text)
    text = re.sub(r"([*\-=_]{2,})", " ", text)
    text = re.sub(r"[^\w\s\-]", "", text)
    return text


def drop_emojis_and_punctuations(text):
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

def to_lowercase(tokens):
    return [w.lower() for w in tokens]

def remove_stopwords(tokens):
    return [
        t for t in tokens
        if t not in stop_words or t in important_words
    ]

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize_words(tokens):
    tagged_tokens = pos_tag(tokens)
    return [lemmatizer.lemmatize(t.lower(), get_wordnet_pos(p)) for t, p in tagged_tokens]

def correct_spellings(tokens):
    corrected_tokens = []
    for token in tokens:
        if token.endswith("_emoji") or token.startswith("exclamation"):
            corrected_tokens.append(token)
        else:
            suggestions = sym_spell.lookup(token, Verbosity.CLOSEST, max_edit_distance=2)
            corrected_tokens.append(suggestions[0].term if suggestions else token)
    return corrected_tokens

def clean_text(text, remove_special=False, clean_emojis=False, drop_emojis = False, lowercase=False, correct_spelling= False, lemmatize=False, remove_stops=False):
    if not isinstance(text, str):
        return str(text)
    if remove_special:
        text = clean_text_regex(text)
    if clean_emojis:
        text = replace_emojis_and_punctuations(text)
    if drop_emojis:
        text = drop_emojis_and_punctuations(text)
    tokens = word_tokenize(text)
    if lowercase:
        tokens = to_lowercase(tokens)
    if remove_stops:
        tokens = remove_stopwords(tokens)
    if correct_spelling:
        tokens = correct_spellings(tokens)
    if lemmatize:
        tokens = lemmatize_words(tokens)
    return " ".join(tokens)

# DEFAULT_OUTPUT_FILE = r"../data/raw/cleaned_youtube_comments.csv"

# # --- Argument parsing ---
# parser = argparse.ArgumentParser(description="YouTube Comments Collector & Cleaner")
# parser.add_argument("--file_csv_path", type=str, required=True, help="csv file of raw comments")
# parser.add_argument("--remove_special", action="store_true", help="Remove special characters from comments")
# parser.add_argument("--clean_emojis", action="store_true", help="Change emojis and punctuations to their text description")
# parser.add_argument("--drop_emojis", action="store_true", help="drop emojis and punctuations")
# parser.add_argument("--lowercase", action="store_true", help="Convert comment text to lowercase")
# parser.add_argument("--correct_spelling", action="store_true", help="Correct spelling of vocabulary")
# parser.add_argument("--lemmatize", action="store_true", help="Apply lemmatization to comments")
# parser.add_argument("--remove_stopwords", action="store_true", help="Remove stopwords from comments")
# parser.add_argument("--extract_tags", action="store_true", help="Extract car model and topic tags")
# parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_FILE, help="Output CSV file name")

# args = parser.parse_args()

# df = pd.read_csv(args.file_csv_path)
# df['clean_comment'] = df['comment'].apply(lambda x: clean_text(
#     x,
#     remove_special=args.remove_special,
#     clean_emojis=args.clean_emojis,
#     drop_emojis=args.drop_emojis,
#     lowercase=args.lowercase,
#     correct_spelling=args.correct_spelling,
#     lemmatize=args.lemmatize,
#     remove_stops=args.remove_stopwords
# ))

# if args.extract_tags:
#     df['car_model'] = df['comment'].apply(extract_car_model)
#     df['topic'] = df['comment'].apply(extract_topic)
    
# output_dir = os.path.dirname(args.output)
# if output_dir and not os.path.exists(output_dir):
#     os.makedirs(output_dir, exist_ok=True)

# df.to_csv(args.output, index=False, encoding="utf-8")
# print(f"Collected {len(df)} comments. Saved to {args.output}")


# --- Collect YouTube comments ---
# rows = []

# # Get video title
# video_resp = requests.get(
#     "https://www.googleapis.com/youtube/v3/videos",
#     params={"part": "snippet", "id": args.video_id, "key": API_KEY}
# ).json()

# video_title = video_resp.get("items", [{}])[0].get("snippet", {}).get("title", "Unknown Title")

# # Get comments
# url = "https://www.googleapis.com/youtube/v3/commentThreads"
# params = {"part": "snippet", "videoId": args.video_id, "maxResults": 100, "key": API_KEY}

# response = requests.get(url, params=params)
# data = response.json()

# for item in data.get("items", []):
#     comment = item["snippet"]["topLevelComment"]["snippet"]
#     original_text = comment["textOriginal"]

#     # Apply cleaning
#     cleaned_text = clean_text(
#         original_text,
#         remove_special=args.remove_special,
#         lowercase=args.lowercase,
#         lemmatize=args.lemmatize,
#         remove_stops=args.remove_stopwords
#     )

#     row = {
#         "video_id": args.video_id,
#         "video_title": video_title,
#         "author": comment["authorDisplayName"],
#         "comment": original_text,
#         "cleaned_comment": cleaned_text,
#         "likes": comment["likeCount"],
#         "published_at": comment["publishedAt"]
#     }

#     # Extract tags if requested
#     if args.extract_tags:
#         row["car_model"] = extract_car_model(original_text)
#         row["topic"] = extract_topic(original_text)

#     rows.append(row)


# # Save to CSV
# df = pd.DataFrame(rows)
