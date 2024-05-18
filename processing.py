import re
import numpy as np
import torch
import torch.nn.functional as F
from deep_translator import GoogleTranslator
from youtube_transcript_api import YouTubeTranscriptApi

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

indices_to_labels = { 0:'e', 1:'n', 2:'a', 3:'c', 4:'o' }
labels_to_indices = { 'e':0, 'n':1, 'a':2, 'c':3, 'o':4 }

label_names = ['Extroversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']

def evalText(fModel, nlp, word_data, text):
    text = GoogleTranslator(source='auto', target='en').translate(text)
    fModel.eval()
    fModel.to(device)
    with torch.no_grad():
        input = torch.zeros((200, 300), device=device, dtype=torch.float32)
        mrc = np.zeros(27)
        words = []
        for word in nlp(text)[:200]:
            if word.lemma_ in word_data:
                mrc += np.array(word_data[word.lemma_])
            if word.has_vector and word.is_alpha:
                words.append(word.vector)

        mrc = torch.tensor(np.array(mrc), device=device, dtype=torch.float32).unsqueeze(0)
        mrc = mrc / 200.0
        mrc = torch.nan_to_num(mrc, nan=0)

        words = torch.tensor(np.array(words), device=device, dtype=torch.float32)
        input[:words.shape[0]] = words
        input = input.unsqueeze(0)

        outputs = F.sigmoid(fModel(input, mrc)).squeeze()
        return outputs
    

def evalLongText(fModel, nlp, word_data, text):
    splited_texts = []
    MAX_WORDS_PER_SPLIT = 200

    sentences = re.split(r'(?<=[.!?])\s+', text)

    current_split = ""
    current_word_count = 0
    for sentence in sentences:
        str_sen =  sentence.strip()
        if str_sen != "" and any(char.isalnum() for char in sentence):
            words_in_sentence = len(str_sen.split())

            if current_word_count + words_in_sentence <= MAX_WORDS_PER_SPLIT:
                current_split += str_sen + " "
                current_word_count += words_in_sentence
            elif current_split.strip() != "":
                splited_texts.append(current_split.strip())
                current_split = str_sen + " "
                current_word_count = words_in_sentence
    if current_split.strip() != "":
        splited_texts.append(current_split.strip())
    err_c = 0
    results = np.zeros(5)
    for t in splited_texts:
        try:
            results += np.array(evalText(fModel, nlp, word_data, t))
        except Exception as e:
            print(e)
            err_c += 1
    
    if err_c == len(splited_texts):
        raise RuntimeError()
    
    results /= len(splited_texts) - err_c

    return results.astype(float).tolist()


def split_text_into_chunks(text, chunk_size=200):
    words = text.split()
    num_chunks = len(words) // chunk_size + (1 if len(words) % chunk_size != 0 else 0)
    chunks = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(words))
        chunk = ' '.join(words[start_idx:end_idx])
        chunks.append(chunk)
    return chunks

def extract_youtube_video_id(url):
    pattern = r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]+)'

    match = re.search(pattern, url)
    
    if match:
        return match.group(1)
    else:
        raise ValueError("Invalid YouTube video URL")

def evalYT(fModel, nlp, word_data, url):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(extract_youtube_video_id(url))
    except Exception as e:
        code = list(YouTubeTranscriptApi.list_transcripts(extract_youtube_video_id(url)))[0].language_code
        transcript = YouTubeTranscriptApi.get_transcript(extract_youtube_video_id(url), languages=[code])

    yt_texts = split_text_into_chunks(" ".join([item['text'] for item in transcript]))
    err_c = 0
    results = np.zeros(5)
    for t in yt_texts:
        try:
            results += np.array(evalText(fModel, nlp, word_data, t))
        except Exception as e:
            print(e)
            err_c += 1
    
    if err_c == len(yt_texts):
        raise RuntimeError()
    results /= len(yt_texts) - err_c
    return results.astype(float).tolist()

