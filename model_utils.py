import os
from transformers import AutoProcessor, AutoModelForCausalLM, pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def generate_captions(frames, device, is_keyframe=None):
    """
    Generate a complete journey summary using the frames.
    This now returns a single narrative summary instead of per-frame captions.
    """
    processor = AutoProcessor.from_pretrained("microsoft/git-base-vatex")
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-vatex").to(device)

    # select keyframes
    if is_keyframe:
        video = [frame for i, frame in enumerate(frames) if is_keyframe[i]]
    else:
        video = frames

    # process video frames
    pixel_values = processor(images=video, return_tensors="pt").pixel_values.to(device)

    # generate caption
    with torch.no_grad():
        generated_ids = model.generate(pixel_values=pixel_values, max_length=50)

    # decode caption
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return [caption] * len(frames), caption

def generate_long_summary(events, landmarks, captions):
    """
    Generate a long-form journey summary (250-400 words) with improved legibility.
    Cleans and filters captions, formats the prompt in natural language, and ensures prompt fits model context.
    """
    repo = os.getenv("JOURNEY_SUMMARY_MODEL", "google/pegasus-cnn_dailymail")
    cache_dir = os.path.join("models", repo.replace("/", "_"))
    tokenizer = AutoTokenizer.from_pretrained(repo, cache_dir=cache_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(repo, cache_dir=cache_dir)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=-1)

    cleaned_captions = [c for c in [clean_caption(c) for c in captions] if c]
    if not cleaned_captions:
        return "Could not generate a summary due to a lack of valid captions."

    cleaned_landmarks = clean_landmarks(landmarks)

    prompt = format_summary_prompt(events, cleaned_landmarks[:10], cleaned_captions[:10])

    # GPT-style: sliding window, dynamic chunking, hierarchical summarization
    tokenizer_max = getattr(tokenizer, 'model_max_length', 1024)
    model_max = getattr(model.config, 'max_position_embeddings', 1024)
    max_input_length = min(tokenizer_max, model_max)
    overlap_tokens = int(max_input_length * 0.2)  # 20% overlap

    def summarize_prompt(p):
        output = summarizer(p, max_length=256, min_length=100, do_sample=False)
        return output[0]['summary_text']

    def chunk_captions_by_tokens(captions, max_tokens, overlap):
        chunks = []
        i = 0
        while i < len(captions):
            chunk = []
            token_count = 0
            j = i
            while j < len(captions):
                test_chunk = chunk + [captions[j]]
                prompt = format_summary_prompt(events[:10], cleaned_landmarks[:10], test_chunk)
                tokens = tokenizer.encode(prompt)
                if token_count + len(tokens) > max_tokens:
                    break
                chunk.append(captions[j])
                token_count = len(tokens)
                j += 1
            if not chunk:
                # Fallback: force at least one caption per chunk
                chunk = [captions[i]]
                j = i + 1
            chunks.append((i, j, chunk))
            # Sliding window: overlap
            i = j - overlap if (j - overlap) > i else j
        return chunks

    def hierarchical_summarize(captions):
        # Step 1: chunk and summarize
        chunks = chunk_captions_by_tokens(captions, max_input_length, overlap=2)
        chunk_summaries = []
        for start, end, chunk in chunks:
            batch_prompt = format_summary_prompt(events[:10], cleaned_landmarks[:10], chunk)
            try:
                chunk_summary = summarize_prompt(batch_prompt)
                chunk_summaries.append(chunk_summary)
            except Exception as e:
                print(f"[Summary] Error during chunk summarization: {e}")
                continue
        # Step 2: if needed, recursively summarize
        combined = "\n".join(chunk_summaries)
        combined_tokens = tokenizer.encode(combined)
        if len(combined_tokens) > max_input_length and len(chunk_summaries) > 1:
            print("[Summary] Recursively summarizing combined output...")
            return hierarchical_summarize(chunk_summaries)
        else:
            return combined

    prompt_tokens = tokenizer.encode(prompt)
    if len(prompt_tokens) <= max_input_length:
        try:
            return summarize_prompt(prompt)
        except Exception as e:
            print(f"[Summary] Error during summarization: {e}")
            return "Could not generate a summary due to an error."
    else:
        return hierarchical_summarize(cleaned_captions)

def clean_caption(caption):
    """Clean a single caption string."""
    if not caption or caption in ["error", "no_change"]:
        return None
    caption = re.sub(r'[^\w\s.,!?-]', '', caption)
    caption = re.sub(r'(\b\w+\b)(?:\s+\1\b)+', r'\1', caption)  # remove repeated words
    caption = caption.strip()
    if len(caption) < 8 or len(caption.split()) < 2:
        return None
    if len(set(caption.lower().split())) == 1:
        return None
    return caption

def clean_landmarks(landmarks):
    """Clean and deduplicate a list of landmark strings."""
    cleaned_landmarks = []
    seen_lm = set()
    for l in landmarks:
        if not l or l in ["error", "none", "no_change"]:
            continue
        l = l.strip()
        if l and l not in seen_lm:
            cleaned_landmarks.append(l)
            seen_lm.add(l)
    return cleaned_landmarks

def format_summary_prompt(events, landmarks, captions):
    """Format the prompt for the summarization model."""
    prompt = (
        "You are an expert journey summarizer. Write a detailed, engaging, and coherent long-form summary (250-400 words) describing the journey. "
        "Include navigation, notable landmarks, road/traffic/weather conditions, and overall impressions.\n\n"
        "Driving events: " + ', '.join(events[:10]) + ".\n"
        "Notable landmarks: " + '; '.join(landmarks) + ".\n"
        "Scene highlights:\n- " + '\n- '.join(captions) + "\n\nSummary:"
    )
    return prompt
