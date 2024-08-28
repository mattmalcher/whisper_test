from pydub import AudioSegment
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import spacy

# Load spaCy model (make sure you have a model installed, e.g., 'en_core_web_sm')
nlp = spacy.load('en_core_web_sm', disable=['ner', 'pos', 'tagger', 'parser'])
# We disable unnecessary components for faster processing

def split_channels(audio_path):
    audio = AudioSegment.from_wav(audio_path)
    audio = audio.set_frame_rate(16000)
    left_channel = audio.split_to_mono()[0]
    right_channel = audio.split_to_mono()[1]
    left_channel = np.array(left_channel.get_array_of_samples())
    right_channel = np.array(right_channel.get_array_of_samples())
    left_channel = left_channel / np.max(np.abs(left_channel))
    right_channel = right_channel / np.max(np.abs(right_channel))
    return left_channel, right_channel, audio.frame_rate

def transcribe_audio_with_timestamps(audio, sr, processor, model):
    input_features = processor(audio, sampling_rate=sr, return_tensors="pt").input_features
    
    # Generate tokens with timestamps
    predicted_ids = model.generate(input_features, return_timestamps=True)
    
    # Decode tokens to text with timestamps
    transcription = processor.batch_decode(predicted_ids, output_word_offsets=True)
    
    # Extract words and their timestamps
    words_and_timestamps = []
    for segment in transcription[0]['chunks']:
        for word in segment['words']:
            words_and_timestamps.append({
                'word': word['text'],
                'start': word['start'],
                'end': word['end']
            })
    
    return words_and_timestamps

def interleave_transcriptions_with_timestamps(caller_transcription, service_transcription):
    # Combine and sort all words by their start time
    all_words = caller_transcription + service_transcription
    all_words.sort(key=lambda x: x['start'])
    
    interleaved = []
    current_speaker = None
    current_sentence = ""
    sentence_start_time = 0
    
    for word in all_words:
        speaker = "[Caller]" if word in caller_transcription else "[Customer Service]"
        
        if speaker != current_speaker:
            if current_sentence:
                doc = nlp(current_sentence.strip())
                for sent in doc.sents:
                    interleaved.append(f"{current_speaker} [{sentence_start_time:.2f}s]: {sent.text}")
            current_speaker = speaker
            current_sentence = ""
            sentence_start_time = word['start']
        
        current_sentence += word['word'] + " "
        
        # Use spaCy for sentence detection
        doc = nlp(current_sentence.strip())
        sentences = list(doc.sents)
        if len(sentences) > 1:
            for sent in sentences[:-1]:
                interleaved.append(f"{current_speaker} [{sentence_start_time:.2f}s]: {sent.text}")
            current_sentence = sentences[-1].text + " "
            sentence_start_time = word['start']
    
    # Add any remaining sentence
    if current_sentence:
        doc = nlp(current_sentence.strip())
        for sent in doc.sents:
            interleaved.append(f"{current_speaker} [{sentence_start_time:.2f}s]: {sent.text}")
    
    return '\n'.join(interleaved)

def main(audio_path):
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    processor.tokenizer.predict_timestamps = True
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

    left_channel, right_channel, sr = split_channels(audio_path)

    caller_transcription = transcribe_audio_with_timestamps(left_channel, sr, processor, model)
    customer_service_transcription = transcribe_audio_with_timestamps(right_channel, sr, processor, model)

    interleaved_transcription = interleave_transcriptions_with_timestamps(caller_transcription, customer_service_transcription)

    print("Interleaved Transcription:")
    print(interleaved_transcription)

    # Optionally, save to a file
    with open("interleaved_transcription.txt", "w") as f:
        f.write(interleaved_transcription)

if __name__ == "__main__":
    audio_path = "example_1.wav"  # Replace with your audio file path
    main(audio_path)