import torch
import numpy as np
import whisper_timestamped as whisper
import spacy

from custom_load_audio import load_audio

# Load spaCy model (make sure you have a model installed, e.g., 'en_core_web_sm')
nlp = spacy.load("en_core_web_sm", disable=["ner", "pos", "tagger", "parser"])
nlp.add_pipe("sentencizer")
# We di


def split_channels(audio_path, threshold_db=-30, dtype="float32"):
    # audio = AudioSegment.from_wav(audio_path)
    # audio = audio.set_frame_rate(16000)

    # left_channel = audio.split_to_mono()[0]
    # right_channel = audio.split_to_mono()[1]

    # left_channel = np.array(left_channel.get_array_of_samples(), dtype=dtype)
    # right_channel = np.array(right_channel.get_array_of_samples(), dtype=dtype)

    left_channel = load_audio(audio_path, channel="0.0.0")

    # Load right channel
    right_channel = load_audio(audio_path, channel="0.0.1")

    # Convert threshold from dB to amplitude
    threshold = 10 ** (threshold_db / 20)

    # Apply noise gate
    left_channel[np.abs(left_channel) < threshold] = 0
    right_channel[np.abs(right_channel) < threshold] = 0

    return left_channel, right_channel


def transcribe_audio_with_timestamps(audio, model):
    result = whisper.transcribe(model, audio, language="en", vad=False)

    # Extract words and their timestamps
    words_and_timestamps = []
    for segment in result["segments"]:
        for word in segment["words"]:
            words_and_timestamps.append(
                {"word": word["text"], "start": word["start"], "end": word["end"]}
            )

    return words_and_timestamps


def interleave_transcriptions_with_timestamps(
    caller_transcription, service_transcription
):
    # Combine and sort all words by their start time
    all_words = caller_transcription + service_transcription
    all_words.sort(key=lambda x: x["start"])

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
                    interleaved.append(
                        f"{current_speaker} [{sentence_start_time:.2f}s]: {sent.text}"
                    )
            current_speaker = speaker
            current_sentence = ""
            sentence_start_time = word["start"]

        current_sentence += word["word"] + " "

        # Use spaCy for sentence detection
        doc = nlp(current_sentence.strip())
        sentences = list(doc.sents)
        if len(sentences) > 1:
            for sent in sentences[:-1]:
                interleaved.append(
                    f"{current_speaker} [{sentence_start_time:.2f}s]: {sent.text}"
                )
            current_sentence = sentences[-1].text + " "
            sentence_start_time = word["start"]

    # Add any remaining sentence
    if current_sentence:
        doc = nlp(current_sentence.strip())
        for sent in doc.sents:
            interleaved.append(
                f"{current_speaker} [{sentence_start_time:.2f}s]: {sent.text}"
            )

    return "\n".join(interleaved)


def main(audio_path):
    model = whisper.load_model("openai/whisper-medium", torch.device("cuda:0"))

    #audio = whisper.load_audio(audio_path)

    left_channel, right_channel = split_channels(audio_path)

    caller_transcript = transcribe_audio_with_timestamps(left_channel, model)

    agent_transcript = transcribe_audio_with_timestamps(right_channel, model)

    interleaved_transcription = interleave_transcriptions_with_timestamps(
        caller_transcript, agent_transcript
    )

    print("Interleaved Transcription:")
    print(interleaved_transcription)

    # Optionally, save to a file
    with open("interleaved_transcription.txt", "w") as f:
        f.write(interleaved_transcription)

    # result = whisper.transcribe(model, audio, language="en")


if __name__ == "__main__":
    audio_path = "example_1.wav"  # Replace with your audio file path
    main(audio_path)
