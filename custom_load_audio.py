from subprocess import CalledProcessError, run
import numpy as np


# hard-coded audio hyperparameters
SAMPLE_RATE = 16000

# Adapted from whisper package to add additional arg for channel
def load_audio(file: str, channel: str = "0.0.0", sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    channel: str
        https://superuser.com/questions/1063185/how-to-select-the-left-audio-channel-with-ffmpeg-and-downmix-to-mono

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the selected audio waveform, in float32 dtype.
    """

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        file,
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sr),
        "-map_channel",
        channel, 
        "-",
    ]

    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    # Convert the raw data to a numpy array
    audio_data = np.frombuffer(out, np.int16)

    # Convert to float32
    return audio_data.astype(np.float32) / 32768.0
