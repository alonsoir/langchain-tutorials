import datetime
import functools
import time
import wave

import pyaudio
import sounddevice as sd
import torch
import torchaudio
from datasets import load_dataset
from jiwer import wer
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import Wav2Vec2Tokenizer

"""
    The idea of this code is to perform speech recognition using the Wav2Vec2 model from the Transformers library. The main steps are:
    1. Load the pre-trained Wav2Vec2 model and processor.
        2. Define a function `map_to_pred` that takes a batch of audio data, processes it through the model, and returns the transcription.
    problem: it is not working. It is not able to recognize my voice using different models. I dont know if it dependes of the mic, it could be,
    or the problem remains in the code. Can you help me to fix it?
"""


# Decorador para medir el tiempo de ejecución de una función
def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"---> {func.__name__} Elapsed time: {elapsed_time:0.4f} seconds")
        print("\n")
        return value

    return wrapper_timer


# Función para capturar audio en tiempo real
def capture_audio(sample_rate=16000, channels=1, duration_seconds=10, dtype="float32"):
    print("capture_audio... Presiona Ctrl+C para detener.")
    audio = sd.rec(
        int(sample_rate * duration_seconds),
        samplerate=sample_rate,
        channels=channels,
        dtype=dtype,
    )
    sd.wait()
    return audio.squeeze()


# Función para transcribir audio
def transcribe_audio(model, tokenizer, waveform):
    input_values = tokenizer(waveform, return_tensors="pt").input_values

    # Obtener la transcripción del modelo
    with torch.no_grad():
        logits = model(input_values).logits

    # Decodificar la salida del modelo
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)[0]
    return transcription


@timer
def load_model_tokenizer():
    # load model and tokenizer
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    # load dummy dataset and read soundfiles
    ds = load_dataset(
        "patrickvonplaten/librispeech_asr_dummy", "clean", split="validation"
    )

    # tokenize
    input_values = processor(
        ds[0]["audio"]["array"], return_tensors="pt", padding="longest"
    ).input_values  # Batch size 1

    # retrieve logits
    logits = model(input_values).logits

    # take argmax and decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    print(f"transcription: {transcription}")


@timer
def evaluation():
    librispeech_eval = load_dataset(path="librispeech_asr", name="clean", split="test")

    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    def map_to_pred(batch):
        input_values = processor(
            batch["audio"][0]["array"], return_tensors="pt", padding="longest"
        ).input_values
        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        batch["transcription"] = transcription
        return batch

    result = librispeech_eval.map(
        map_to_pred, batched=True, batch_size=1, remove_columns=["audio"]
    )

    print("WER:", wer(result["text"], result["transcription"]))


@timer
def main():
    # Cargar el modelo y el tokenizer preentrenado
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")

    # Configurar el backend de audio. sox_io/soundfile
    torchaudio.set_audio_backend("sox_io")

    while True:
        try:
            audio = capture_audio()
            transcription = transcribe_audio(model, tokenizer, torch.tensor(audio))
            print("Texto reconocido:", transcription)
        except KeyboardInterrupt:
            print("Captura de audio detenida.")
            break


@timer
def test_spanish():
    # Cargar el modelo y el tokenizador
    model = Wav2Vec2ForCTC.from_pretrained("sil-ai/wav2vec2-bloom-speech-spa")
    processor = Wav2Vec2Processor.from_pretrained("sil-ai/wav2vec2-bloom-speech-spa")

    # Configurar la grabación de audio
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 5
    start_time = datetime.datetime.now()
    p = pyaudio.PyAudio()

    stream = p.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )

    print("Grabando...")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Grabación finalizada.")

    stream.stop_stream()
    stream.close()
    p.terminate()
    end_time = datetime.datetime.now()

    start_timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    end_timestamp = end_time.strftime("%Y%m%d_%H%M%S")
    WAVE_OUTPUT_FILENAME = f"audio_{start_timestamp}_{end_timestamp}.wav"
    wf = wave.open(WAVE_OUTPUT_FILENAME, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()

    # Cargar el archivo de audio grabado
    with open(WAVE_OUTPUT_FILENAME, "rb") as f:
        audio_data = f.read()

    # Procesar el audio
    input_values = processor(
        audio_data, return_tensors="pt", sampling_rate=16000
    ).input_values

    # Realizar la transcripción
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    print(f"Transcripción: {transcription}")


if __name__ == "__main__":
    test_spanish()
    # load_model_tokenizer()
    # evaluation()
    print("Done!")
