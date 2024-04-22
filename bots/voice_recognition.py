import functools
import time
import torch
import torchaudio
import sounddevice as sd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, Wav2Vec2Processor
from datasets import load_dataset
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
from jiwer import wer


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
def capture_audio(sample_rate=16000, channels=1, duration_seconds=10, dtype='float32'):
    print("capture_audio... Presiona Ctrl+C para detener.")
    audio = sd.rec(int(sample_rate * duration_seconds), samplerate=sample_rate, channels=channels, dtype=dtype)
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
    ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")

    # tokenize
    input_values = processor(ds[0]["audio"]["array"], return_tensors="pt",
                             padding="longest").input_values  # Batch size 1

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
        input_values = processor(batch["audio"][0]["array"], return_tensors="pt", padding="longest").input_values
        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        batch["transcription"] = transcription
        return batch

    result = librispeech_eval.map(map_to_pred, batched=True, batch_size=1, remove_columns=["audio"])

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


if __name__ == "__main__":
    main()
    # load_model_tokenizer()
    # evaluation()
    print("Done!")
