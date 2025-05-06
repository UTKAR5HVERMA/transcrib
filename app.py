from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pyannote.audio import Pipeline
import whisper
from pydub import AudioSegment
import os
import uuid
import tempfile

# Set ffmpeg path (make sure this path is valid on your system)
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"

# Load models at startup
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization"
)
whisper_model = whisper.load_model("base")

app = FastAPI()

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # Save uploaded MP3 file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_mp3:
            tmp_mp3.write(await file.read())
            mp3_path = tmp_mp3.name

        # Convert MP3 to WAV
        wav_path = mp3_path.replace(".mp3", ".wav")
        AudioSegment.from_file(mp3_path).export(wav_path, format="wav")

        # Run speaker diarization on WAV
        diarization = pipeline(wav_path)

        # Load audio for slicing
        audio = AudioSegment.from_wav(wav_path)

        results = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start_ms = int(turn.start * 1000)
            end_ms = int(turn.end * 1000)
            segment = audio[start_ms:end_ms]

            # Export each segment to temporary WAV
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as seg_file:
                segment.export(seg_file.name, format="wav")
                transcription = whisper_model.transcribe(seg_file.name)

            results.append({
                "speaker": speaker,
                "start": round(turn.start, 2),
                "end": round(turn.end, 2),
                "text": transcription["text"].strip()
            })

            os.remove(seg_file.name)

        # Cleanup
        os.remove(mp3_path)
        os.remove(wav_path)

        return JSONResponse(content=results)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
