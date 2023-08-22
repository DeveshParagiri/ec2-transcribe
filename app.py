from faster_whisper import WhisperModel
import time
from datetime import timedelta
from emailing import email
from datetime import datetime
import os


model_size = "base"

model = WhisperModel(model_size, device="cpu", compute_type="int8")

def transcribefaster(file_in):
    start_time = time.time()
    segments, _ = model.transcribe(file_in)
    segments = list(segments)
    result = ''
    for segment in segments:
        result += segment.text
    end_time = time.time()
    duration = str(timedelta(seconds=end_time-start_time))
    response = {"text":result,"duration":duration}
    name = f'Transcription_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")}.txt'
    with open(name,'w+') as f:
        f.write(f'{response["text"]} \n\nDURATION: {response["duration"]} \n')
    email(["devesh.paragiri@gmail.com","bhuvana.kundumani@gmail.com","pjsudhakar@hotmail.com"], name)
    os.remove(name)
    return "Transcription sent to client successfully!"

print(transcribefaster("audiofiles/realtest.wav"))
