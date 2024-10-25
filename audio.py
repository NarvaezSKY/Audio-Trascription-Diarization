from flask import Flask, request, jsonify
import whisper
import os
from pyannote.audio import Pipeline

whisper_model = whisper.load_model('base.en')
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization", 
    use_auth_token="[YOUR TOKEN HERE]"
)

app = Flask(__name__)

def process_transcription(audio_file):
    diarization_result = diarization_pipeline(audio_file, num_speakers=2)
    result = whisper_model.transcribe(audio_file)
    final_text = []

    for segment in result['segments']:
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text'].strip()

        if text:            
            speaker_assigned = False
            for turn, _, speaker in diarization_result.itertracks(yield_label=True):
                
                if turn.start <= end_time and turn.end >= start_time and not speaker_assigned:
                    subsegment_start = max(start_time, turn.start)
                    subsegment_end = min(end_time, turn.end)

                    final_text.append(
                        f"Speaker {speaker}: {text} [{subsegment_start:.2f}s - {subsegment_end:.2f}s]"
                    )
                    speaker_assigned = True
                    break

    return ' '.join(final_text)


@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio = request.files['audio']
    audio_path = os.path.join('temp_audio', audio.filename)
    
    audio.save(audio_path)

    transcription_result = process_transcription(audio_path)
    
    if os.path.exists(audio_path):
        os.remove(audio_path)

    return jsonify({'transcription': transcription_result})

if __name__ == '__main__':
    
    os.makedirs('temp_audio', exist_ok=True)
    app.run(debug=True)
