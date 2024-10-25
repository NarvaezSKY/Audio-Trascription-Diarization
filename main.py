from flask import Flask, request, jsonify
import whisper
import yt_dlp
import os
from pyannote.audio import Pipeline

whisper_model = whisper.load_model('base.en')  
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization", 
    use_auth_token="[YOUR TOKEN HERE]"
)

app = Flask(__name__)

ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
    'outtmpl': '%(id)s.%(ext)s',
    'quiet': True,
}

def process_transcription(audio_file):
    diarization_result = diarization_pipeline(audio_file, num_speakers=2)
    result = whisper_model.transcribe(audio_file)
    final_text = []

    for segment in result['segments']:
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text'].strip()

        if text:
            longest_overlap = 0
            selected_speaker = None

            for turn, _, speaker in diarization_result.itertracks(yield_label=True):

                overlap_start = max(start_time, turn.start)
                overlap_end = min(end_time, turn.end)
                overlap_duration = overlap_end - overlap_start

                if overlap_duration > 0 and overlap_duration > longest_overlap:
                    longest_overlap = overlap_duration
                    selected_speaker = speaker

            if selected_speaker:
                final_text.append(
                    f"Speaker {selected_speaker}: {text} [{start_time:.2f}s - {end_time:.2f}s]"
                )

    return ' '.join(final_text)

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    data = request.json
    url = data.get('url')

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=False)
        duration = info_dict.get('duration', 0)
        video_id = info_dict.get('id', None)
        audio_file = f"{video_id}.mp3"

        if duration > 3600:
            return jsonify({'error': 'Video duration exceeds 1 hour, please provide a shorter video'}), 400

        ydl.download([url])

    if not os.path.exists(audio_file):
        return jsonify({'error': 'Audio file not found or download failed'}), 500

    transcription_result = process_transcription(audio_file)

    if os.path.exists(audio_file):
        os.remove(audio_file)

    return jsonify({'transcription': transcription_result})

if __name__ == '__main__':
    app.run(debug=True)
