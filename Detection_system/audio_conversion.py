from pydub import AudioSegment

# Load the stereo audio file (replace 'stereo_audio.wav' with your file)
stereo_audio = AudioSegment.from_file("Recording (22).wav")

# Convert stereo audio to mono audio
mono_audio = stereo_audio.set_channels(1)

# Save the mono audio to a new file
mono_audio.export("mono_audio.wav", format="wav")