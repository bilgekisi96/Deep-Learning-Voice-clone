import os
import torch
import torchaudio

from transformers import AutoProcessor, AutoModelForCTC



# Define the path where the .wav files are located
wav_directory = r"C:\soiceclone\First_wavs"  # Use raw string to avoid escape characters

# Define the output file name
output_file = os.path.join(wav_directory, "list.txt")

# Define the range of .wav files (1 to 165)
wav_files_range = range(1, 31)

# Initialize the list to store file paths and transcripts
file_and_transcripts = []

processor = AutoProcessor.from_pretrained("ozcangundes/wav2vec2-large-xlsr-53-turkish")
model = AutoModelForCTC.from_pretrained("ozcangundes/wav2vec2-large-xlsr-53-turkish")
# Initialize the wav2vec model and processor
#model = AutoModelForCTC.from_pretrained("mpoyraz/wav2vec2-xls-r-300m-cv7-turkish")
#processor = AutoProcessor.from_pretrained("mpoyraz/wav2vec2-xls-r-300m-cv7-turkish")

# Iterate through the .wav files
for i in wav_files_range:
    wav_file = os.path.join(wav_directory, f"{i}.wav")

    # Check if the .wav file exists
    if os.path.exists(wav_file):
        print(f"Processing file: {wav_file}")
        try:
            # Load the waveform and sample rate from the .wav file
            waveform, sample_rate = torchaudio.load(wav_file)

            # If stereo audio (2 channels), convert to mono by averaging the channels
            if waveform.shape[0] == 2:
                waveform = torch.mean(waveform, dim=0, keepdim=True)  # Downmix to mono

            # Resample the waveform to 16kHz
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

            # Ensure the tensor has the correct shape [batch_size, channels, seq_len]
            # Reshape if the tensor has extra dimensions (e.g., [1, 1, 1, N])
            if waveform.dim() == 4:  # Shape: [1, 1, 1, N]
                waveform = waveform.reshape(-1)  # Convert to [1, 1, N]
            elif waveform.dim() == 3:  # Shape: [1, 1, N] or [channels, N]
                waveform = waveform  # Already in the correct shape
            elif waveform.dim() == 2:  # Shape: [N]
                waveform = waveform.reshape(-1)  # Add batch and channel dimensions

            # Process the input waveform for the model
            input_values = processor(waveform, return_tensors="pt", sampling_rate=16000).input_values

            # Forward pass through the model
            logits = model(input_values).logits

            # Get predicted ids
            predicted_ids = torch.argmax(logits, dim=-1)

            # Decode the transcript
            transcript = processor.decode(predicted_ids[0])
            print(f"Transcript for {wav_file}: {transcript}")

        except Exception as e:
            print(f"Error processing file {wav_file}: {e}")
            continue

        # Append the desired path format and transcript to the list
        file_and_transcripts.append(f"/content/TTS-TT2/wavs/{i}.wav|{transcript}")
    else:
        print(f"File not found: {wav_file}")

# Write the file paths and transcripts to the output file
with open(output_file, "w",encoding="utf-8") as f:
    for line in file_and_transcripts:
        f.write(f"{line}\n")

print(f"File '{output_file}' created successfully.")
