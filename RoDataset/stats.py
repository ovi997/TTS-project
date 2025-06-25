import os
import glob
import wave
import json
import matplotlib.pyplot as plt
import string

# # Step 1: Get durations of each .wav file
# wav_folder = './wavs'  # Update this path
# wav_files = glob.glob(os.path.join(wav_folder, '*.wav'))
# durations = []

# for wav_file in wav_files:
#     with wave.open(wav_file, 'rb') as wf:
#         frames = wf.getnframes()
#         rate = wf.getframerate()
#         duration = frames / float(rate)
#         durations.append(duration)

# Step 2: Export durations to JSON
json_path = 'durations.json'
# with open(json_path, 'w') as jf:
#     json.dump(durations, jf)

# Step 3: Read durations from JSON
with open(json_path, 'r') as jf:
    loaded_durations = json.load(jf)

# # Step 4: Bin durations into specified categories
# labels = [
#     '0-1 sec', '1-2 sec', '2-3 sec', '3-4 sec', '4-5 sec',
#     '5-6 sec', '6-7 sec', '7-8 sec', '8-9 sec', '9-10 sec',
#     '10-15 sec', '15-20 sec', '20+ sec'
# ]
# bins = [0,1,2,3,4,5,6,7,8,9,10,15,20,float('inf')]
# counts = [0] * (len(bins) - 1)

# for d in loaded_durations:
#     for i in range(len(bins)-1):
#         if bins[i] <= d < bins[i+1]:
#             counts[i] += 1
#             break

# # Step 5: Plot the distribution
# plt.figure(figsize=(10, 6))
# plt.bar(labels, counts)
# plt.xticks(rotation=45, ha='right')
# plt.xlabel('Durata')
# plt.ylabel('Numărul de înregistrări')
# plt.title('Distribuția duratelor înregistrărilor')
# plt.tight_layout()
# plt.show()

# # Notify the user
# print(f"Processed {len(loaded_durations)} files. Durations saved to '{json_path}'.")

wav_folder = './wavs'
txt_folder = './txts'
wav_files = sorted(glob.glob(os.path.join(wav_folder, '*.wav')))
durations = loaded_durations

# 2. Gather transcripts
txt_files = sorted(glob.glob(os.path.join(txt_folder, '*.txt')))
transcripts = {}
for txt_path in txt_files:
    base = os.path.splitext(os.path.basename(txt_path))[0]
    with open(txt_path, 'r', encoding='utf-8') as f:
        transcripts[base] = f.read()

# 3. Compute text-based stats
total_words = 0
total_chars = 0
word_set = set()
words_per_clip = []

for wav_path, duration in zip(wav_files, durations):
    base = os.path.splitext(os.path.basename(wav_path))[0]
    text = transcripts.get(base, '')
    total_chars += len(text)
    
    # Simple tokenization on whitespace
    tokens = text.strip().split()
    num_words = len(tokens)
    words_per_clip.append(num_words)
    total_words += num_words
    
    # Build distinct word set (lowercased, stripped of punctuation)
    for token in tokens:
        clean = token.lower().strip(string.punctuation)
        if clean:
            word_set.add(clean)

# 4. Aggregate statistics
stats = {
    'Total Clips': len(wav_files),
    'Total Words': total_words,
    'Total Characters': total_chars,
    'Total Duration (sec)': sum(durations),
    'Mean Clip Duration (sec)': (sum(durations) / len(durations)) if durations else 0,
    'Min Clip Duration (sec)': min(durations) if durations else 0,
    'Max Clip Duration (sec)': max(durations) if durations else 0,
    'Mean Words per Clip': (total_words / len(words_per_clip)) if words_per_clip else 0,
    'Distinct Words': len(word_set)
}

with open('stats.json', 'w') as jf:
    json.dump(stats, jf, indent=4)

# 5. Output stats as JSON
# print(json.dumps(stats, indent=2))