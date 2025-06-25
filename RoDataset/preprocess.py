import os
import soundfile as sf
import pandas as pd
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import gc
import random

_FILENAMES = os.listdir('./original_wavs')

class BuildRDSpeech:
    def __init__(self):
        pass

    def split_chunks():
        pass

model = load_silero_vad()
def return_voice_timestamps(audio_path):
    wav = read_audio(audio_path)
    speech_timestamps = get_speech_timestamps(
      wav,
      model,
      return_seconds=True,
      min_silence_duration_ms=350,
      threshold=0.5,  # Return speech timestamps in seconds (default is samples)
    )

    del wav
    gc.collect()

    return speech_timestamps

def split_audio(filename, timestamps, fn_map):
    wav, sr = sf.read(f'RoDataset\{filename}')
    # timestamps = convert_seconds_to_samples(timestamps=timestamps, sr=sr)
    temp = timestamps.copy()
    # start_sample = temp[0]['start']
    # stop_sample = temp[0]['end']
    idx = 0
    total_chunks_len = 0
    while len(temp) > 0:
        min_duration = 5
        max_duration = 9

        current_timestamp = temp.pop(0)
        start = current_timestamp['start']
        end = current_timestamp['end']

        segment_duration = current_timestamp['end'] - current_timestamp['start']

        if segment_duration < max_duration:
            temp, start, end = merge_segments(temp, start, end, min_duration, max_duration)

        sf.write(f'RDSpeech/RD_{fn_map[filename]}_{"{:06d}".format(idx)}.wav', wav[int(start*sr):int(end*sr)], sr)
        idx += 1
        total_chunks_len += (end - start)

    return total_chunks_len


def merge_segments(_timestamps, _start, _end, _min_dur, _max_dur):
    _segment_duration = _end - _start
    if _min_dur <= _segment_duration or len(_timestamps) == 0:
        return _timestamps, _start, _end
    else:
        _current_timestamp = _timestamps.pop(0)
        tmp_end = _current_timestamp['end']
        _segment_duration = tmp_end - _start

        if _segment_duration >= _max_dur:
            _timestamps.insert(0, _current_timestamp)
            return _timestamps, _start, _end

        return merge_segments(_timestamps, _start, tmp_end, _min_dur, _max_dur)


def compute_segments_len(timestamps):
    return sorted([i['end'] - i['start'] for i in timestamps], reverse=True)


def convert_seconds_to_samples(timestamps, sr):
    for item in timestamps:
        item['start'] = item['start'] * sr
        item['end'] = item['end'] * sr

    return timestamps

def mapping(iterable):
    return {key: "{:04d}".format(val) for val, key in enumerate(iterable)}

def convert_seconds_to_hours(time):
    hours, minutes = divmod(time/3600, 1)
    minutes, seconds = divmod(minutes*60, 1)
    seconds, _ = divmod(seconds*60, 1)
    result = f'Total audio content: {"{:02d}".format(int(hours))}:{"{:02d}".format(int(minutes))}:{"{:02d}".format(int(seconds))} hours.'
    
    print(result)
    return(result)

def compute_audio_content_len(filenames):
    audio_content_len = 0
    for i, filename in enumerate(filenames):
        print(f'Loaded {i+1}/{len(filenames)} audio.')
        wav, sr = sf.read(f'RDSpeech\{filename}')
        audio_content_len += len(wav) / sr

    return convert_seconds_to_hours(audio_content_len)


def split_dataset(root_path, fn_mapping):

    txts = os.listdir(os.path.join(root_path, "txts"))

    ood_filenames = ['Mihai_Eminescu-E_trist_ca_nimeni_sa_te_stie.mp3', 
                     'Florian_Cristescu-Familia-Roade-Mult_Capitolul_01.mp3', 
                     'Florian_Cristescu-Familia-Roade-Mult_Capitolul_02.mp3',
                     'Florian_Cristescu-Familia-Roade-Mult_Capitolul_03.mp3',
                     'Florian_Cristescu-Familia-Roade-Mult_Capitolul_04.mp3',
                     'Florian_Cristescu-Familia-Roade-Mult_Capitolul_05.mp3',
                     'Florian_Cristescu-Familia-Roade-Mult_Capitolul_06.mp3',
                     ]
    valid_filenames = ['A.P.Cehov-Calugarul_negru-Capitolul_01.mp3',
                       'A.P.Cehov-Calugarul_negru-Capitolul_02.mp3',
                       'A.P.Cehov-Calugarul_negru-Capitolul_03.mp3',
                       'A.P.Cehov-Calugarul_negru-Capitolul_04.mp3',
                       'A.P.Cehov-Calugarul_negru-Capitolul_05.mp3',
                       'A.P.Cehov-Calugarul_negru-Capitolul_06.mp3',
                       'A.P.Cehov-Calugarul_negru-Capitolul_07.mp3',
                       'A.P.Cehov-Calugarul_negru-Capitolul_08.mp3',
                       'A.P.Cehov-Calugarul_negru-Capitolul_09.mp3',
                       'Anton_Pavlovici_Cehov-Doamna_cu_catelul-Capitolul_01.mp3',
                       'Anton_Pavlovici_Cehov-Doamna_cu_catelul-Capitolul_02.mp3',
                       'Anton_Pavlovici_Cehov-Doamna_cu_catelul-Capitolul_03.mp3',
                       'Anton_Pavlovici_Cehov-Doamna_cu_catelul-Capitolul_04.mp3',
                       ]

    ood_fid = [fn_mapping[i] for i in ood_filenames]
    valid_fid = [fn_mapping[i] for i in valid_filenames]

    num_train = 0
    num_valid = 0
    num_ood = 0

    for i, filename in enumerate(txts):
        print(filename)
        print(f"Processing record {i+1}/{len(txts)}")
        # try:
        with open(f'./txts/{filename}', errors="replace") as f:
            transcript = f.read()
        current_fid = filename.split('_')[1]

        if current_fid in ood_fid:
            out_file = './OOD_texts.txt'
            num_ood += 1
        elif current_fid in valid_fid:
            out_file = './train_list.txt'
            num_valid += 1
        else:
            out_file = './val_list.txt'
            num_train += 1

        data = f'{filename.replace(".txt", ".wav")}|{transcript}|0\n'
        with open(out_file, 'a', encoding="utf-8") as f:
            f.write(data)
        # except:
        #     print(filename)

    print(f"""
          Train samples: {num_train} 
          Valid samples: {num_valid}
          OOD   samples: {num_ood}
          """)


def main():
    # if not os.path.exists('RDSpeech'):
    #     os.makedirs('RDSpeech')

    filenames_mapping = mapping(_FILENAMES)
    split_dataset('./', filenames_mapping)
    # print(filenames_mapping)
    # audio_content = 0
    # for i, filename in enumerate(_FILENAMES):
    #     if (i+1) <= 42:
    #         continue
    #     print(f'{filename} | {i+1}/{len(_FILENAMES)}')
    #     timestamps = return_voice_timestamps(audio_path=f'RoDataset\{filename}')
    #     audio_content += split_audio(filename=filename, timestamps=timestamps, fn_map=filenames_mapping)

    # audio_content = convert_seconds_to_hours(audio_content)
    
    # splitted_filenames = os.listdir('RDSpeech')
    # audio_content = compute_audio_content_len(splitted_filenames)

    # for fn, id in filenames_mapping.items():
    #     with open('RDSpeech_medatada.txt', 'a') as f:
    #         f.writelines([f"{fn}|{id}\n"])  

    # with open('RDSpeech_medatada.txt', 'a') as f:
    #     f.writelines([audio_content])      

if __name__ == '__main__':
    main()


    # RD_0026_000260.wav