from phonemizer import phonemize
from phonemizer.separator import Separator
import os
import sys
from memory_profiler import profile
import gc
import pandas as pd

def mapping(iterable):
    return {key: "{:04d}".format(val) for val, key in enumerate(iterable)}

_FILENAMES = os.listdir('./original_wavs')


_filenames_mapping = mapping(_FILENAMES)
_txts = os.listdir(os.path.join('./', "txts"))

_ood_filenames = ['Mihai_Eminescu-E_trist_ca_nimeni_sa_te_stie.mp3',
                    'Florian_Cristescu-Familia-Roade-Mult_Capitolul_01.mp3',
                    'Florian_Cristescu-Familia-Roade-Mult_Capitolul_02.mp3',
                    'Florian_Cristescu-Familia-Roade-Mult_Capitolul_03.mp3',
                    'Florian_Cristescu-Familia-Roade-Mult_Capitolul_04.mp3',
                    'Florian_Cristescu-Familia-Roade-Mult_Capitolul_05.mp3',
                    'Florian_Cristescu-Familia-Roade-Mult_Capitolul_06.mp3',
                    ]
_valid_filenames = ['A.P.Cehov-Calugarul_negru-Capitolul_01.mp3',
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

_ood_fid = [_filenames_mapping[i] for i in _ood_filenames]
_valid_fid = [_filenames_mapping[i] for i in _valid_filenames]

# @profile
def _split_phonemes_dataset(txts, error_file=None):
    if error_file is None:
      error_file = "./split_phonemes/error_files.txt"

    num_train = 0
    num_valid = 0
    num_ood = 0

    for i, filename in enumerate(txts):
        print(f"Processing record {i+1}/{len(txts)}")
        try:
            with open(f'./txts/{filename}', errors="replace") as f:
                transcript = f.read()
            current_fid = filename.split('_')[1]

            phonemes = phonemize(transcript,
                                    language='ro',
                                    backend='espeak',
                                    separator=Separator(phone=None, word=' '),
                                    strip=True,
                                    preserve_punctuation=True,
                                    njobs=4)
            
            if current_fid in _ood_fid:
                out_file = './split_phonemes/OOD_texts.txt'
                num_ood += 1
            elif current_fid in _valid_fid:
                out_file = './split_phonemes/val_list.txt'
                num_valid += 1
            else:
                out_file = './split_phonemes/train_list.txt'
                num_train += 1

            data = f'{filename.replace(".txt", ".wav")}|{phonemes}|0\n'
            with open(out_file, 'a', encoding="utf-8") as f:
                f.write(data)
        except:
            with open(error_file, 'a', encoding="utf-8") as f:
                f.write(f"{filename}\n")
            continue

    del data
    del phonemes
    del current_fid
    gc.collect()

    # print(f"""
    #       Train samples: {num_train}
    #       Valid samples: {num_valid}
    #       OOD   samples: {num_ood}
    #       """)
    
    # return num_ood, num_valid, num_train


def sharding(filenames, n_shards):
  shards = [[] for _ in range(n_shards)]
  for i, filename in enumerate(filenames):
    shards[i % n_shards].append(filename)
  return shards

def load_dictionary(path):
    csv = pd.read_csv(path, header=None).values
    word_index_dict = {word: index for word, index in csv}
    return word_index_dict

if __name__ == "__main__":
    # _shards = sharding(_txts, 50)
    # print(len(_shards))

    # for i, shard in enumerate(_shards[0:1]):
    #     print(f"Shard {i+1}/ {len(_shards)}")
    # _split_phonemes_dataset(_shards[50])

    # all_phonemes = []


    # with open("./split_phonemes/OOD_texts.txt") as f:
    #     all_phonemes += f.readlines()

    # with open("./split_phonemes/train_list.txt") as f:
    #     all_phonemes += f.readlines()

    # with open("./split_phonemes/val_list.txt") as f:
    #     all_phonemes += f.readlines()

    # all_phonemes = [item.replace("\n","").split("|")[1].replace(" ", "") for item in all_phonemes]
    # all_phonemes = list(set(list("".join(all_phonemes))))
    # all_phonemes.remove('"')
    # all_phonemes.remove(",")
    # all_phonemes.remove(".")
    # all_phonemes.remove("?")
    # all_phonemes.remove("«")
    # all_phonemes.remove("»")
    # all_phonemes.remove("!")

    # print(all_phonemes)
    # print(len(all_phonemes))

    # special_tokens = [("<pad>", 0), ("<sos>", 1), ("<eos>", 2), ("<unk>", 3), (" ", 4), (",", 5),(".", 6), (":", 7), ("!", 8), ("?", 9)]
    # phonemes_tokens = [f'"{item}",{i+10}\n' for i, item in enumerate(all_phonemes)]

    # with open("word_index_dict.txt", "a") as f:
    #     for item in special_tokens:
    #         f.write(f'"{item[0]}",{item[1]}\n')

    #     for item in phonemes_tokens:
    #         f.write(item)


    df = load_dictionary("word_index_dict.txt")
    print(df)