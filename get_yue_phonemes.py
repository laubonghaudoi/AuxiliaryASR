from ToJyutping import get_jyutping_list
import pandas as pd


def add_phonemes(file, phonemes_set):
    df = pd.read_csv(file)
    sent = df['transcription'].tolist()
    jyutping = [get_jyutping_list(s) for s in sent]

    for sent in jyutping:
        for char in sent:
            if char[1]:
                # jyutping is not None, it is a valid character
                phonemes_set.add(char[1])
            else:
                # jyutping is None, it is a punctuation or special character
                phonemes_set.add(char[0])


if __name__ == '__main__':
    phonemes = set()
    add_phonemes('saamgwokjinji/metadata.csv', phonemes)
    add_phonemes('seoiwuzyun/metadata.csv', phonemes)
    add_phonemes('mouzaakdung/metadata.csv', phonemes)

    phonemes.remove(' ')
    phonemes = sorted(list(phonemes))

    with open('yue_phoneme_index.txt', 'w', encoding='utf-8') as f:
        f.write(""""<pad>",0
"<sos>",1
"<eos>",2
"<unk>",3
" ",4
",",5
"，",5
"；",5
"、",5
".",6
"。",6
":",7
"：",7
"！",8
"!",8
"?",9
"？",9
"《",10
"》",11
"“",10
"”",11
"‘",10
"’",11
"(",10
")",11
"（",10
"）",11
"",10
"[",10
"]",11
"【",10
"】",11
"「",10
"」",11
"—",12
"…",12
""")

        for i, p in enumerate(phonemes):
            f.write(f"\"{p}\",{i + 13}\n")
