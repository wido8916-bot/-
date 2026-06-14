import streamlit as st
import numpy as np
import soundfile as sf
from jamo import h2j, j2hcj
import io
import os

# 1. 자음과 모음의 고정된 음역대 스펙트럼 매핑 (반음 기준)
CONSONANT_SPECTRUM = {
    'ㄱ': 0, 'ㄴ': 1, 'ㄷ': 2, 'ㄹ': 3, 'ㅁ': 4, 'ㅂ': 5, 'ㅅ': 6,
    'ㅇ': 7, 'ㅈ': 8, 'ㅊ': 9, 'ㅋ': 10, 'ㅌ': 11, 'ㅍ': 12, 'ㅎ': 13
}

VOWEL_SPECTRUM = {
    'ㅏ': 0, 'ㅑ': 1, 'ㅓ': 2, 'ㅕ': 3, 'ㅗ': 4,
    'ㅛ': 5, 'ㅜ': 6, 'ㅠ': 7, 'ㅡ': 8, 'ㅣ': 9
}

def generate_pure_tone(base_waveform, semitones, duration=0.5, is_vowel=False, sr=22050):
    """
    가족 목소리의 배음 구조를 필터로 사용하여,
    수학적으로 고정된 스펙트럼의 맑고 긴 악기 소리를 생성합니다.
    """
    n_samples = int(sr * duration)
    if len(base_waveform) == 0:
        return np.zeros(n_samples)
    
    # 데이터 루프 돌려 길이 맞추기
    extended = np.tile(base_waveform, int(np.ceil(n_samples / len(base_waveform))))
    trimmed = extended[:n_samples]
    
    # 자음과 모음의 기본 옥타브 분리 (모음을 중심음으로 설정)
    base_freq = 440.0 if is_vowel else 220.0
    freq = base_freq * (2 ** (semitones / 12.0))
    
    t = np.linspace(0, duration, n_samples, endpoint=False)
    
    if is_vowel:
        # 모음: 0.5초 동안 한 음으로 길게 뻗는 맑고 울림 있는 목관 악기풍 사운드
        source = np.sin(2 * np.pi * freq * t)
        # 긴 호흡을 위한 부드러운 패드형 인벨로프
        envelope = np.ones(n_samples)
        fade_in = int(sr * 0.08)
        fade_out = int(sr * 0.1)
        envelope[:fade_in] = np.linspace(0, 1, fade_in)
        envelope[-fade_out:] = np.linspace(1, 0, fade_out)
    else:
        # 자음: 글자의 시작과 끝을 장식하는 맑은 현악기(피치카토)풍 사운드
        source = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * 2 * np.pi * freq * t)
        envelope = np.exp(-15 * t)  # 타격 후 맑게 감쇄
        
    signal = trimmed * source * envelope
    return signal

def decompose_text(text):
    # 이중자모음 분해 딕셔너리
    double_jamo = {
        'ㄲ': 'ㄱㄱ', 'ㄸ': 'ㄷㄷ', 'ㅃ': 'ㅂㅂ', 'ㅆ': 'ㅅㅅ', 'ㅉ': 'ㅈㅈ',
        'ㄳ': 'ㄱㅅ', 'ㄵ': 'ㄴㅈ', 'ㄶ': 'ㄴㅎ', 'ㄺ': 'ㄹㄱ', 'ㄻ': 'ㄹㅁ', 'ㄼ': 'ㄹㅂ', 'ㄽ': 'ㄹㅅ', 'ㄾ': 'ㄹㅌ', 'ㄿ': 'ㄹㅍ', 'ㅀ': 'ㄹㅎ', 'ㅄ': 'ㅂㅅ',
        'ㅐ': 'ㅏㅣ', 'ㅒ': 'ㅑㅣ', 'ㅔ': 'ㅓㅣ', 'ㅖ': 'ㅕㅣ', 'ㅘ': 'ㅗㅏ', 'ㅙ': 'ㅗㅐ', 'ㅚ': 'ㅗㅣ', 'ㅝ': 'ㅜㅓ', 'ㅞ': 'ㅜㅔ', 'ㅟ': 'ㅜㅣ', 'ㅢ': 'ㅡㅣ'
    }
    result = []
    for char in text:
        if char == " ":
            result.append("SPACE")
            continue
        jamo_list = j2hcj(h2j(char))
        syllable = {'초': [], '중': [], '종': []}
        pos = '초'
        for j in jamo_list:
            if j in 'ㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣㅐㅒㅔㅖㅘㅙㅚㅝㅞㅟㅢ':
                pos = '중'
            elif pos == '중':
                pos = '종'
            decomposed = double_jamo.get(j, j)
            for d in decomposed:
                syllable[pos].append(d)
        result.append(syllable)
    return result

st.set_page_config(page_title="고정 스펙트럼 가족 TTS", layout="centered")
st.title("🎼 가족 자/모음 고정 스펙트럼 앙상블 TTS")
st.write("자음(ㄱ~ㅎ)과 모음(ㅏ~ㅣ)에 완벽한 스펙트럼 규칙을 부여하여 모음이 한 음으로 길게 뻗어가는 맑은 선율을 만듭니다.")

mom_path = "mom_voice.npy"
dad_path = "dad_voice.npy"
me_path = "me_voice.npy"

st.sidebar.subheader("📁 음색 데이터 연결 상태")
for p in [mom_path, dad_path, me_path]:
    if os.path.exists(p):
        st.sidebar.success(f"⭕ {p} 연결됨")
    else:
        st.sidebar.error(f"❌ {p} 없음")

input_text = st.text_input("연주할 문장을 입력하세요", "사랑해")

if st.button("스펙트럼 연주하기"):
    try:
        mom_base = np.load(mom_path)
        dad_base = np.load(dad_path)
        me_base = np.load(me_path)
        
        sr = 22050
        char_duration = 0.5   # 글자 정적 0.5초
        space_duration = 0.3  # 공백 정적 0.3초
        
        space_unit = np.zeros(int(sr * space_duration))
        decomposed_data = decompose_text(input_text)
        final_audio = []

        for item in decomposed_data:
            if item == "SPACE":
                final_audio.append(space_unit)
            else:
                char_signal = np.zeros(int(sr * char_duration))
                active_layers = 0
                
                # 1. 중성 (아빠 목소리 중심 모음 패드) -> 지정된 스펙트럼 음으로 0.5초간 '길게' 연주
                if item['중']:
                    vowel_signal = np.zeros(int(sr * char_duration))
                    for v in item['중']:
                        if v in VOWEL_SPECTRUM:
                            semi = VOWEL_SPECTRUM[v]
                            vowel_signal += generate_pure_tone(dad_base, semi, char_duration, is_vowel=True, sr=sr)
                    char_signal += vowel_signal * 1.2
                    active_layers += 1
                
                # 2. 초성 (엄마 목소리 자음) -> 스펙트럼에 맞춘 맑은 어택음
                if item['초']:
                    consonant_signal = np.zeros(int(sr * char_duration))
                    for c in item['초']:
                        if c in CONSONANT_SPECTRUM:
                            semi = CONSONANT_SPECTRUM[c]
                            consonant_signal += generate_pure_tone(mom_base, semi, char_duration, is_vowel=False, sr=sr)
                    char_signal += consonant_signal * 0.8
                    active_layers += 1
                
                # 3. 종성 (원재 목소리 자음) -> 받침이 있을 때만 스펙트럼 음 연주
                if item['종']:
                    tail_signal = np.zeros(int(sr * char_duration))
                    for c in item['종']:
                        if c in CONSONANT_SPECTRUM:
                            semi = CONSONANT_SPECTRUM[c]
                            tail_signal += generate_pure_tone(me_base, semi, char_duration, is_vowel=False, sr=sr)
                    char_signal += tail_signal * 0.8
                    active_layers += 1
                
                if active_layers > 0:
                    char_signal /= active_layers
                    
                final_audio.append(char_signal)
        
        if final_audio:
            combined = np.concatenate(final_audio)
            if np.max(np.abs(combined)) > 0:
                combined = combined / np.max(np.abs(combined)) * 0.8
                
            out_bio = io.BytesIO()
            sf.write(out_bio, combined, sr, format='WAV')
            st.audio(out_bio.getvalue())
            st.success(f"🎵 규칙 기반 스펙트럼 연주 완료!")
            
    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
