import streamlit as st
import numpy as np
import soundfile as sf
from jamo import h2j, j2hcj
import io
import os

# 1. 고정 스펙트럼 매핑
CONSONANT_SPECTRUM = {
    'ㄱ': 0, 'ㄴ': 1, 'ㄷ': 2, 'ㄹ': 3, 'ㅁ': 4, 'ㅂ': 5, 'ㅅ': 6,
    'ㅇ': 7, 'ㅈ': 8, 'ㅊ': 9, 'ㅋ': 10, 'ㅌ': 11, 'ㅍ': 12, 'ㅎ': 13
}

VOWEL_SPECTRUM = {
    'ㅏ': 0, 'ㅑ': 1, 'ㅓ': 2, 'ㅕ': 3, 'ㅗ': 4,
    'ㅛ': 5, 'ㅜ': 6, 'ㅠ': 7, 'ㅡ': 8, 'ㅣ': 9
}

def generate_pure_tone(base_waveform, semitones, duration, is_vowel=False, sr=22050):
    """
    지정된 duration(길이)만큼 끊김 없이 통째로 하나의 파형을 생성하는 함수
    """
    n_samples = int(sr * duration)
    if len(base_waveform) == 0:
        return np.zeros(n_samples)
    
    # 반복 배치로 원하는 총 길이 맞추기
    extended = np.tile(base_waveform, int(np.ceil(n_samples / len(base_waveform))))
    trimmed = extended[:n_samples]
    
    base_freq = 440.0 if is_vowel else 220.0
    freq = base_freq * (2 ** (semitones / 12.0))
    t = np.linspace(0, duration, n_samples, endpoint=False)
    
    if is_vowel:
        # 모음: 늘어난 시간(예: 1.0초) 동안 한 호흡으로 곧게 뻗는 사인파
        source = np.sin(2 * np.pi * freq * t)
        envelope = np.ones(n_samples)
        fade_in = int(sr * 0.08)
        fade_out = int(sr * 0.1)
        envelope[:fade_in] = np.linspace(0, 1, fade_in)
        envelope[-fade_out:] = np.linspace(1, 0, fade_out)
    else:
        # 자음: 타격음 성격 유지
        source = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * 2 * np.pi * freq * t)
        envelope = np.exp(-15 * t)
        
    return trimmed * source * envelope

def decompose_text(text):
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
            if j in 'ㅏㅑㅓㅕㅗ ㅛㅜㅠㅡㅣㅐㅒㅔㅖㅘㅙㅚㅝㅞㅟㅢ':
                pos = '중'
            elif pos == '중':
                pos = '종'
            decomposed = double_jamo.get(j, j)
            for d in decomposed:
                syllable[pos].append(d)
        result.append(syllable)
    return result

st.set_page_config(page_title="연속음 통합 가족 TTS", layout="centered")
st.title("🎼 가족 자/모음 연속음 통합 앙상블 TTS")
st.write("같은 자음이나 모음이 연속되면 소리가 끊기지 않고 하나의 긴 선율로 이어져 연주됩니다.")

mom_path = "mom_voice.npy"
dad_path = "dad_voice.npy"
me_path = "me_voice.npy"

st.sidebar.subheader("📁 음색 데이터 연결 상태")
for p in [mom_path, dad_path, me_path]:
    if os.path.exists(p):
        st.sidebar.success(f"⭕ {p} 연결됨")
    else:
        st.sidebar.error(f"❌ {p} 없음")

input_text = st.text_input("연주할 문장을 입력하세요", "하나")

if st.button("붙임줄 연주하기"):
    try:
        mom_base = np.load(mom_path)
        dad_base = np.load(dad_path)
        me_base = np.load(me_path)
        
        sr = 22050
        base_dur = 0.5
        space_dur = 0.3
        
        decomposed_data = decompose_text(input_text)
        
        # ─── 2. 연속된 자/모음 그룹화 알고리즘 (글자 묶기) ───
        grouped_chunks = []
        current_chunk = []
        
        for item in decomposed_data:
            if item == "SPACE":
                if current_chunk:
                    grouped_chunks.append(current_chunk)
                    current_chunk = []
                grouped_chunks.append("SPACE")
            else:
                if not current_chunk:
                    current_chunk.append(item)
                else:
                    # 이전 글자의 모음과 현재 글자의 모음이 같은지 확인
                    prev_vowels = current_chunk[-1]['중']
                    curr_vowels = item['중']
                    if prev_vowels == curr_vowels and prev_vowels:  # 모음이 같으면 한 청크로 묶음
                        current_chunk.append(item)
                    else:
                        grouped_chunks.append(current_chunk)
                        current_chunk = [item]
        if current_chunk:
            grouped_chunks.append(current_chunk)

        # ─── 3. 합성 및 연주 ───
        final_audio = []
        
        for chunk in grouped_chunks:
            if chunk == "SPACE":
                final_audio.append(np.zeros(int(sr * space_dur)))
            else:
                # 묶인 글자 수만큼 총 연주 시간 계산 (예: 2글자면 1.0초)
                chunk_len = len(chunk)
                total_duration = chunk_len * base_dur
                chunk_signal = np.zeros(int(sr * total_duration))
                
                # 1. 중성 (모음) 통합 생성 -> 끊김 없이 통째로 1.0초 질주
                # 대표로 첫 글자의 모음을 가져와 전체 시간만큼 통째로 뽑아냅니다.
                v_list = chunk[0]['중']
                if v_list:
                    vowel_signal = np.zeros(len(chunk_signal))
                    for v in v_list:
                        if v in VOWEL_SPECTRUM:
                            semi = VOWEL_SPECTRUM[v]
                            vowel_signal += generate_pure_tone(dad_base, semi, total_duration, is_vowel=True, sr=sr)
                    chunk_signal += vowel_signal * 1.2

                # 2. 초성 및 종성 (자음들)은 각 글자의 타이밍 위치에 얹기
                for idx, item in enumerate(chunk):
                    offset_samples = int(sr * idx * base_dur)
                    
                    # 초성 배치
                    if item['초']:
                        for c in item['초']:
                            if c in CONSONANT_SPECTRUM:
                                semi = CONSONANT_SPECTRUM[c]
                                tone = generate_pure_tone(mom_base, semi, base_dur, is_vowel=False, sr=sr)
                                # 해당 글자의 시작 위치에 자음 타격음 더하기
                                chunk_signal[offset_samples:offset_samples+len(tone)] += tone * 0.8
                                
                    # 종성 배치
                    if item['종']:
                        for c in item['종']:
                            if c in CONSONANT_SPECTRUM:
                                semi = CONSONANT_SPECTRUM[c]
                                tone = generate_pure_tone(me_base, semi, base_dur, is_vowel=False, sr=sr)
                                chunk_signal[offset_samples:offset_samples+len(tone)] += tone * 0.8
                                
                # 정규화 및 결합
                chunk_signal /= 2.0
                final_audio.append(chunk_signal)
                
        if final_audio:
            combined = np.concatenate(final_audio)
            if np.max(np.abs(combined)) > 0:
                combined = combined / np.max(np.abs(combined)) * 0.8
                
            out_bio = io.BytesIO()
            sf.write(out_bio, combined, sr, format='WAV')
            st.audio(out_bio.getvalue())
            st.success("🎵 붙임줄(Tie) 알고리즘 연주가 완료되었습니다!")
            
    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
