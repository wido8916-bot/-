import streamlit as st
import numpy as np
import soundfile as sf
from jamo import h2j, j2hcj
import io
import os

def change_pitch(waveform, semitones, sr=22050):
    """수학적 리샘플링을 통해 오디오의 음높이를 조절하고, 길이를 완벽히 고정합니다."""
    if semitones == 0:
        return waveform
    factor = 2 ** (semitones / 12.0)
    indices = np.arange(0, len(waveform), factor)
    indices = indices[indices < len(waveform)]
    pitched = np.interp(indices, np.arange(len(waveform)), waveform)
    
    target_len = len(waveform)
    if len(pitched) > target_len:
        pitched = pitched[:target_len]
    elif len(pitched) < target_len:
        padded = np.zeros(target_len)
        padded[:len(pitched)] = pitched
        pitched = padded
        
    return pitched

def apply_envelope_and_reverb(waveform, sr=22050, decay_rate=0.7):
    """소리에 부드러운 여운(A.D.S.R)과 공간감(Reverb)을 줍니다."""
    n_samples = len(waveform)
    envelope = np.exp(-decay_rate * np.linspace(0, 3, n_samples))
    smoothed = waveform * envelope
    
    delay_samples = int(sr * 0.08)
    reverb = np.zeros(n_samples + delay_samples)
    reverb[:n_samples] += smoothed
    reverb[delay_samples:] += smoothed * 0.35
    reverb[delay_samples*2:] += smoothed * 0.15
    
    return reverb

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
            if j in 'ㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣㅐㅒㅔㅖㅘㅙㅚㅝㅞㅟㅢ':
                pos = '중'
            elif pos == '중':
                pos = '종'
            decomposed = double_jamo.get(j, j)
            for d in decomposed:
                syllable[pos].append(d)
        result.append(syllable)
    return result

st.set_page_config(page_title="가족 목소리 앙상블 TTS", layout="centered")
st.title("🎵 가족 목소리 앙상블 합창 TTS")

mom_path = "mom_voice.npy"
dad_path = "dad_voice.npy"
me_path = "me_voice.npy"

st.sidebar.subheader("📁 음색 데이터 연결 상태")
for p in [mom_path, dad_path, me_path]:
    if os.path.exists(p):
        st.sidebar.success(f"⭕ {p} 연결됨")
    else:
        st.sidebar.error(f"❌ {p} 없음")

input_text = st.text_input("연주할 문장을 입력하세요", "원재 사랑해")

if st.button("음악으로 연주하기"):
    try:
        mom_base = np.load(mom_path)
        dad_base = np.load(dad_path)
        me_base = np.load(me_path)
        
        sr = 22050
        space_unit = np.zeros(int(sr * 0.4))
        decomposed_data = decompose_text(input_text)
        final_audio = []

        for i, item in enumerate(decomposed_data):
            if item == "SPACE":
                final_audio.append(space_unit)
            else:
                pitch_offset = (i % 3) * 2
                
                for _ in item['초']:
                    pitched = change_pitch(mom_base, 0 + pitch_offset, sr)
                    final_audio.append(apply_envelope_and_reverb(pitched, sr))
                for _ in item['중']:
                    pitched = change_pitch(dad_base, 4 + pitch_offset, sr)
                    final_audio.append(apply_envelope_and_reverb(pitched, sr))
                for _ in item['종']:
                    pitched = change_pitch(me_base, 7 + pitch_offset, sr)
                    final_audio.append(apply_envelope_and_reverb(pitched, sr))
        
        if final_audio:
            combined = np.concatenate(final_audio)
            if np.max(np.abs(combined)) > 0:
                combined = combined / np.max(np.abs(combined)) * 0.85
                
            out_bio = io.BytesIO()
            sf.write(out_bio, combined, sr, format='WAV')
            st.audio(out_bio.getvalue())
            st.success(f"'{input_text}' 음악 연주 완료!")
    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
