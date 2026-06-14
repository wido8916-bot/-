import streamlit as st
import numpy as np
import soundfile as sf
from jamo import h2j, j2hcj
import io
import os

def change_pitch(waveform, semitones, sr=22050):
    """음높이를 바꾸고 결과 길이를 원본과 완벽하게 일치시키는 안전한 함수"""
    if semitones == 0 or len(waveform) == 0:
        return waveform
    
    factor = 2 ** (semitones / 12.0)
    indices = np.arange(0, len(waveform), factor)
    indices = indices[indices < len(waveform)]
    pitched = np.interp(indices, np.arange(len(waveform)), waveform)
    
    # 에러의 원인이 되는 길이를 무조건 원본 크기로 강제 고정
    target_len = len(waveform)
    if len(pitched) > target_len:
        pitched = pitched[:target_len]
    elif len(pitched) < target_len:
        padded = np.zeros(target_len)
        padded[:len(pitched)] = pitched
        pitched = padded
    return pitched

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

input_text = st.text_input("연주할 문장을 입력하세요", "사랑해")

if st.button("음악으로 연주하기"):
    try:
        mom_base = np.load(mom_path)
        dad_base = np.load(dad_path)
        me_base = np.load(me_path)
        
        sr = 22050
        space_unit = np.zeros(int(sr * 0.3)) # 띄어쓰기는 0.3초 평화로운 무음
        decomposed_data = decompose_text(input_text)
        final_audio = []

        for item in decomposed_data:
            if item == "SPACE":
                final_audio.append(space_unit)
            else:
                # 1. 초성 (엄마 목소리) -> 근음 (도, 0 반음)
                for _ in item['초']:
                    pitched = change_pitch(mom_base, 0, sr)
                    final_audio.append(pitched)
                
                # 2. 중성 (아빠 목소리) -> 따뜻한 3도 화음 (미, +4 반음)
                for _ in item['중']:
                    pitched = change_pitch(dad_base, 4, sr)
                    final_audio.append(pitched)
                
                # 3. 종성 (원재 목소리) -> 풍성한 5도 화음 (솔, +7 반음)
                for _ in item['종']:
                    pitched = change_pitch(me_base, 7, sr)
                    final_audio.append(pitched)
        
        if final_audio:
            # 모든 소리 조각을 안전하게 일렬로 결합
            combined = np.concatenate(final_audio)
            
            # 볼륨 안정화 처리
            if np.max(np.abs(combined)) > 0:
                combined = combined / np.max(np.abs(combined)) * 0.85
                
            out_bio = io.BytesIO()
            sf.write(out_bio, combined, sr, format='WAV')
            st.audio(out_bio.getvalue())
            st.success(f"'{input_text}' 연주가 성공적으로 완료되었습니다!")
            
    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
