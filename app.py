import streamlit as st
import numpy as np
import soundfile as sf
from jamo import h2j, j2hcj
import io
import os

def decompose_text(text):
    """한글 텍스트를 초성, 중성, 종성 단위로 분해하고 이중자모를 결합형으로 변환"""
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

st.set_page_config(page_title="가족 목소리 합창 TTS", layout="centered")
st.title("👨‍👩‍👦 가족 목소리 합창 TTS")
st.write("가족 목소리 전체를 분석한 평균 음색 데이터(.npy)를 사용하여 글자를 음악처럼 연주합니다.")

# 추출된 데이터 파일 경로
mom_path = "mom_voice.npy"
dad_path = "dad_voice.npy"
me_path = "me_voice.npy"

# 파일 연결 확인 사이드바
st.sidebar.subheader("📁 음색 데이터 연결 상태")
for p in [mom_path, dad_path, me_path]:
    if os.path.exists(p):
        st.sidebar.success(f"⭕ {p} 준비 완료")
    else:
        st.sidebar.error(f"❌ {p} 파일 없음")

input_text = st.text_input("변환할 문장을 입력하세요", "원재 사랑해")

if st.button("소리로 변환하기"):
    try:
        # 데이터 파일 직접 로드 (매우 빠르고 에러 없음!)
        mom_unit = np.load(mom_path)
        dad_unit = np.load(dad_path)
        me_unit = np.load(me_path)
        
        sr = 22050
        space_unit = np.zeros(int(sr * 0.3)) # 0.3초 공백

        decomposed_data = decompose_text(input_text)
        final_audio = []

        for item in decomposed_data:
            if item == "SPACE":
                final_audio.append(space_unit)
            else:
                for _ in item['초']: final_audio.append(mom_unit)
                for _ in item['중']: final_audio.append(dad_unit)
                for _ in item['종']: final_audio.append(me_unit)
        
        if final_audio:
            combined = np.concatenate(final_audio)
            out_bio = io.BytesIO()
            sf.write(out_bio, combined, sr, format='WAV')
            st.audio(out_bio.getvalue())
            st.success(f"'{input_text}' 변환 완료!")
            st.write("🎶 초성: 엄마 / 중성: 아빠 / 종성: 원재 목소리의 평균 배음이 순서대로 연주되었습니다.")
    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
