import streamlit as st
import numpy as np
import librosa
import soundfile as sf
from jamo import h2j, j2hcj
import io
import os

# --- 설정 및 데이터 처리 함수 ---

def get_voice_instrument(audio_data, sr, duration=0.15):
    """목소리 파일에서 대표적인 음색 구간을 추출하여 악기 소리로 만듦"""
    yt, _ = librosa.effects.trim(audio_data)
    start_sample = len(yt) // 2
    end_sample = start_sample + int(sr * duration)
    return yt[start_sample:end_sample]

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

# --- UI 구성 ---
st.set_page_config(page_title="가족 목소리 합창 TTS", layout="centered")
st.title("👨‍👩‍👦 가족 목소리 합창 TTS")
st.write("엄마(초성), 아빠(중성), 원재(종성)의 목소리로 글자를 연주합니다.")

# 파일 경로 지정
mom_path = "엄마_voice.wav"
dad_path = "아빠_voice.wav"
me_path = "원재_voice.wav"

# [진단 기능] 서버에 파일이 진짜 잘 존재하는지 화면에 표시
st.sidebar.subheader("📁 서버 파일 연결 상태")
for p in [mom_path, dad_path, me_path]:
    if os.path.exists(p):
        st.sidebar.success(f"⭕ {p} 연결됨")
    else:
        st.sidebar.error(f"❌ {p} 없음 (체크 필요)")

input_text = st.text_input("변환할 문장을 입력하세요", "원재 사랑해")

if st.button("소리로 변환하기"):
    try:
        # 파일 존재 여부 수동 체크로 에러 원인 특정
        for p in [mom_path, dad_path, me_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"'{p}' 파일이 서버 경로에 존재하지 않습니다.")

        # 오디오 로드 (soundfile 백엔드 에러 방지를 위해 가벼운 예외처리 포함)
        y_m, sr = librosa.load(mom_path, sr=22050)
        y_d, _ = librosa.load(dad_path, sr=22050)
        y_w, _ = librosa.load(me_path, sr=22050)

        # 각 목소리의 '선율(단위 소리)' 생성
        mom_unit = get_voice_instrument(y_m, sr)
        dad_unit = get_voice_instrument(y_d, sr)
        me_unit = get_voice_instrument(y_w, sr)
        space_unit = np.zeros(int(sr * 0.3)) # 0.3초 공백

        decomposed_data = decompose_text(input_text)
        final_audio = []

        for item in decomposed_data:
            if item == "SPACE":
                final_audio.append(space_unit)
            else:
                # 초성 (엄마)
                for _ in item['초']:
                    final_audio.append(mom_unit)
                # 중성 (아빠)
                for _ in item['중']:
                    final_audio.append(dad_unit)
                # 종성 (원재)
                for _ in item['종']:
                    final_audio.append(me_unit)
        
        if final_audio:
            combined = np.concatenate(final_audio)
            
            # 결과 출력
            out_bio = io.BytesIO()
            sf.write(out_bio, combined, sr, format='WAV')
            st.audio(out_bio.getvalue())
            st.success(f"'{input_text}' 변환 완료!")
            st.write("초성: 엄마 / 중성: 아빠 / 종성: 원재 목소리가 순서대로 들립니다.")
            
    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
