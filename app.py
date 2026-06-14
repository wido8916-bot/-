import streamlit as st
import numpy as np
import librosa
import soundfile as sf
from jamo import h2j, j2hcj
import io

# --- 설정 및 데이터 처리 함수 ---

def get_voice_instrument(audio_data, sr, duration=0.15):
    """목소리 파일에서 대표적인 음색 구간을 추출하여 악기 소리로 만듦"""
    # 무음 제거 후 일정한 특징을 가진 구간 추출
    yt, _ = librosa.effects.trim(audio_data)
    # 목소리의 중간 지점에서 0.15초 추출
    start_sample = len(yt) // 2
    end_sample = start_sample + int(sr * duration)
    return yt[start_sample:end_sample]

def decompose_text(text):
    """한글 텍스트를 초성, 중성, 종성 단위로 분해하고 이중자모를 결합형으로 변환"""
    # 이중 자음/모음 분해 사전
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
        
        # 한 글자를 초/중/종성으로 분리
        jamo_list = j2hcj(h2j(char))
        
        syllable = {'초': [], '중': [], '종': []}
        
        # 0: 초성, 1: 중성, 2: 종성 (종성은 없을 수 있음)
        # 실제 jamo 라이브러리는 글자마다 길이를 다르게 주므로 순서대로 매핑
        pos = '초'
        for j in jamo_list:
            # 기본적으로 자음이면 초성/종성, 모음이면 중성
            if j in 'ㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣㅐㅒㅔㅖㅘㅙㅚㅝㅞㅟㅢ':
                pos = '중'
            elif pos == '중': # 이미 중성이 나온 뒤 자음이 나오면 종성
                pos = '종'
            
            # 이중 자모 분해 적용
            decomposed = double_jamo.get(j, j)
            for d in decomposed:
                syllable[pos].append(d)
        
        result.append(syllable)
    return result

# --- UI 구성 ---
st.set_page_config(page_title="가족 목소리 합창 TTS", layout="centered")
st.title("👨‍👩‍👦 가족 목소리 합창 TTS")
st.write("엄마(초성), 아빠(중성), 원재(종성)의 목소리로 글자를 연주합니다.")

# 파일 로드 (제공된 파일 이름 기준)
# 실제 실행 시에는 업로드 버튼을 사용하거나 서버의 파일을 읽습니다.
mom_path = "엄마_voice.m4a"
dad_path = "아빠_voice.m4a"
me_path = "원재_voice.m4a"

input_text = st.text_input("변환할 문장을 입력하세요", "원재 사랑해")

if st.button("소리로 변환하기"):
    try:
        # 오디오 로드 (m4a 대응을 위해 librosa 사용)
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
        st.error(f"오류가 발생했습니다: {e}. 파일이 현재 경로에 있는지 확인해주세요.")
