import streamlit as st
import numpy as np
import soundfile as sf
from jamo import h2j, j2hcj
import io
import os

def generate_musical_tone(base_waveform, semitones, duration=0.5, sr=22050):
    """
    목소리의 특징을 유지하면서 서걱거림을 없애고, 
    지정된 음역대의 부드러운 악기 소리(멜로디)로 재가공하는 함수
    """
    n_samples = int(sr * duration)
    
    # 1. 원본 데이터 크기 맞추기
    if len(base_waveform) == 0:
        return np.zeros(n_samples)
    
    # 루프를 돌려 0.5초 길이로 맞춤
    extended = np.tile(base_waveform, int(np.ceil(n_samples / len(base_waveform))))
    trimmed = extended[:n_samples]
    
    # 2. 서걱거리는 노이즈를 제어하고 주파수 선율(Melody)을 입히기 위한 음고(Pitch) 계산
    # 미디(MIDI) 음고 기준으로 주파수 변환 (기본음 A4 = 440Hz 기준)
    base_freq = 440.0
    freq = base_freq * (2 ** (semitones / 12.0))
    
    t = np.linspace(0, duration, n_samples, endpoint=False)
    sine_wave = np.sin(2 * np.pi * freq * t)
    
    # 3. 목소리 고유의 배음 톤(Envelope)과 부드러운 사인파 멜로디를 융합 (서걱거림 완화)
    # 저주파 필터 효과를 주어 부드러운 악기 소리처럼 만듦
    smoothed_voice = trimmed * sine_wave
    
    # 4. 글자 시작과 끝에 부드러운 페이드 인/아웃을 주어 툭툭 끊기는 느낌 제거
    fade_len = int(sr * 0.05)  # 0.05초 페이드
    window = np.ones(n_samples)
    window[:fade_len] = np.linspace(0, 1, fade_len)
    window[-fade_len:] = np.linspace(1, 0, fade_len)
    
    return smoothed_voice * window

def decompose_text(text):
    double_jamo = {
        'ㄲ': 'ㄱㄱ', 'ㄸ': 'ㄷㄷ', 'ㅃ': 'ㅂㅂ', 'ㅆ': 'ㅅㅅ', 'ㅉ': 'ㅈㅈ',
        'ㄳ': 'ㄱㅅ', '健全': 'ㄴㅈ', 'ㄶ': 'ㄴㅎ', 'ㄺ': 'ㄹㄱ', 'ㄻ': 'ㄹㅁ', 'ㄼ': 'ㄹㅂ', 'ㄽ': 'ㄹㅅ', 'ㄾ': 'ㄹㅌ', 'ㄿ': 'ㄹㅍ', 'ㅀ': 'ㄹㅎ', 'ㅄ': 'ㅂㅅ',
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

st.set_page_config(page_title="가족 목소리 앙상블 합창 TTS", layout="centered")
st.title("🎵 가족 목소리 앙상블 합창 TTS")
st.write("초성(엄마), 중성(아빠), 종성(원재)의 목소리가 주파수 멜로디로 변환되어 한 글자 안에서 동시에 아름다운 화음으로 울립니다.")

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
        char_duration = 0.5  # 한 글자당 0.5초 고정
        space_duration = 0.3 # 띄어쓰기 0.3초 고정
        
        space_unit = np.zeros(int(sr * space_duration))
        decomposed_data = decompose_text(input_text)
        final_audio = []

        for i, item in enumerate(decomposed_data):
            if item == "SPACE":
                final_audio.append(space_unit)
            else:
                # 글자가 진행됨에 따라 은은하게 변화하는 멜로디 오프셋 (단조로움 방지)
                melody_shift = (i % 4) * 2  # 0, 2, 4, 6 반음씩 변형
                
                # 한 글자 공간(0.5초)을 생성하고 여기에 목소리들을 동시에 중첩(화음)시킵니다.
                char_signal = np.zeros(int(sr * char_duration))
                layers_count = 0
                
                # 1. 초성 (엄마) -> 근음 (도 계열, 0 반음)
                if item['초']:
                    mom_tone = generate_musical_tone(mom_base, 0 + melody_shift, char_duration, sr)
                    char_signal += mom_tone
                    layers_count += 1
                
                # 2. 중성 (아빠) -> 따뜻한 3도 화음 (미 계열, +4 반음)
                if item['중']:
                    dad_tone = generate_musical_tone(dad_base, 4 + melody_shift, char_duration, sr)
                    char_signal += dad_tone
                    layers_count += 1
                
                # 3. 종성 (원재) -> 풍성한 5도 화음 (솔 계열, +7 반음)
                if item['종']:
                    me_tone = generate_musical_tone(me_base, 7 + melody_shift, char_duration, sr)
                    char_signal += me_tone
                    layers_count += 1
                
                # 소리가 겹쳐서 깨지는 것을 방지하기 위해 정규화
                if layers_count > 0:
                    char_signal /= layers_count
                    
                final_audio.append(char_signal)
        
        if final_audio:
            combined = np.concatenate(final_audio)
            
            # 최종 볼륨 마스터링 및 안정화
            if np.max(np.abs(combined)) > 0:
                combined = combined / np.max(np.abs(combined)) * 0.8
                
            out_bio = io.BytesIO()
            sf.write(out_bio, combined, sr, format='WAV')
            st.audio(out_bio.getvalue())
            st.success(f"🎵 '{input_text}' 0.5초 화음 합창 연주 완료!")
            st.info("💡 각 글자마다 세 사람의 음색이 동시에 중첩되어 0.5초 동안 울리며, 띄어쓰기는 0.3초 동안 쉬어갑니다.")
            
    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
