import streamlit as st
import numpy as np
import soundfile as sf
from jamo import h2j, j2hcj
import io
import os

def generate_instrument_tone(base_waveform, instrument_type, semitones, duration=0.5, sr=22050):
    """
    목소리 데이터를 기반으로 완전히 다른 3가지 악기(하프, 바순, 드럼)의 특성을 합성하는 함수
    """
    n_samples = int(sr * duration)
    if len(base_waveform) == 0:
        return np.zeros(n_samples)
    
    # 길이 맞추기 (반복 배치)
    extended = np.tile(base_waveform, int(np.ceil(n_samples / len(base_waveform))))
    trimmed = extended[:n_samples]
    
    # 주파수 설정 (음역대)
    base_freq = 440.0
    freq = base_freq * (2 ** (semitones / 12.0))
    t = np.linspace(0, duration, n_samples, endpoint=False)
    
    if instrument_type == "harp":
        # 🎻 하프: 맑은 사인파 + 칼같이 튕겼다가 빠르게 사라지는 포물선형 감쇄 (Pluck 효과)
        source = np.sin(2 * np.pi * freq * t)
        envelope = np.exp(-12 * t) # 매우 빠른 감쇄
        # 현악기의 찰랑거리는 고음역대 배음 추가
        source += 0.2 * np.sin(2 * 2 * np.pi * freq * t) * np.exp(-15 * t)
        signal = trimmed * source * envelope

    elif instrument_type == "bassoon":
        # 🎷 바순: 풍성한 목관 울림을 위해 배음(Harmonics)이 쌓인 삼각파/톱니파 조합 + 부드러운 곡선형 울림
        source = 0.6 * np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(3 * np.pi * freq * t)
        # 관악기 특유의 부드러운 불어넣기(Fade-in)와 잔잔한 유지
        envelope = np.ones(n_samples)
        attack_len = int(sr * 0.1)
        decay_len = int(sr * 0.15)
        envelope[:attack_len] = np.sin(np.linspace(0, np.pi/2, attack_len))
        envelope[-decay_len:] = np.cos(np.linspace(0, np.pi/2, decay_len))
        signal = trimmed * source * envelope

    elif instrument_type == "drum":
        # 🥁 드럼: 불규칙한 타격 잡음(노이즈) + 아주 강한 저음 타격(Thump)의 충격파
        # 종성 목소리의 서걱거림을 타악기의 스네어/킥 드럼 질감으로 치환
        low_thump = np.sin(2 * np.pi * 60 * t) * np.exp(-30 * t) # 60Hz 쿵 소리
        noise_burst = trimmed * np.exp(-25 * t) # 목소리 톤을 스네어 브러쉬 느낌으로 감쇄
        signal = (low_thump * 0.7) + (noise_burst * 0.4)
        
    # 최종 글자 경계면 클릭 노이즈 방지 페이딩
    fade = int(sr * 0.02)
    window = np.ones(n_samples)
    window[:fade] = np.linspace(0, 1, fade)
    window[-fade:] = np.linspace(1, 0, fade)
    
    return signal * window

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

st.set_page_config(page_title="가족 오케스트라 앙상블 TTS", layout="centered")
st.title("🎼 가족 목소리 오케스트라 합창 TTS")
st.write("초성(엄마-하프), 중성(아빠-바순), 종성(원재-드럼)이 결합되어 음악적인 소리를 만들어냅니다.")

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
        char_duration = 0.5
        space_duration = 0.3
        
        space_unit = np.zeros(int(sr * space_duration))
        decomposed_data = decompose_text(input_text)
        final_audio = []

        for i, item in enumerate(decomposed_data):
            if item == "SPACE":
                final_audio.append(space_unit)
            else:
                # 글자가 갈수록 다채로워지도록 은은한 스케일 오프셋 부여
                melody_shift = (i % 3) * 2 
                char_signal = np.zeros(int(sr * char_duration))
                
                # 1. 초성 (엄마) -> 하프 (부드럽고 투명하게 튕기는 높은 도, +12 반음)
                if item['초']:
                    mom_tone = generate_instrument_tone(mom_base, "harp", 12 + melody_shift, char_duration, sr)
                    char_signal += mom_tone * 0.9
                
                # 2. 중성 (아빠) -> 바순 (포근하고 묵직한 중간 미, +4 반음)
                if item['중']:
                    dad_tone = generate_instrument_tone(dad_base, "bassoon", 4 + melody_shift, char_duration, sr)
                    char_signal += dad_tone * 1.1
                
                # 3. 종성 (원재) -> 드럼 (받침이 있을 때만 하단에서 쿵-탁 치고 빠짐)
                if item['종']:
                    me_tone = generate_instrument_tone(me_base, "drum", 0, char_duration, sr)
                    char_signal += me_tone * 1.3
                
                final_audio.append(char_signal)
        
        if final_audio:
            combined = np.concatenate(final_audio)
            
            # 클리핑 방지 정규화
            if np.max(np.abs(combined)) > 0:
                combined = combined / np.max(np.abs(combined)) * 0.8
                
            out_bio = io.BytesIO()
            sf.write(out_bio, combined, sr, format='WAV')
            st.audio(out_bio.getvalue())
            st.success(f"🎨 '{input_text}' 오케스트라 연주 완료!")
            
    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
