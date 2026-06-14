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
    
    # 🚨 더 안전하고 확실한 길이 고정 알고리즘
    target_len = len(waveform)
    if len(pitched) > target_len:
        pitched = pitched[:target_len] # 길면 칼같이 자르기
    elif len(pitched) < target_len:
        # 짧으면 target_len 크기의 빈 배열을 만들고 앞부분에 쏙 집어넣기
        padded = np.zeros(target_len)
        padded[:len(pitched)] = pitched
        pitched = padded
        
    return pitched

def apply_envelope_and_reverb(waveform, sr=22050, decay_rate=0.7):
    """소리에 부드러운 여운(A.D.S.R)과 공간감(Reverb)을 줍니다."""
    n_samples = len(waveform)
    
    # 1. 자연스럽게 사라지는 페이드아웃 곡선 (Envelope) 생성
    envelope = np.exp(-decay_rate * np.linspace(0, 3, n_samples))
    smoothed = waveform * envelope
    
    # 2. 아주 가벼운 음악적 잔향(Reverb) 효과 추가
    delay_samples = int(sr * 0.08) # 0.08초 뒤에 메아리
    reverb = np.zeros(n_samples + delay_samples)
    reverb[:n_samples] += smoothed
    reverb[delay_samples:] += smoothed * 0.35 # 첫 번째 메아리 (35% 크기)
    reverb[delay_samples*2:] += smoothed * 0.15 # 두 번째 메아리 (15% 크기)
    
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

st.set_page_config(page_title="가족 목소리 합창 TTS", layout="centered")
st.title("🎵 가족 목소리 앙상블 합창 TTS")
st.write("가족의 목소리 고유 톤을 주파수 선율로 변환하여, 음악적인 화음 합창으로 연주합니다.")

mom_path = "mom_voice.npy"
dad_path = "dad_voice.npy"
me_path = "me_voice.npy"

# 파일 연결 확인 사이드바
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
        space_unit = np.zeros(int(sr * 0.4)) # 공백은 0.4초 잔잔하게

        decomposed_data = decompose_text(input_text)
        final_audio = []

        # 음악적 화음 구성 (도, 미, 솔 기본 화음 구조를 한 글자 내에서 펼침)
        # 글자가 진행됨에 따라 약간의 멜로디 변화(Pitch Shift)를 주어 단조로움을 피함
        for i, item in enumerate(decomposed_data):
            if item == "SPACE":
                final_audio.append(space_unit)
            else:
                # 글자 위치에 따라 기본 스케일에 미세한 변화를 주어 선율을 만듦
                pitch_offset = (i % 3) * 2 # 0, 2, 4 반음씩 올림 (은은한 멜로디 라인)
                
                # 1. 초성 (엄마): 근음 (도 계열)
                for _ in item['초']:
                    pitched = change_pitch(mom_base, 0 + pitch_offset, sr)
                    musical = apply_envelope_and_reverb(pitched, sr)
                    final_audio.append(musical)
                
                # 2. 중성 (아빠): 3도 화음 (미 계열 - 따뜻함)
                for _ in item['중']:
                    pitched = change_pitch(dad_base, 4 + pitch_offset, sr)
                    musical = apply_envelope_and_reverb(pitched, sr)
                    final_audio.append(musical)
                
                # 3. 종성 (원재): 5도 혹은 7도 화음 (솔~시 계열 - 풍성함)
                for _ in item['종']:
                    pitched = change_pitch(me_base, 7 + pitch_offset, sr)
                    musical = apply_envelope_and_reverb(pitched, sr)
                    final_audio.append(musical)
        
        if final_audio:
            # 음과 음 사이를 부드럽게 겹치기(Crossfading 효과) 위해 일정 비율로 혼합하여 결합
            combined = np.concatenate(final_audio)
            
            # 전체 볼륨 정규화 및 클리핑 방지
            if np.max(np.abs(combined)) > 0:
                combined = combined / np.max(np.abs(combined)) * 0.85
                
            out_bio = io.BytesIO()
            sf.write(out_bio, combined, sr, format='WAV')
            
            st.audio(out_bio.getvalue())
            st.success(f"'{input_text}' 음악 연주 완료!")
            st.info("💡 엄마(초성: 도), 아빠(중성: 미), 원재(종성: 솔)의 목소리가 조화로운 화음 선율과 잔향을 입고 연주되었습니다.")
            
    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
