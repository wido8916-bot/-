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

def generate_glided_timeline(pitch_targets, base_waveform, is_vowel=False, sr=22050):
    """
    목소리 소스 위에서 목표 음정들 사이를 부드러운 곡선(포르타멘토)으로 연결합니다.
    """
    total_samples = len(pitch_targets)
    if total_samples == 0:
        return np.zeros(0)
    
    base_freq = 330.0 if is_vowel else 180.0
    freq_timeline = np.zeros(total_samples)
    current_freq = base_freq * (2 ** (pitch_targets[0] / 12.0))
    
    # 아빠(모음)는 완만하게(0.0015), 엄마(자음)는 상대적으로 빠르게(0.006) 음 이동
    gliding_speed = 0.0015 if is_vowel else 0.006
    
    for t in range(total_samples):
        target_freq = base_freq * (2 ** (pitch_targets[t] / 12.0))
        current_freq += (target_freq - current_freq) * gliding_speed
        freq_timeline[t] = current_freq
        
    dt = 1.0 / sr
    phases = np.cumsum(2 * np.pi * freq_timeline * dt)
    source_signal = np.sin(phases)
    
    if not is_vowel:
        source_signal += 0.2 * np.sin(2 * phases)
        
    extended_base = np.tile(base_waveform, int(np.ceil(total_samples / len(base_waveform))))
    trimmed_base = extended_base[:total_samples]
    
    return trimmed_base * source_signal

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
        
        # 단독 자음 처리용 예외문
        if char in 'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㄲㄸㅃㅆㅉ':
            decomposed = double_jamo.get(char, char)
            syllable = {'초': [d for d in decomposed], '중': [], '종': []}
            result.append(syllable)
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

st.set_page_config(page_title="곡선 음이동 가족 TTS", layout="centered")
st.title("🎼 가족 자/모음 주파수 곡선 융합 TTS")

mom_path = "mom_voice.npy"
dad_path = "dad_voice.npy"
me_path = "me_voice.npy"

st.sidebar.subheader("📁 음색 데이터 연결 상태")
for p in [mom_path, dad_path, me_path]:
    if os.path.exists(p):
        st.sidebar.success(f"⭕ {p} 연결됨")
    else:
        st.sidebar.error(f"❌ {p} 없음")

input_text = st.text_input("연주할 문장 또는 자음을 입력하세요", "ㅋㅋㅋ")

if st.button("곡선 선율 연주하기"):
    try:
        mom_base = np.load(mom_path)
        dad_base = np.load(dad_path)
        me_base = np.load(me_path)
        
        sr = 22050
        base_dur = 0.5
        space_dur = 0.3
        
        decomposed_data = decompose_text(input_text)
        
        # ─── 2. 전체 타임라인 구성 ───
        timeline = []
        current_time = 0.0
        for item in decomposed_data:
            if item == "SPACE":
                current_time += space_dur
            else:
                timeline.append({'start': current_time, 'data': item})
                current_time += base_dur
                
        total_samples = int(sr * current_time)
        
        mom_targets = np.zeros(total_samples)
        dad_targets = np.zeros(total_samples)
        dad_active = np.zeros(total_samples) # 모음 존재 여부 마스킹
        
        for t_item in timeline:
            start_s = int(sr * t_item['start'])
            end_s = start_s + int(sr * base_dur)
            item = t_item['data']
            
            if item['초'] and item['초'][0] in CONSONANT_SPECTRUM:
                mom_targets[start_s:end_s] = CONSONANT_SPECTRUM[item['초'][0]]
                
            if item['중'] and item['중'][0] in VOWEL_SPECTRUM:
                dad_targets[start_s:end_s] = VOWEL_SPECTRUM[item['중'][0]]
                dad_active[start_s:end_s] = 1.0 # 모음이 있는 글자만 1.0 활성화
                
        # ─── 3. 엔진 가동 및 출력 계산 ───
        mom_signal = generate_glided_timeline(mom_targets, mom_base, is_vowel=False, sr=sr)
        dad_signal = generate_glided_timeline(dad_targets, dad_base, is_vowel=True, sr=sr)
        
        # 모음이 없는 자음 독주 구간에서는 아빠 소리를 완전히 소거
        dad_signal = dad_signal * dad_active
        
        # 종성(원재 - 드럼): 쿵. 쿵. 쿵. 확실하게 끊어치는 스탁카토 렌더링
        me_signal = np.zeros(total_samples)
        for t_item in timeline:
            item = t_item['data']
            # 모음이 있고 종성(받침)이 존재할 때만 드럼 작동
            if item['중'] and item['종']:
                start_s = int(sr * t_item['start'])
                
                # 0.5초 중 오직 0.15초만 타격하고 뒤는 무음으로 칼같이 커트
                drum_dur = 0.15 
                drum_samples = int(sr * drum_dur)
                
                t_local = np.linspace(0, drum_dur, drum_samples, endpoint=False)
                low_thump = np.sin(2 * np.pi * 60 * t_local) * np.exp(-40 * t_local) # 감쇄 속도를 높임
                
                extended_me = np.tile(me_base, int(np.ceil(drum_samples / len(me_base))))
                noise_burst = extended_me[:drum_samples] * np.exp(-30 * t_local)
                
                # 0.15초 타격 사운드 윈도우 페이딩
                drum_wave = (low_thump * 0.9) + (noise_burst * 0.2)
                fade = int(sr * 0.02)
                if len(drum_wave) > fade:
                    drum_wave[-fade:] *= np.linspace(1, 0, fade)
                
                me_signal[start_s:start_s+drum_samples] = drum_wave
        
        # ─── 4. 최종 마스터 믹싱 ───
        master_signal = (mom_signal * 0.6) + (dad_signal * 1.0) + (me_signal * 1.4)
        
        if len(master_signal) > int(sr * 0.1):
            fade_len = int(sr * 0.05)
            master_signal[:fade_len] *= np.linspace(0, 1, fade_len)
            master_signal[-fade_len:] *= np.linspace(1, 0, fade_len)
            
        if np.max(np.abs(master_signal)) > 0:
            master_signal = master_signal / np.max(np.abs(master_signal)) * 0.85
            
        out_bio = io.BytesIO()
        sf.write(out_bio, master_signal, sr, format='WAV')
        st.audio(out_bio.getvalue())
        st.success("🎨 종성 타격감 강화 및 자음 독주 시스템 적용 완료!")
        
    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
