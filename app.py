import streamlit as st
import numpy as np
import soundfile as sf
from jamo import h2j, j2hcj
import io
import os

# 1. 자음/모음 스펙트럼 매핑 (반음 단위)
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
    목소리 소스 위에서 목표 음정들(pitch_targets) 사이를 
    부드러운 곡선(포르타멘토)을 그리며 이동하는 파형을 생성합니다.
    """
    # 전체 샘플 수 계산
    total_samples = len(pitch_targets)
    if total_samples == 0:
        return np.zeros(0)
    
    # 기본 음역대 축 설정
    base_freq = 330.0 if is_vowel else 180.0
    
    # 주파수 곡선(타임라인) 생성
    freq_timeline = np.zeros(total_samples)
    current_freq = base_freq * (2 ** (pitch_targets[0] / 12.0))
    
    # ─── 곡선의 완만함(Glide Smoothness) 설정 ───
    # 아빠(모음)는 훨씬 완만하게(0.0015), 엄마(자음)는 상대적으로 빠르게(0.006) 음이 이동함
    gliding_speed = 0.0015 if is_vowel else 0.006
    
    for t in range(total_samples):
        target_freq = base_freq * (2 ** (pitch_targets[t] / 12.0))
        # 현재 주파수가 목표 주파수를 향해 곡선을 그리며 서서히 접근 (지수 감쇄 전이)
        current_freq += (target_freq - current_freq) * gliding_speed
        freq_timeline[t] = current_freq
        
    # 주파수 타임라인을 기반으로 위상(Phase) 누적 연산
    dt = 1.0 / sr
    phases = np.cumsum(2 * np.pi * freq_timeline * dt)
    source_signal = np.sin(phases)
    
    if not is_vowel:
        # 자음은 배음을 살짝 얹어서 명확하게 분리
        source_signal += 0.2 * np.sin(2 * phases)
        
    # 3,307 크기의 오리지널 엑기스 목소리 소스를 전체 길이에 맞춰 루프 배치 후 합성
    extended_base = np.tile(base_waveform, int(np.ceil(total_samples / len(base_waveform))))
    trimmed_base = extended_base[:total_samples]
    
    # 소스 융합
    glided_wave = trimmed_base * source_signal
    
    return glided_wave

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

st.set_page_config(page_title="곡선 음이동 가족 TTS", layout="centered")
st.title("🎼 가족 자/모음 주파수 곡선 융합 TTS")
st.write("각 자/모음 스펙트럼 기준에 맞춰 소리가 생성되며, 음이 변할 때 아빠의 목소리는 더욱 완만한 곡선을 그리며 미끄러지듯 이동합니다.")

mom_path = "mom_voice.npy"
dad_path = "dad_voice.npy"
me_path = "me_voice.npy"

st.sidebar.subheader("📁 음색 데이터 연결 상태")
for p in [mom_path, dad_path, me_path]:
    if os.path.exists(p):
        st.sidebar.success(f"⭕ {p} 연결됨")
    else:
        st.sidebar.error(f"❌ {p} 없음")

input_text = st.text_input("연주할 문장을 입력하세요", "나비")

if st.button("곡선 선율 연주하기"):
    try:
        mom_base = np.load(mom_path)
        dad_base = np.load(dad_path)
        me_base = np.load(me_path)
        
        sr = 22050
        base_dur = 0.5
        space_dur = 0.3
        
        decomposed_data = decompose_text(input_text)
        
        # ─── 2. 전체 타임라인 맵 구성 ───
        timeline = []
        current_time = 0.0
        for item in decomposed_data:
            if item == "SPACE":
                current_time += space_dur
            else:
                timeline.append({'start': current_time, 'data': item})
                current_time += base_dur
                
        total_samples = int(sr * current_time)
        
        # 주파수 타겟 배열 초기화 (초기값은 각 영역의 첫 타겟값으로 채움)
        mom_targets = np.zeros(total_samples)
        dad_targets = np.zeros(total_samples)
        me_targets = np.zeros(total_samples)
        me_active = np.zeros(total_samples) # 종성 유무 체크용
        
        # 타임라인을 돌며 각 샘플 시점의 '목표 음(반음 수)'을 맵핑
        for t_item in timeline:
            start_s = int(sr * t_item['start'])
            end_s = start_s + int(sr * base_dur)
            item = t_item['data']
            
            # 엄마 초성 목표음 설정
            if item['초'] and item['초'][0] in CONSONANT_SPECTRUM:
                mom_targets[start_s:end_s] = CONSONANT_SPECTRUM[item['초'][0]]
                
            # 아빠 중성 목표음 설정
            if item['중'] and item['중'][0] in VOWEL_SPECTRUM:
                dad_targets[start_s:end_s] = VOWEL_SPECTRUM[item['중'][0]]
                
            # 원재 종성 설정 (드럼 타격 영역 활성화)
            if item['종']:
                me_active[start_s:end_s] = 1.0
                
        # ─── 3. 포르타멘토(Glide) 엔진 적용 및 사운드 생성 ───
        # 엄마 선율 생성 (자음 규칙 곡선)
        mom_signal = generate_glided_timeline(mom_targets, mom_base, is_vowel=False, sr=sr)
        
        # 아빠 선율 생성 (★모음 규칙, 완만하고 중후한 곡선)
        dad_signal = generate_glided_timeline(dad_targets, dad_base, is_vowel=True, sr=sr)
        
        # 원재 종성 처리 (연속음 제외, 종성 위치에서만 묵직한 드럼 타격 연출)
        me_signal = np.zeros(total_samples)
        t_axis = np.linspace(0, current_time, total_samples, endpoint=False)
        for t_item in timeline:
            item = t_item['data']
            if item['종']:
                start_s = int(sr * t_item['start'])
                end_s = start_s + int(sr * base_dur)
                dur_samples = end_s - start_s
                
                # 드럼 충격파 생성
                t_local = np.linspace(0, base_dur, dur_samples, endpoint=False)
                low_thump = np.sin(2 * np.pi * 55 * t_local) * np.exp(-35 * t_local)
                
                # 원재 3307 npy 소스 결합
                extended_me = np.tile(me_base, int(np.ceil(dur_samples / len(me_base))))
                noise_burst = extended_me[:dur_samples] * np.exp(-20 * t_local)
                
                me_signal[start_s:end_s] = (low_thump * 0.8) + (noise_burst * 0.3)
        
        # ─── 4. 최종 오케스트라 믹싱 ───
        master_signal = (mom_signal * 0.6) + (dad_signal * 1.0) + (me_signal * 1.2)
        
        # 전체 시작과 끝 볼륨 정리 (클릭 노이즈 방지)
        if len(master_signal) > int(sr * 0.1):
            fade_len = int(sr * 0.05)
            master_signal[:fade_len] *= np.linspace(0, 1, fade_len)
            master_signal[-fade_len:] *= np.linspace(1, 0, fade_len)
            
        # 마스터링 정규화
        if np.max(np.abs(master_signal)) > 0:
            master_signal = master_signal / np.max(np.abs(master_signal)) * 0.85
            
        out_bio = io.BytesIO()
        sf.write(out_bio, master_signal, sr, format='WAV')
        st.audio(out_bio.getvalue())
        st.success("🎨 자/모음 개별 스펙트럼 및 아빠의 완만한 글라이드 연주 완료!")
        
    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
