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

def generate_pure_tone(base_waveform, semitones, duration, is_vowel=False, is_drum=False, sr=22050):
    """
    지정된 duration만큼 끊김 없이 통째로 하나의 파형을 생성하는 함수.
    시작과 끝에만 부드러운 페이딩을 주어 연속음일 때 중간이 끊기지 않게 합니다.
    """
    n_samples = int(sr * duration)
    if len(base_waveform) == 0:
        return np.zeros(n_samples)
    
    extended = np.tile(base_waveform, int(np.ceil(n_samples / len(base_waveform))))
    trimmed = extended[:n_samples]
    
    if is_drum:
        # 종성 (원재 - 드럼): 타격음 성격 유지 (쿵- 소리 후 목소리 노이즈 감쇄)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        low_thump = np.sin(2 * np.pi * 60 * t) * np.exp(-30 * t)
        noise_burst = trimmed * np.exp(-25 * t)
        return (low_thump * 0.7) + (noise_burst * 0.4)
        
    # 선율 악기 (초성/중성)
    base_freq = 440.0 if is_vowel else 220.0
    freq = base_freq * (2 ** (semitones / 12.0))
    t = np.linspace(0, duration, n_samples, endpoint=False)
    
    if is_vowel:
        source = np.sin(2 * np.pi * freq * t)
    else:
        # 초성 자음: 맑은 현악기(하프)풍 지속음 선율
        source = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * 2 * np.pi * freq * t)
        
    # 전체 연주 시간의 처음과 끝에만 페이드 인/아웃 적용 (중간 뚝 끊김 방지)
    envelope = np.ones(n_samples)
    fade_in = int(sr * 0.08)
    fade_out = int(sr * 0.1)
    envelope[:fade_in] = np.linspace(0, 1, fade_in)
    envelope[-fade_out:] = np.linspace(1, 0, fade_out)
    
    return trimmed * source * envelope

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

st.set_page_config(page_title="고급 연속음 통합 가족 TTS", layout="centered")
st.title("🎼 가족 자/모음 동시 연속 통합 앙상블 TTS")
st.write("초성 자음이나 중성 모음이 연속되면 소리가 끊기지 않고 롱톤으로 이어지며, 종성은 드럼 톤으로 연주됩니다.")

mom_path = "mom_voice.npy"
dad_path = "dad_voice.npy"
me_path = "me_voice.npy"

st.sidebar.subheader("📁 음색 데이터 연결 상태")
for p in [mom_path, dad_path, me_path]:
    if os.path.exists(p):
        st.sidebar.success(f"⭕ {p} 연결됨")
    else:
        st.sidebar.error(f"❌ {p} 없음")

input_text = st.text_input("연주할 문장을 입력하세요", "나는")

if st.button("오케스트라 연주하기"):
    try:
        mom_base = np.load(mom_path)
        dad_base = np.load(dad_path)
        me_base = np.load(me_path)
        
        sr = 22050
        base_dur = 0.5
        space_dur = 0.3
        
        decomposed_data = decompose_text(input_text)
        
        # ─── 2. 텍스트 연속성 분석 및 독립 렌더링 ───
        # 전체 오디오 타임라인 길이를 먼저 계산
        total_samples = 0
        timeline_map = [] # 각 요소의 시작 시간과 타입을 기록
        
        current_time = 0.0
        for item in decomposed_data:
            if item == "SPACE":
                current_time += space_dur
            else:
                timeline_map.append({'type': 'char', 'start': current_time, 'data': item})
                current_time += base_dur
        
        total_audio_len = int(sr * current_time)
        master_signal = np.zeros(total_audio_len)
        
        # 연속음을 묶어서 한 번에 길게 그리기 위한 탐색
        # [초성 자음 연속성 처리]
        visited_cho = [False] * len(timeline_map)
        for i in range(len(timeline_map)):
            if visited_cho[i]: continue
            
            start_idx = i
            end_idx = i
            current_cho = timeline_map[i]['data']['초']
            
            # 다음 글자들의 초성이 같은지 전수 조사
            while end_idx + 1 < len(timeline_map) and timeline_map[end_idx + 1]['data']['초'] == current_cho and current_cho:
                end_idx += 1
                
            # 연속된 덩어리 길이 계산
            for k in range(start_idx, end_idx + 1): visited_cho[k] = True
            
            if current_cho:
                start_time = timeline_map[start_idx]['start']
                duration = (end_idx - start_idx + 1) * base_dur
                
                # 통째로 긴 자음 선율 생성
                consonant_signal = np.zeros(int(sr * duration))
                for c in current_cho:
                    if c in CONSONANT_SPECTRUM:
                        semi = CONSONANT_SPECTRUM[c]
                        consonant_signal += generate_pure_tone(mom_base, semi, duration, is_vowel=False, sr=sr)
                
                start_sample = int(sr * start_time)
                master_signal[start_sample:start_sample+len(consonant_signal)] += consonant_signal * 0.7

        # [중성 모음 연속성 처리]
        visited_jung = [False] * len(timeline_map)
        for i in range(len(timeline_map)):
            if visited_jung[i]: continue
            
            start_idx = i
            end_idx = i
            current_jung = timeline_map[i]['data']['중']
            
            while end_idx + 1 < len(timeline_map) and timeline_map[end_idx + 1]['data']['중'] == current_jung and current_jung:
                end_idx += 1
                
            for k in range(start_idx, end_idx + 1): visited_jung[k] = True
            
            if current_jung:
                start_time = timeline_map[start_idx]['start']
                duration = (end_idx - start_idx + 1) * base_dur
                
                vowel_signal = np.zeros(int(sr * duration))
                for v in current_jung:
                    if v in VOWEL_SPECTRUM:
                        semi = VOWEL_SPECTRUM[v]
                        vowel_signal += generate_pure_tone(dad_base, semi, duration, is_vowel=True, sr=sr)
                
                start_sample = int(sr * start_time)
                master_signal[start_sample:start_sample+len(vowel_signal)] += vowel_signal * 1.0

        # [종성 자음 처리 - 연속성 제외, 매 글자 독립 드럼 타격]
        for i in range(len(timeline_map)):
            item = timeline_map[i]['data']
            if item['종']:
                start_time = timeline_map[i]['start']
                # 종성은 0.5초 공간의 뒷부분에 배치되거나 0.5초 전체에 타격 효과를 줌
                drum_signal = np.zeros(int(sr * base_dur))
                for _ in item['종']:
                    drum_signal += generate_pure_tone(me_base, 0, base_dur, is_vowel=False, is_drum=True, sr=sr)
                
                start_sample = int(sr * start_time)
                master_signal[start_sample:start_sample+len(drum_signal)] += drum_signal * 1.2

        # 최종 마스터링 및 출력
        if np.max(np.abs(master_signal)) > 0:
            master_signal = master_signal / np.max(np.abs(master_signal)) * 0.8
            
        out_bio = io.BytesIO()
        sf.write(out_bio, master_signal, sr, format='WAV')
        st.audio(out_bio.getvalue())
        st.success(f"🎵 '{input_text}' 융합 오케스트라 연주 완료!")
        
    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
