# CCTV-Style Vlog Recorder

Orin Nano Super 최적화 브이로그 녹화 시스템
- YOLO11 객체 감지
- 얼굴 영역 글리치 블러 효과
- 타임스탬프 & FPS 오버레이

## 주요 기능

1. **실시간 객체 감지**: YOLO11n으로 사람, 물체 등 80개 클래스 감지
2. **얼굴 익명화**: Person 감지 시 얼굴 영역에 글리치 효과 자동 적용
3. **바운딩 박스**: 모든 감지된 객체에 라벨 + 신뢰도 표시
4. **타임스탬프**: 현재 날짜/시간 표시 (좌측 상단)
5. **FPS 카운터**: 실시간 프레임레이트 표시 (우측 상단)
6. **비디오 녹화**: MP4 형식으로 저장

## 글리치 효과 종류

- `rgb_shift`: RGB 채널 시프트 (사이버펑크 느낌)
- `pixelate`: 픽셀화 (모자이크)
- `noise`: 디지털 노이즈
- `scanlines`: 스캔라인 효과 (VHS 느낌)
- `combined`: 모든 효과 조합 (기본값, 가장 강력)

## 사용법

### 1. 기본 실행 (미리보기만, 녹화 안함)

```bash
python3 vlog_recorder.py
```

### 2. 실시간 녹화

```bash
# 자동 파일명 생성 (vlog_20260315_143025.mp4)
python3 vlog_recorder.py --record

# 커스텀 파일명
python3 vlog_recorder.py --output my_vlog.mp4
```

### 3. 글리치 효과 변경

```bash
# RGB 시프트만
python3 vlog_recorder.py --record --glitch rgb_shift

# 픽셀화만
python3 vlog_recorder.py --record --glitch pixelate

# 노이즈만
python3 vlog_recorder.py --record --glitch noise
```

### 4. 다른 카메라 사용

```bash
# USB 카메라 1번
python3 vlog_recorder.py --camera 1 --record

# RealSense 등 특정 카메라
python3 vlog_recorder.py --camera 2 --record
```

### 5. 감지 민감도 조정

```bash
# 낮은 신뢰도 객체도 감지 (더 많이 감지)
python3 vlog_recorder.py --confidence 0.2 --record

# 높은 신뢰도 객체만 감지 (확실한 것만)
python3 vlog_recorder.py --confidence 0.5 --record
```

## 키보드 단축키

- `q`: 종료
- `s`: 스크린샷 저장 (screenshot_YYYYMMDD_HHMMSS.jpg)

## 전체 옵션

```bash
python3 vlog_recorder.py --help
```

### 옵션 설명

- `--camera`: 카메라 장치 ID (기본: 0)
- `--model`: YOLO 모델 (기본: yolo11n.pt)
- `--confidence`: 감지 신뢰도 임계값 (기본: 0.3)
- `--glitch`: 글리치 효과 종류 (기본: combined)
- `--output`: 출력 비디오 파일 경로
- `--record`: 자동 파일명으로 녹화 시작

## 추천 설정

### 실내 CCTV 브이로그 (기본)
```bash
python3 vlog_recorder.py --record --glitch combined --confidence 0.3
```

### 깔끔한 스타일 (RGB 시프트만)
```bash
python3 vlog_recorder.py --record --glitch rgb_shift --confidence 0.35
```

### 레트로 VHS 스타일
```bash
python3 vlog_recorder.py --record --glitch scanlines --confidence 0.3
```

### 강력한 익명화 (픽셀화)
```bash
python3 vlog_recorder.py --record --glitch pixelate --confidence 0.4
```

## 성능 최적화 팁

1. **해상도**: 기본 1280x720, 필요시 스크립트에서 조정
2. **모델 크기**:
   - `yolo11n.pt`: 가장 빠름 (권장)
   - `yolo11s.pt`: 더 정확함 (약간 느림)
3. **TensorRT 변환**: 더 빠른 추론을 위해 TensorRT로 변환 가능

## 문제 해결

### 카메라가 안 열릴 때
```bash
# 사용 가능한 카메라 확인
ls /dev/video*

# 권한 확인
sudo usermod -aG video $USER
```

### FPS가 낮을 때
- 신뢰도 임계값 높이기 (--confidence 0.4)
- 더 작은 모델 사용 (yolo11n.pt)
- 해상도 낮추기 (스크립트 수정)

### 글리치 효과가 약할 때
스크립트의 `GlitchEffect.apply_glitch()` 함수에서 파라미터 조정:
- `offset`: RGB 시프트 강도
- `pixel_size`: 픽셀화 크기
- `intensity`: 노이즈 강도

## 예제

```bash
# 완벽한 CCTV 브이로그 설정
python3 vlog_recorder.py \
    --record \
    --glitch combined \
    --confidence 0.3 \
    --camera 0
```

실행 후 화면에 실시간으로 감지 결과가 표시되며,
사람 얼굴에는 자동으로 글리치 블러가 적용됩니다.

녹화를 종료하려면 `q`를 누르세요.
