# CUDA By Example Project

CUDA 프로그래밍 학습을 위한 예제 프로젝트입니다.

## 프로젝트 구조

```
CUDA_BY_EXAMPLE/
├── src/           # CUDA 소스 파일들 (.cu)
├── include/       # 헤더 파일들 (.h, .cuh)
├── bin/           # 컴파일된 실행 파일들
├── build/         # 빌드 중간 파일들
└── .vscode/       # VS Code 설정 파일들
```

## 빌드 방법

### VS Code에서 빌드
- `Ctrl+Shift+P` → "Tasks: Run Task" → "Build CUDA" 선택
- 또는 `Ctrl+Shift+B`로 기본 빌드 태스크 실행

### 명령줄에서 빌드
```bash
# 개별 파일 컴파일
nvcc -o bin/example src/example.cu -Iinclude

# Makefile 사용 (생성 후)
make
```

## 디버깅

VS Code에서 F5를 눌러 디버깅을 시작할 수 있습니다.

## 프로파일링

```bash
# nvprof 사용
nvprof ./bin/example

# nvvp 사용 (GUI)
nvvp
```

## 요구사항

- NVIDIA GPU
- CUDA Toolkit
- GCC/G++
- nvcc 컴파일러