# GitHub 업로드 안내

이 폴더는 원자료·캐시·학습 모델·대용량 개별 예측을 제외한 경량 저장소입니다.

## 포함된 핵심 자료

- `README.md`: 프로젝트 개요와 실행법
- `research/`: 주 실험 파이프라인
- `research_extensions/`: 후보 기간 민감도와 오류 분석
- `research_confirmation/`: Super User 사전 고정 추가 확인
- `research_ranking/`: GPU RankNet 탐색
- `tests/`: 자동 검사
- `results/`: 요약 표·그림·manifest
- `report/`: 보수적 최종 보고서

## 업로드 명령

PowerShell에서 이 폴더를 연 뒤 다음 순서로 실행합니다.

```powershell
git status
git add .
git commit -m "Add temporal user reconnection prediction project"
git branch -M main
git remote add origin 본인의_GitHub_저장소_URL
git push -u origin main
```

이미 `origin`이 등록되어 있다면 `git remote add origin`은 실행하지 않습니다. 원자료와 모델은 `.gitignore`로 제외되어 있으며, 공개 SNAP 데이터는 코드 실행 과정에서 다시 받을 수 있습니다.

## 실행 확인

```powershell
python -m pip install -r requirements-research.txt
python -m pytest -q
.\start_app.ps1
```

보고서 재생성은 다음과 같습니다.

```powershell
python build_team_report.py
```
