# 프로젝트 문제점 및 해결 방안

> 코드 분석 중 발견된 문제점 목록. 전체 분석 완료 후 순차 수정 예정.

---

## 1. 2025시즌 데이터 누락

**문제**
`collect.py`의 `collect_seasons(start=2020, end=2024)`가 하드코딩되어 있어 2025시즌 데이터가 없음. 현재 2026년이므로 학습 데이터가 1~2시즌 부족한 상태.

**해결**
`end` 파라미터를 `date.today().year - 1`로 동적으로 설정하여 항상 직전 시즌까지 수집.

```python
from datetime import date
collect_seasons(start=2021, end=date.today().year - 1)
```

---

## 2. 2020시즌 데이터 품질 문제

**문제**
2020시즌은 COVID로 60경기 단축 시즌. 일반 시즌(162경기)과 패턴이 달라 모델 학습에 노이즈가 될 수 있음.

**해결**
수집 시작 연도를 2021로 변경. (#1 해결과 동시 적용)

---

## 3. GitHub Actions에서 매일 전체 시즌 재수집 (비효율)

**문제**
`retrain.yml`이 매일 실행될 때 2020~2024 전체를 다시 수집함. 시간 낭비 + API 부하.

**해결**
초기 수집(2021~직전 시즌)과 일별 증분 수집(당해 시즌)을 분리.

- 초기 수집: 수동 1회 실행 또는 별도 workflow
- 매일 배치: 현재 시즌만 수집 (`collect_seasons(start=current_year, end=current_year)`)

---

## 4. Airflow DAG와 GitHub Actions 간 수집 범위 불일치

**문제**
- Airflow DAG: 현재 연도만 수집 (`current_year`)
- GitHub Actions: 2020~2024 하드코딩
두 파이프라인이 서로 다른 범위를 수집하여 일관성 없음.

**해결**
두 곳 모두 동일한 로직 사용. `collect_seasons`에 명확한 "초기 모드 / 증분 모드" 파라미터 추가하거나, 공통 설정값으로 관리.

---

## 5. GitHub Actions 실행 후 모델이 사라짐 (저장 미완성)

**문제**
GitHub Actions는 실행 후 환경이 초기화됨. 학습된 모델(`mlruns/`)과 데이터(`data/`)가 모두 사라져 재학습 의미가 없음.

**해결 (개인 프로젝트 기준 현실적인 방법)**
MLflow 모델을 GitHub Actions Artifact로 업로드하여 보존.

```yaml
# retrain.yml에 추가
- name: Upload MLflow artifacts
  uses: actions/upload-artifact@v4
  with:
    name: mlruns
    path: mlruns/
```

---

## 6. 팀 시즌 통계(ERA/OPS)가 매일 갱신되지 않음

**문제**
현재 배치는 경기 결과만 매일 업데이트하고, 팀 통계(ERA/OPS)는 초기 수집 시 값에서 변하지 않음. 시즌 중 예측 시 오래된 통계가 사용됨.

**해결**
`collect_task`에서 경기 결과와 함께 현재 시즌 팀 통계도 매일 갱신.
`fetch_team_season_stats(current_year)` 호출을 배치에 추가하고 `team_stats.parquet`을 덮어쓰면 됨. 이미 함수가 구현되어 있어 추가 코드 최소화.
시즌 초 통계 불안정 문제는 현재 6월 기준으로 60경기 이상 누적되어 있어 당장은 보정 불필요.

---

## 7. `home_advantage` 피처가 실질적으로 무의미

**문제**
`home_advantage = 1.0`으로 모든 경기에서 동일한 값. 모델 입력으로 들어가지만 어떤 경기도 구별하지 못함.

**해결**
제거. (추후 팀별 홈/원정 승률 차이로 교체 고려)

---

## 8. `_add_rolling_win_rate()`에 사용되지 않는 데드 코드 존재

**문제**
`features.py` 21~38행의 첫 번째 `for` 루프 (`home_rates`, `away_rates`, `result_home`, `result_away`)는 실행되지만 결과를 아무데도 사용하지 않음. 코드 작성 중 남겨진 미완성 흔적.

**해결**
첫 번째 루프와 관련 변수 선언 전체 제거.

---

## 9. `import`가 함수 중간에 위치

**문제**
`features.py` 41행: `from collections import deque`가 함수 내부 중간에 선언됨. Python에서는 동작하지만 관례에 어긋나며 가독성을 해침.

**해결**
파일 상단 임포트 블록으로 이동.

---

## 10. FEATURE_COLS가 세 파일에 중복 선언

**문제**
`features.py`, `train.py`, `test_features.py` 세 곳에 동일한 `FEATURE_COLS` 리스트가 각각 선언되어 있음. 피처를 추가/삭제할 때 세 파일을 모두 수정해야 하며, 한 곳만 수정하면 불일치 버그 발생.

**해결**
`src/data/features.py`에만 선언하고 나머지 파일에서 import.

```python
# train.py, test_features.py
from src.data.features import FEATURE_COLS
```

---

## 11. `holdout_season`이 2024로 하드코딩

**문제**
`load_data(holdout_season=2024)`가 고정값. 2025시즌 데이터가 추가되면 테스트셋을 수동으로 바꿔야 함.

**해결**
`date.today().year - 1`로 동적으로 설정하여 항상 직전 시즌이 테스트셋이 되도록.

```python
from datetime import date
load_data(holdout_season=date.today().year - 1)
```

---

## 12. MLflow URI가 Docker 내부 주소로 고정

**문제**
`api.py`의 기본 `MLFLOW_TRACKING_URI`가 `http://mlflow:5000`으로 설정되어 있음. Docker Compose 내부 네트워크 주소라서 로컬에서 단독 실행 시 MLflow에 연결되지 않음.

**해결**
기본값을 로컬 파일 기반으로 변경.

```python
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
```

---

## 13. 예측 시 피처를 수동으로 입력해야 함 (자동 조회 없음)

**문제**
`/predict` API는 ERA, OPS, 최근 승률을 사람이 직접 입력해야 함. "오늘 LAD vs NYY 예측해줘"처럼 팀명만 넣으면 자동으로 최신 수치를 조회해주는 기능이 없음.

**해결**
Streamlit 대시보드에서 팀명을 선택하면 `collect.py`의 API 호출 로직을 재사용해서 최신 ERA, OPS, 최근 승률을 자동으로 조회하여 채움.

---

## 14. 테스트가 실질적인 로직을 검증하지 않음

**문제**
`test_features.py`의 세 테스트 모두 `FEATURE_COLS` 상수값만 확인. rolling 승률 계산의 리키지 방지 여부, 팀 통계 조인 정확도, NaN 처리 등 실제 중요한 로직은 전혀 검증하지 않음.

**해결**
의미 있는 테스트 추가:
- `_add_rolling_win_rate()`에 샘플 데이터를 넣어 리키지 없이 승률이 계산되는지 검증
- `build_features()` 결과 DataFrame의 컬럼, 행 수, NaN 없음 검증
