# 피처 개선안

> 현재 모델의 예측 정확도를 높이기 위한 추가 피처 후보 목록.
> 구현 전 데이터 수집 가능 여부 및 API 확인 필요.

---

## 현재 사용 중인 피처 (6개)

| 피처 | 설명 | 출처 |
|------|------|------|
| `home_win_rate_l10` | 홈팀 최근 10경기 승률 | schedules.parquet 계산 |
| `away_win_rate_l10` | 원정팀 최근 10경기 승률 | schedules.parquet 계산 |
| `home_era` | 홈팀 시즌 누적 ERA | MLB Stats API |
| `away_era` | 원정팀 시즌 누적 ERA | MLB Stats API |
| `home_ops` | 홈팀 시즌 누적 OPS | MLB Stats API |
| `away_ops` | 원정팀 시즌 누적 OPS | MLB Stats API |

---

## 추가 피처 후보

### 1. 선발투수 ERA / WHIP

**설명**
팀 전체 ERA가 아닌 오늘 경기의 선발투수 개인 성적. 선발투수가 경기 결과에 가장 큰 영향을 미치는 단일 변수임.

**추가 피처**
- `home_starter_era`: 홈팀 선발투수 시즌 ERA
- `away_starter_era`: 원정팀 선발투수 시즌 ERA
- `home_starter_whip`: 홈팀 선발투수 시즌 WHIP
- `away_starter_whip`: 원정팀 선발투수 시즌 WHIP

**수집 방법**
MLB Stats API `/schedule` 응답의 `probablePitcher` 필드에서 선발투수 ID 확인 후
`/people/{pitcherId}/stats?stats=season&group=pitching` 으로 개인 스탯 조회.

**난이도**: 중간 (API 2회 호출 필요)

---

### 2. 팀 득점력 (Runs Scored / Runs Allowed)

**설명**
OPS가 타격 효율을 나타내지만, 실제 득점(RS)과 실점(RA)은 승패와 더 직접적인 관계가 있음.
피타고라스 승률(`RS² / (RS² + RA²)`)로 기대 승률을 계산할 수 있음.

**추가 피처**
- `home_runs_scored_per_game`: 홈팀 경기당 평균 득점
- `away_runs_scored_per_game`: 원정팀 경기당 평균 득점
- `home_runs_allowed_per_game`: 홈팀 경기당 평균 실점
- `away_runs_allowed_per_game`: 원정팀 경기당 평균 실점

**수집 방법**
MLB Stats API `/teams/{teamId}/stats?stats=season&group=hitting` 응답의 `runs` 필드.

**난이도**: 낮음 (이미 수집 중인 API 확장)

---

### 3. 불펜 ERA (최근 N경기)

**설명**
시즌 누적 ERA는 초반 부진/호투를 평균내서 희석시킴.
최근 2주(약 15경기) 기준 불펜 ERA가 현재 상태를 더 잘 반영.

**추가 피처**
- `home_bullpen_era_l15`: 홈팀 최근 15경기 불펜 ERA
- `away_bullpen_era_l15`: 원정팀 최근 15경기 불펜 ERA

**수집 방법**
MLB Stats API `/teams/{teamId}/stats?stats=byDateRange&group=pitching` 으로 기간별 투구 스탯 조회.
선발/불펜 구분은 `gameType` 또는 등판 이닝으로 필터링 필요.

**난이도**: 높음 (불펜만 따로 분리하는 로직 필요)

---

### 4. 홈/원정 상대 전적 (Head-to-Head)

**설명**
특정 팀 간에는 구장 특성, 투구 스타일 등으로 인해 편향된 상대 전적이 존재함.
당해 시즌 또는 최근 3년 상대 전적 승률을 피처로 활용.

**추가 피처**
- `h2h_home_win_rate`: 홈팀 vs 원정팀 최근 3시즌 홈 승률

**수집 방법**
이미 수집된 `schedules.parquet`에서 두 팀 간 경기만 필터링해서 계산 가능.
별도 API 호출 불필요.

**난이도**: 낮음 (기존 데이터로 계산 가능)

---

### 5. 주요 선수 부상 여부

**설명**
핵심 타자나 선발투수의 부상은 승패에 큰 영향을 미치지만 현재 모델에서 전혀 반영되지 않음.

**추가 피처**
- `home_il_count`: 홈팀 현재 부상자 명단(IL) 인원 수
- `away_il_count`: 원정팀 현재 부상자 명단 인원 수

**수집 방법**
MLB Stats API `/teams/{teamId}/roster?rosterType=injuries` 로 부상자 명단 조회 가능.

**난이도**: 중간

---

### 6. 날씨 (야외 구장)

**설명**
돔 구장은 해당 없지만, 야외 구장은 바람 방향/세기, 기온, 습도가 타구 비거리와 투수 구위에 영향을 줌.
특히 강풍(Wrigley Field 등)은 득점에 직접적인 영향.

**추가 피처**
- `wind_speed`: 풍속 (mph)
- `wind_direction`: 풍향 (타자 유리/불리 방향)
- `temperature`: 기온 (°F)

**수집 방법**
Weather API (OpenWeatherMap 등) 또는 MLB Stats API `/game/{gamePk}/boxscore`의 날씨 정보.
야외 구장 목록을 별도로 관리해야 함 (돔 구장은 날씨 피처 0으로 고정).

**난이도**: 중간 (외부 API 추가 필요)

---

## 우선순위 요약

| 순위 | 피처 | 기대 효과 | 난이도 |
|------|------|----------|--------|
| 1 | 선발투수 ERA/WHIP | 높음 | 중간 |
| 2 | 득점/실점 (RS/RA) | 중간 | 낮음 |
| 3 | 상대 전적 (H2H) | 중간 | 낮음 |
| 4 | 불펜 ERA (최근) | 중간 | 높음 |
| 5 | 부상자 명단 | 중간 | 중간 |
| 6 | 날씨 | 낮음 | 중간 |

---

## 구현 시 주의사항

- **리키지 방지**: 모든 피처는 경기 시작 전 시점의 데이터만 사용해야 함. 선발투수 스탯도 해당 경기 결과 반영 전 값 사용.
- **결측치 처리**: 선발투수 미발표 경기, 시즌 초 데이터 부족 등에 대한 기본값(fallback) 전략 필요.
- **피처 저장**: 새 피처를 `features.parquet`에 추가하면 기존 모델과 호환되지 않으므로 모델 재학습 필수.
