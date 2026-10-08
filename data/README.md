# 원본 데이터 준비

원본 데이터는 이용 조건 때문에 저장소에 올리지 않았습니다. 아래 방법으로 받아 넣은 뒤 실행합니다.

1. KAMP(인공지능 제조 플랫폼)에서 **「자원 최적화 AI 데이터셋」**을 내려받습니다.
2. 압축 파일 안의 `okm_augumented_2021.csv`를 이 폴더의 `raw/` 아래에 둡니다. 파일 이름의 철자(augumented)는 원본 그대로입니다.

```
data/raw/okm_augumented_2021.csv
```

내려받은 파일이 분석에 쓴 파일과 같은지 확인하려면, 파일의 SHA-256 값(파일 내용으로 계산한 고유 식별값)이 아래와 같은지 비교하세요.
Windows에서는 `certutil -hashfile data\raw\okm_augumented_2021.csv SHA256` 명령으로 확인할 수 있습니다.

```
8f7af2e49366c93e1d6f5fdef4b5e350066c1792ac463c2c2886e370f4674830
```

# 외부 자료 준비 (02-1c 노트북, 선택)

02-1c 노트북은 공개된 다른 공장의 15분 전력 자료로 Chronos-2를 미세조정합니다. 두 자료 모두 CC BY 4.0입니다. 아래 파일을 받아 `data/external/`에 둡니다. 없으면 `run_all.py --with-chronos`도 02-1c는 건너뛰고 저장된 예측을 씁니다.

| 파일 | 출처 |
|---|---|
| `LoadProfile_20IPs_2016.csv`, `LoadProfile_30IPs_2017.csv` | Braeuer, F. (2020). Load profile data of 50 industrial plants in Germany for one year. Zenodo. https://doi.org/10.5281/zenodo.3899018 |
| `Steel_industry_data.csv` | Steel Industry Energy Consumption, UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption (zip을 풀어 나온 파일) |

```
curl -L -o data/external/LoadProfile_20IPs_2016.csv https://zenodo.org/api/records/3899018/files/LoadProfile_20IPs_2016.csv/content
curl -L -o data/external/LoadProfile_30IPs_2017.csv https://zenodo.org/api/records/3899018/files/LoadProfile_30IPs_2017.csv/content
curl -L -o data/external/steel.zip "https://archive.ics.uci.edu/static/public/851/steel+industry+energy+consumption.zip"
```

02-1c 노트북이 세 파일의 SHA-256 값을 확인합니다.
