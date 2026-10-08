# 참고 자료

## 데이터

- 중소벤처기업부, Korea AI Manufacturing Platform(KAMP), 자원 최적화 AI 데이터셋, KAIST, 2021.12.27., www.kamp-ai.kr
- 같은 데이터셋의 「자원 최적화 AI 데이터셋 분석실습 가이드북」: 공정 설명, 품질 지수, 무작위 분할 실습, 한 시점 비용 최소화 실습의 비교 기준으로 사용

## 요금 제도와 피크 기준

| 내용 | 출처 |
|---|---|
| 최대수요전력의 정의(15분 단위) | [한전 기본공급약관 제2조](https://home.kepco.co.kr/kepco/front/html/CY/D/C/CYDCHP00101.html) |
| 요금적용전력: 직전 12개월 중 7~9월·12~2월과 당월 최대수요, 계약전력 30% 하한 | [한전 기본공급약관 제68조](https://home.kepco.co.kr/kepco/front/html/CY/D/C/CYDCHP00108.html) |
| 계절·시간대 구분, 공휴일은 경부하·토요일 최대부하는 중간부하로 계량, 기본요금 대상은 중간·최대부하 시간대 | [한전 전기공급약관 별표 3](https://cyber.kepco.co.kr/ckepco/front/jsp/CY/D/C/CYDCHP00403.jsp) |
| 2021년 산업용(을) 요금표(선택Ⅰ 7,220원/kW·월, 시간대별 단가) | [2021.1.1 시행 전기요금표(한국물가정보)](http://data.cmpi.or.kr/program/board/2021.01.01%EC%8B%9C%ED%96%89.pdf) |
| 현행 요금표와 시간대 | [한전 산업용 전기요금표](https://cyber.kepco.co.kr/ckepco/front/jsp/CY/E/E/CYEEHP00103.jsp) |
| 최대전력 관리장치의 정의와 목표전력, 지원 대상(계약전력 500kW 이상) | [한전 선택공급약관 부하관리기기 지원제도](https://home.kepco.co.kr/kepco/front/html/CY/D/C/CYDCHP00303.html) |
| 관리장치의 1단계(주의)·2단계(차단) 운영 방식 | [최대수요전력 제어 설명(업체 자료)](https://cq4l.com/%EC%B5%9C%EB%8C%80%EC%88%98%EC%9A%94-%EC%A0%84%EB%A0%A5-%EC%A0%9C%EC%96%B4/) |

## 예측 모델과 평가

| 내용 | 출처 |
|---|---|
| Chronos-2 사전학습 시계열 모델 | [Chronos-2 논문(arXiv 2510.15821)](https://arxiv.org/html/2510.15821), [chronos-forecasting](https://github.com/amazon-science/chronos-forecasting) |
| 에너지 분야 사전학습 모델 비교 | [FETS 벤치마크(arXiv 2604.22328)](https://arxiv.org/abs/2604.22328) |
| 시계열에서의 교차 검증 | [Bergmeir, Hyndman, Koo (2018)](https://research.monash.edu/en/publications/a-note-on-the-validity-of-cross-validation-for-evaluating-autoreg/) |
| 기준 곡선과 당일 보정 | [Cho 외, Modelling and forecasting daily electricity load curves](https://arxiv.org/pdf/1611.08632), [Obst, de Vilmarest, Goude (2021)](https://ar5iv.labs.arxiv.org/html/2009.06527) |
| 단순 평균 결합이 강한 이유 | [Wang, Hyndman 외, 예측 결합 리뷰](https://arxiv.org/pdf/2205.04216) |
| 미탐지·오경보 비용 비대칭 | [Elkan (2001), The Foundations of Cost-Sensitive Learning](https://cseweb.ucsd.edu/~elkan/rescale.pdf) |
| 분위수 conformal 보정, 적응형 conformal | [Romano, Patterson, Candès (2019)](https://www.semanticscholar.org/paper/Conformalized-Quantile-Regression-Romano-Patterson/6f9dc6f8519e927d948a13aa7ae0df336f443eb9), [Gibbs & Candès (2021)](https://papers.neurips.cc/paper/2021/file/0d441de75945e5acbc865406fc9a2559-Paper.pdf) |

## 피크 저감과 투자 판단

| 내용 | 출처 |
|---|---|
| 압축공기는 일반 공장 전력의 약 10% | [미국 에너지부, Compressed Air Tip Sheet](https://www.energy.gov/sites/prod/files/2014/05/f16/compressed_air1.pdf) |
| 피트형 열처리로 사양(무부하 승온 시간·소비전력 상한) | [제조사 사양표](https://www.china-electric.net/pit_type_heat_treatment_tempering_furnace.html) |
| 수요관리·피크 저감 정책 배경 | [에너지경제연구원 KIP1903](https://www.keei.re.kr/keei/download/KIP1903.pdf) |
| 소규모 ESS 경제성(배터리·PCS 단가, 유지비) | 한국산학기술학회논문지 24(4), 2023, doi:10.5762/KAIS.2023.24.4.61 |
| ESS 수명·효율 | PNNL Energy Storage Cost and Performance Database (2022), NREL Annual Technology Baseline (2024) |
| 20kWh 초과 전기저장시설의 설비 기준 | 화재안전기술기준(NFTC 607), 한국전기설비규정(KEC 512) |
| 2026년 4월 산업용 시간대 개편 | 기후에너지환경부 보도자료(2026-03-13) |

## 피크 관리 도우미

| 내용 | 출처 |
|---|---|
| 경보 관리 기준(운전원 1명당 시간당 경보 수) | [IEC 62682, ISA-18.2, EEMUA 191 요약](https://industrialmonitordirect.com/blogs/knowledgebase/industrial-alarm-system-standards-iec-62682-isa-182-and-eemua-191) |
