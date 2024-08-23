# MCL_Openlane

[Openlane Dataset](https://github.com/OpenDriveLab/OpenLane) 전처리

### E01_V01_data_processing

Raw data로부터 원하는 데이터만 남기기

### E01_V04_lane_filtering

노이즈가 있는 x,z좌표 Median Filter 적용 및 이상치 제거

### E01_V05_lane_merge_spline

차선 사이의 각도를 근거로 하나의 차선으로 합칠 수 있는 차선쌍 합치기

### E01_V06_interpolation_only

차선 중간중간에 비어있는 부분 채우기

### E01_V06_lane_selection_parallel

너무 짧거나 이상한 곡선 표시, 너무 짧은 차선들에 대해 인접한 차선과 평행할 것이라는 근거로 연장하기

### E01_V07_ALS*

ALS 모델을 활용하여 원하는 y범위(3~103)에 걸친 Extrapolation 수행

### E01_V08_lane_selection_after_als*

ALS결과들 중 잘 나온 차선만 추출

### E01_V09_editor

수작업을 하기위한 자체 툴 제작