# openlane_editor

```bash
<Original Data>
└─ output_v01_validation
     └─ pickle
          ├─ datalist.pickle
          └─ results
         │   ├─ segment-1
         │   ...
         │   └─ segment-123123

E01_V09_editor
├─ code_v02
│   ├─ options
│   │  ├─ args.py
│   │  └─ config.py
│   │
│   ├─ ui_tools
│   │  ├─ utils
│   │  │  ├─ utils.py
│   │  │  └─ vis_utils.py
│   │  ├─ ui.py
│   │  └─ window.py
│   │
│   └─ main.py
│
└─ output_v02_{현재 mode}
    └─ pickle
         ├─ results
         │   ├─ segment-1
         │   ...
         │   └─ segment-123123
         ├─ datalist_video.pickle (자동 생성)
         └─ datalist.pickle (자동 생성)

```

## <조작키>

좌우 방향키: 다음/이전 이미지

상하 방향키: 다음/이전 영상(폴더)

Esc: 프로그램 종료 및 저장

delete: 현재 선택된 차선 삭제

end: 현재 프레임의 모든 차선 삭제 (= 현재 프레임 삭제와 동일한 역할)

r 키: 초기값으로 리셋

## 각 Window 기능

image: 마우스 클릭으로 차선 선택, 편집 불가능

xy/yz plane: 가로 방향 마우스 드래그로 차선 편집 (y 고정)

progress: 클릭 후 cmd 창에 원하는 영상 번호 입력하여 이동 (잘못 눌렀을 경우 cmd에 enter키로 취소) / trackbar 드래그 해서 원하는 이미지 또는 영상으로 이동

## 결과물

Annotation File: E01_V09_editor - output_v02_{현재 mode} - pickle - results - segment-* 내부 파일

빈 프레임 제외한 프레임 리스트: E01_V09_editor - output_v02_{현재 mode} - pickle - datalist.pickle