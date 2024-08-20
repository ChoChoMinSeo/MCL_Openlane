# openlane_editor

```bash
E00_V01_data_processing
└─ output_v01_validation
     └─ pickle
          └─ datalist.pickle

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
└─ output_v02_validation
    └─ pickle
         ├─ results
         │   ├─ segment-1
         │   ...
         │   └─ segment-100
         ├─ datalist_video.pickle
         └─ datalist.pickle

```

좌우 방향키: 다음/이전 이미지

상하 방향키: 다음/이전 영상(폴더)

delete: 현재 선택된 차선 삭제

xy/yz plane: 마우스 드래그로 차선 편집

progress: 클릭 후 cmd 창에 원하는 영상 번호 입력하여 이동 / trackbar 드래그 해서 원하는 이미지 또는 영상으로 이동