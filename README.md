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

```
.
├── LICENSE
├── README.md
├── main.py
├── options
│   ├── __pycache__
│   │   ├── args.cpython-311.pyc
│   │   └── config.cpython-311.pyc
│   ├── args.py
│   └── config.py
├── tree.txt
└── ui_tools
    ├── __pycache__
    │   ├── ui.cpython-311.pyc
    │   ├── utils.cpython-311.pyc
    │   └── window.cpython-311.pyc
    ├── ui.py
    ├── utils
    │   ├── __pycache__
    │   │   ├── utils.cpython-311.pyc
    │   │   └── vis_utils.cpython-311.pyc
    │   ├── utils.py
    │   └── vis_utils.py
    └── window.py

6 directories, 17 files
```