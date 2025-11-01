Code 1: Minutiae Extraction & Patch Generation
Code 2: MobileNet-V3 with FRO Fusion Rule

Workflow:
Run Code 1 first to extract minutiae-based patches

Then run Code 2 to train MobileNet-V3 and apply FRO fusion

Database Structure 

C:\Users\Mohamed\livdet2015_crossmatch\
├── Testing\
│   ├── Fake\
│   │   ├── Body Double\
│   │   ├── Ecoflex\
│   │   ├── Gelatin\
│   │   ├── OOMOO\
│   │   └── Playdoh\
│   └── Live\
└── Training\
    ├── Fake\
    │   ├── Ecoflex\
    │   └── Playdoh\
    └── Live\



