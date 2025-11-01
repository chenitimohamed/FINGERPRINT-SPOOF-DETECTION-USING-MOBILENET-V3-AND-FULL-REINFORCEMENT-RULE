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



////
The MobileNet-v3 CNN model takes around 6-8 hours to converge using a single Nvidia GTX
1080 Ti GPU with approximately 96; 000 local patches from 2; 000 fingerprint images (2; 000
images  46 patches/fingerprint image) in the training set. The average spoof detection time for
an input image, including minutiae detection, local patch extraction and alignment, inference of
spoofness scores for local patches, and producing the final spoof detection decision, is 100 ms
using a Nvidia RTX 3090 GPU.