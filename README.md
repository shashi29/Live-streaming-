# Live-streaming-

app/
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   ├── jquery.min.js
│   │   └── script.js
│   └── vendor/
│       └── bootstrap.min.css
├── templates/
│   └── index.html
├── app.py
├── requirements.txt
└── README.md

backend/
├── app.py
├── requirements.txt
├── models/
│   ├── __init__.py
│   ├── video.py
│   └── word.py
├── controllers/
│   ├── __init__.py
│   ├── video_controller.py
│   └── word_controller.py
├── services/
│   ├── __init__.py
│   ├── speech_to_text.py
│   ├── video_muter.py
│   └── video_streamer.py
└── static/
    ├── videos/
    │   └── example_video.mp4
    └── muted_videos/
        └── example_video_muted.mp4