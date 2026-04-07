# AI Vision Projects

A collection of AI vision systems built with OpenCV and MediaPipe, exploring computer vision techniques for various applications.

## Overview

This repository contains multiple computer vision projects that leverage cutting-edge libraries like OpenCV and MediaPipe to build intelligent vision systems. Each project demonstrates different applications of AI-powered image and video processing.

## Projects

### 1. **Face Puzzle** 🧩
A computer vision application for face detection and puzzle-solving using facial recognition.

- **Technology Stack**: OpenCV, MediaPipe, NumPy
- **Features**: Face detection, facial landmark detection, puzzle mechanics
- **Location**: `./face-puzzle/`
- **Details**: [Read more](./face-puzzle/README.md)

### 2. **Posture Detection System** 🧘
A real-time posture analysis and correction system using pose estimation.

- **Technology Stack**: MediaPipe, OpenCV, Flask/FastAPI (Backend), HTML/CSS/JavaScript (Frontend)
- **Features**: Real-time pose detection, posture analysis, web interface
- **Location**: `./posture-detection-system/`
- **Details**: [Read more](./posture-detection-system/README.md)

## Technology Stack

### Core Libraries
- **OpenCV** - Computer vision processing and image manipulation
- **MediaPipe** - Pre-built ML solutions for vision tasks (pose, face, hand detection)
- **NumPy** - Numerical computing and array operations
- **Pillow (PIL)** - Image processing utilities

### Backend
- **Python** - Primary programming language
- **Flask or FastAPI** - Web framework for APIs

### Frontend
- **HTML/CSS/JavaScript** - User interface and visualization

## Getting Started

Each project has its own virtual environment and dependencies. To get started with a specific project:

1. Navigate to the project directory
2. Activate the virtual environment
3. Install dependencies: `pip install -r requirements.txt`
4. Follow the project-specific README for usage instructions

## Project Structure

```
AI-Vision-Projects/
├── face-puzzle/              # Face detection puzzle application
│   ├── face_puzzle.py
│   ├── face_puzzle_test.py
│   ├── puzzle_env/          # Virtual environment
│   └── README.md
│
├── posture-detection-system/ # Posture analysis system
│   ├── backend/             # Flask/FastAPI backend
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── models.py
│   │   ├── services/
│   │   ├── routes/
│   │   ├── models/          # Pre-trained ML models
│   │   └── requirements.txt
│   │
│   ├── frontend/            # Web interface
│   │   ├── index.html
│   │   ├── app.js
│   │   └── styles.css
│   │
│   └── README.md
│
└── README.md               # This file
```

## Contributing

To add a new AI vision project to this collection:

1. Create a new project directory
2. Set up a Python virtual environment
3. Document the project with a comprehensive README
4. Update this main README with project information

## Requirements

- Python 3.8+
- pip (Python package manager)

## License

[Add your license information here]

## Resources

- [OpenCV Documentation](https://docs.opencv.org/)
- [MediaPipe Documentation](https://mediapipe.dev/)
- [MediaPipe Solutions](https://mediapipe.readthedocs.io/)

---

*Last Updated: April 2026*
