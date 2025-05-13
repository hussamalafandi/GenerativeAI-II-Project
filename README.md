# 🌪️ AI-Powered Hurricane Analysis System

An advanced question-answering system using RAG (Retrieval-Augmented Generation) technology and Google's Gemini model.  
✔ Supports both Arabic and English  
✔ Works with post-August 2024 reports  
✔ Maintains conversation context  


## 📋 Table of Contents
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Examples](#-examples)
- [Troubleshooting](#-troubleshooting)


## 📌 Requirements
- Python 3.9 or newer
- API key from [Google AI Studio](https://aistudio.google.com/)
- PDF files containing hurricane data

## 🛠️ Installation
1. Clone the repository:
```bash
git clone https://github.com/hussamalafandi/GenerativeAI-II-Project/tree/rahaf-aswad
cd GenerativeAI-II-Project
Install dependencies:

bash
pip install -r requirements.txt
Add report files to the data/ directory:

bash
mkdir data
cp /path/to/your/report.pdf data/hurricane_report_2024.pdf
Create environment file:

bash
echo "GEMINI_API_KEY=your_api_key_here" > .env
▶️ Usage
bash
python main.py
Example Queries:
> What was the maximum recorded wind speed?
> Which areas were most affected?
> What was the evacuation plan?
🗂️ Project Structure
hurricane-analysis/
├── data/                   # Hurricane reports (PDF)
├── src/
│   ├── document_loader.py  # Document processing
│   ├── rag_system.py       # Core system
│   └── utils.py           # Helper functions
├── main.py                # Main application
├── requirements.txt       # Dependencies
└── README.md              # This file
💡 Examples
Question	Sample Answer
"What was the hurricane category?"	"The hurricane reached category 4"
"Number of emergency shelters"	"12 shelters were opened"
"Names of affected areas"	"Miami, Tampa, Orlando"
⚠️ Troubleshooting
API Key Errors:

bash
export GEMINI_API_KEY="your_key_here"  # Linux/Mac
set GEMINI_API_KEY="your_key_here"     # Windows
Inaccurate Answers:

Ensure documents contain post-August 2024 data

Verify PDFs are text-based (not image scans)