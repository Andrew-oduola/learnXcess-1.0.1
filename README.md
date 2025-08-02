# LearnXcess
[Prototype Link](https://learnxcess.streamlit.app) <br>
[Landing Page link](https://learnxcess-one.vercel.app)

**LearnXcess** is an AI-powered tool that transforms your study materials into high-quality flashcards, ready for review or export to Anki. Powered by Google Gemini, it supports multiple input methods and flashcard formats for flexible, effective learning.

## Features

- **Input Methods:**
  - Generate flashcards from a topic/subject name
  - Upload study files (PDF, TXT, DOCX)
  - Paste text directly
  - Extract and use YouTube video transcripts

- **Flashcard Types:**
  - Q&A (Question & Answer)
  - Definition (Term & Definition)
  - Fill in the Gap
  - MCQ (Multiple Choice Questions)

- **Export:**
  - Download generated flashcards as an Anki deck (`.apkg`)

- **Customization:**
  - Choose number of flashcards, deck name, and card types
  - Select knowledge level (Beginner, Intermediate, Advanced)

## Getting Started

### Prerequisites

- Python 3.8+
- [Google Gemini API access](https://ai.google.dev/)
- [Streamlit](https://streamlit.io/)
- [Anki](https://apps.ankiweb.net/) (for importing decks)

### Installation

1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd learnXcess
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Set up API keys:**
   - Create a `.env` file or edit `.streamlit/secrets.toml` with your Google Gemini API key(s):

     ```
     GEMINI_API_KEYS=your_key1,your_key2
     ```

### Running the App

```sh
streamlit run main.py
```

The app will open in your browser at [http://localhost:8501](http://localhost:8501).

## Usage

1. **Choose Input Method:**  
   Select how you want to provide study material (topic, file, text, or YouTube link).

2. **Configure Flashcards:**  
   Pick flashcard types, number of cards, and deck name.

3. **Generate Flashcards:**  
   Click "Generate Flashcards" to let the AI create your deck.

4. **Preview & Export:**  
   Browse generated cards, then click "Create Anki Deck" to download your `.apkg` file.

## File Structure

- `main.py` — Main Streamlit application
- `requirements.txt` — Python dependencies
- `.env` / `.streamlit/secrets.toml` — API keys
- `.devcontainer/` — Dev container configuration

## Troubleshooting

- **No API keys found:**  
  Ensure your `.env` or `.streamlit/secrets.toml` contains valid `GEMINI_API_KEYS`.

- **File upload issues:**  
  Only PDF, TXT, and DOCX files are supported.

- **YouTube transcript errors:**  
  Some videos may not have transcripts or may have them disabled.

## License

This project is for educational use. See [LICENSE](LICENSE) if available.

---


Made with ❤️ using Streamlit and Google Gemini.

