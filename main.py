import streamlit as st
import google.generativeai as genai
import os
import json
import re
import textwrap
import genanki
import uuid
import tempfile
from pathlib import Path
from typing import List, Dict, Union
import traceback
from dotenv import load_dotenv
import random
# Add these new imports at the top
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound


# Constants
MAX_CHARS = 30000  # Safe input size
DEFAULT_MODEL = 'gemini-1.5-flash'

# Load environment variables
load_dotenv()

GEMINI_API_KEYS = ["AIzaSyDlip38FAQFVVjoo-d4w0Jh5ufa0imA8wE", "AIzaSyBCURyNysCPQZyX71fbbM2B8bnX3rYoeRM"]

# Configuration with error handling
def get_api_keys():
    """Secure key loading with fallback logic"""
    keys = []
    
    # 1. Try Streamlit secrets (for cloud deployment)
    try:
        if 'GEMINI_API_KEYS' in st.secrets:
            keys.extend([k.strip() for k in st.secrets["GEMINI_API_KEYS"].split(",") if k.strip()])
    except Exception:
        pass
    
    # 2. Try environment variables (for local development)
    load_dotenv()
    env_keys = os.getenv("GEMINI_API_KEYS", "")
    if env_keys:
        keys.extend([k.strip() for k in env_keys.split(",") if k.strip()])
    
    # 3. Validation
    if not keys:
        st.error("No API keys found. Configure secrets.toml or .env file.")
        st.stop()
    
    return keys

# Initialize with key rotation
keys = get_api_keys()
for attempt, key in enumerate(random.sample(keys, len(keys)), 1):
    try:
        genai.configure(api_key=key)
        st.session_state.current_key = key[-6:] + "..."  # Store partial for logging
        break
    except Exception as e:
        if attempt == len(keys):
            st.error(f"❌ All API keys failed: {str(e)}")
            st.stop()

# Flashcard Types
MODES = {
    "Q&A": {
        "description": "Standard question and answer format",
        "fields": ["Question", "Answer"],
        "template": {
            'qfmt': '{{Question}}',
            'afmt': '{{FrontSide}}<hr id="answer">{{Answer}}'
        }
    },
    "Definition": {
        "description": "Term and definition format",
        "fields": ["Term", "Definition"],
        "template": {
            'qfmt': 'What is the definition of <b>{{Term}}</b>?',
            'afmt': '{{FrontSide}}<hr id="answer">{{Definition}}'
        }
    },
    "Fill in the Gap": {
        "description": "Fill-in-the-blank questions",
        "fields": ["Question", "Answer"],
        "template": {
            'qfmt': '{{Question}}',
            'afmt': '{{FrontSide}}<hr id="answer">{{Answer}}'
        }
    },
    "MCQ": {
        "description": "Multiple choice questions",
        "fields": ["Question", "Options", "Correct Answer"],
        "template": {
            'qfmt': '{{Question}}<br><br>{{Options}}',
            'afmt': '{{FrontSide}}<hr id="answer">Correct answer: {{Correct Answer}}'
        }
    }
}

def initialize_session_state():
    """Initialize session state variables with error handling."""
    try:
        defaults = {
            'flashcards': [],
            'deck_created': False,
            'processing': False,
            'input_method': "topic",
            'current_card_index': 0,
            'topic': "",
            'level': "Intermediate"
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    except Exception as e:
        st.error(f"❌ Session state initialization failed: {str(e)}")
        st.stop()

def display_intro():
    """Display the application introduction with error handling."""
    try:
        st.title("🎓 LearnXcess 2.0")
        st.markdown("""
        Transform your study materials into effective flashcards with AI. Choose from:
        1. **Generate from topic** - Just provide a subject/topic
        2. **Upload a file** - PDF, TXT, or DOCX
        3. **Paste text** - Direct content input
        """)
    except Exception as e:
        st.error(f"❌ Failed to display introduction: {str(e)}")

def get_user_input():
    """Get input method and content from user with error handling."""
    try:
        # Define the available input methods
        input_methods = [
            "Generate from topic", 
            "Upload a file", 
            "Paste text",
            "YouTube video"
        ]
        
        # Ensure session state has a valid input method
        if 'input_method' not in st.session_state or st.session_state.input_method not in input_methods:
            st.session_state.input_method = input_methods[0]  # Default to first option
            
        input_method = st.radio(
            "Choose your input method:",
            input_methods,
            index=input_methods.index(st.session_state.input_method),
            horizontal=True,
            key="input_method_selector"
        )
        st.session_state.input_method = input_method
        
        content = ""
        file_path = ""
        
        if input_method == "Generate from topic":
            # Initialize topic and level if not in session state
            if 'topic' not in st.session_state:
                st.session_state.topic = ""
            if 'level' not in st.session_state:
                st.session_state.level = "Intermediate"
            
            st.session_state.topic = st.text_input(
                "Enter the topic/subject you want to learn about:", 
                value=st.session_state.topic,
                placeholder="e.g., Quantum Mechanics, French Revolution, Python OOP",
                key="topic_input"
            )
            
            st.session_state.level = st.selectbox(
                "Knowledge level:", 
                ["Beginner", "Intermediate", "Advanced"],
                index=["Beginner", "Intermediate", "Advanced"].index(st.session_state.level),
                key="level_selector"
            )
            
            if st.session_state.topic:
                content = f"TOPIC: {st.session_state.topic}\nLEVEL: {st.session_state.level}"
        
        elif input_method == "Upload a file":
            uploaded_file = st.file_uploader(
                "Upload your study material (PDF, TXT, DOCX)", 
                type=["pdf", "txt", "docx"],
                key="file_uploader"
            )
            if uploaded_file:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        file_path = tmp.name
                    content = extract_text_from_file(file_path)
                except Exception as e:
                    st.error(f"❌ Error processing uploaded file: {str(e)}")
                    content = ""
        
        elif input_method == "Paste text":
            if 'text_content' not in st.session_state:
                st.session_state.text_content = ""
                
            content = st.text_area(
                "Paste your study content here:", 
                value=st.session_state.text_content,
                height=200,
                key="text_input_area"
            )
            st.session_state.text_content = content
            
        elif input_method == "YouTube video":
            youtube_url = st.text_input(
                "Enter YouTube video URL:",
                placeholder="https://www.youtube.com/watch?v=...",
                key="youtube_url"
            )
            if youtube_url:
                with st.spinner("Fetching transcript..."):
                    content = get_youtube_transcript(youtube_url)
                    if content:
                        st.success("✅ Successfully fetched transcript")
                    else:
                        st.warning("No transcript content available")
        
        return content, file_path
    
    except Exception as e:
        st.error(f"❌ Error getting user input: {str(e)}")
        return "", ""

def extract_text_from_file(file_path: str) -> str:
    """Extract text from different file types with error handling."""
    try:
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == ".pdf":
            import PyPDF2
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                return "\n".join([page.extract_text() for page in reader.pages])
        elif file_ext == ".docx":
            import docx
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        else:  # Assume plain text
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        return ""

def display_mode_selection() -> tuple:
    """Display flashcard mode selection options with error handling."""
    try:
        st.subheader("Flashcard Options")
        
        cols = st.columns(2)
        selected_modes = []
        
        with cols[0]:
            st.markdown("**Flashcard Types**")
            for mode in MODES:
                if st.checkbox(
                    f"{mode} - {MODES[mode]['description']}", 
                    value=True,
                    key=f"mode_{mode}_checkbox"
                ):
                    selected_modes.append(mode)
        
        with cols[1]:
            num_flashcards = st.slider(
                "Number of flashcards to generate",
                min_value=5,
                max_value=100,
                value=20,
                step=5,
                key="num_flashcards_slider"
            )
            
            deck_name = st.text_input(
                "Name your flashcard deck",
                value="My AI-Generated Deck",
                key="deck_name_input"
            )
        
        return selected_modes, num_flashcards, deck_name
    
    except Exception as e:
        st.error(f"❌ Error in mode selection: {str(e)}")
        return [], 20, "My AI-Generated Deck"

def split_text(text: str, max_length: int) -> List[str]:
    """Split text into chunks without cutting sentences."""
    paragraphs = textwrap.wrap(text, max_length, break_long_words=False, break_on_hyphens=False)
    return paragraphs

def clean_json_response(text: str) -> str:
    """Clean and extract JSON from AI response."""
    raw = text.strip()
    if raw.startswith("```json") or raw.startswith("```"):
        raw = re.sub(r"```(json)?", "", raw)
        raw = raw.strip("` \n")
    return raw

def build_prompt(chunk: str, num_flashcards: int, selected_modes: List[str]) -> str:
    """Build the prompt for the AI model."""
    mode_instructions = []
    
    if "Q&A" in selected_modes:
        mode_instructions.append("- Q&A: Clear question and answer format. Questions should be direct and answers concise.")
    
    if "Definition" in selected_modes:
        mode_instructions.append("- Definition: Important terms with clear definitions. Format as 'Term: [term]' and 'Definition: [definition]'.")
    
    if "Fill in the Gap" in selected_modes:
        mode_instructions.append("- Fill in the Gap: Sentences with key information replaced by '_____'. Include the correct answer.")
    
    if "MCQ" in selected_modes:
        mode_instructions.append("- MCQ: Multiple choice questions with 4 options. Clearly indicate the correct answer. include 'options' as a list and 'correct_option' index make distractors plausible")
    
    instructions = "\n".join(mode_instructions)
    
    prompt = f"""
I need you to generate {num_flashcards} high-quality flashcards from the following content. 
The flashcards should be varied and cover the most important concepts.

Required formats:
{instructions}

For each flashcard, provide:
1. A "type" field indicating the format ({selected_modes})
2. Appropriate fields for the type 

Guidelines:
- Focus on key concepts and relationships
- Make questions clear and unambiguous
- Keep answers concise but complete
- Avoid trivial or obvious questions
- Include examples where helpful

Return ONLY a well-formatted JSON array of flashcards. Each flashcard should be an object with the appropriate fields.

Content to convert to flashcards:
{chunk}
"""
    return prompt

def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from URL."""
    if "youtu.be" in url:
        return url.split("/")[-1].split("?")[0]
    parsed_url = urlparse(url)
    return parse_qs(parsed_url.query).get("v", [None])[0]

def get_youtube_transcript(url: str) -> str:
    """Get transcript from YouTube video."""
    try:
        video_id = extract_video_id(url)
        if not video_id:
            st.error("❌ Invalid YouTube URL")
            return ""
            
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry['text'] for entry in transcript])
        
    except TranscriptsDisabled:
        st.error("❌ Transcripts are disabled for this video")
        return ""
    except NoTranscriptFound:
        st.error("❌ No transcript available for this video")
        return ""
    except Exception as e:
        st.error(f"❌ Error getting transcript: {str(e)}")
        return ""


def generate_flashcards(content: str, num_flashcards: int, selected_modes: List[str]) -> List[Dict]:
    """Generate flashcards using the AI model."""
    if not content or not selected_modes:
        st.warning("Please provide content and select at least one flashcard type.")
        return []
    
    try:
        model = genai.GenerativeModel(DEFAULT_MODEL)
        chunks = split_text(content, MAX_CHARS)
        flashcard_results = []
        
        with st.spinner("Generating flashcards..."):
            # Reset deck creation status when generating new flashcards
            st.session_state.deck_created = False
            st.session_state.processing = True
            
            for i, chunk in enumerate(chunks):
                per_chunk_count = max(3, num_flashcards // len(chunks))
                prompt = build_prompt(chunk, per_chunk_count, selected_modes)
                
                response = model.generate_content(prompt)
                if response.text:
                    try:
                        cleaned = clean_json_response(response.text)
                        flashcards = json.loads(cleaned)
                        flashcard_results.extend(flashcards)
                    except json.JSONDecodeError as e:
                        st.error(f"Could not parse response from chunk {i+1}: {e}")
                        st.text("Raw response:")
                        st.code(response.text)
                else:
                    st.warning(f"No response for chunk {i+1}")
            
            st.session_state.flashcards = flashcard_results
            st.session_state.processing = False
            
            return flashcard_results
    except Exception as e:
        st.error(f"Error generating flashcards: {e}")
        st.session_state.processing = False
        return []


def generate_from_topic(topic: str, level: str, num_flashcards: int, selected_modes: List[str]) -> List[Dict]:
    """Generate flashcards from just a topic name with error handling."""
    try:
        model = genai.GenerativeModel(DEFAULT_MODEL)
        
        prompt = f"""
Generate {num_flashcards} high-quality flashcards about {topic} at {level} level.
Focus on the most important concepts, facts, and relationships.

Flashcard formats to include: {", ".join(selected_modes)}

For each flashcard, provide:
1. A "type" field {", ".join(selected_modes)}
2. Appropriate content fields
3. For MCQs: include "options" list and "correct_option" index

Guidelines:
- Cover fundamental concepts first
- Include examples where helpful
- Make questions clear and answers concise
- Ensure factual accuracy
- Vary question difficulty appropriately for {level} level
- For technical topics, include both theory and practical applications

Return ONLY a JSON array of flashcards.
"""
        with st.spinner(f"Generating {num_flashcards} flashcards about {topic}..."):
            response = model.generate_content(prompt)
            st.session_state.deck_created = False
            
            if not response.text:
                st.warning("⚠️ No response received from the AI model")
                return []
            
            try:
                cleaned = clean_json_response(response.text)
                flashcards = json.loads(cleaned)
                return flashcards
            except json.JSONDecodeError as e:
                st.error(f"❌ Failed to parse AI response: {str(e)}")
                st.text("Raw response for debugging:")
                st.code(response.text)
                return []
    
    except Exception as e:
        st.error(f"❌ Error generating from topic: {str(e)}")
        return []

# [Keep all your existing helper functions like split_text, clean_json_response, etc.]
# Add the same level of error handling to each function



def create_anki_deck(flashcards: List[Dict], deck_name: str) -> str:
    """Create an Anki deck package file using the Basic model."""
    try:
        # Use Anki's built-in Basic model
        my_model = genanki.Model(
            1607392319,  # This is the standard Basic model ID
            'Basic',
            fields=[
                {'name': 'Front'},
                {'name': 'Back'},
            ],
            templates=[
                {
                    'name': 'Card 1',
                    'qfmt': '{{Front}}',
                    'afmt': '{{FrontSide}}<hr id="answer">{{Back}}',
                },
            ],
            css="""
            .card {
                font-family: Arial;
                font-size: 20px;
                text-align: center;
                color: black;
                background-color: white;
            }
            """
        )
        
        # Create a deck with a unique ID
        deck_id = abs(hash(deck_name)) % (10**10)
        my_deck = genanki.Deck(deck_id, deck_name)
        
        # Track unique cards to prevent duplicates
        unique_questions = set()
        cards_added = 0
        
        for card in flashcards:
            card_type = card.get('type', 'Q&A')
            question = card.get('question', '') or card.get('term', '')
            
            # Skip if we've already seen this question
            if question in unique_questions:
                continue
            unique_questions.add(question)
            
            # Prepare fields based on card type
            if card_type == 'Q&A':
                front = card.get('question', '')
                back = card.get('answer', '')
            elif card_type == 'Definition':
                front = f"What is the definition of: <b>{card.get('term', '')}</b>"
                back = card.get('definition', '')
            elif card_type == 'Fill in the Gap':
                front = card.get('question', '')
                back = card.get('answer', '')
            elif card_type == 'MCQ':
                options = card.get('options', [])
                correct_option = card.get('correct_option', 0)
                formatted_options = "<br>".join(f"{'• ' + opt}" for opt in options)
                front = f"{card.get('question', '')}<br><br>{formatted_options}"
                back = f"Correct answer: {options[correct_option] if options else ''}"
            
            # Create and add note
            my_note = genanki.Note(
                model=my_model,
                fields=[front, back],
                tags=[card_type.replace(" ", "")]
            )
            my_deck.add_note(my_note)
            cards_added += 1
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".apkg") as tmp:
            genanki.Package(my_deck).write_to_file(tmp.name)
            st.session_state.deck_created = True
            st.success(f"Created deck with {cards_added} cards (expected {len(flashcards)})")
            return tmp.name
            
    except Exception as e:
        st.error(f"❌ Error creating Anki deck: {str(e)}")
        st.error("Please check your flashcards for duplicate questions or invalid formats")
        return ""


def display_flashcards(flashcards: List[Dict]):
    """Display generated flashcards in an interactive way."""
    st.subheader("Generated Flashcards Preview")
    
    if not flashcards:
        st.info("No flashcards generated yet. Provide some content and click 'Generate Flashcards'.")
        return
    
    tab1, tab2 = st.tabs(["Card Browser", "Raw JSON"])
    
    with tab1:
        cols = st.columns([1, 4, 1])
        current_index = st.session_state.get('current_card_index', 0)
        
        with cols[1]:
            if current_index < len(flashcards):
                card = flashcards[current_index]
                card_type = card.get('type', 'Unknown')
                
                st.markdown(f"**Card {current_index + 1} of {len(flashcards)}** ({card_type})")
                
                if card_type == 'Q&A':
                    st.markdown(f"**Q:** {card.get('question', '')}")
                    with st.expander("Show Answer"):
                        st.markdown(f"**A:** {card.get('answer', '')}")
                
                elif card_type == 'Definition':
                    st.markdown(f"**Term:** {card.get('term', '')}")
                    with st.expander("Show Definition"):
                        st.markdown(f"**Definition:** {card.get('definition', '')}")
                
                elif card_type == 'Fill in the Gap':
                    st.markdown(f"**Question:** {card.get('question', '')}")
                    with st.expander("Show Answer"):
                        st.markdown(f"**Answer:** {card.get('answer', '')}")
                
                elif card_type == 'MCQ':
                    st.markdown(f"**Question:** {card.get('question', '')}")
                    options = card.get('options', [])
                    correct_option = card.get('correct_option', 0)
                    
                    st.markdown("**Options:**")
                    for i, option in enumerate(options):
                        prefix = "✓" if i == correct_option else "○"
                        st.markdown(f"{prefix} {option}")
        
        with cols[0]:
            if st.button("◀ Previous", key="prev_btn") and current_index > 0:
                st.session_state.current_card_index = current_index - 1
                st.rerun()  # Changed from experimental_rerun()
        
        with cols[2]:
            if st.button("Next ▶", key="next_btn") and current_index < len(flashcards) - 1:
                st.session_state.current_card_index = current_index + 1
                st.rerun()  # Changed from experimental_rerun()
    
    with tab2:
        st.json(flashcards)

def display_flashcards(flashcards: List[Dict]):
    """Display generated flashcards in an interactive way."""
    st.subheader("Generated Flashcards Preview")
    
    if not flashcards:
        st.info("No flashcards generated yet. Provide some content and click 'Generate Flashcards'.")
        return
    
    tab1, tab2 = st.tabs(["Card Browser", "Raw JSON"])
    
    with tab1:
        cols = st.columns([1, 4, 1])
        current_index = st.session_state.get('current_card_index', 0)
        
        with cols[1]:
            if current_index < len(flashcards):
                card = flashcards[current_index]
                card_type = card.get('type', 'Unknown')
                
                st.markdown(f"**Card {current_index + 1} of {len(flashcards)}** ({card_type})")
                
                if card_type == 'Q&A':
                    st.markdown(f"**Q:** {card.get('question', '')}")
                    with st.expander("Show Answer"):
                        st.markdown(f"**A:** {card.get('answer', '')}")
                
                elif card_type == 'Definition':
                    st.markdown(f"**Term:** {card.get('term', '')}")
                    with st.expander("Show Definition"):
                        st.markdown(f"**Definition:** {card.get('definition', '')}")
                
                elif card_type == 'Fill in the Gap':
                    st.markdown(f"**Question:** {card.get('question', '')}")
                    with st.expander("Show Answer"):
                        st.markdown(f"**Answer:** {card.get('answer', '')}")
                
                elif card_type == 'MCQ':
                    st.markdown(f"**Question:** {card.get('question', '')}")
                    options = card.get('options', [])
                    correct_option = card.get('correct_option', 0)
                    
                    st.markdown("**Options:**")
                    for i, option in enumerate(options):
                        prefix = "✓" if i == correct_option else "○"
                        st.markdown(f"{prefix} {option}")
        
        with cols[0]:
            if st.button("◀ Previous") and current_index > 0:
                st.session_state.current_card_index = current_index - 1
                st.rerun()
        
        with cols[2]:
            if st.button("Next ▶") and current_index < len(flashcards) - 1:
                st.session_state.current_card_index = current_index + 1
                st.rerun()
    
    with tab2:
        st.json(flashcards)


def main():
    """Main application function with comprehensive error handling."""
    try:
        initialize_session_state()
        display_intro()
        
        # Get user input
        content, file_path = get_user_input()
        
        # Get flashcard options
        selected_modes, num_flashcards, deck_name = display_mode_selection()
        
        # Generate flashcards button
        if st.button("Generate Flashcards", disabled=st.session_state.processing, key="generate_btn"):
            initialize_session_state()
            if not content:
                st.warning("⚠️ Please provide some input first")
            else:
                try:
                    if st.session_state.input_method == "Generate from topic":
                        flashcards = generate_from_topic(
                            st.session_state.topic,
                            st.session_state.level,
                            num_flashcards,
                            selected_modes
                        )
                    else:
                        flashcards = generate_flashcards(content, num_flashcards, selected_modes)
                    
                    if flashcards:
                        st.session_state.flashcards = flashcards
                        st.success(f"✅ Successfully generated {len(flashcards)} flashcards!")
                
                except Exception as e:
                    st.error(f"❌ Flashcard generation failed: {str(e)}")
                    st.text(traceback.format_exc())
        
        # Display and export functionality
        if st.session_state.flashcards:
            display_flashcards(st.session_state.flashcards)
            
            if st.button("Create Anki Deck", disabled=st.session_state.deck_created, key="anki_btn"):
                try:
                    deck_path = create_anki_deck(st.session_state.flashcards, deck_name)
                    if deck_path:
                        with open(deck_path, "rb") as f:
                            st.download_button(
                                label="Download Anki Deck",
                                data=f,
                                file_name=f"{deck_name.replace(' ', '_')}.apkg",
                                mime="application/apkg",
                                key="download_btn"
                            )
                except Exception as e:
                    st.error(f"❌ Failed to create Anki deck: {str(e)}")
    
    except Exception as e:
        st.error(f"❌ Critical application error: {str(e)}")
        st.text(traceback.format_exc())

if __name__ == "__main__":
    # Windows-specific event loop policy
    if os.name == 'nt':
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    main()