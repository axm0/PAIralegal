import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import os
from datetime import datetime
from pathlib import Path
import logging
import re
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
import threading


class InsuranceChatbotEvaluator:
    def __init__(self):
        self.setup_logging()
        self.qdrant = QdrantClient("http://localhost:6333")
        self.evaluation_file = Path("C:/Developer/Workspace/llama3.2/evaluations/evaluations.txt")
        self.evaluation_file.parent.mkdir(parents=True, exist_ok=True)
        self.data_root = "C:/Developer/Workspace/llama3.2/data"
        self.file_cache = {}
        self.lock = threading.Lock()

        with st.spinner("Loading embedding model..."):
            self.embed_model = SentenceTransformer('intfloat/e5-large-v2')
        with st.spinner("Loading LLaMA model..."):
            self.model_path = "C:/Developer/Models/Llama-3.2-3B-Instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.bos_token = '<s>'
            self.tokenizer.eos_token = '</s>'
            self.model.resize_token_embeddings(len(self.tokenizer))

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'chatbot_eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )

    def find_file_in_directory(self, filename: str) -> str:
        """Find a file by name in the data directory structure."""
        # Add .pdf extension if not present
        if not filename.lower().endswith(('.pdf', '.txt', '.xls', '.xlsm')):
            filename = filename + '.pdf'

        # Normalize the search path
        search_path = os.path.normpath(self.data_root)

        for root, _, files in os.walk(search_path):
            for file in files:
                # Compare normalized filenames
                if os.path.normcase(file) == os.path.normcase(filename):
                    return os.path.join(root, file)

        logging.error(f"File not found: {filename}")
        return None

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text_parts = []
                for page_num, page in enumerate(pdf.pages, 1):
                    with self.lock:
                        try:
                            page_text = page.extract_text(x_tolerance=3, y_tolerance=3, layout=True,
                                                          keep_blank_chars=True)
                            if page_text:
                                text_parts.append(page_text)
                        except Exception as e:
                            logging.error(f"Error extracting text from page in {pdf_path}: {str(e)}")
                if not text_parts:
                    return self.extract_text_from_scanned_pdf(pdf_path)
                return "\n".join(text_parts)
        except Exception as e:
            logging.error(f"Error in PDF extraction for {pdf_path}, falling back to OCR: {str(e)}")
            return self.extract_text_from_scanned_pdf(pdf_path)

    def extract_text_from_scanned_pdf(self, pdf_path: str) -> str:
        try:
            images = convert_from_path(pdf_path, dpi=400)

            def process_image(image):
                custom_config = r'''--oem 3 --psm 6 
                    -c tessedit_char_whitelist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,!?()-:;\'\"$@ "
                    -c preserve_interword_spaces=1
                    -c textord_heavy_nr=1
                    -c tessedit_do_invert=0
                    -c tessedit_enable_dict_correction=1'''
                gray_image = image.convert('L')
                return pytesseract.image_to_string(gray_image, config=custom_config, lang='eng')

            texts = [process_image(image) for image in images]
            return "\n".join(texts)
        except Exception as e:
            logging.error(f"Error in OCR processing for {pdf_path}: {str(e)}")
            return ""

    def extract_and_cache_file(self, filepath: str) -> str:
        """Extract text from a file and cache it."""
        if filepath in self.file_cache:
            return self.file_cache[filepath]

        try:
            if filepath.endswith('.pdf'):
                text = self.extract_text_from_pdf(filepath)
            elif filepath.endswith(('.xls', '.xlsm')):
                text = "Excel file content extraction not implemented"
            else:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()

            self.file_cache[filepath] = text
            return text
        except Exception as e:
            logging.error(f"Error extracting text from {filepath}: {str(e)}")
            return ""

    def get_relevant_context(self, query: str, specific_files: list = None, top_k: int = 3) -> tuple:
        """Enhanced context retrieval with specific file support."""
        if specific_files:
            contexts = []
            document_names = []

            for filename in specific_files:
                filepath = self.find_file_in_directory(filename)
                if filepath:
                    text = self.extract_and_cache_file(filepath)
                    if text:
                        preview_text = text[:2000] + ("..." if len(text) > 2000 else "")
                        context = f"""
                        Source: {filename}
                        Content:
                        {preview_text}
                        """
                        contexts.append(context)
                        document_names.append(filename)
                else:
                    contexts.append(f"File not found: {filename}")
                    document_names.append(filename)

            return "\n" + "=" * 30 + "\n".join(contexts), document_names

        query_embedding = self.embed_model.encode(query, normalize_embeddings=True)
        search_result = self.qdrant.search(
            collection_name="insurance_docs",
            query_vector=query_embedding,
            limit=top_k,
            score_threshold=0.7,
            with_payload=True
        )

        if not search_result:
            return "", []

        contexts = []
        document_names = []

        for result in search_result:
            metadata = result.payload.get('metadata', {})
            text = result.payload.get('text', '')
            score = result.score
            file_path = result.payload.get('file_path', 'Unknown')
            document_names.append(os.path.basename(file_path))

            preview_text = text[:2000] + ("..." if len(text) > 2000 else "")
            context = f"""
            Source: {os.path.basename(file_path)}
            Relevance: {score:.2f}
            ID: {metadata.get('sfma_id', 'Unknown')}

            Content:
            {preview_text}
            """
            contexts.append(context)

        return "\n" + "=" * 30 + "\n".join(contexts), document_names

    def extract_filenames_from_query(self, query: str) -> list:
        """Extract quoted filenames from the query."""
        return re.findall(r"'([^']+)'", query)

    def evaluate_response(self, prompt: str, response: str) -> str:
        """
        Evaluate the response quality with enhanced prompt and validation.
        Returns: 'accuracy_score, relevance_score' (e.g., '4, 5')
        """
        eval_prompt = f"""You are an AI evaluator. Rate the following response on two metrics:
    1. Accuracy (1-5): How accurate and factual is the information provided
    2. Relevance (1-5): How well the response addresses the original query

    Respond ONLY with two numbers in this format: X, Y
    Example correct responses: '4, 5' or '3, 2' or '5, 4'
    Do not include any other text, explanations, or characters.

    Original Query: {prompt}

    Response to Evaluate: {response}

    Your rating (just two numbers):"""

        try:
            input_ids = self.tokenizer.encode(eval_prompt, return_tensors='pt').to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=8,
                    temperature=0.1,  # Reduced temperature for more deterministic output
                    top_p=0.1,  # Reduced top_p for more focused output
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=False,  # Deterministic generation
                    num_beams=1  # Simple greedy decoding
                )

            # Extract only the generated response, not the prompt
            evaluation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            evaluation = evaluation.split("Your rating (just two numbers):")[-1].strip()

            # Clean up the response
            evaluation = re.sub(r'[^0-9,\s]', '', evaluation).strip()

            # Validate format
            if ',' in evaluation:
                nums = [int(n.strip()) for n in evaluation.split(',')]
                if len(nums) == 2 and all(1 <= n <= 5 for n in nums):
                    return f"{nums[0]}, {nums[1]}"

            # If no valid rating found, calculate based on response characteristics
            accuracy = self._calculate_accuracy_score(response)
            relevance = self._calculate_relevance_score(prompt, response)
            return f"{accuracy}, {relevance}"

        except Exception as e:
            logging.error(f"Error in evaluate_response: {str(e)}")
            return "3, 3"

    def _calculate_accuracy_score(self, response: str) -> int:
        """Calculate accuracy score based on response characteristics."""
        score = 3  # Default score

        # Check for detailed analysis
        if len(response) > 500:
            score += 1

        # Check for structured content
        if "###" in response or "####" in response:
            score += 1

        # Limit to valid range
        return min(max(score, 1), 5)

    def _calculate_relevance_score(self, prompt: str, response: str) -> int:
        """Calculate relevance score based on prompt and response correlation."""
        score = 3  # Default score

        # Check if response contains key terms from prompt
        prompt_terms = set(prompt.lower().split())
        response_terms = set(response.lower().split())
        common_terms = prompt_terms.intersection(response_terms)

        if len(common_terms) > len(prompt_terms) * 0.5:
            score += 1

        if len(response) > 200 and any(term in response.lower() for term in ['analysis', 'overview', 'summary']):
            score += 1

        # Limit to valid range
        return min(max(score, 1), 5)

    def save_evaluation(self, prompt: str, response: str, doc_names: list, ratings: str):
        try:
            with open(self.evaluation_file, 'a', encoding='utf-8') as f:
                f.write("------------------------------------\n")
                f.write(f"Rating: {ratings}\n")
                f.write(f"Prompt: {prompt}\n")
                f.write(f"Response: {response}\n")
                f.write(f"Referenced Documents: {', '.join(doc_names)}\n")
                f.write("------------------------------------\n\n")
            logging.info(f"Saved evaluation to {self.evaluation_file}")
        except Exception as e:
            logging.error(f"Error saving evaluation: {str(e)}")
            raise

    def generate_response(self, query: str, context: str) -> tuple:
        """Enhanced response generation with file-specific handling."""
        specific_files = self.extract_filenames_from_query(query)

        if specific_files:
            if len(specific_files) == 2 and "compare" in query.lower():
                prompt = f"""You are PAIralegal, an AI assistant specializing in insurance regulations. Compare the following two documents and identify key differences between them.

                Document 1: {specific_files[0]}
                Document 2: {specific_files[1]}

                Context from Documents:
                {context}

                Please provide a detailed comparison focusing on key differences:"""
            else:
                prompt = f"""You are PAIralegal, an AI assistant specializing in insurance regulations. Analyze the following specific document(s): {', '.join(specific_files)}

                Context from Documents:
                {context}

                Please provide a detailed analysis of the document content:"""
        else:
            prompt = f"""You are PAIralegal, an AI assistant specializing in insurance regulations. Answer the following question using only the provided insurance documents. Be specific and detailed. If you can't find exact information, say so clearly.

            Question: {query}

            Context from Insurance Documents:
            {context}

            Detailed Answer:"""

        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.model.device)
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()

        ratings = self.evaluate_response(query, response)

        return response, ratings


def set_custom_style():
    st.markdown("""
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .main-header {
            text-align: center;
            color: #2E4057;
        }
        .subheader {
            text-align: center;
            color: #666;
            font-size: 1.2em;
            margin-bottom: 2em;
        }
        .file-instructions {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)


def display_file_instructions():
    with st.expander("📝 How to Reference Specific Files"):
        st.markdown("""
        To analyze specific files, use single quotes around the filename in your query:

        - Single file analysis: "What is the file 'CA PP Symbol Filing Exhibits with Upd Exh 14' about?"
        - Compare files: "Compare 'CA PP Symbol Filing Exhibits with Upd Exh 14' with 'Exhibit 7.A (Proxy Weight Template) - (Part 1)'"

        Available files will be shown in the sidebar.
        """)


def display_available_files(data_root: str):
    with st.sidebar:
        st.markdown("### 📁 Available Files")
        files = []
        for root, _, filenames in os.walk(data_root):
            for filename in filenames:
                if filename.endswith(('.pdf', '.txt', '.xls', '.xlsm')):
                    files.append(filename)

        if st.checkbox("Show Available Files"):
            st.write("Click to copy filename:")
            for file in sorted(files):
                if st.button(f"📄 {file}", key=file):
                    st.write(f"'{file}' copied!")
                    st.text_input("", value=f"'{file}'", key=f"copy_{file}")


def initialize_session_state():
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []


def main():
    """
    Main function for the PAIralegal Streamlit application.
    Handles initialization, UI setup, and chat interaction flow.
    """
    # Configure the Streamlit page
    st.set_page_config(
        page_title="PAIralegal - Insurance Regulations Assistant",
        page_icon="⚖️",
        layout="wide"
    )

    # Set up the UI styling
    set_custom_style()

    # Display the main header
    st.markdown('<h1 class="main-header">🤖 PAIralegal</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subheader">Your AI-Powered Insurance Regulations Assistant</p>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    # Initialize session state and display instructions
    initialize_session_state()
    display_file_instructions()

    # Initialize the chatbot if not already done
    if st.session_state.chatbot is None:
        try:
            with st.spinner("Initializing PAIralegal... This may take a few minutes."):
                st.session_state.chatbot = InsuranceChatbotEvaluator()
                display_available_files(st.session_state.chatbot.data_root)
                st.success("PAIralegal is ready to assist you!")
        except Exception as e:
            st.error(f"Error initializing PAIralegal: {str(e)}")
            st.error("Please ensure Qdrant is running at http://localhost:6333")
            return

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Ask me about insurance regulations..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            try:
                # Extract specific files from query if any
                specific_files = st.session_state.chatbot.extract_filenames_from_query(prompt)

                # Search for relevant context
                with st.spinner("Searching documents..."):
                    context, doc_names = st.session_state.chatbot.get_relevant_context(
                        prompt,
                        specific_files
                    )

                if not context:
                    response = (
                        "I couldn't find any relevant information in the insurance documents. "
                        "Could you please rephrase your question or check if the specified files exist?"
                    )
                    st.markdown(response)
                else:
                    # Generate response
                    with st.spinner("Analyzing regulations..."):
                        response, ratings = st.session_state.chatbot.generate_response(
                            prompt,
                            context
                        )

                        # Save evaluation
                        st.session_state.chatbot.save_evaluation(
                            prompt,
                            response,
                            doc_names,
                            ratings
                        )

                    # Display referenced documents
                    with st.expander("📚 View Referenced Documents"):
                        st.markdown(context)

                    # Display the response
                    st.markdown(response)

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                error_message = (
                    "I encountered an error while processing your request. "
                    f"Error details: {str(e)}"
                )
                st.error(error_message)
                logging.error(f"Error processing query: {str(e)}", exc_info=True)

    # Sidebar content
    with st.sidebar:
        # Options section
        st.title("⚙️ Options")
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

        # About section
        st.markdown("---")
        st.markdown("### About PAIralegal")
        st.markdown("""
            PAIralegal is your AI-powered assistant for navigating insurance regulations. 

            Features:
            - 📚 Access to comprehensive insurance documentation
            - 🔍 Intelligent document search
            - 💡 Detailed regulatory insights
            - 📝 Clear, documented responses
            - 📊 Automated response evaluation
            - 📄 Specific file analysis and comparison
        """)

        # Query examples section
        st.markdown("---")
        st.markdown("### Query Examples")
        st.markdown("""
            Try these example queries:
            1. General query: "What are the coverage limits for personal auto insurance?"
            2. Single file analysis: "What is the file 'CA PP Symbol Filing Exhibits with Upd Exh 14' about?"
            3. File comparison: "Compare 'CA AT 2021-07-19.pdf' with 'CA AT 2022-07-18 Compare.pdf'"
        """)

if __name__ == "__main__":
    main()