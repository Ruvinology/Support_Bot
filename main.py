from transformers import AutoModelForCausalLM, AutoTokenizer
from gtts import gTTS
import torch
import os
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit, QLineEdit
from PyQt6.QtCore import Qt

# Load DialoGPT model and tokenizer
model_name = "microsoft/DialoGPT-medium"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set device to GPU if available, else fallback to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Function to generate a response
def generate_response(user_input):
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt').to(device)
    attention_mask = torch.ones(new_user_input_ids.shape, dtype=torch.long).to(device)
    chat_history_ids = model.generate(
        new_user_input_ids,
        attention_mask=attention_mask,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        top_p=0.9,
        top_k=50,
        temperature=0.7,
        no_repeat_ngram_size=2,
        do_sample=True
    )
    bot_output = tokenizer.decode(chat_history_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return bot_output

# Function to convert text to speech (TTS)
def text_to_speech(text):
    tts = gTTS(text, lang='en', slow=False)
    save_path = os.path.join(os.path.expanduser("~"), "Desktop", "bot_response.mp3")
    tts.save(save_path)
    os.system(f"start {save_path}")

# PyQt Chatbot GUI
class ChatbotGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Hanako san - The supporting partner")
        self.setGeometry(100, 100, 500, 400)

        self.layout = QVBoxLayout()
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.input_box = QLineEdit()
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.handle_input)

        self.layout.addWidget(self.chat_display)
        self.layout.addWidget(self.input_box)
        self.layout.addWidget(self.send_button)
        self.setLayout(self.layout)

    def handle_input(self):
        user_text = self.input_box.text().strip()
        if user_text:
            self.chat_display.append(f"You: {user_text}")
            bot_reply = generate_response(user_text)
            self.chat_display.append(f"Hanako san: {bot_reply}\n")
            text_to_speech(bot_reply)
            self.input_box.clear()

# Run the PyQt application
app = QApplication([])
window = ChatbotGUI()
window.show()
app.exec()


