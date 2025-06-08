import pyttsx3

def speak(text: str):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":
    test_str = "This is a test of the TTS system on Raspberry Pi."
    print("Speaking:", test_str)
    speak(test_str)

