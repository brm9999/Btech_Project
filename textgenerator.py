import speech_recognition as sr
r=sr.Recognizer()
PATH="audio.wav"

with sr.AudioFile(PATH) as source:
        audio=r.listen(source)
try:
        txt=r.recognize_google(audio)
        print("You said     :        ",txt)
except sr.UnknownValueError:
        print("Could not understand audio")
except sr.RequestError as e:
        print("Could not request results;{0}".format(e))
                  