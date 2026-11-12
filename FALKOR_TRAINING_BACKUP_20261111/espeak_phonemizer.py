import piper_phonemize


class Phonemizer:
    def __init__(self, voice):
        self.voice = voice

    def phonemize(self, text):
        resultado = piper_phonemize.phonemize_espeak(
            text,
            self.voice
        )
        return "".join(resultado[0])
