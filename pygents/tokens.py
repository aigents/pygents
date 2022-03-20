import abc

class Tokenizer(abc.ABC):

    def __init__(self, debug=True):
        self.debug = debug

    def tokenize(self,text):
        return text.split()

assert str(Tokenizer().tokenize("ab c")) == "['ab', 'c']"


