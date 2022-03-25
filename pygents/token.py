import abc

class Tokenizer(abc.ABC):

    def __init__(self, debug=True):
        self.debug = debug

    def tokenize(self,text):
        return text.split()

assert str(Tokenizer().tokenize("ab c")) == "['ab', 'c']"


def tokenize_detaching_head(text,chars="'\"{[("):
    tokens = []
    for head in range(len(text)):
        found = chars.find(text[head])
        if found >= 0:
            tokens.append(chars[found])
        else:
            return tokens, text[head:]
    return tokens, None
assert str(tokenize_detaching_head("test")) == "([], 'test')"
assert str(tokenize_detaching_head("'\"")) == '(["\'", \'"\'], None)'
assert str(tokenize_detaching_head("\"'test")) == "(['\"', \"'\"], 'test')"


def tokenize_detaching_tail(text,chars="'\":,;.!?}])"):
    tokens = []
    length = len(text)
    for i in range(length):
        tail = length - i - 1
        found = chars.find(text[tail])
        if found >= 0:
            tokens.append(chars[found])
        else:
            return tokens, text[:tail + 1]
    return tokens, None

assert str(tokenize_detaching_tail("test")) == "([], 'test')"
assert str(tokenize_detaching_tail("test'")) == "([\"'\"], 'test')"
assert str(tokenize_detaching_tail("test.\"")) == "(['\"', '.'], 'test')"
assert str(tokenize_detaching_tail("test').\"")) == "(['\"', '.', ')', \"'\"], 'test')"

    
def tokenize_split_with_delimiters_and_quotes(text):
    tokens = []
    splits = text.split(' ')
    for split in splits:
        if len(tokens) > 0:
            tokens.append(' ')
        head, token = tokenize_detaching_head(split)
        tokens.extend(head)
        if token is not None and len(token) > 0: 
            tail, token = tokenize_detaching_tail(token)
            if token is not None and len(token) > 0:
                tokens.append(token)
            tokens.extend(tail)
    return tokens
assert str(tokenize_split_with_delimiters_and_quotes("man says hi")) == "['man', ' ', 'says', ' ', 'hi']"
assert str(tokenize_split_with_delimiters_and_quotes("man (tom) says 'hi there!' to me.")) == "['man', ' ', '(', 'tom', ')', ' ', 'says', ' ', \"'\", 'hi', ' ', 'there', \"'\", '!', ' ', 'to', ' ', 'me', '.']"

