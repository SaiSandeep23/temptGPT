import spacy
import tensorflow as tf

def main():
    # Testing Spacy installation
    # loading small english language model
    # (run the command "python -m spacy download en_core_web_sm
    # " if you're getting error in main folder)
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("To test the installation on spacy")
    print("spaCy Tokenization")
    for token in doc:
        print(token.text)

    # Testing tensorflow installation
    print("\n TF version:",tf.__version__)
    print("Hello world")

if __name__ == "__main__":
    main()