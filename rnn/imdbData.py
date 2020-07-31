import numpy as np
from tensorflow.keras.datasets import imdb
#from tensorflow.keras.utils import categorical
from tensorflow.keras.preprocessing import sequence

class IMDBdata:
    def __init__(self,num_words=10000,skip_top=20,maxlen=80):
        super().__init__()
        self.num_classes =2 
        self.num_words = num_words
        self.skip_top = skip_top
        self.maxlen = maxlen
        #word to indx
        self.word_index = imdb.get_word_index()
        self.word_index = {k:(v+3) for k,v in self.word_index.items()}
        self.word_index["<PAD>"] = 0
        self.word_index["<START>"] = 1
        self.word_index["<UNKOWN>"] = 2
        #indx to word
        self.index_word={v:k for k,v in self.word_index.items()}

        # print(self.word_index)
        # print(len(self.word_index))
        # print(self.index_word)
        # load data set
        (self.x_train,self.y_train),(self.x_test,self.y_test) = imdb.load_data(
            num_words = self.num_words,
            skip_top = self.skip_top,
            maxlen = self.maxlen
        )

        self.x_train_text = np.array([[self.index_word[v] for v in sample] for sample in self.x_train])

    def get_review_text(self,indices):
        return [self.index_word[v] for v in indices]

if __name__ == "__main__":
    imdb_data = IMDBdata()
    print(imdb_data.get_review_text(imdb_data.x_train[0]))
    print(imdb_data.y_train[0])
