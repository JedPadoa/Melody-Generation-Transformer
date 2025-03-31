import json
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
import pickle

RAW_DATA_PATH = "first_1000.json"

class Processor():
    
    def __init__(self, raw_data, batch_size = 64):
        self.raw_data = raw_data
        self.abc_notations = self._get_abc_notation(raw_data)[:200]
        self.tokenizer = Tokenizer(filters="", lower=False, split=" ")
        self.max_sequence_length = None
        self.num_tokens = None
        self.batch_size = batch_size
        
    def process(self):
        parsed_notation = [song.split(" ") for song in self.abc_notations]
        tokenized_symbols = self._tokenize_symbols(
            parsed_notation
        )
        self.max_sequence_length = max([len(symbols) for symbols in tokenized_symbols])
        self.num_tokens = len(self.tokenizer.word_index)
        input_sequences, target_sequences = self._generate_sequences(
            tokenized_symbols
        )
        tf_training_dataset = self._convert_to_tf_dataset(
            input_sequences, target_sequences
        )
        
        with open('tokenizer.pkl', 'wb') as f:
            pickle.dump(self.tokenizer, f)
            
        return tf_training_dataset
    
    def _generate_sequences(self, songs):
        """
        Creates input-target pairs from tokenized melodies.

        Parameters:
            melodies (list): A list of tokenized melodies.

        Returns:
            tuple: Two numpy arrays representing input sequences and target sequences.
        """
        input_sequences, target_sequences = [], []
        for song in songs:
            for i in range(1, len(song)):
                input_seq = song[:i]
                target_seq = song[1 : i + 1]  # Shifted by one time step
                padded_input_seq = self._pad_sequence(input_seq)
                padded_target_seq = self._pad_sequence(target_seq)
                input_sequences.append(padded_input_seq)
                target_sequences.append(padded_target_seq)
        return np.array(input_sequences), np.array(target_sequences)
    
    def _convert_to_tf_dataset(self, input_sequences, target_sequences):
        """
        Converts input and target sequences to a TensorFlow Dataset.

        Parameters:
            input_sequences (list): Input sequences for the model.
            target_sequences (list): Target sequences for the model.

        Returns:
            batched_dataset (tf.data.Dataset): A batched and shuffled
                TensorFlow Dataset.
        """
        dataset = tf.data.Dataset.from_tensor_slices(
            (input_sequences, target_sequences)
        )
        shuffled_dataset = dataset.shuffle(buffer_size=1000)
        batched_dataset = shuffled_dataset.batch(self.batch_size)
        return batched_dataset
    
    def _tokenize_symbols(self, songs):
        """
        Tokenizes and encodes a list of abc notation.

        Parameters:
            songs (list): A list of notation to be tokenized and encoded.

        Returns:
            tokenized_notation: A list of tokenized and encoded melodies.
        """
        self.tokenizer.fit_on_texts(songs)
        tokenized_notation = self.tokenizer.texts_to_sequences(songs)
        return tokenized_notation
    
    def _get_abc_notation(self, songs):
        with open(songs, 'r') as songs:
            songs = json.load(songs)
            
        abc_notations = []
        
        for song in songs:
            abc_notations.append(song['abc notation'])
        
        return [line.replace("\n", " ") for line in abc_notations] #removes \n from strings
    
    def _pad_sequence(self, sequence):
        """
        Pads a sequence to the maximum sequence length.

        Parameters:
            sequence (list): The sequence to be padded.

        Returns:
            list: The padded sequence.
        """
        return sequence + [0] * (self.max_sequence_length - len(sequence))
    
    @property
    def number_of_tokens_with_padding(self):
        """
        Returns the number of tokens in the vocabulary including padding.

        Returns:
            int: The number of tokens in the vocabulary including padding.
        """
        return self.num_tokens + 1