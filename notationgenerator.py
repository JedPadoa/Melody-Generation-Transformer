import tensorflow as tf
from preprocessing import Processor
from transformer import Transformer
import pickle 

# Global parameters
EPOCHS = 1
BATCH_SIZE = 64
RAW_DATA_PATH = "first_1000.json"
MAX_POSITIONS_IN_POSITIONAL_ENCODING = 300


class ABCGenerator:
    """
    Class to generate ABC notation score using a trained Transformer model.

    This class encapsulates the inference logic for generating notation
    based on a starting sequence.
    """

    def __init__(self, transformer, tokenizer, max_length=100):
        """
        Initializes the MelodyGenerator.

        Parameters:
            transformer (Transformer): The trained Transformer model.
            tokenizer (Tokenizer): Tokenizer used for encoding notation.
            max_length (int): Maximum length of the generated notation.
        """
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.max_length = max_length

    def generate(self, start_sequence):
        """
        Generates a melody based on a starting sequence.

        Parameters:
            start_sequence (list of str): The starting sequence of the melody.

        Returns:
            str: The generated melody.
        """
        print("start sequence: ", start_sequence)
        input_tensor = self._get_input_tensor(start_sequence)

        num_notes_to_generate = self.max_length - len(input_tensor[0])

        for _ in range(num_notes_to_generate):
            predictions = self.transformer(
                input_tensor, input_tensor, False, None, None, None
            )
            predicted_note = self._get_note_with_highest_score(predictions)
            input_tensor = self._append_predicted_note(
                input_tensor, predicted_note
            )

        generated_melody = self._decode_generated_sequence(input_tensor)

        return generated_melody

    def _get_input_tensor(self, start_sequence):
        """
        Gets the input tensor for the Transformer model.

        Parameters:
            start_sequence (list of str): The starting sequence of the melody.

        Returns:
            input_tensor (tf.Tensor): The input tensor for the model.
        """
        input_sequence = self.tokenizer.texts_to_sequences([start_sequence])
        input_tensor = tf.convert_to_tensor(input_sequence, dtype=tf.int64)
        return input_tensor

    def _get_note_with_highest_score(self, predictions):
        """
        Gets the note with the highest score from the predictions.

        Parameters:
            predictions (tf.Tensor): The predictions from the model.

        Returns:
            predicted_note (int): The index of the predicted note.
        """
        latest_predictions = predictions[:, -1, :]
        predicted_note_index = tf.argmax(latest_predictions, axis=1)
        predicted_note = predicted_note_index.numpy()[0]
        return predicted_note

    def _append_predicted_note(self, input_tensor, predicted_note):
        """
        Appends the predicted note to the input tensor.

        Parameters:
            input_tensor (tf.Tensor): The input tensor for the model.

        Returns:
            (tf.Tensor): The input tensor with the predicted note
        """
        return tf.concat([input_tensor, [[predicted_note]]], axis=-1)

    def _decode_generated_sequence(self, generated_sequence):
        """
        Decodes the generated sequence of notes.

        Parameters:
            generated_sequence (tf.Tensor): Tensor with note indexes generated.

        Returns:
            generated_melody (str): The decoded sequence of notes.
        """
        generated_sequence_array = generated_sequence.numpy()
        generated_notation = self.tokenizer.sequences_to_texts(
            generated_sequence_array
        )[0]
        return generated_notation
    
if __name__ == "__main__":
    """
    Loads tokenizer from saved pkl file, 
    compiles model, 
    generates tokens
    and maps output back to abc notation
    """
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    transformer_model = tf.keras.models.load_model('transformer_model', custom_objects={"Transformer": Transformer})
    
    transformer_model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]  # You can also include other metrics as needed
    )
    
    print("Generating notation...")
    
    notation_generator = ABCGenerator(
        transformer_model, tokenizer
    )
    
    start_sequence = ['X:56']
    new_notation = notation_generator.generate(start_sequence)
    
    print(f"Generated notation: {new_notation}")