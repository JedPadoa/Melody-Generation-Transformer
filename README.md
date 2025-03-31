
This is a Transformer neural network trained on the [**Irishman dataset**](https://huggingface.co/datasets/sander-wood/irishman) to generate a melody autoregressively in ABC notation given an initial seed

**Instructions to run**

- Create a virtual python env with python 3.10
- Install required dependencies: *$ pip install -r requirements.txt*
- Run training module *$ python train.py* : a tokenizer.pkl file and transformer folder will be created
- Run generator module *$ notationgenerator.py* : Prints generated ABC notation
- If needed, tweak the initial sequence in notationgenerator.py
- truncate_data.py does not need to be run. This was created solely to truncate the raw training data to 1000 elements

# Completion Report #

**Preprocessing**
- created a processor class to handle preprocessing duties
- removed metatdata from each of the elements in the json dictionary
- parsed each song notation string into an array split by spaces and tokenized
- generated input-target pairs for training, returned in main process method
- dumped tokenizer into pkl file for later use

**Training** 

- created a training model to run in order to train the model
- preprocesses data and generates training tensors
- orgainizes, trains, and saves transformer model

**Generation**

- Created class ABCGenerator which encapsulates generation tasks
- Generates input tensor from start sequence
- generates and appends prediction token to input tensor at each step of for loop
- main function loads pkl tokenizer and saved model
- instantiates ABCGenerator class with tokenizer and model and generates a sequence






