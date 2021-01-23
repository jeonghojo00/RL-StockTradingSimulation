### Georgia Institute of Technology: Machine Learning for Trading

## Project: Develop a Trader with Machine Learning algorithms

# Stock Trading Simulation with Q-Learning Algorithm (Reinforcement Learning)

## Objective
To build a strategy learner that makes a trading decision (buy/sell/hold) for given period. 
The Result shall outperform the Benchmark for both in-sample period and out-sample period for the each stock ticker in a given set of stocks. 
The input of the learner shall be only Technical indicators, not fundamental indicatoras, and at least 3 of indicators shall be used.

## Install
- Python 3
- NumPy, Pandas
- TensorFlow
- Keras

## Translation
English Text = 'he saw a old yellow truck'
French Text(Translated)        = 'il a vu un vieux camion jaune'
French Text(Google Translator) = 'il a vu un vieux camion jaune'

## Result with the Final Model
Accuracy: 98%
Training times: Ran 25 epochs and achieved 98% accuracy at 11th epoch with 463us/step time consumed per epoch

## Building the Pipeline
Below is a summary of the various preprocessing and modeling steps. The high-level steps include:

Preprocessing: Load sentences, cleaning, tokenization, padding.
Modeling: Build the RNN model, train, and testing. 
Prediction: Translate English to French, and compare the output translations to what it is supposed to be.
Iteration: iterate on the model, experimenting with different architectures

## Preprocessing
1. Cleaning
* Description: Uppercase letters -> Lowercase letters, Punctuations -> spaces 
               % Current dataset does not require Cleaning because it has been preprocessed already in the ways mentioned above.
  ![Alt](images/PreprocessClean.png)
        
2. Tokenization
* Description: Turn each sentence into a sequence of words ids using Keras's Tokenizer function. Use this function to tokenize english_sentences and french_sentences in the cell below.
* tokenize(x)
    Input(x): List of sentences/strings to be tokenized
    Output  : Tuple of (tokenized x data, tokenizer used to tokenize x)
  ![Alt](images/PreprocessToken.png)
  
3. Padding
* Description: Since all the sequences need to be the same length, we add padding to the end of the sequences(padding='post') to make them the same length.
* pad(x, length=None)
    Input(x)     : List of sequences
    Input(length): Length to pad the sequence to.  If None, use length of longest sequence in x.
    Output       : Padded numpy array of sequences
  ![](images/PreprocessPad.png)

## Modeling
4 RNN Models
* 4 RNN models have been developed and compared to use them as reference for the final model: Simple RNN, RNN with Embedding, Bidirectional RNN, and Encoder-Decoder RNN. 
  1. Simple RNN
    * Description: Simple RNN with a GRU layer
    ![](images/SimpleRNN.png)
  
    * Result
    ![](images/SimpleRNNresult.png)
      
  2. Embedding layers implemented
    * Description: Embedding Layer and GRU Layer
    ![](images/EmbedRNN.png)
    
    * Result
    ![](images/SimpleRNNresult.png)
  
  3. Bidirectional layers implemented
    * Description: Bidirectional Layer with GRU model
    ![](images/BidirecRNN.png)

    * Result
    ![](images/BidRNNresult.png)
    
  4. Encoder-Decoder model
    * Description: Encoder Layer with GRU and Decoder Layer with GRU
    * Result
    ![](images/EncdecRNNresult.png)
     
  5. (Final model) Combination of layers for higher accuracy
    * Description: Embed Layer, Bidirectional with LSTM, Dropout 20%, 
    * Result
    ![](images/FinalRNNresult1.png)
    ![](images/FinalRNNresult2.png)    
  
## Future Improvements
Here are few improvements that I could try to improve

1. Train with other languages
2. Speed up time taken per step. ex) Decrease the time to 150us/step level with 98% accuracy.
3. Decrease number of epoches to reach to 98% accuracy
4. Compare GRU and LSTM: I tried the final model with  the GRU at first, but the accuracy was not high as the LSTM. Next time, I will experiment how they are different and what are advantages and disadvantages to use them. 
