# Time Series Forecasting with Seq2Seq Model

An Inmplementation of the Seq2seq model in the tensorflow library 
for time series forecasting

## Data
The data I used is based on the followng Kaggle challenge:
https://www.kaggle.com/dgawlik/nyse
The part I used consists of daily stock prices for various companies.
I focused on Yahoo for this particular exercise but can be applied to any time series
(I haven't addressed the multivariate case but It should work as well with some tweaks)


##Preprocessing
In order to be fed to the model, the data needsto be turned into a supervised setting with:
	- X as [n_samples, in_time_steps]
	- y as [n_samples, max_out_time_steps]
	
an hyper parameter allow to reduce the number of out_time_steps considered.
There is a some more of preprocessing that goes on in Seq2seq model.
See https://www.tensorflow.org/tutorials/seq2seq for more details in the more classic,
Neural Machine Translation Setting.
All the preprocessing is found under /notebooks/Exploration.ipynb

##Model
The model use a multilayer rnn as an Encoder and another as a decoder. This is the basic
architechture to make sequences to sequences prediction. however, the seq2seq library introduces 
the attention mechanism the allows the decoder to take information from every encoding state to
predict each time steps of the output. This allows to alleviate the model capacity restriction 
that arises when encoding a time series into a single vector. 
The other concept introduces is that of a Helper that defines of the inputs to the decoder
are presented at each time step. Since the seq2seq library is usually aimed at Neural 
Machine Translation that deals with discret object, I had to define a Custom Helper to
perform the Time series forecasting. 
Besides that, my implementation follow the aforementionned tutorial.
The tunable hyperparameters, such as the batch size, the number and size of the layers
or the dropout can be changed at the end of the model
the model is under models/seq2seq_ts.py

##Monitoring
I have set up summaries to monitor the training and evalution loss every 50 epochs.
They can be visualized along with the graph structure using tensorboard.

The model with he current hyperparameter tends to overfit, one could further tune them.
(Early stopping would probably take care of the overfitting)

##TO DO
Make the predictions for the holdout set and visualize them.

