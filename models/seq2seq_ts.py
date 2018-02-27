# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 03:15:58 2018

@author: franck
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
import os
from glob import glob

class Data_Reader:
    """Defines reader instance for training, evaluation and testing"""
    
    def __init__(self, n_features, records_length):
        """iniializes the parameters of the reader instance"""
        self.n_features = n_features
        self.records_length = records_length
        
        assert self.n_features <= self.records_length
        
    
    def read_data(self, filename_queue):
        """ Handles the data files: Since X and y have the same format for train
        , test and eval, we use one reader for all with a bunch of arguments"""
        
        
        
        # Reader op for the train dataset
        reader = tf.TextLineReader(skip_header_lines=1)
        
        
        
        
        #The reader reads one example
        _, record_string = reader.read(filename_queue)
    
        ##We need to decode the string as a csv and process it so data is feature be
        #a float
        record_defaults = [[1.0] for _ in range(self.records_length)]
        train_features = tf.decode_csv(record_string, 
                                       record_defaults=record_defaults)
        train_features = tf.stack(train_features, axis=0)[:self.n_features
                                 ]
        #print(type(train_features))
        #return record_string
        #print(len(train_features))
        return train_features


def input_pipeline(files_folder, train_graph, eval_graph, in_time_steps, 
                   out_time_steps, out_records_length, batch_size, num_epochs=None):
    """Defines the input pipeline that uses the different readers """
    
    
    
    #train_graph inputs
    with train_graph.as_default():
        #We need to create a queue that the reader is going to use 
        train_input_filenames = [files_folder + 'X_train_YHOO.csv']
        train_output_filenames = [files_folder + 'y_train_YHOO.csv']
        
        train_input_filename_queue = tf.train.string_input_producer(
          train_input_filenames, num_epochs=num_epochs, shuffle=True)
        train_output_filename_queue = tf.train.string_input_producer(
          train_output_filenames, num_epochs=num_epochs, shuffle=True)
            
        train_input = (
                Data_Reader(in_time_steps, in_time_steps)
                .read_data(train_input_filename_queue)
                )
        train_output = (
                Data_Reader(out_time_steps, out_records_length)
                .read_data(train_output_filename_queue)
                )
        train_decoder_input = tf.concat([tf.constant([-1.0]), train_output], axis=0)
        train_decoder_output = tf.concat([train_output, tf.constant([-1.0])], axis=0)
        
        # min_after_dequeue defines how big a buffer we will randomly sample
        #   from -- bigger means better shuffling but slower start up and more
        #   memory used.
        # capacity must be larger than min_after_dequeue and the amount larger
        #   determines the maximum we will prefetch.  Recommendation:
        #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
        min_after_dequeue = batch_size
        capacity = min_after_dequeue + 3 * batch_size
        
        (train_input_batch, 
             train_decoder_input_batch, 
             train_decoder_output) = tf.train.shuffle_batch(
          [train_input, train_decoder_input, train_decoder_output], 
          batch_size=batch_size, capacity=capacity,
          min_after_dequeue=min_after_dequeue)
        
        #print(train_input_batch.get_shape().as_list())
        assert train_input_batch.graph is train_graph
        
    #eval graph inputs
    with eval_graph.as_default():
        
        #We need to create a queue that the reader is going to use 
        eval_input_filenames = [files_folder + 'X_val_YHOO.csv']
        eval_output_filenames = [files_folder + 'y_val_YHOO.csv']
        
        
        eval_input_filename_queue = tf.train.string_input_producer(
          eval_input_filenames, num_epochs=1, shuffle=True)
        eval_output_filename_queue = tf.train.string_input_producer(
          eval_output_filenames, num_epochs=1, shuffle=True)
        
        eval_input = (
                Data_Reader(in_time_steps, in_time_steps)
                .read_data(eval_input_filename_queue)
                )
        eval_output = (
                Data_Reader(out_time_steps, out_records_length)
                .read_data(eval_output_filename_queue)
                )
        eval_decoder_input = tf.concat([tf.constant([-1.0]), eval_output], axis=0)
        eval_decoder_output = tf.concat([eval_output, tf.constant([-1.0])], axis=0)
        
        
        
        (eval_input_batch, 
             eval_decoder_input_batch, 
             eval_decoder_output_batch) = tf.train.shuffle_batch(
          [eval_input, eval_decoder_input, eval_decoder_output], 
          batch_size=1, capacity=capacity,
          min_after_dequeue=1)
        
        assert eval_input_batch.graph is eval_graph
    
    
    return (train_input_batch, 
         train_decoder_input_batch, 
         train_decoder_output, eval_input_batch, 
         eval_decoder_input_batch, 
         eval_decoder_output_batch)



class Seq2seq_ts:
    
    """ Model implementation for many to many time series predictions"""
    
    def __init__(self, files_folder, in_time_steps, out_time_steps,
                          out_records_length,
                          batch_size, num_epochs, max_gradient_norm,
                          logdir, learning_rate, num_units, keep_prob,
                          num_layers_encoder, num_layers_decoder):
        """ model parameters initialization"""
        self.files_folder = files_folder
        self.in_time_steps = in_time_steps
        self.out_time_steps = out_time_steps
        self.out_records_length = out_records_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.max_gradient_norm = max_gradient_norm
        self.logdir = logdir
        self.learning_rate = learning_rate
        self.num_units = num_units
        self.keep_prob = keep_prob
        self.num_layers_encoder = num_layers_encoder
        self.num_layers_decoder = num_layers_decoder
        
        ##We need to init the variables for the encoder, the decoder and the 
        #dense layer since we want them to be shared by the train and 
        #eval/infer graphs
        self.train_graph = tf.Graph()
        self.eval_graph = tf.Graph()
        #self.encoder_cell = tf.nn.rnn_cell.GRUCell(self.num_units_encoder)
        #self.decoder_cell = tf.nn.rnn_cell.GRUCell(self.num_units_decoder)
        #self.projection_layer = tf.layers.Dense(
        #        1, use_bias=False)
        
        
    def batch_generator(self):
        """Defining the model
        input/output batches for training and evaluation"""
        
        
        (self.train_input_batch, 
        self.train_decoder_input_batch, 
        self.train_decoder_output_batch, 
        self.eval_input_batch, 
        self.eval_decoder_input_batch, 
        self.eval_decoder_output_batch) = input_pipeline(self.files_folder,
                          self.train_graph, self.eval_graph, 
                          self.in_time_steps, self.out_time_steps, self.out_records_length,
                          self.batch_size, self.num_epochs)
        
        #reshaping the tensors to comply with the shape requirements of the 
        #encoder and decoder
        with self.train_graph.as_default() as train_graph:
            with tf.name_scope('variables'):
                
                self.train_input_batch =  tf.transpose(self.train_input_batch, 
                                                     [1,0],
                                                      name='train_input_batch')
                #print(self.train_input_batch.get_shape().as_list())
                self.train_input_batch = tf.expand_dims(self.train_input_batch,
                                                    -1)
    
                self.train_decoder_input_batch = tf.transpose(self.train_decoder_input_batch, 
                                                     [1,0],
                                                      name='train_decoder_input_batch')
                self.train_decoder_input_batch = tf.expand_dims(
                                                    self.train_decoder_input_batch,
                                                    -1)
    
                self.train_decoder_output_batch =tf.transpose(self.train_decoder_output_batch, 
                                                     [1,0])
                
                self.train_decoder_output_batch = tf.expand_dims(self.train_decoder_output_batch, 
                                                     -1
                                                     )
            
            assert self.train_decoder_input_batch.graph is train_graph
    
        with self.eval_graph.as_default():
            with tf.name_scope('variables'):
                #print(self.eval_input_batch.get_shape().as_list())
                self.eval_input_batch =tf.transpose(self.eval_input_batch, [1,0]
                                                     ) 
                #print(self.eval_input_batch.get_shape().as_list())
                self.eval_input_batch = tf.expand_dims(self.eval_input_batch,
                                                      -1
                                                      )
                   
                    
                self.eval_decoder_input_batch = tf.transpose(self.eval_decoder_input_batch, 
                                                     [1,0]) 
                self.eval_decoder_input_batch = tf.expand_dims(self.eval_decoder_input_batch, 
                                                     -1) 
                
                self.eval_decoder_output_batch = tf.transpose(self.eval_decoder_output_batch, 
                                                     [1,0])
                self.eval_decoder_output_batch = tf.expand_dims(self.eval_decoder_output_batch, 
                                                     -1)
        """
        (self.train_input_batch, 
        self.train_decoder_input_batch, 
        self.train_decoder_output_batch) = input_pipeline_test(self.files_folder, 
                          self.in_time_steps, self.out_time_steps, 
                          self.batch_size, self.num_epochs)
        
        self.train_input_batch =  tf.reshape(self.train_input_batch, 
                                             [self.in_time_steps,
                                              self.batch_size, 1])
        self.train_decoder_input_batch = tf.reshape(self.train_decoder_input_batch, 
                                             [self.out_time_steps +1,
                                              self.batch_size, 1])
        self.train_decoder_output_batch =tf.reshape(self.train_decoder_output_batch, 
                                             [self.out_time_steps +1,
                                              self.batch_size, 1])
        """
        
    def generate_batch(self, input_generator, decoder_input_generator,
                       decoder_output_generator, sess):
        """Generates a new batch"""

        inp, decoder_inp, decoder_out = sess.run([input_generator, 
                                                  decoder_input_generator,
                                                  decoder_output_generator])
        
        return inp, decoder_inp, decoder_out


        
    def create_encoder(self, graph, mode='Train'):
        """ Creates the encoder part of the graph """
        
        # Build RNN cell
        #encoder_cell = tf.nn.rnn_cell.GRUCell(self.num_units_encoder)
        
        # Run Dynamic RNN
        #   encoder_outputs: [max_time, None, num_units]
        #   encoder_state: [None, num_units]
        with graph.as_default():
            with tf.name_scope('encoder'):
                input_tensor = tf.placeholder(shape = [self.in_time_steps, None, 1],
                                              dtype=tf.float32, name='input_tensor')
                
                def lstm_cell():
                    encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_units,
                                                            state_is_tuple=True)
                    drop = tf.contrib.rnn.DropoutWrapper(encoder_cell, 
                                                         output_keep_prob=self.keep_prob)
                    return drop
                
                stacked_lstm = tf.contrib.rnn.MultiRNNCell(
                        [lstm_cell() for _ in range(self.num_layers_encoder)])
    
                
                size = self.batch_size if mode == 'Train' else 1
                initial_state=stacked_lstm.zero_state(size, tf.float32)
                
                encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                    stacked_lstm, input_tensor, 
                    initial_state = initial_state,
                    time_major=True,
                )
                
                dropout_encoder_outputs = tf.nn.dropout(encoder_outputs,
                                                        keep_prob=self.keep_prob)
    
        return dropout_encoder_outputs, encoder_state, input_tensor

        
    def create_helper(self, graph, mode='Train'):
        """
        Helper: allows the model to know when the processing of the input is 
        done and presents the right input to the decoder.
        We need to separate between
        the training and eval/infer modes because they do not decode the 
        same way"""
        
        with graph.as_default():
            with tf.name_scope('helper'):
                if mode == 'Train':
                    decoder_input = tf.placeholder(
                                                shape=[self.out_time_steps +1,
                                                       None,
                                                       1], dtype=tf.float32)
                    helper = tf.contrib.seq2seq.TrainingHelper(
                        decoder_input, 
                        sequence_length=tf.constant([self.out_time_steps+1]*self.batch_size),
                        time_major=True)
                    return helper, decoder_input
                else:
                    
                    # define inference custom helper
                    def initialize_fn():
                        finished = tf.tile([False], [1])
                        enc_inp_end = self.eval_input_batch[self.in_time_steps -1, 0, 0]
                        start_inputs = tf.reshape(enc_inp_end, shape=[1, 1]) 
                        return (finished, start_inputs)
                    
                    def sample_fn(time, outputs, state):
                        return tf.constant([0])
                    
                    def next_inputs_fn(time, outputs, state, sample_ids):
                        finished = time >= self.out_time_steps
                        next_inputs = outputs
                        return (finished, next_inputs, state)
                    
                    helper = tf.contrib.seq2seq.CustomHelper(
                        initialize_fn = initialize_fn,
                        sample_fn = sample_fn,                      
                        next_inputs_fn = next_inputs_fn)
                    return helper
        
    
    def create_attention_mechanism(self, encoder_outputs, graph):
        """ Create the attention mechanism part of the graph"""
        # attention_states: [batch_size, max_time, num_units]
        with graph.as_default():
            with tf.name_scope('attention'):
                memory_sequence_length = tf.placeholder(shape=[None], 
                                                        dtype=tf.int32,
                                                        name='memory_sequence_length')
                attention_states = tf.transpose(encoder_outputs, [1, 0, 2])
                
                 
                # Create an attention mechanism
                attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                    self.num_units, attention_states,
                    memory_sequence_length=memory_sequence_length, scale=True)
                
                def lstm_cell():
                    decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_units,
                                                            state_is_tuple=True)
                    drop = tf.contrib.rnn.DropoutWrapper(decoder_cell, 
                                                         output_keep_prob=self.keep_prob)
                    return drop
                
                stacked_lstm = tf.contrib.rnn.MultiRNNCell(
                        [lstm_cell() for _ in range(self.num_layers_decoder)])
                
                
                
                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                stacked_lstm, attention_mechanism,
                attention_layer_size=self.num_units)

        
        return decoder_cell, memory_sequence_length
    


    def create_decoder(self,  helper, encoder_state, 
                       decoder_cell, graph, mode='Train'):
        """Creates the decoder part of the graph """
        
        # Build RNN cell
        #decoder_cell = tf.nn.rnn_cell.GRUCell(self.num_units_decoder)
        
        # Helper: allows the model to know when the processing of the input is 
        #done and presents the right input to the decoder
        
        
        #Projection layer is going to convert the decoder output to a vector of
        #the right shape for each time step: here with univariate TS, the shape is 1
        #projection_layer = tf.layers.Dense(
        #        1, use_bias=False)

        with graph.as_default():
            with tf.name_scope('decoder'):
                # Training Decoder
                size = self.batch_size if mode=='Train'else 1
                
                projection_layer = tf.layers.Dense(
                    1, use_bias=True)
                
                initial_state = decoder_cell.zero_state(dtype=tf.float32, batch_size=size)
                initial_state = initial_state.clone(cell_state=encoder_state)
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_cell, helper,
                    initial_state=initial_state,
                    output_layer=projection_layer)
                # Dynamic decoding
                (outputs, _, final_sequence_lengths) = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=True)
                
                predictions = outputs.rnn_output
                

        
        return predictions, final_sequence_lengths
        
    def create_loss(self, predictions, graph):
        
        """Computes the loss of the model: Here we are going to use a regular
        MSE loss"""
        
        with graph.as_default():
            with tf.name_scope('loss'):
                output_tensor = tf.placeholder(shape=[
                                                self.out_time_steps +1,
                                                None,
                                                1],
                                              dtype=tf.float32,
                                              name='output_tensor')
                loss = tf.losses.mean_squared_error(output_tensor[:-1,:,:], 
                                                    predictions[:-1,:,:])
                #self.eval_loss = tf.losses.mean_squared_error(output_ts, predictions)
        return loss, output_tensor
        
    def create_optimizer(self, loss, graph):
        """Gradient computation and optimization part of the graph"""
        
        with graph.as_default():
            with tf.name_scope('optimizer'):
                # Calculate and clip gradients
                params = tf.trainable_variables()
                gradients = tf.gradients(loss, params)
                clipped_gradients, _ = tf.clip_by_global_norm(
                    gradients, self.max_gradient_norm)
                
                # Optimization
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                update_step = optimizer.apply_gradients(
                    zip(clipped_gradients, params))
        
        return update_step
    
    def create_summaries(self, graph, model, mode='Train'):
        """Create a summary for the loss and the eval loss"""
        with graph.as_default():
            with tf.name_scope('summaries'):
                if mode == 'Train':
                    tf.summary.scalar('{}_loss'.format(mode), model['loss'])
                    #tf.summary.scalar('eval_loss', model['loss'])
                    #because we have several summaries, we should merge them all
                    #into one op to makethem easier to manage
                    summary_op = tf.summary.merge_all()
        
                    assert summary_op.graph is graph
                    return summary_op
    
                else:
                    #Since we only want the avg loss at the end for inference
                    #We create a placeholder for this loss and we feed it the 
                    #value we computed at the end of the computation for all
                    #the examples to run the summary
                    model['avg_loss'] = tf.placeholder(shape=[], dtype=tf.float32)
                    tf.summary.scalar('{}_loss'.format(mode), model['avg_loss'])
                    #tf.summary.scalar('eval_loss', model['loss'])
                    #because we have several summaries, we should merge them all
                    #into one op to makethem easier to manage
                    summary_op = tf.summary.merge_all()
        
                    assert summary_op.graph is graph
                    return summary_op
                    
                    
    def create_utils(self, graph):
        """Create the clobal step tensor for monitoring"""
        with graph.as_default():
            global_step = tf.Variable(0, dtype=tf.int32
                                       , trainable=False, name='global_step')
            saver = tf.train.Saver()
            
        return global_step, saver

    def build_graphs(self):
        """ Build the graph for our model """
        self.batch_generator()
        #Creating the training graph
        self.model_train = {}
        (self.model_train['encoder_outputs'],
         self.model_train['encoder_state'],
         self.model_train['input_tensor']) = self.create_encoder(self.train_graph,
                                                                 mode='Train')
        (self.model_train['helper'],
         self.model_train['decoder_input'])= self.create_helper(
                                                        self.train_graph,
                                                        mode='Train' 
                                                        )
        (self.model_train['decoder_cell'],
         self.model_train['memory_sequence_length'])= self.create_attention_mechanism(
                                                self.model_train['encoder_outputs'], 
                                                self.train_graph)
        (self.model_train['predictions'],
         self.model_train['final_sequence_lengths'])= self.create_decoder(
                                            self.model_train['helper'],
                                            self.model_train['encoder_state'],
                                            self.model_train['decoder_cell'],
                                            self.train_graph,
                                            mode='Train'
                                                    )
        (self.model_train['loss'],
         self.model_train['output_tensor'])= self.create_loss(
                                        self.model_train['predictions'], 
                                        self.train_graph)
        self.model_train['optimizer_op'] = self.create_optimizer(
                                                self.model_train['loss'], 
                                                self.train_graph)
        self.model_train['summary_op'] = self.create_summaries(
                                            self.train_graph,
                                            self.model_train,
                                            mode='Train'
                                            )
        (self.model_train['global_step'],
         self.model_train['saver'])= self.create_utils(
                                            self.train_graph
                                            )
        
        #creating the eval graph
        self.model_eval = {}
        (self.model_eval['encoder_outputs'],
         self.model_eval['encoder_state'],
         self.model_eval['input_tensor']) = self.create_encoder(self.eval_graph,
                                                                mode='Eval')
        self.model_eval['helper']= self.create_helper(
                                             self.eval_graph, mode='Eval'
                                             )
        
        (self.model_eval['decoder_cell'],
         self.model_eval['memory_sequence_length'])= self.create_attention_mechanism(
                                                self.model_eval['encoder_outputs'],
                                                self.eval_graph)
        
        (self.model_eval['predictions'],
         self.model_eval['final_sequence_lengths'])= self.create_decoder(
                                            self.model_eval['helper'],
                                            self.model_eval['encoder_state'],
                                            self.model_eval['decoder_cell'],
                                            self.eval_graph,
                                            mode='Eval'
                                                    )
        (self.model_eval['loss'],
         self.model_eval['output_tensor']) = self.create_loss(
                                        self.model_eval['predictions'],
                                        self.eval_graph)
        self.model_eval['summary_op'] = self.create_summaries(
                                            self.eval_graph,
                                            self.model_eval,
                                            mode='Eval'
                                            )
        (self.model_eval['global_step'],
         self.model_eval['saver'])= self.create_utils(
                                            self.eval_graph
                                            )
        
        
    
            
            
    def test_input_pipeline(self):
        """ Test the the input pipeline is working properly by
        printing the batches"""
        
        #print(os.path.abspath(self.files_folder))
        config = tf.ConfigProto(inter_op_parallelism_threads=2)
        with tf.Session(config=config) as sess:
            sess.run([tf.global_variables_initializer(), 
                      tf.local_variables_initializer()])
            coord = tf.train.Coordinator()
            print('fine until here')
            threads = tf.train.start_queue_runners(coord=coord)
            (encoder_outputs) = sess.run(self.model_train['encoder_outputs'],
                                   feed_dict ={
                               self.model_train['input_tensor']:self.eval_input_batch.eval(session=sess),
                               self.model_train['output_tensor']:self.eval_decoder_output_batch.eval(session=sess),
                               self.model_train['memory_sequence_length']:(tf.ones([1], tf.int32)*(self.in_time_steps +1)).eval(session=sess),
                               self.model_train['initial_state']:tf.zeros(
                                              shape=[1, 
                                              self.num_units]).eval(session=sess)
                                           })
            print(encoder_outputs.shape)
            coord.request_stop()
            
            # Wait for threads to finish.
            coord.join(threads)
            
    def validation_step(self, train_step):
        """validation loss during the training"""
        config = tf.ConfigProto(inter_op_parallelism_threads=2)
        with tf.Session(config=config, graph=self.eval_graph) as val_sess:
            summary_writer = tf.summary.FileWriter(self.logdir, val_sess.graph)
            
            #restore the current parameters of the training graph
            self.model_eval['saver'].restore(val_sess, self.checkpoint_path)

            
            # Initialize the variables (like the epoch counter).
            val_sess.run([tf.local_variables_initializer()])
            # Start input enqueue threads.
            eval_coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=val_sess, coord=eval_coord)
            avg_val_loss = 0.0
            val_cpt = 0
            try:
                while not eval_coord.should_stop():
                    # Run training steps or whatever
                    (input_batch_value, 
                    decoder_input_batch_value,
                    decoder_output_batch_value) = self.generate_batch(
                                                    self.eval_input_batch,
                                                    self.eval_decoder_input_batch,
                                                    self.eval_decoder_output_batch,
                                                    val_sess)
                    #print(sess.run(self.train_input_batch))
                    val_loss= val_sess.run(self.model_eval['loss'],
                                           feed_dict ={
                                       self.model_eval['input_tensor']:input_batch_value,
                                       self.model_eval['output_tensor']:decoder_output_batch_value,
                                       self.model_eval['memory_sequence_length']:(tf.ones([1], tf.int32)*(self.in_time_steps +1)).eval(session=val_sess)
                                                   })
                    avg_val_loss += val_loss
                    val_cpt += 1
            except tf.errors.OutOfRangeError:
                print('End of validation step ')
            finally:
                #Create the summary for the step
                summary = val_sess.run(self.model_eval['summary_op'],
                                       feed_dict={
                                           self.model_eval['avg_loss']:avg_val_loss
                                               })
                
                # When done, ask the threads to stop.
                eval_coord.request_stop()
            
            # Wait for threads to finish.
            eval_coord.join(threads)
            
            summary_writer.add_summary(summary, global_step=train_step)
            
            return avg_val_loss /val_cpt
            
    def train(self):
        """trains the model, running an evaluation periodically to check the
        evolution of the validation loss"""
        
        #create the summaries
        #self.create_summaries()
        
        ######PHASE 2: Execute the computations
        config = tf.ConfigProto(inter_op_parallelism_threads=2)
        with tf.Session(config=config, graph=self.train_graph) as sess:
            # Initialize the variables (like the epoch counter).
            sess.run([tf.global_variables_initializer(),
                      tf.local_variables_initializer()])
            summary_writer = tf.summary.FileWriter(self.logdir, sess.graph)
            # Start input enqueue threads.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            avg_loss = 0.0
            cpt = 0
            try:
                while not coord.should_stop():
                    # Run training steps or whatever
                    
                    #print(sess.run(self.train_input_batch))
                    (input_batch_value, 
                    decoder_input_batch_value,
                    decoder_output_batch_value) = self.generate_batch(
                                                    self.train_input_batch,
                                                    self.train_decoder_input_batch,
                                                    self.train_decoder_output_batch,
                                                    sess)
                    (loss_batch, 
                     _, 
                     summary,
                     predictions) = sess.run([self.model_train['loss'], 
                                       self.model_train['optimizer_op'],
                                       self.model_train['summary_op'],
                                       self.model_train['predictions']],
                                           feed_dict ={
                                       self.model_train['input_tensor']:input_batch_value,
                                       self.model_train['decoder_input']:decoder_input_batch_value,
                                       self.model_train['output_tensor']:decoder_output_batch_value,
                                       self.model_train['memory_sequence_length']:(tf.ones([self.batch_size], tf.int32)*(self.in_time_steps +1)).eval(session=sess)
                                                   })
                    avg_loss += loss_batch
                    summary_writer.add_summary(summary, global_step=cpt)
                    
                    cpt +=1
                        
                    if cpt % 50 == 0:
                        print('Average loss at step {}: {:5.1f}'.format( cpt +1,
                                                                      avg_loss / (cpt + 1)))
                        
                        self.checkpoint_path = self.model_train['saver'].save(sess=sess, save_path=logdir +'/checkpoint_directory/seq2seq_ts',
                               global_step=self.model_train['global_step'])
                        
                        avg_val_loss = self.validation_step(train_step=cpt)
                        print('Average validation loss at step {}: {:5.1f}'.format( cpt +1,
                                                                      avg_val_loss ))
                        
                        print('Training Example:\n {}'.format( decoder_output_batch_value[:,0,:]))
                        print('Predicted Value:\n {}'.format(predictions[:,0,:]))
            
            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()
            
            # Wait for threads to finish.
            coord.join(threads)
             

files_folder = '../data/processed/'
in_time_steps = 28
out_time_steps = 14
out_records_length = 28
batch_size = 32
num_epochs = 100
max_gradient_norm = 5
logdir = '../logs'
learning_rate= 0.002
num_units = 64
num_layers_encoder = 2
num_layers_decoder = 2
keep_prob = .8


def main():
    #print(os.path.abspath(files_folder))
    seq2seq = Seq2seq_ts(files_folder, in_time_steps, out_time_steps, 
                         out_records_length,
                          batch_size, num_epochs, max_gradient_norm,
                          logdir, learning_rate, num_units, keep_prob,
                          num_layers_encoder, num_layers_decoder)
    
    #seq2seq.create_batch()
    
    seq2seq.build_graphs()
    #seq2seq.test_input_pipeline()
    seq2seq.train()
    
if __name__ == '__main__':
    main()