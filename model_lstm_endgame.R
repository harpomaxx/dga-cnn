# keras model used in ENDGAME LSTM 2016 paper
keras_model_lstm_endgame<-function(x){
  
  input_shape <- dim(x)[2]
  inputs<-layer_input(shape = input_shape) 
  
  embeding<- inputs %>% layer_embedding(length(valid_characters_vector), 128 , input_length = input_shape)
  
  lstm <- embeding %>%
    layer_lstm(units = 128) %>%
    layer_dropout(rate = 0.5) %>%
    layer_dense(1, activation = 'sigmoid')
  
  #compile model
  model_endgame <- keras_model(inputs = inputs, outputs = lstm)
  model_endgame %>% compile(
    optimizer = 'rmsprop',
    loss = 'binary_crossentropy',
    metrics = c('accuracy')
  )
  return(model_endgame)
}

funcs[["lstm_endgame"]]=keras_model_lstm_endgame