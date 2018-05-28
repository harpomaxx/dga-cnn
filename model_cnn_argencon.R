# keras model used in ARGECON 2018 (rejected) paper

default_keras_model_cnn_argencon_parameters_tune=list(
  nb_filter = c(256,128,63,32),
  kernel_size = c(8,4,2),
  embedingdim = c(100,50,32),
  hidden_size = c(1024,512,128)
)

default_keras_model_cnn_argencon_parameters=list(
  nb_filter = 128,
  #nb_filter = 256,
  kernel_size = 8,
  #kernel_size = 4,
  embedingdim = 100,
  hidden_size = 1024
)

keras_model_cnn_argencon<-function(x,parameters=default_keras_model_cnn_argencon_parameters)
{
  input_shape <- dim(x)[2]
  inputs<-layer_input(shape = input_shape) 
  embeding<- inputs %>% layer_embedding(length(valid_characters_vector), parameters$embedingdim , input_length = input_shape)
  conv1d <- embeding %>%
    layer_conv_1d(filters = parameters$nb_filter, kernel_size = parameters$kernel_size, activation = 'relu', padding='valid',strides=1) %>%
    layer_flatten() %>%
    layer_dense(parameters$hidden_size,activation='relu') %>%
    layer_dense(1,activation = 'sigmoid')
  
  #compile model
  model <- keras_model(inputs = inputs, outputs = conv1d) 
  model %>% compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = c('accuracy')
  )
  return (model)
}
# Registering new model
funcs[["cnn_argencon"]]=keras_model_cnn_argencon