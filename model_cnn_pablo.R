# Last CNN from Pablo's Repo

default_keras_model_cnn_pablo=list(
  nb_filter = 32,
  kernel_size = 4,
  embedingdim = 32,
  hidden_size = 64
)

sum_1d<-function(x){
  k<-backend()
  k$sum(x,axis=1L)
  
}

keras_model_cnn_pablo<-function(x,parameters=default_keras_model_cnn_pablo)
{
  
  input_shape <- dim(x)[2]
  inputs<-layer_input(shape = input_shape) 
  embeding<- inputs %>% layer_embedding(length(valid_characters_vector), parameters$embedingdim , input_length = input_shape)
  conv1d <- embeding %>%
    layer_conv_1d(filters = parameters$nb_filter, kernel_size = parameters$kernel_size, activation = 'selu', padding='valid',strides=1) %>%
    layer_batch_normalization() %>%
    #layer_flatten() %>%
    layer_lambda(sum_1d,dtype='float64') %>%
    layer_dense(parameters$hidden_size,activation='selu') %>%
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
funcs[["cnn_pablo"]]=keras_model_cnn_pablo