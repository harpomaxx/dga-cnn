library(keras)
# Compile keras model used in ARGECON 2018 paper
keras_model_cnn_argencon<-function(x)
{
  input_shape <- dim(x)[2]
  inputs<-layer_input(shape = input_shape) 
  
  nb_filter <- 256
  kernel_size <- 4
  embedingdim <- 100
  
  embeding<- inputs %>% layer_embedding(length(valid_characters_vector), embedingdim , input_length = input_shape)
  
  conv1d <- embeding %>%
    layer_conv_1d(filters = nb_filter, kernel_size = kernel_size, activation = 'relu', padding='valid',strides=1) %>%
    layer_flatten() %>%
    layer_dense(1024,activation='relu') %>%
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
# Train model
train_model <- function(x,y, model){
  history<-model %>% fit(x,y,epochs = 5, batch_size = 4096, validation_split = 0.2,verbose = T)
  model %>% save_model_hdf5("cnn-nomlp.h5")
  return(list(model=model,history=history))
}
