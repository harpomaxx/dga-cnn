library(keras)

valid_characters <- "$abcdefghijklmnopqrstuvwxyz0123456789-_."
valid_characters_vector <- strsplit(valid_characters,split="")[[1]]
tokens <- 0:length(valid_characters_vector)
names(tokens) <- valid_characters_vector

# convert dataset to matrix of tokens  
tokenize <- function(data,labels){
  #string_char_vector <- strsplit(string,split="")[[1]]
  x_data <- sapply(
    lapply(data,function(x) strsplit(tolower(x),split="")),function(x) lapply(x[[1]], function(x) tokens[[x]]))
  padded_token<-pad_sequences(x_data,maxlen=45,padding='post', truncating='post')
  return (list(encode=padded_token,domain=data, label=labels))
  #return (list(encode=padded_token))
} 
# convert vector with char tokens to one-hot encodings
to_onehot <- function(data,shape){
  train <- array(0,dim=c(shape[1],shape[2],shape[3]))
  for (i in 1:shape[1]){
    for (j in 1:shape[2])
      train[i,j,data[i,j]] <- 1
  }
  return (train)
}

# Create a dataset using tokenizer 
build_dataset<- function(data,maxlen){
  dataset<-tokenize(data[,1],data[,2])
  shape=c(nrow(dataset$encode),maxlen,length(valid_characters_vector))
  #dataset$encode<-to_onehot(dataset$encode,shape)
  return(dataset)
}

train_test_sample<-function(x,percent=0.7){
  smp_size <- floor(percent * nrow(x))
  train_ind <- sample(seq_len(nrow(x)), size = smp_size)
  return (train_ind)
}

build_train_test<-function(datasetfile,maxlen){
  dataset<-read_csv(datasetfile)
  dataset$domain1<-str_split(dataset$domain,"\\.",simplify = T)[,1]
  dindex<-train_test_sample(dataset,0.4)
  train_dataset<-dataset[dindex,]
  test_dataset<-dataset[-dindex,]
  # Dataset transformation usually requires a lot of time. Some sort of caching needed
  train_dataset_keras<-build_dataset(as.matrix(train_dataset),maxlen)
  save(train_dataset_keras,file = "datasets/.train_dataset_keras.rd")
  test_dataset_keras<-build_dataset(as.matrix(test_dataset),maxlen)
  save(test_dataset_keras,file = "datasets/.test_dataset_keras.rd")
  return(list(train=train_dataset_keras,test=test_dataset_keras))
}