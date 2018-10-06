library(keras)
library(plumber)

# loading the DGA model for classifier 
#model<-load_model_hdf5("/home/harpo/Dropbox/ongoing-work/git-repos/dga-wb-r/docker/app/pmodel.h5")
model<-load_model_hdf5("/app/pmodel.h5")

valid_characters <- "$abcdefghijklmnopqrstuvwxyz0123456789-_."
valid_characters_vector <- strsplit(valid_characters,split="")[[1]]
tokens <- 0:length(valid_characters_vector)
names(tokens) <- valid_characters_vector


# testing function
#* @get /echo
function(msg="Hi!"){
  list(msg = paste("The message is: ", msg))
}

# DGA prediction function
#* @get /predict
#* @serializer unboxedJSON
function(domain){
  domain_encoded <-
	   
	  	sapply( 
			unlist(strsplit(tolower(domain),split="")), function(x) tokens [[x]] 
			) 
	  		
		       #	{tokens[[x]] })
  domain_encoded<-pad_sequences(t(domain_encoded),maxlen=45,padding='post', truncating='post')

  prediction<-predict(model,domain_encoded)
  return(list(domain=domain,class=ifelse(prediction[1]>0.9,"dga","normal"),probability=prediction[1]))
}
