source("create_csv.R")
source("preprocess.R")
source("build_model.R")


suppressPackageStartupMessages(library("optparse"))
suppressPackageStartupMessages(library("caret"))
suppressPackageStartupMessages(library("e1071"))

option_list <- list(
  make_option("--generate", action="store_true",  help = "generate train and test files", default=FALSE)
)
opt <- parse_args(OptionParser(option_list=option_list))


set.seed(12121) # For ensuring repeatibility 

## Configuration paths 
results_dir='./results/'
models_dir='./models/'
datasets_dir='./datasets/'
dataset_default='argencon.csv.gz'

# Function implemeting k-fold cross validation 
# modelfun: reference to the function that create the keras model
# data : dataset used for crossvalidation
#       k : the number of folds in k-fold (default 5)

evaluate_model <- function(data,k=5, modelfun = keras_model_cnn_argencon){
  knum=k
  result=c()
  result_per_subclass=c()
  
  folds <- createFolds(factor(data$label), k = knum, list = FALSE)
  for (k in 1:knum){
    
    train_dataset_x<-data$encode[ which(folds !=k ),]
    train_dataset_y<-ifelse(grepl("normal",data$label[ which(folds !=k)]) ,0,1)
    
    test_dataset_x<-data$encode[ which(folds == k),]
    test_dataset_y<-ifelse(grepl("normal",data$label[ which( folds ==k)]),0,1)
    
    model_learned<-train_model(x=train_dataset_x,
                               y=train_dataset_y,
                               model=modelfun(train_dataset_x))
    
    predsprobs<-model_learned$model %>% predict(test_dataset_x, batch_size=4096)
    preds<-ifelse(predsprobs>0.9,1,0)
    confmatrix<-confusionMatrix(as.factor(preds),as.factor(test_dataset_y),positive = '1')
    recall<-data.frame(label=data$label[ which( folds ==k)], class=test_dataset_y,predicted_class=preds) %>% 
      group_by(label) %>% summarise(recall=sum(predicted_class==class)/n(),support=n()) 
    
    result<-rbind(result,cbind(k=k,value=as.data.frame(confmatrix$byClass) %>% rownames_to_column()))
    result_per_subclass=rbind(result_per_subclass,cbind(k=k,recall))
  }
  return (list(result=result, resultperclass=result_per_subclass))
}

##
## MAIN Section
##
execid='cnn-test' # id used during the experiment. Output file will used either
maxlen=45         # the maximum length of the domain name considerd for input of the NN

#dataset<-create_csv("argencon.csv")

if (!file.exists("datasets/.train_dataset_keras.rd")){
	print(" []  train and test files not found. Generating")
	opt$generate<-TRUE
}


if ( opt$generate){
  dataset<-read_csv(paste(datasets_dir,dataset_default,sep=""))
  dataset$domain1<-str_split(dataset$domain,"\\.",simplify = T)[,1]

  dindex<-train_test_sample(dataset,0.4)
  train_dataset<-dataset[dindex,]
  test_dataset<-dataset[-dindex,]

# Dataset transformation usually requires a lot of time. Some sort of caching needed

  print("[] Generating Datasets")
  train_dataset_keras<-build_dataset(as.matrix(train_dataset),maxlen)
  save(train_dataset_keras,file = "datasets/.train_dataset_keras.rd")
  test_dataset_keras<-build_dataset(as.matrix(test_dataset),maxlen)
  save(test_dataset_keras,file = "datasets/.test_dataset_keras.rd")
} else {
  print("[] Loading Datasets ")
  load(file='datasets/.train_dataset_keras.rd')
  load(file='datasets/.test_dataset_keras.rd')
}

print("[] Evaluating model ")
results<-evaluate_model(modelfun=keras_model_cnn_argencon,data=train_dataset_keras,k=5)


## Save results to csv
print("[] Saving results ")
names(results$result)<-c("k","metric","value")
write_csv(results$result,col_names = T,path=paste(results_dir,"results_",execid,".csv",sep=""))
write_csv(results$resultperclass,col_names = T,path=paste(results_dir,"results_per_subclass_",execid,".csv",sep=""))

# SAVE model
#TODO
