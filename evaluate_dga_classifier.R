source("create_csv.R")
source("preprocess.R")
source("build_model.R")
source("tune.R")


suppressPackageStartupMessages(library("optparse"))
suppressPackageStartupMessages(library("caret"))
suppressPackageStartupMessages(library("e1071"))

option_list <- list(
  make_option("--generate", action="store_true",  help = "Generate train and test files", default=FALSE),
  make_option("--experimenttag", action="store", type="character", default="default-experiment", help = "Set experiment tag id "),
  make_option("--modelid", action="store", type="numeric", default=1, help = "Select between different models"),
  make_option("--list-available-models", action="store_true", help = "List different models", dest="list_models",default=FALSE),
  make_option("--tune", action="store_true", help = "Tune the selected model",default=FALSE),
  make_option("--maxlen", action="store", type="numeric", default=45, help = "Set the maximun length of the domain name considered")
)
opt <- parse_args(OptionParser(option_list=option_list))


set.seed(12121) # For ensuring repeatibility 
# tensorflow session setup
#library(tensorflow)
#get_session<-function(gpu_fraction=0.333){
#  gpu_options = tf$GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
#                              allow_growth=TRUE)
#return (tf$Session(config=tf$ConfigProto(gpu_options=gpu_options)))
#}

#keras::k_set_session(get_session())


## Configuration paths 
results_dir='./results/'
models_dir='./models/'
datasets_dir='./datasets/'
dataset_default='JISA2018.csv.gz'


get_predictions <- function(model_learned, test_dataset_x,threshold=0.9) {
  predsprobs<-model_learned$model %>% predict(test_dataset_x, batch_size=4096)
  preds<-ifelse(predsprobs>threshold,1,0)
  return (preds)
}

calculate_recall <-function(dataset){
  recall<-dataset %>% group_by(label) %>% summarise(recall=sum(predicted_class==class)/n(),support=n()) 
  return(recall)
}


# Function implemeting k-fold cross validation 
# modelfun: reference to the function that create the keras model
# data : dataset used for crossvalidation
#       k : the number of folds in k-fold (default 5)

evaluate_model_cv <- function(data,k=5, modelfun = keras_model_cnn_argencon,model_parameters=NULL,experimentname="default"){
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
                               model=modelfun(train_dataset_x,parameters=model_parameters),
                               modelname = paste(experimentname,"_model",sep="")
                               )
    preds<-get_predictions(model_learned,test_dataset_x)
    
    confmatrix<-confusionMatrix(as.factor(preds),as.factor(test_dataset_y),positive = '1')
    result<-rbind(result,cbind(k=k,value=as.data.frame(confmatrix$byClass) %>% rownames_to_column()))
    
    #recall<-data.frame(label=data$label[ which( folds ==k)], class=test_dataset_y,predicted_class=preds) %>% 
    #  group_by(label) %>% summarise(recall=sum(predicted_class==class)/n(),support=n()) 
    recall<-calculate_recall(data.frame(label=data$label[ which( folds ==k)], class=test_dataset_y,predicted_class=preds))
    result_per_subclass=rbind(result_per_subclass,cbind(k=k,recall))
   
    rm(model_learned)
    gc()
    
    keras::k_clear_session()
  }
  names(result)<-c("k","metric","value")
  return (list(result=result, resultperclass=result_per_subclass))
}

################################################################################################
## MAIN Section
################################################################################################
#opt$experimenttag='cnn-test' # id used during the experiment. Output file will used either
maxlen=opt$maxlen         # the maximum length of the domain name considerd for input of the NN

#dataset<-create_csv("argencon.csv")

if (opt$list_models){
  print (names(funcs))
  quit()
}

# Generate new tokenized datasets from .csv files
if (!file.exists("datasets/.train_dataset_keras.rd")){
	print(" []  train and test files not found. Generating")
	opt$generate<-TRUE
}

if ( opt$generate){
  print("[] Generating Datasets")
  datasets<-build_train_test(paste(datasets_dir,dataset_default,sep=""),maxlen)
  train_dataset_keras<-datasets$train
  test_dataset_keras<-datasets$test
} else {
  print("[] Loading Datasets ")
  load(file='datasets/.train_dataset_keras.rd')
  load(file='datasets/.test_dataset_keras.rd')
}

if (opt$tune){
  print("[] Tuning model hyperparameters")
  model_results<-tune_model(dataset=train_dataset_keras,modelid=opt$modelid,experimentname=opt$experimenttag )
  write_csv(models_results,col_names = T,path=paste(results_dir,"results_tuning_",opt$experimenttag,".csv",sep=""))
  quit()
}
print("[] Evaluating model on trainset")
selected_parameters<- 
  eval(
    parse(
      text=paste("default_keras_model_",names(funcs)[1],"_parameters",sep="") # TODO: verify existence
    )
  )
results<-evaluate_model_cv(modelfun=funcs[[opt$modelid]],model_parameters=selected_parameters,data=train_dataset_keras,k=5,experimentname = opt$experimenttag)


## Save results to csv
print("[] Saving results ")
names(results$result)<-c("k","metric","value")
write_csv(results$result,col_names = T,path=paste(results_dir,"results_",opt$experimenttag,".csv",sep=""))
write_csv(results$resultperclass,col_names = T,path=paste(results_dir,"results_per_subclass_",opt$experimenttag,".csv",sep=""))

# SAVE model
#TODO
