source("create_csv.R")
source("preprocess.R")
source("build_model.R")
library(caret)
set.seed(12121) # For ensuring repeatibility 

## Configuration paths 
results_dir='./results/'
models_dir='./models/'
datasets_dir='./datasets/'

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


execid='cnn-test'
maxlen=45
#dataset<-create_csv("argencon.csv")

dataset<-read_csv(paste(datasets_dir,"argencon.csv",sep=""))
dataset$domain1<-str_split(dataset$domain,"\\.",simplify = T)[,1]

dindex<-train_test_sample(dataset,0.4)
train_dataset<-dataset[dindex,]
test_dataset<-dataset[-dindex,]

## Dataset transformation usually requires a lot of time. Some sort of caching needed
print("[] Generate Datasets")
train_dataset_keras<-build_dataset(as.matrix(train_dataset),maxlen)
test_dataset_keras<-build_dataset(as.matrix(test_dataset),maxlen)
print("[] Evaluate model ")
results<-evaluate_model(modelfun=keras_model_cnn_argencon,data=train_dataset_keras,k=5)


## Save results to csv
names(results$result)<-c("k","metric","value")
write_csv(results$result,col_names = T,path=paste(results_dir,"results_",execid,".csv",sep=""))
write_csv(results$resultperclass,col_names = T,path=paste(results_dir,"results_per_subclass",execid,".csv",sep=""))
