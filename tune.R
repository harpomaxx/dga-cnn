
tune_model<-function(dataset,experimentname,modelid){
  models_results<-c()
  parameters_combinations <- expand.grid(
    eval(
      parse(
        text=paste("default_keras_model_",names(funcs)[1],"_parameters_tune",sep="") # TODO: verify existence
      )
    )
  )
  for (i in 1:nrow(parameters_combinations)){
    comb <-parameters_combinations[i,]
    print(comb)
    results<-evaluate_model_cv(modelfun=funcs[[modelid]],model_parameters=comb,data=dataset,k=5,experimentname = experimentname)
    value<-results$result %>% filter(metric=='F1') %>% group_by(metric) %>% summarise(F1=mean(value)) %>% select(F1)
    models_results<-rbind(models_results,cbind(value=value,comb))
    print(models_results)
    write_csv(models_results,col_names = T,path=paste(results_dir,"results_tuning_",opt$experimenttag,".csv",sep=""))
  }
  return (model_results)
}
