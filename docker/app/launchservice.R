library(plumber)
r <- plumb("/app/dga-classifier-service.R") 
#r <- plumb("/home/harpo/Dropbox/ongoing-work/git-repos/dga-wb-r/docker/app/dga-classifier-service.R") 

r$run(host = "0.0.0.0",port=8000)
