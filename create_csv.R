suppressPackageStartupMessages(library("readr"))
suppressPackageStartupMessages(library("tibble"))
suppressPackageStartupMessages(library("dplyr"))
suppressPackageStartupMessages(library("stringr"))
# Function for creating a csv from sql db

create_csv<- function (datafilename){
  con <- DBI::dbConnect(RMySQL::MySQL(), 
                        host = "localhost",
                        user = "root",
                        dbname="DGA",
                        password = "dios"
  )
  domains_db <- tbl(con, "Domains")
  domains_tbl<-collect(domains_db)
  domains_tbl<-domains_tbl %>%  mutate(label=str_replace(label,"DGA.360","DGA")) %>% mutate(label=tolower(label))
  domains_tbl<-domains_tbl %>%  mutate(label=str_replace(label,"normal$","normal.bambenek")) %>% mutate(label=tolower(label))
  domains_tbl<-domains_tbl %>%  mutate(label=str_replace(label,"conflicker","conficker")) %>% mutate(label=tolower(label))
  readr::write_csv(domains_tbl, path = paste('./datasets/',datafilename,sep=''))
  return (domains_tbl)
}
