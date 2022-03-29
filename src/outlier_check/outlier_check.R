dn <- read.csv("Amyloid Positivity/Gatekeeping_Amyloid_Positivity/Gatekeeping-Amyloid-Positivity/data/outlier_values_smoothed.csv")
data_to_check <- read.csv("Amyloid Positivity/Gatekeeping_Amyloid_Positivity/Gatekeeping-Amyloid-Positivity/data/munich_fdg.csv")

colnames(data_to_check) <- gsub(" ", "", colnames(data_to_check))
names <- colnames(subset(data_to_check, select=-ID))

out <- list()
cat("OUTLIER DETECTION\nIDs that appear are outside of 3IQR")
for (i in 1:length(names)){
  d <- subset(data_to_check, select=names[i])
  test <- which((d < dn["lowerbound",names[i]]) | d > dn["upperbound", names[i]])
  if (length(test)>0){
    cat(names[i])
    catt(data_to_check$ID[which((d < dn["lowerbound",names[i]]) | d > dn["upperbound", names[i]])])
  }
}
