rm(list=ls())
df <- read.csv2("1_Amyloid Positivity/Gatekeeping_Amyloid_Positivity/Gatekeeping-Amyloid-Positivity/data/parcels_all_nomask.csv", dec = ".")
merge <- read.csv2("1_Amyloid Positivity/Gatekeeping_Amyloid_Positivity/Gatekeeping-Amyloid-Positivity/data/ADNI_merge_Amyloid_4_noNA.csv", dec = ".")
#merge <- read.csv("1_Amyloid Positivity/Gatekeeping_Amyloid_Positivity/Gatekeeping-Amyloid-Positivity/data/ADNI_merge_Amyloid_3.csv", dec = ".")
imc <- read.csv2('1_Amyloid Positivity/Gatekeeping_Amyloid_Positivity/Gatekeeping-Amyloid-Positivity/data/munich_pib_new_noNA.csv')

#df <- df[df$ID %in% merge$PTID[merge$AMY_STAT!="ERROR"],]
df <- df[df$ID %in% merge$PTID[merge$Amyloid_OK=="ok"],]
merge <- merge[merge$PTID %in% df$ID,]
#table(merge$Amyloid_OK, useNA = "always")
table(merge$AMY_STAT, useNA = "always")

iqr_val <- 3
# Outlier analyses
colnames(df) <- gsub(" ", "", colnames(df))
names <- colnames(subset(df, select=-ID))
ids <- df$ID

# Find outliers in current dataset
outliers <- list()
outlier_hubs <- data.frame(matrix(NA,nrow = length(names)))
outlier_hubs$ROI <- names
for (i in 1:length(names)){
  d <- subset(df, select=names[i])
  b <- boxplot(d, range = iqr_val)
  roi_outliers <- length(which(t(d) %in% b$out))
  outlier_hubs$n[outlier_hubs$ROI == names[i]] <- roi_outliers
  outliers <- c(outliers, ids[which(t(d) %in% b$out)])
}

length(unique(outliers))
#dev.new(width=20,height=10, unit = 'in', noRStudioGD = TRUE)
par(mar=c(9,4,2,10)+.1)
imp_ROI <- list()
imp_ind <- which(outlier_hubs$n>0)
for (i in 1:90){
  if (i %in% imp_ind){
    imp_ROI <- c(imp_ROI, outlier_hubs$ROI[i])
  } else {imp_ROI <- c(imp_ROI,"")}
}
par(family="serif")
barplot(outlier_hubs$n, ylim = c(0,15), xlim = c(0,90),
        names.arg = imp_ROI, las = 2, cex.names = 0.51,
        cex.lab = 0.8,
        xlab = "", ylab = "Number of Participants")
list(unique(outliers))
data_wo_outliers <- df[!(df$ID %in% outliers),]
merge_wo_outliers <- merge[merge$PTID %in% data_wo_outliers$ID,]

# --> total reduction of sample size to 537 before outlier correction

write.csv(data_wo_outliers,
          "1_Amyloid Positivity/Gatekeeping_Amyloid_Positivity/Gatekeeping-Amyloid-Positivity/data/parcels_rev1.csv", row.names = FALSE)
write.csv(merge_wo_outliers,
          "1_Amyloid Positivity/Gatekeeping_Amyloid_Positivity/Gatekeeping-Amyloid-Positivity/data/ADNI_merge_nooutliers_rev1.csv", row.names = FALSE)


library(comprehenr)
library(dplyr)
age <- c(merge_wo_outliers$AGE, imc$Age)
mean(age, na.rm = TRUE)
sd(age, na.rm = TRUE)
sex_adni <- to_vec(for(i in 1:nrow(merge)) if(merge$PTGENDER[i]=="Female") 1 else 2)
sex <- c(sex_adni, imc$Gender)
table(sex)
amyloid <- c(merge_wo_outliers$Amyloid_Status, imc$SUMMARY_bin)
table(amyloid)
apoe <- c(merge_wo_outliers$APOE4, imc$APOE)
table(apoe)

mean(age[amyloid == 0])
sd(age[amyloid == 0])
mean(age[amyloid == 1], na.rm = TRUE)
sd(age[amyloid == 1], na.rm = TRUE)

t.test(age[amyloid==0 & apoe==0], age[amyloid==1 & apoe==0])
t.test(age[apoe>0 & amyloid==0], age[apoe>0 & amyloid==1])

t.test(age[amyloid == 1], age[amyloid == 0], na.omit = TRUE)

table(sex[amyloid==0])
table(sex[amyloid==1])

write.csv(data_wo_outliers,"C:/Users/doeringe/Documents/Amyloid Positivity/Gatekeeping_Amyloid_Positivity/Gatekeeping-Amyloid-Positivity/data/parcels_nooutliers_iqr3_amy_final.csv", row.names = FALSE)

# Save outlier boundaries for other datasets
dn <- data.frame(lapply(subset(df, select=-ID),quantile,probs=c(0.25,0.75)))
dn["IQR",] <- lapply(subset(df, select=-ID),IQR)
dn["IQR",] <- as.numeric(as.character(dn["IQR",]))
dn["upperbound",] <- dn["75%",]+(iqr_val*dn["IQR",])
dn["lowerbound",] <- dn["25%",]-(iqr_val*dn["IQR",])

write.csv(dn, "Amyloid Positivity/Gatekeeping_Amyloid_Positivity/Gatekeeping-Amyloid-Positivity/data/outlier_values_smoothed.csv", row.names = FALSE)

