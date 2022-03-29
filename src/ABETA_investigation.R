rm(list=ls())
pat <- read.csv2(paste("1_Amyloid Positivity/Gatekeeping_Amyloid_Positivity/",
                       "Gatekeeping-Amyloid-Positivity/data/",
                       "ADNI_merge_Amyloid_3.csv", sep=""),
                 na.strings = "", dec = ".")
colnames(pat)[which(names(pat) == "Amyloid_Status")] <- "AMY_STAT"

# only include data complete for APOE4, AGE and SEX
pat$APOE4 <- as.numeric(pat$APOE4)
pat <- pat[!is.na(pat$APOE4),]
pat <- pat[!is.na(pat$AGE),]
pat <- pat[!is.na(pat$PTGENDER),]
pat$av45_bin <- as.numeric(pat$av45_bin)
pat$fbb_bin <- as.numeric(pat$fbb_bin)
pat$pib_bin <- as.numeric(pat$pib_bin)
pat$AMYSTAT_PET <- ifelse(!is.na(pat$av45_bin), pat$av45_bin, 
                         ifelse(!is.na(pat$fbb_bin), pat$fbb_bin, pat$pib_bin))
table(pat$AMYSTAT_PET)
write.csv(pat, paste("1_Amyloid Positivity/Gatekeeping_Amyloid_Positivity/",
                    "Gatekeeping-Amyloid-Positivity/data/",
                    "ADNI_merge_Amyloid_rev1_full.csv", sep=""), row.names = F)

ab4240 <- read.csv2(paste("1_Amyloid Positivity/Gatekeeping_Amyloid_Positivity/",
                          "Gatekeeping-Amyloid-Positivity/data/",
                          "UPENNBIOMKAB4240_merge.csv", sep=""),
                   dec = ".")


ab4240 <- ab4240[ab4240$VISCODE2=="bl",]
ab4240$AMYSTAT_ab4240 <- ifelse(as.numeric(ab4240$AB4240)<0.059, 1, 0)
ab4240 <- subset(ab4240, select = c(RID, ABETA40, ABETA42, AMYSTAT_ab4240, PTAU, TAU))
df <- merge(pat, ab4240, by="RID", all.x = TRUE)

df$av45_bin <- as.numeric(df$av45_bin)
df$fbb_bin <- as.numeric(df$fbb_bin)
df$pib_bin <- as.numeric(df$pib_bin)
df$csfab_bin <- as.numeric(df$csfab_bin)

table(df$AMYSTAT_PET, df$AMYSTAT_ab4240)
# quite consistent, but only very few individuals

attach(df)
df$Amyloid_ratio <- ifelse(!is.na(AMYSTAT_PET) & is.na(AMYSTAT_ab4240),
                           AMYSTAT_PET, ifelse(
                           is.na(AMYSTAT_PET) & !is.na(AMYSTAT_ab4240),
                           AMYSTAT_ab4240, ifelse(
                           !is.na(AMYSTAT_PET) & !is.na(AMYSTAT_ab4240),
                           ifelse(AMYSTAT_PET == AMYSTAT_ab4240, AMYSTAT_PET,
                           "ERROR"),
                           NA)))
df <- df[!is.na(df$Amyloid_ratio),]
table(df$Amyloid_ratio)
colnames(df)[which(names(df) == "Amyloid_ratio")] <- "AMY_STAT"
row.names(df) <- NULL
detach(df)
# --> decrease of 13, instead of 53
write.csv(df, paste("1_Amyloid Positivity/Gatekeeping_Amyloid_Positivity/",
          "Gatekeeping-Amyloid-Positivity/data/",
          "ADNI_merge_Amyloid_final_with_ratio.csv", sep=""), row.names = F)

# FREQUENCY ANALYSIS CSF AB42 VS CSF AB42 RATIO WITH CURRENT SAMPLE
#pat <- read.csv2(paste("1_Amyloid Positivity/Gatekeeping_Amyloid_Positivity/",
#                       "Gatekeeping-Amyloid-Positivity/data/",
#                       "ADNImerge_amypos_nooutliers_nomask3.csv", sep=""),
#                 na.strings = "", dec = ".")

ab4240$pTauAB42 <- ab4240$PTAU/ab4240$ABETA42
ab4240$AMYSTAT_ab42 <- ifelse(ab4240$ABETA42<1100, 1, 0)
ab4240$AMYSTAT_ptab42 <- ifelse(ab4240$pTauAB42>0.022, 1, 0)

df <- merge(pat, ab4240, by="RID")

df$av45_bin <- as.numeric(df$av45_bin)
df$fbb_bin <- as.numeric(df$fbb_bin)
df$pib_bin <- as.numeric(df$pib_bin)

df$AMYSTAT_PET <- ifelse(!is.na(df$av45_bin), df$av45_bin, 
                         ifelse(!is.na(df$fbb_bin), df$fbb_bin, df$pib_bin))
# exclude three individuals from previous analyses
df <- df[is.na(as.numeric(df$csfab)),]

# (1) CSF AB42 vs CSF AB42/40
table(df$AMYSTAT_ab42, df$AMYSTAT_ab4240)

# 2 CSF AB42 vs PET AB
table(df$AMYSTAT_PET, df$AMYSTAT_ab42)

# 3 CSF AB42/40 vs PET AB
table(df$AMYSTAT_PET, df$AMYSTAT_ab4240)
t <- c(df$PTID[df$AMYSTAT_PET != df$AMYSTAT_ab42],
       df$PTID[df$AMYSTAT_PET != df$AMYSTAT_ab4240])
table(t)

# 4 CSF pTau/AB4240 vs PET AB
table(df$AMYSTAT_PET, df$AMYSTAT_ptab42)
t <- c(df$PTID[df$AMYSTAT_PET != df$AMYSTAT_ptab42],
       df$PTID[df$AMYSTAT_PET != df$AMYSTAT_ab42])
table(t)
pat$ptab <- as.numeric(pat$PTAU)/as.numeric(pat$ABETA)
pat$ptab_bin <- ifelse(pat$ptab>0.022, 1, 0)
table(pat$ptab_bin, pat$csfab_bin)

