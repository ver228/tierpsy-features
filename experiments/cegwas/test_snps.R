library(cegwas)
library(tidyverse)
library(dplyr)

dat <- dplyr::select(snps, CHROM, POS, REF, ALT, MY16)  %>%
       dplyr::filter(is.na(MY16))
       
dd <- dplyr::select(snps, -CHROM, -POS, -REF, -ALT)


write.csv(snps, file='CeNDR_snps.csv', row.names=F)