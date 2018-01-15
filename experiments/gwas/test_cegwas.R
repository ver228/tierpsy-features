library(cegwas)
library(tidyverse)

fname = '~/OneDrive\ -\ Imperial\ College\ London/classify_strains/manual_features/CeNDR/F_tierpsy_features_CeNDR.csv'
df = read.csv(fname, row.names = 1)

index_cols = c("Unnamed..0", "id", "directory", "base_name", "exp_name")
missing_strains <- setdiff(pheno$strain, colnames(geno))

pheno <- df %>% 
      dplyr::select(-one_of(index_cols)) %>% 
      dplyr::filter(! strain %in% missing_strains)

geno <- snps %>% dplyr::mutate(marker = paste0(CHROM, "_", POS)) %>% 
  dplyr::select(marker, everything(), -REF, -ALT) %>%
  as.data.frame()





setdiff(list(pheno['strain']), list(colnames(geno)))

# pheno = data.frame(t(df))
# pheno = cbind(trait = colnames(df), pheno) #add the traits column to the first row
# rownames(pheno) = seq(nrow(pheno)) #eliminate the columns names
# 
# processed_phenotypes <- process_pheno(pheno)
# mapping_df <- gwas_mappings(processed_phenotypes)
# processed_mapping_df <- process_mappings(mapping_df, phenotype_df = processed_phenotypes, CI_size = 50, snp_grouping = 200)
# 
# 
# pdf("manhatan_plots.pdf", width=10, height=4)
# manplot(processed_mapping_df)
# dev.off()
# 
# pdf("phenotypes_boxplot.pdf")
# pxg_plot(processed_mapping_df)
# dev.off()
# 
# pdf("gene_variants.pdf")
# gene_variants(processed_mapping_df)
# dev.off()

