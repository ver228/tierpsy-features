#https://github.com/AndersenLab/cegwas
#devtools::install_github("AndersenLab/cegwas")

library(cegwas)
library(tidyverse)
library(doParallel)

process_csv = function(fname){
  print(fname)
  df = read.csv(fname)
  df[,'strain'] = cbind(sapply(df['strain'] , function(x)gsub('\\s+', '',x)))
  
  # remove strains that are not available
  valid_rows = df$strain %in% colnames(snps)
  print(unique(df[!valid_rows, 'strain']))
  df = df[valid_rows, ]
  
  #remove columns that are c(not features
  index_cols = c("X", "strain", "id", "directory", "base_name", "exp_name" )
  df_f = df[-which(names(df) %in% index_cols)]
  #get the mean value for each feature (the code does not accept more than one value per strain...)
  feats = aggregate(df_f, by=df['strain'], FUN=mean, na.rm=TRUE)
  rownames(feats) = feats$strain
  feats = within(feats, rm('strain'))
  
  #prepare data for process_pheno
  pheno = data.frame(t(feats))
  pheno = cbind(trait = colnames(feats), pheno) #add the traits column to the first row
  rownames(pheno) = seq(nrow(pheno)) #eliminate the columns names
  
  #cegwas analysis
  
  nn = 10
  process_row  = function(nn){
    require(cegwas)
    require(tidyverse)
    print(pheno[nn,'trait'])
    processed_phenotypes <- process_pheno(pheno[nn, ])
    mapping_df <- gwas_mappings(processed_phenotypes)
    processed_mapping_df <- process_mappings(mapping_df, phenotype_df = processed_phenotypes, CI_size = 50, snp_grouping = 200)
    
    #output = c(processed_phenotypes = processed_phenotypes, 
    #           mapping_df = mapping_df,
    #           processed_mapping_df = processed_mapping_df
    #           )
    return(processed_mapping_df)
  }
  
  process_row_err = function (n) tryCatch(process_row(n), error=function(e) NULL)
  
  cl <- makeCluster(21)
  registerDoParallel(cl)
  tot = 20#nrow(pheno)
  results = foreach(i=1:tot) %dopar% process_row_err(i)
  stopCluster(cl)
  
  
  
  plot_path = gsub('.csv', '_manhatan_plots.pdf', fname)
  pdf(plot_path, width=10, height=4)
  for (ii in 1:length(results)){
    tryCatch(manplot(results[[ii]]), error=function(e) NULL)
  }
  dev.off()
  
  results_path = gsub('.csv', '_cegwas.RData', fname)
  save(results, results_path)
  
}

tierpsy_feat_file = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/manual_features/CeNDR/tierpsy_features_CeNDR.csv'

main_dir = '/Users/ajaver/OneDrive - Imperial College London/classify_strains/trained_models/vae_w_embeddings'
files <- list.files(path = main_dir, pattern="*.csv", full.names=T, recursive=FALSE)
files = c(tierpsy_feat_file, files)
lapply(files[[1]], process_csv)

#process_csv(fname)
# pdf("phenotypes_boxplot.pdf")
# pxg_plot(processed_mapping_df)
# dev.off()
# 
# pdf("gene_variants.pdf")
# gene_variants(processed_mapping_df)
# dev.off()

