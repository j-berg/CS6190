# Install and load Recount3 library
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

if ("recount3" %in% rownames(installed.packages()) == FALSE) {
  BiocManager::install("recount3", dependencies=c("Depends", "Imports", "LinkingTo"))
} else {
  print("recount3 package already installed")
}

library(recount3)


#
