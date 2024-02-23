# Settings
library(lubridate)
library(tidyverse)
library(data.table)
theme_set(theme_bw())

# Prepare data
data <- fread("Videos/Data/dnk.csv", colClasses = c("eom"="character"))
data[, eom := eom %>% fast_strptime(format="%Y%m%d") %>% as.Date()]
data <- data[common==1 & obs_main==1 & exch_main==1 & primary_sec==1] # Already done in SAS
data[, size_grp := size_grp %>% factor(levels = c("mega", "large", "small", "micro", "nano"))]

# Extract important variables
char_info <- readxl::read_xlsx("Videos/Data/Factor Details.xlsx", range = "A1:O256") %>%  # Available from: https://github.com/bkelly-lab/ReplicationCrisis/tree/master/GlobalFactors
  select(abr_jkp, name_new, direction, significance) %>%
  filter(!is.na(abr_jkp)) %>% # Add significance == 1 to only include characteristics found to be significant, in the original paper.
  setDT()
data <- data[, c("id", "eom", "size_grp", "me", "ret_exc_lead1m", char_info$abr_jkp), with=F]

# Add information for capped value-weights
nyse_cutoffs <- fread("Videos/Data/nyse_cutoffs.csv", colClasses = c("eom"="character"))  # Available from:  https://www.dropbox.com/sh/qz5qo5hypjofpg7/AAD6I-2oYoGsb--h2N-zYHb6a?dl=0
nyse_cutoffs[, eom := eom %>% as.Date(format = "%Y%m%d")]
nyse_cutoffs <- nyse_cutoffs[, .(eom, nyse_p80)]
data <- nyse_cutoffs[data, on = .(eom)]
data[, me_cap := pmin(me, nyse_p80)]  # Check: data[size_grp=="mega", .(id, eom, me, me_cap, nyse_p80)]

# Winsorize returns
return_cutoffs <- fread("Videos/data/return_cutoffs.csv", colClasses = c("eom"="character")) # Available from:  https://www.dropbox.com/sh/qz5qo5hypjofpg7/AAD6I-2oYoGsb--h2N-zYHb6a?dl=0
return_cutoffs[, eom := eom %>% as.Date(format="%Y%m%d")]
return_cutoffs <- return_cutoffs[, .(eom, ret_exc_0_1, ret_exc_99_9)]
return_cutoffs <- return_cutoffs[, eom := eom %>% floor_date(unit="m") - 1]# Subtract 1 month to eom to align ret_exc_lead1m
data <- return_cutoffs[data, on = "eom"]
data[ret_exc_lead1m>ret_exc_99_9, ret_exc_lead1m := ret_exc_99_9]
data[ret_exc_lead1m<ret_exc_0_1, ret_exc_lead1m := ret_exc_0_1]

# Application: Coverage -------------------------
# Market equity coverage by Size Group
data[!is.na(me), .(n = .N), by = .(eom, size_grp)] %>%
  ggplot(aes(eom, n, colour = size_grp, linetype = size_grp)) +
  geom_line(size=1.5) +
  labs(y = "Stocks with non-missing market equity", colour = "Size group:", linetype = "Size group:") +
  theme(
    legend.position = "top",
    axis.title.x = element_blank()
  )

# Book-to-Market equity coverage by Size Group
data[!is.na(be_me), .(n = .N), by = .(eom, size_grp)] %>%
  ggplot(aes(eom, n, colour = size_grp, linetype = size_grp)) +
  geom_line(size=1.5) +
  labs(y = "Stocks with non-missing book-to-market equity", colour = "Size group:", linetype = "Size group:") +
  theme(
    legend.position = "top",
    axis.title.x = element_blank()
  )

# Application: Long-short factor ----------------
var <- "market_equity"
n_pfs <- 5
pf_data <- data[, c("id", "eom", "me", "me_cap", "ret_exc_lead1m", var), with = F]
pf_data %>% setnames(old = var, new = "sort_var")
pf_data <- pf_data[!is.na(sort_var) & !is.na(ret_exc_lead1m) & !is.na(me)]
pf_data[, prank := frank(sort_var)/.N, by = eom]
pf_data[, pf := ceiling(prank*n_pfs)]  # Check number of stocks in each portfolio: pf_data[, .N, by = .(pf, eom)] %>% ggplot(aes(eom, N, colour = factor(pf))) + geom_point()
# Portfolio returns
pfs <- pf_data[, .(
  n_stocks = .N,
  ret_ew = mean(ret_exc_lead1m),
  ret_vw = sum(ret_exc_lead1m*me)/sum(me),
  ret_vw_cap = sum(ret_exc_lead1m*me_cap)/sum(me_cap)
), by = .(pf, eom)]
pfs[, eom_ret := ceiling_date(eom + 1, unit="m")-1]

# Summary statistics
pfs_ss <- pfs[, .(
  ret = mean(ret_ew)*12,
  sd = sd(ret_ew)*sqrt(12),
  sr = mean(ret_ew)/sd(ret_ew)*sqrt(12)
), by = pf][order(pf)]
print(pfs_ss)

name <- char_info[abr_jkp==var, name_new]
pfs_ss %>% 
  ggplot(aes(pf, ret)) +
  geom_col() +
  labs(y = "Annualized Return", x = "Portfolio", title = paste0("Portfolios: ", name))

# Factor return
fct <- pfs[, .(
  fct_ew = ret_ew[pf == n_pfs] - ret_ew[pf == 1],
  fct_vw = ret_vw[pf == n_pfs] - ret_vw[pf == 1],
  fct_vw_cap = ret_vw_cap[pf == n_pfs] - ret_vw_cap[pf == 1]
), by = eom_ret]
fct_dir <- char_info[abr_jkp == var, as.integer(direction)]
fct[, fct_ew := fct_ew * fct_dir] # Sign factor following the literature
fct[, fct_vw := fct_vw * fct_dir] 
fct[, fct_vw_cap := fct_vw_cap * fct_dir] 
fct[, long := if_else(fct_dir==1, paste0("High ", var), paste0("Low ", var))]

# Cumulative return
fct %>%
  arrange(eom_ret) %>%
  mutate(cumret = cumsum(fct_ew)) %>%
  ggplot(aes(eom_ret, cumret)) +
  geom_line() +
  labs(y = "Cumulative Return", title=paste0("Factor: ", name)) +
  theme(axis.title.x = element_blank())
