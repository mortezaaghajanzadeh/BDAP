# Settings
library(tidyverse)
library(data.table)
theme_set(theme_bw())

# Corona by factor-country
factors <- fread("Videos/Data/jkp_factors.csv")
factors <- factors[n_stocks_min >= 5]

factors %>%
  filter(year(date)==2020 & month(date)==3 & location %in% c("usa", "dnk")) %>%
  group_by(name) %>%
  mutate(sort_var = ret[location == "usa"]) %>%
  ggplot(aes(reorder(name, sort_var), ret)) +
  geom_col() +
  coord_flip() +
  theme(
    axis.title.y = element_blank() 
  ) +
  facet_wrap(~location) +
  labs(title = "Corona-Factors: Denmark vs. US")

# Corona performance by cluster
themes <- fread("Videos/Data/jkp_themes.csv")

themes %>%
  filter(year(date)==2020 & month(date)==3 & location %in% c("usa", "dnk")) %>%
  group_by(name) %>%
  mutate(sort_var = ret[location == "usa"]) %>%
  ggplot(aes(reorder(name, sort_var), ret)) +
  geom_col() +
  coord_flip() +
  theme(
    axis.title.y = element_blank() 
  ) +
  facet_wrap(~location) +
  labs(title = "Corona-Themes: Denmark vs. US")

# Corona performance by region-cluster
themes_region <- fread("Videos/Data/jkp_themes_regions.csv")

themes_region %>%
  filter(year(date)==2020 & month(date)==3 & 
         location %in% c("developed", "emerging", "frontier")) %>%
  group_by(name) %>%
  mutate(sort_var = ret[location == "developed"]) %>%
  ggplot(aes(reorder(name, sort_var), ret)) +
  geom_col() +
  coord_flip() +
  theme(
    axis.title.y = element_blank() 
  ) +
  facet_wrap(~location) +
  labs(title = "Corona-Themes: Developed vs. Emerging vs. Frontier")

# Post-2005 theme performance: Value, Momentum and Low Risk
themes_region %>%
  filter(date >= as.Date("2005-01-01") & 
         name %in% c("value", "momentum", "quality") & 
         location %in% c("developed", "emerging", "frontier")) %>%
  group_by(location, name) %>%
  arrange(location, name) %>%
  mutate(cumret = cumsum(ret)) %>%
  ggplot(aes(date, cumret, colour = name)) +
  geom_line() +
  facet_wrap(~location) +
  labs(title = "Post-2005 Theme Performance")
