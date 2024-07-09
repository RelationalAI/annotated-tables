require(dplyr)
require(ggplot2)
require(stringr)
require(tidyr)
require(data.table)
require(lubridate)
library(RColorBrewer)
library(rjson)


#####
wd <- "./git/TabPFN/tabpfn"

res_path <- file.path(wd, "dump_results.csv")
res <- fread(res_path)


res <- res %>% mutate(win = transformer_roc>autogluon_roc) %>% drop_na()
roc_wins_total <- length(which(res$win))

# model <- glm(win ~ number_of_input_columns + missingness + number_of_classes + number_of_rows , data=res)
# summary(model)
# with(summary(model), 1 - deviance/null.deviance)

model <- aov(win ~ number_of_input_columns + missingness + number_of_classes + number_of_rows, data=res)
summary(model)

# res <- res %>% mutate(win = factor(win, levels = c(TRUE, FALSE)))
res <- res %>% mutate(Winner = if_else(win, "TabPFN", "AutoGluon"))


# div plot. change legends
ggplot() + geom_point(data = res, mapping = aes(transformer_roc, autogluon_roc,
                                                color=Winner), size =0.2, alpha=0.25) + 
  xlab("TabPFN AUROC (OVO)") + 
  ylab("AutoGluon AUROC (OVO)")+ guides(alpha = "none")+
  labs(color = "Winner")+
  scale_color_brewer(palette = "Set2")+
  theme(legend.position = 'bottom')+ guides(colour = guide_legend(override.aes = list(alpha = 1)))

ggsave(file.path(wd, "dot60.png"), width =4, height = 4)

shapiro.test(res$transformer_roc)
shapiro.test(res$autogluon_roc)
ww <- res %>% select("transformer_roc", "autogluon_roc") %>% na.omit()

wt <- wilcox.test(ww$transformer_roc, ww$autogluon_roc, paired = FALSE)
wt
wt <- wilcox.test(ww$transformer_roc, ww$autogluon_roc, paired = TRUE)
wt
wt <- wilcox.test(ww$transformer_roc, ww$autogluon_roc, "greater", paired=TRUE)
wt

ww <- res %>% select("transformer_roc", "autogluon_roc") %>% na.omit()
dat2 <- data.frame(
  model = c(rep("tabpfn", length(res$transformer_roc)), rep("autogluon", length(res$autogluon_roc))),
  roc = c(res$transformer_roc, res$autogluon_roc)
)

wt <- wilcox.test(res$transformer_roc, res$autogluon_roc, "greater", paired=TRUE)
wt
wt <- wilcox.test(res$transformer_cross_entropy, res$autogluon_cross_entropy, "less", paired=TRUE)
wt


ggplot(dat2) +
  aes(x = model, y = roc) +
  geom_boxplot(fill = "#0c4c8a") +
  theme_minimal()

test <- wilcox.test(dat2$roc ~ dat2$model)
test

################

wd <- "./git/TabPFN/tabpfn"

res_path <- file.path(wd, "dump_results_300.csv")
res300 <- fread(res_path)


res300 <- res300 %>% mutate(win = transformer_roc>autogluon_roc) %>% drop_na()
roc_wins_total <- length(which(res300$win))

# model <- glm(win ~ number_of_input_columns + missingness + number_of_classes + number_of_rows , data=res300)
# summary(model)
# with(summary(model), 1 - deviance/null.deviance)

model <- aov(win ~ number_of_input_columns + missingness + number_of_classes + number_of_rows, data=res300)
summary(model)

# res300 <- res300 %>% mutate(win = factor(win, levels = c(TRUE, FALSE)))
res300 <- res300 %>% mutate(Winner = if_else(win, "TabPFN", "AutoGluon"))




ggsave(file.path(wd, "dot300.png"), width =4, height = 4)

# ww <- res300 %>% select("transformer_roc", "autogluon_roc") %>% na.omit()
wt <- wilcox.test(res300$transformer_roc, res300$autogluon_roc, "less", paired=TRUE)
wt
wt <- wilcox.test(res300$transformer_cross_entropy, res300$autogluon_cross_entropy, "greater", paired=TRUE)
wt

# summary(res300)
# 
# 
# sd(res300$transformer_cross_entropy)
# sd(res300$autogluon_cross_entropy)
# mean(res300$transformer_cross_entropy)
# mean(res300$autogluon_cross_entropy)
# 
# sd(res300$transformer_roc)
# sd(res300$autogluon_roc)
# mean(res300$transformer_roc)
# mean(res300$autogluon_roc)



# 
# summary(res)
# 
# sd(res$transformer_cross_entropy)
# sd(res$autogluon_cross_entropy)
# mean(res$transformer_cross_entropy)
# mean(res$autogluon_cross_entropy)
# 
# sd(res$transformer_roc)
# sd(res$autogluon_roc)
# mean(res$transformer_roc)
# mean(res$autogluon_roc)

### wt table

wt <- wilcox.test(res$transformer_roc, res$autogluon_roc, "greater", paired=TRUE)
wt
wt$p.value
wt <- wilcox.test(res$transformer_cross_entropy, res$autogluon_cross_entropy, "less", paired=TRUE)
wt
wt$p.value
wt <- wilcox.test(res300$transformer_roc, res300$autogluon_roc, "less", paired=TRUE)
wt
wt$p.value
wt <- wilcox.test(res300$transformer_cross_entropy, res300$autogluon_cross_entropy, "greater", paired=TRUE)
wt
wt$p.value


# ww <- res %>% select("transformer_roc", "autogluon_roc") %>% na.omit()
dat1 <- data.frame(
  model = c(rep("TabPFN", length(res$transformer_roc)), rep("AutoGluon", length(res$autogluon_roc))),
  roc = c(res$transformer_roc, res$autogluon_roc)
)
dat1<- dat1 %>% mutate(model = factor(model, levels = c("TabPFN", "AutoGluon")))
ggplot(dat1) +
  aes(x = model, y = roc) +
  geom_boxplot() +
  scale_y_continuous(limits = quantile(dat1$roc, c(0.1, 0.9)))+
  labs(y= "AUROC", x="Model")
ggsave(file.path(wd, "box60roc.png"), width =2, height=4)

dat2 <- data.frame(
  model = c(rep("TabPFN", length(res$transformer_cross_entropy)), rep("AutoGluon", length(res$autogluon_cross_entropy))),
  cross_entropy = c(res$transformer_cross_entropy, res$autogluon_cross_entropy)
)
dat2<- dat2 %>% mutate(model = factor(model, levels = c("TabPFN", "AutoGluon")))
ggplot(dat2) +
  aes(x = model, y = cross_entropy) +
  geom_boxplot() +
  scale_y_continuous(limits = quantile(dat2$cross_entropy, c(0.1, 0.9)))+
  labs(y= "Cross Entropy", x="Model")
ggsave(file.path(wd, "box60ce.png"), width =2, height=4)


dat3 <- data.frame(
  model = c(rep("TabPFN", length(res300$transformer_roc)), rep("AutoGluon", length(res300$autogluon_roc))),
  roc = c(res300$transformer_roc, res300$autogluon_roc)
)
dat3<- dat3 %>% mutate(model = factor(model, levels = c("TabPFN", "AutoGluon")))
ggplot(dat3) +
  aes(x = model, y = roc) +
  geom_boxplot()+
  scale_y_continuous(limits = quantile(dat3$roc, c(0.1, 0.9)))+
  labs(y= "AUROC", x="Model")
ggsave(file.path(wd, "box300roc.png"), width =2, height=4)


dat4 <- data.frame(
  model = c(rep("TabPFN", length(res300$transformer_cross_entropy)), rep("AutoGluon", length(res300$autogluon_cross_entropy))),
  cross_entropy = c(res300$transformer_cross_entropy, res300$autogluon_cross_entropy)
)
dat4<- dat4 %>% mutate(model = factor(model, levels = c("TabPFN", "AutoGluon")))
ggplot(dat4) +
  aes(x = model, y = cross_entropy) +
  geom_boxplot() +
  scale_y_continuous(limits = quantile(dat4$cross_entropy, c(0.1, 0.9)))+
  labs(y= "Cross Entropy", x="Model")
ggsave(file.path(wd, "box300ce.png"), width =2, height=4)

### box plot 

one_minute_roc_diff <- res$transformer_roc - res$autogluon_roc
summary(one_minute_roc_diff)
five_minute_roc_diff <- res300$transformer_roc - res300$autogluon_roc
summary(five_minute_roc_diff)



roc_diff <- c(one_minute_roc_diff, five_minute_roc_diff)
time <- c(rep(60,length(one_minute_roc_diff)), rep(300, length(five_minute_roc_diff)))
time_diff <- data.table(roc_diff = roc_diff, time = time)

ggplot(time_diff, aes(y = factor(time), x = roc_diff))+
  geom_boxplot(outlier.shape = NA) +
  scale_x_continuous(limits = c(-0.05,0.05))
#quantile(one_minute_roc_diff, c(0.1, 0.9)))



### Incremental


wd <- "./git/nirel/worksheets"

inc_path <- file.path(wd, "incremental.csv")
inc <- fread(inc_path)

inc <- inc %>% pivot_longer(!Examples, names_to = 'type', values_to = 'percentage')
inc <- inc %>% mutate(type = factor(inc$type, levels = c("% Rel Error", "% Diff. Results", "% Rel Correct")))

ggplot(inc, aes(fill=type, y=percentage, x=Examples)) + 
  geom_bar(position="stack", stat="identity")+
  scale_fill_brewer(palette = "Set2") +
  xlab("Number of Few-Shot Examples") + 
  ylab("Percentage") +
  labs(fill = "Execution Result")
  #scale_fill_manual(name = "Execution Result")

ggsave(file.path(wd, "incremental.png"), width =5, height = 2)
