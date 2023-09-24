library(tidyverse)
library(GGally)
library(lubridate)
library(e1071)
library(caTools)
library(caret)
library(tidymodels)
library(klaR)
library(naivebayes)


# The file describes which customer acts as a reseller.
# -> classified customers / response variables
classification <- read_csv("classification.csv")

# The file provides the information about the customers and their sales orders.
# -> customers that shared sales orders / link to features
customers <- read_csv("customers.csv")

# The file contains information about the sales orders.
# -> features / independent variables
sales_orders <- read_csv("sales_orders.csv")
sales_orders_header <- read_csv("sales_orders_header.csv")
business_units <- read_csv("business_units.csv")
service_map <- read_csv("service_map.csv")

# This file is an example solution template.
submission_random <- read_csv("submission_random.csv")

# This file is for writing out the values.
unseen <- read_csv("pub_K8PzhiD.csv")

# testing if there are NA values in "classification"
classification %>% mutate_all(is.na) %>% summarise_all(sum)
# 512 customers were not classified
# 2049 customers are not in the test set

# testing if there are NA values in "customers"
customers %>% mutate_all(is.na) %>% summarise_all(sum)
# no NA values

# joining the "classification" and "customers" on "Customer_ID"
joined_classCus <- full_join(classification, customers, 
                             by = c('Customer_ID'))

# joining the "joined_classCus" and "sale_orders" on "Item_Position"

# converting the type of "Item_Position" in "sales_orders" from dbl to chr
joined_classCus <- joined_classCus %>%  
  mutate(Item_Position = as.double(Item_Position))

# In some cases, there does not exist a matching Item_Position in either 
# of both tables. Set those entries to zero and re-match the tables.

# defining a function that sets the Item_Position values of unmatched entries to 0
assignZeroToUnmatched <- function(t1, t2, j1, j2) {
  t1 <- t1 %>% mutate(Key = seq(1, nrow(t1), 1))
  onlyLeft <- anti_join(t1, t2, 
                        by = c(j1, j2))
  t1[match(onlyLeft$Key, t1$Key),]$Item_Position <- 0
  t1 <- t1[,!names(t1) %in% "Key"]
  
  return(t1)
}

# Assignment of 0s to the unmatched "joined_classCus"
temp <- joined_classCus
joined_classCus <- assignZeroToUnmatched(joined_classCus, sales_orders, 'Item_Position', 'Sales_Order')

# Assignment of 0s to the unmatched "sales_orders"
sales_orders <- assignZeroToUnmatched(sales_orders, temp, 'Item_Position', 'Sales_Order')

# final join of joined_classCus and sales_order
joined_classCusSo <- full_join(joined_classCus, sales_orders, 
                                 by = c('Item_Position', 'Sales_Order'))

# since there are two Net_Value attributes in both tables (sales_orders and sales_orders_header)
# and they are different from each other, we have to rename them
joined_classCusSo <- joined_classCusSo %>% rename(Net_Value_LI = Net_Value)

sales_orders_header <- sales_orders_header %>% rename(Net_Value_Total = Net_Value)

# joining the "joined_classCusSo" and "sales_orders_header" on "Sales_Order"
joined_classCusSoH <- inner_join(joined_classCusSo, sales_orders_header, by = c('Sales_Order'))

# In this scenario, there exists a matching partner of the Material_Class value 
# in the service-maps.csv file. Otherwise, the sales order does not correspond 
# to a service. If there is no corresponding service, then Material_Class value is 0.
joined_classCusSoH <- joined_classCusSoH %>% mutate(Material_Class = as.double(Material_Class))

joined_classCusSoH[!joined_classCusSoH$Material_Class %in% service_map$MATKL_service,]$Material_Class <- 0

# joining the "joined_classCusSoH" and "business_units" on "Cost_Center"
finalDF <- left_join(joined_classCusSoH, business_units, by = c('Cost_Center'))

# an overview
glimpse(finalDF)


# displaying the columns that have NA values
names(which(colSums(is.na(finalDF))>0))

# modifying features

# handling NA values of columns Num_Items, Material_Code, Net_Value_Total, Cost_Center,
# YHKOKRS, Business_Unit

# attributes which are possibly significant:

# dependent variable: Reseller
# independent variables: 

# Material_Class (discretization, can be in interaction with Material_Code), Cost_Center (discretization),
# Net_Value_LI, Sales_Organization (discretization), Creator (discretization), 
# Document_Type (discretization), Delivery (discretization), Net_Value_Total, Business_Unit
#(discretization), Type (discretization)

# converting characters into categorical
finalDF <- finalDF %>% mutate_at(.vars = vars(Material_Code, Material_Class, 
                                              Cost_Center, Sales_Organization, 
                                              Creator, Document_Type, Delivery, Business_Unit, Type), .funs = ~as.factor(.))

# dealing with NA values

# replacing all the numerical NA values with 0
finalDF <- finalDF %>% mutate(Num_Items = if_else(is.na(Num_Items), 0, Num_Items))
finalDF <- finalDF %>% mutate(Net_Value_LI = if_else(is.na(Net_Value_LI), 0, Net_Value_LI))

# creating a separate category for NA values
finalDF <- finalDF %>% mutate(across(.cols = c(Type, Material_Code, Cost_Center, YHKOKRS, Business_Unit), 
                                     .fns = ~addNA(.)))

# unseen data set
unseenSub <- finalDF %>% filter(!is.na(Test_set_id))
unseen <- full_join(unseenSub, unseen, by = c('Test_set_id' = 'id'))

# total data set
finalDF <- finalDF %>% filter(is.na(Test_set_id))
finalDF <- finalDF %>% filter(!is.na(Reseller))

# setting a seed to generate the same results
set.seed(2022)

# creating 10 folds
tr <- trainControl(method = "cv", number = 1)

# selecting features
xAtt <- finalDF %>% dplyr::select(Material_Class)

# training the model using 10 fold cross  validation
modelNB <- train(x = xAtt, y = as.factor(finalDF$Reseller), method = "naive_bayes", trControl = tr, metric = "Accuracy",  
                 maximize = ifelse(metric == "RMSE", FALSE, TRUE))

predictions <- predict(modelNB, unseen, type="class")

unseen <- unseen %>% mutate(prediction = predictions)

unseenPredictions <- unseen %>% group_by(Test_set_id) %>% count(prediction) %>% top_n(1) %>% dplyr::select(-n)

unseenPredictions <- unseenPredictions[!duplicated(unseenPredictions$Test_set_id),]

unseenPredictions <- unseenPredictions %>% rename(id = Test_set_id)

unseenPredictions <- unseenPredictions %>% mutate(prediction = as.double(prediction))

unseenPredictions <- unseenPredictions %>% mutate(prediction = case_when(prediction == 1 ~ 0,
                                                                         prediction == 2 ~ 1))

write_csv(unseenPredictions, 'predictions_first.csv')

