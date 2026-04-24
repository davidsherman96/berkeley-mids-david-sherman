# Load required libraries
library(tidyverse)
library(caret)

# Load the dataset
df <- read.csv('/Users/abrahmar/Downloads/cubic_zirconia.csv')

# Drop the unwanted column "Unnamed: 0"
df <- df[, !names(df) %in% "Unnamed: 0"]

# Impute missing values with median for numeric columns and mode for factor columns
numeric_columns <- sapply(df, is.numeric)
df[numeric_columns] <- lapply(df[numeric_columns], function(x) ifelse(is.na(x), median(x, na.rm = TRUE), x))
factor_columns <- sapply(df, is.factor)
df[factor_columns] <- lapply(df[factor_columns], function(x) ifelse(is.na(x), names(which.max(table(x))), x))

# Identify and remove outliers using the IQR method (keeping values within 1.5 * IQR)
df <- df %>% filter_all(all_vars(. >= quantile(., 0.25, na.rm = TRUE) - 1.5 * IQR(., na.rm = TRUE) &
                                   . <= quantile(., 0.75, na.rm = TRUE) + 1.5 * IQR(., na.rm = TRUE)))

# Create a new dataframe with only "price" and "cut" variables
new_df <- df %>% select(price, cut)

# Create a model matrix with dummy variables for the "cut" variable
new_df_dummy <- model.matrix(~ cut - 1, data = new_df)  # The "- 1" term ensures that no intercept column is created

# Convert the model matrix to a data frame
new_df_dummy <- as.data.frame(new_df_dummy)

# Add the original "price" column to the new data frame
new_df_dummy$price <- new_df$price

# Set seed for reproducibility
set.seed(42)

# Calculate the number of rows for the training set (70% of the total rows)
train_rows <- sample(1:nrow(new_df_dummy), size = round(0.7 * nrow(new_df_dummy)))

# Split the data into training and testing datasets
train_data <- new_df_dummy[train_rows, ]
test_data <- new_df_dummy[-train_rows, ]


# Train the simple linear regression model with only "price" and "cut" variables
model <- lm(price ~ ., data = train_data)

# Summary of the linear regression model
summary(model)

# Predict on the testing data
predictions <- predict(model, newdata = test_data)

# Evaluate the model performance
RMSE <- sqrt(mean((test_data$price - predictions)^2))
R2 <- 1 - (sum((test_data$price - predictions)^2) / sum((test_data$price - mean(test_data$price))^2))

# Display RMSE and R-squared
cat("RMSE:", RMSE, "\n")
cat("R-squared:", R2, "\n")

# plot

plot(model, which=2)
