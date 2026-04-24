# Import the required libraries
library(tidyverse)
library(caret)
library(lmtest)

# Reload the data file
df <- read.csv('/Users/abrahmar/Downloads/cubic_zirconia.csv')

# Preprocessing
df$depth[is.na(df$depth)] <- median(df$depth, na.rm = TRUE)
df <- subset(df, select = -c(X))

# Create a new DataFrame for regression analysis (predict "price" based on "cut" and "carat")
df_regression <- df %>%
  mutate(cut = as.factor(cut))

# Create dummy variables for "cut" column using model.matrix
# The first dummy variable is automatically removed to use as a reference category
cut_dummies <- model.matrix(~ cut - 1, data = df_regression)

# Convert the dummy variable matrix to a data frame
cut_dummies_df <- as.data.frame(cut_dummies)

# Combine the original data frame with the dummy variables data frame
df_regression <- cbind(df_regression, cut_dummies_df)

# Remove the original "cut" column
df_regression$cut <- NULL

# Select relevant columns for regression analysis
X_regression <- df_regression %>%
  select(starts_with("carat"), starts_with("cut"))

y_regression <- df_regression$price

# Set the random seed for reproducibility
set.seed(42)

# Determine the number of training samples (70% of the total)
num_train_samples <- floor(0.7 * nrow(X_regression))

# Randomly sample indices for the training set
train_indices <- sample(seq_len(nrow(X_regression)), size = num_train_samples)

# Create the training datasets using the sampled indices
X_train_regression_revised <- X_regression[train_indices, ]
y_train_regression_revised <- y_regression[train_indices]

# Create the testing datasets using the remaining indices
X_test_regression_revised <- X_regression[-train_indices, ]
y_test_regression_revised <- y_regression[-train_indices]

# Train the linear regression model with selected features (excluding "cut_Ideal")
lr_model_regression_revised <- lm(y_train_regression_revised ~ ., data = X_train_regression_revised)

# Predict on the testing data
y_pred_regression_revised <- predict(lr_model_regression_revised, X_test_regression_revised)

# Evaluate the model performance
rmse_regression_revised <- sqrt(mean((y_test_regression_revised - y_pred_regression_revised)^2))
r2_regression_revised <- summary(lr_model_regression_revised)$r.squared

# Display the coefficients of the model
coefficients_regression_revised <- coef(lr_model_regression_revised)
intercept_regression_revised <- coef(lr_model_regression_revised)[1]

coefficients_regression_revised
intercept_regression_revised
rmse_regression_revised
r2_regression_revised

# Display the coefficients of the model and p-values
summary(model)

model_summary <- summary(model)
f_statistic <- model_summary$fstatistic[1]
cat("F-statistic:", f_statistic)

# plot

plot(lr_model_regression_revised, which=1)

