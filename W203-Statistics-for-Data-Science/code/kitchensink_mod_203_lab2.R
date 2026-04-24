# Load required libraries
library(tidyverse)
library(caret)

# Load the dataset
df <- read.csv('/Users/abrahmar/Downloads/cubic_zirconia.csv')

# Drop the unwanted column "Unnamed: 0"
df <- df[, !names(df) %in% "Unnamed: 0"]

# Impute missing values using median imputation
df$carat[is.na(df$carat)] <- median(df$carat, na.rm = TRUE)
df$depth[is.na(df$depth)] <- median(df$depth, na.rm = TRUE)
df$table[is.na(df$table)] <- median(df$table, na.rm = TRUE)
df$x[is.na(df$x)] <- median(df$x, na.rm = TRUE)
df$y[is.na(df$y)] <- median(df$y, na.rm = TRUE)
df$z[is.na(df$z)] <- median(df$z, na.rm = TRUE)

# Handle outliers using IQR method
IQR <- IQR(df$price)
lower_bound <- quantile(df$price, 0.25) - 1.5 * IQR
upper_bound <- quantile(df$price, 0.75) + 1.5 * IQR
df <- df[df$price >= lower_bound & df$price <= upper_bound,]

# Engineer features
df$volume <- df$x * df$y * df$z
df$depth_percentage <- (2 * df$z) / (df$x + df$y)
df$cut_color_interaction <- interaction(df$cut, df$color)
df$cut_clarity_interaction <- interaction(df$cut, df$clarity)

# Create a new dataframe with only the specified variables
new_df <- df %>% select(price, carat, color, clarity, cut, volume, depth_percentage, cut_color_interaction, cut_clarity_interaction)

# Create dummy variables for "cut," "color," "clarity," "cut_color_interaction," and "cut_clarity_interaction" variables
dummies <- dummyVars(~ cut + color + clarity + cut_color_interaction + cut_clarity_interaction, data = new_df, fullRank = TRUE)
new_df_dummy <- data.frame(predict(dummies, new_df))
new_df_dummy$price <- new_df$price
new_df_dummy$carat <- new_df$carat
new_df_dummy$volume <- new_df$volume
new_df_dummy$depth_percentage <- new_df$depth_percentage

# Create interaction terms for "cut" and "color" as well as "cut" and "clarity"
new_df$cut_color_interaction <- interaction(new_df$cut, new_df$color, drop = TRUE)
new_df$cut_clarity_interaction <- interaction(new_df$cut, new_df$clarity, drop = TRUE)

# Create a model matrix with dummy variables for individual factors and interactions
new_df_dummy <- model.matrix(~ cut + color + clarity + cut_color_interaction + cut_clarity_interaction - 1, data = new_df)

# Convert the model matrix to a data frame
new_df_dummy <- as.data.frame(new_df_dummy)

# Add the original columns "price", "carat", "volume", and "depth_percentage" to the data frame
new_df_dummy$price <- new_df$price
new_df_dummy$carat <- new_df$carat
new_df_dummy$volume <- new_df$volume
new_df_dummy$depth_percentage <- new_df$depth_percentage

# Split the data into training (70%) and testing (30%) datasets
set.seed(42)
train_index <- createDataPartition(new_df_dummy$price, p = 0.7, list = FALSE)
train_data <- new_df_dummy[train_index, ]
test_data <- new_df_dummy[-train_index, ]

# Set seed for reproducibility
set.seed(42)

# Calculate the number of rows for the training set (70% of the total rows)
train_rows <- sample(1:nrow(new_df_dummy), size = round(0.7 * nrow(new_df_dummy)))

# Split the data into training and testing datasets
train_data <- new_df_dummy[train_rows, ]
test_data <- new_df_dummy[-train_rows, ]

# Train the simple linear regression model with the specified variables
model <- lm(price ~ ., data = train_data)

# Summary of the linear regression model
summary(model)

# Predict on the testing data
predictions <- predict(model, newdata = test_data)

# Evaluate the model performance
RMSE <- sqrt(mean((test_data$price - predictions)^2))
R2 <- 1 - (sum((test_data$price - predictions)^2) / sum((test_data$price - mean(test_data$price))^2))

# Display resuls of performance
cat("RMSE:", RMSE, "\n")
cat("R-squared:", R2, "\n")

# plot

plot(model, which=1)