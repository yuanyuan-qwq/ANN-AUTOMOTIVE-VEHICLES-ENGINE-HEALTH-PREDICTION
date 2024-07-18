library(tibble)
library(readr)
library(ggplot2)
library(keras)
library(tensorflow)
library(caret)
library(RColorBrewer)



raw_data <- read.csv("engine_data.csv")

glimpse(raw_data)      # preview the data set with different column

# visualize data using ggplot.
ggplot(raw_data, aes(x = 1:nrow(raw_data), y = Engine.rpm)) + geom_line()

ggplot(raw_data[1:1000,], aes(x= 1:1000, y = Engine.rpm)) +geom_line()

# Data Cleaning
#=========================================

outlier_threshold <- 3 # Adjust the threshold as needed
for (col in names(raw_data)) {
  if (is.numeric(raw_data[[col]])) {
    z_scores <- abs(scale(raw_data[[col]]))
    raw_data <- raw_data[z_scores < outlier_threshold,]
  }
}


#===========================================

X<- data.matrix(raw_data[, -ncol(raw_data)])
Y <- data.matrix(raw_data[, ncol(raw_data)])

set.seed(42)

indexes <- createDataPartition(y = Y, times = 1, p = 0.7, list = FALSE)
x_train <- X[indexes, ]
y_train <- Y[indexes, ]
x_test <- X[-indexes, ]
y_test <- Y[-indexes, ]


dim(x_train)
dim(x_test)
y_train
dim(y_test)


# scaling and normalization

#2 is "column-wise" means that the function is applied to each column of a dataset individually
mean <- apply(x_train, 2, mean)
std <- apply(x_train,2, sd)
x_train <- scale(x_train, center = mean , scale = std)

mean1 <- apply(x_test, 2, mean)
std1 <- apply(x_test,2, sd)
x_test <- scale(x_test, center = mean , scale = std)


normalize <- function(x){
  return ((x - min(x)) / (max(x) - min(x)))    # min max normalization
}


max <- apply(x_train, 2, max)
min <- apply(x_train, 2 ,min)

x_train_transform  <- apply(x_train, 2 , normalize)

max <- apply(x_test, 2, max)
min <- apply(x_test, 2 ,min)

x_test_transform <- apply(x_test, 2 , normalize)

x_test_transform 
x_train_transform


x_train_t <- as_tensor(x_train_transform)
y_train_t <- as_tensor(y_train)
x_test_t <- as_tensor(x_test_transform)
y_test_t <- as_tensor(y_test)

list(NULL, dim(x_train_t)[[-1]])





model <- keras_model_sequential() %>% 
  layer_dense(units = 128, input_shape = 6) %>%
  layer_dense(units = 128, activation = "relu") %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 32, activation = "sigmoid") %>% 
  layer_dense(units = 1, )


summary(model)


model %>% compile(optimizer = optimizer_adam(learning_rate = 0.001), loss = "mae")

history <- model %>%  fit( x=x_train_t , y = y_train_t ,validation_split = 0.1, batch_size = 10, shuffle = TRUE,epochs = 50)

history$metrics$loss
history$metrics$val_loss

predictions <- predict(model, x_test_t, batch_size = 10)

predictions

predictions_rounded <- round(predictions, digits = 0)
predictions_rounded




checking <- cbind(predictions_rounded,y_test_t)
checking

predictions_rounded <- as.vector(predictions_rounded)
y_test_t <- as.vector(y_test_t)

predictions_factor <- factor(predictions_rounded, levels = c(0, 1))
y_test_factor <- factor(y_test_t, levels = c(0, 1))

# Create the confusion matrix
confusion_matrix <- confusionMatrix(predictions_factor, y_test_factor)

# Convert the confusion matrix table to a matrix
confusion_matrix_mat <- as.matrix(confusion_matrix)


# Generate a color palette for the heatmap
col_palette <- c("#F0F9E8", "#BAE4BC", "#7BCCC4", "#43A2CA", "#0868AC")


# Plot the confusion matrix as a heatmap
heatmap(confusion_matrix_mat, col = col_palette,
        main = "Confusion Matrix", xlab = "Actual", ylab = "Predicted",
        add.expr = {
          for (i in 1:nrow(confusion_matrix_mat)) {
            for (j in 1:ncol(confusion_matrix_mat)) {
              text(j, i, confusion_matrix_mat[i, j], col = "black", cex = 0.8)
            }
          }
        })




# Calculate Accuracy, Recall, and F1-score
accuracy <- sum(diag(confusion_matrix_mat)) / sum(confusion_matrix_mat)
recall <- confusion_matrix_mat[2, 2] / sum(confusion_matrix_mat[, 2])
precision <- confusion_matrix_mat[2, 2] / sum(confusion_matrix_mat[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)

# Print the results
cat("Accuracy:", accuracy, "\n" , "Recall:", recall, "\n", "F1-score:", f1_score, "\n")


