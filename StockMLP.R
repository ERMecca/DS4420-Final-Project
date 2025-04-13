#install.packages(c("neuralnet", "scales", "caret", "lubridate", "ggplot2"))
library(neuralnet)
library(scales)
library(caret)
library(lubridate)
library(ggplot2)
library(fastDummies)


load_data <- function(file_path){
  data <- read.csv(file_path, stringsAsFactors = FALSE)
  
  if("date" %in% colnames(data)){
    data$date <- as.Date(data$date)
    data <- data[order(data$date), ]
  }
  
  return(data)
}

prep_data <- function(data, encoding = TRUE){
  proc_data <- data
  
  if(encoding && "Name" %in% colnames(proc_data)){
    enc_data <- dummy_cols(proc_data, select_columns = "Name",
                            remove_selected_columns = TRUE,
                            remove_first_dummy = FALSE)
    proc_data <- enc_data
  }
  
  if("date" %in% colnames(proc_data)){
    proc_data$date <- NULL
  }
  
  proc_data <- proc_data[, sapply(proc_data, is.numeric)]
  
  return(proc_data)
}

normalize_data <- function(data){
  norm_params <- lapply(data, function(x){
    c(min = min(x, na.rm = TRUE), max = max(x, na.rm = TRUE))
  })
  
  norm_data <- as.data.frame(lapply(1:ncol(data), function(i){
    if(norm_params[[i]]["max"] == norm_params[[i]]["min"]){
      rep(.5, length(data[[i]]))
    } else {
      (data[[i]] - norm_params[[i]]["min"]) /
        (norm_params[[i]]["max"] - norm_params[[i]]["min"])
    }
  }))
  
  colnames(norm_data) <- colnames(data)
  
  return(list(norm_data = norm_data, norm_params = norm_params))
}

denormalize <- function(norm_vals, col_name, norm_params){
  min_val <- norm_params[[col_name]]["min"]
  max_val <- norm_params[[col_name]]["max"]
  
  if(max_val == min_val){
    return(rep(min_val, length(norm_vals)))
  }
  
  denorm_vals <- norm_vals * (max_val - min_val) + min_val
  return(denorm_vals)
}

train_multi_stock_mlp <- function(file_path, test_size = .2, hidden_layers = c(32, 16, 8),
                                  threshold = .01, stepmax = 1e+05, max_stocks = NULL){
  
  cat("Loading data...\n")
  data <- load_data(file_path)
  
  num_stocks <- length(unique(data$Name))
  cat("Number of unique stocks:", num_stocks, "\n")
  
  if(!is.null(max_stocks) && max_stocks < num_stocks) {
    cat("Limiting to", max_stocks, "stocks for testing\n")
    set.seed(123)
    selected_stocks <- sample(unique(data$Name), max_stocks)
    data <- data[data$Name %in% selected_stocks, ]
    num_stocks <- length(unique(data$Name))
    cat("Reduced to", num_stocks, "stocks\n")
  }
  
  cat("Prepping data with stock encoding\n")
  mlp_data <- prep_data(data, encoding = TRUE)
  
  cat("Processed data dimensions:", dim(mlp_data), "\n")
  cat("Number of features after one-hot encoding:", ncol(mlp_data) - 1, "\n")
  
  cat("Normalizing data...\n")
  norm_result <- normalize_data(mlp_data)
  normalized_data <- norm_result$norm_data
  norm_params <- norm_result$norm_params
  
  cat("Checking for NA or Inf values in normalized data...\n")
  has_na <- any(is.na(normalized_data))
  cat("Has NA values:", has_na, "\n")
  
  if(has_na) {
    normalized_data[is.na(normalized_data)] <- 0
    cat("Replaced NA values with 0\n")
  }
  
  cat("Splitting data...\n")
  train_size <- round((1 - test_size) * nrow(normalized_data))
  
  train_data <- normalized_data[1:train_size, ]
  test_data <- normalized_data[(train_size+1):nrow(normalized_data), ]
  
  predictors <- colnames(normalized_data)[colnames(normalized_data) != "close"]
  formula_str <- paste("close ~", paste(predictors, collapse = " + "))
  nn_formula <- as.formula(formula_str)
  
  cat("Training MLP model with", length(hidden_layers), "hidden layers:",
      paste(hidden_layers, collapse = ", ", "neurons\n"))
  
  if (length(predictors) > 100) {
    cat("Large number of features (", length(predictors), 
        "). Training may take a long time.\n")
  }
  
  set.seed(123)
  mlp_model <- neuralnet(
    formula = nn_formula,
    data = train_data,
    hidden = hidden_layers,
    linear.output = TRUE,
    threshold = threshold,
    stepmax = stepmax,
    algorithm = "rprop+",
    lifesign = "full",
    lifesign.step = 100
  )
  
  cat("Evaluating model...\n")
  
  test_predictions <- compute(mlp_model, test_data[, predictors])
  test_predictions_norm <- test_predictions$net.result
  
  test_predictions_actual <- denormalize(test_predictions_norm, "close", norm_params)
  test_actual_values <- denormalize(test_data$close, "close", norm_params)
  
  mae <- mean(abs(test_predictions_actual - test_actual_values))
  mape <- mean(abs((test_actual_values - test_predictions_actual) / test_actual_values)) * 100
  rmse <- sqrt(mean((test_predictions_actual - test_actual_values)^2))
  
  cat("Overall Metrics:\n")
  cat("Mean Absolute Error: $", round(mae, 2), "\n")
  cat("Mean Absolute Percentage Error: ", round(mape, 2), "%\n")
  cat("Root Mean Squared Error: $", round(rmse, 2), "\n")
  
  if ("Name" %in% colnames(data)) {
    test_stocks <- data$Name[(train_size+1):nrow(data)]
    
    stock_performance <- data.frame()
    
    for (stock in unique(test_stocks)) {
      stock_indices <- which(test_stocks == stock)
      
      if (length(stock_indices) > 0) {
        stock_actual <- test_actual_values[stock_indices]
        stock_pred <- test_predictions_actual[stock_indices]
        
        stock_mae <- mean(abs(stock_pred - stock_actual))
        stock_mape <- mean(abs((stock_actual - stock_pred) / stock_actual)) * 100
        stock_rmse <- sqrt(mean((stock_pred - stock_actual)^2))
        
        stock_performance <- rbind(stock_performance, data.frame(
          Stock = stock,
          MAE = stock_mae,
          MAPE = stock_mape,
          RMSE = stock_rmse,
          SampleSize = length(stock_indices)
        ))
      }
    }
    
    stock_performance <- stock_performance[order(stock_performance$MAPE), ]
    
    cat("\nTop 5 Performing Stocks (by MAPE):\n")
    print(head(stock_performance, 5))
    
    cat("\nBottom 5 Performing Stocks (by MAPE):\n")
    print(tail(stock_performance, 5))
    
    perf_file <- "stock_performance_metrics.csv"
    write.csv(stock_performance, perf_file, row.names = FALSE)
    cat("\nFull per-stock performance metrics saved to:", perf_file, "\n")
 
    filtered_performance <- stock_performance[stock_performance$MAPE < 20, ]
    cat("Excluded", nrow(stock_performance) - nrow(filtered_performance), 
        "stocks with MAPE > 20% for visualization purposes\n")
    
    hist_plot <- ggplot(filtered_performance, aes(x = MAPE)) +
      geom_histogram(bins = 30, fill = "skyblue", color = "black") +
      labs(title = "Distribution of Prediction Error Across Stocks (Excluding Outliers)",
           x = "Mean Absolute Percentage Error (%)",
           y = "Number of Stocks") +
      theme_minimal()
       
    print(hist_plot)
  }
  
  return(list(
    model = mlp_model, 
    norm_params = norm_params, 
    predictors = predictors,
    stock_names = unique(data$Name),
    mae = mae,
    mape = mape,
    rmse = rmse
  ))
}

predict_stock_price <- function(model_result, new_data, stock_name = NULL) {
  mlp_model <- model_result$model
  norm_params <- model_result$norm_params
  predictors <- model_result$predictors
  
  if (!is.null(stock_name)) {
    new_data_copy <- new_data
    
    if (!"Name" %in% colnames(new_data_copy)) {
      new_data_copy$Name <- stock_name
    } else {
      new_data_copy$Name <- stock_name
    }
    
    new_data <- prep_data(new_data_copy, encoding = TRUE)
  }
  
  missing_cols <- setdiff(predictors, colnames(new_data))
  if (length(missing_cols) > 0) {
    for (col in missing_cols) {
      new_data[[col]] <- 0
    }
  }
  
  new_data <- new_data[, c(predictors), drop = FALSE]
  
  normalized_new_data <- as.data.frame(lapply(names(new_data), function(col) {
    if (col %in% names(norm_params)) {
      min_val <- norm_params[[col]]["min"]
      max_val <- norm_params[[col]]["max"]
      
      if (max_val == min_val) {
        return(rep(0.5, nrow(new_data)))
      } else {
        return((new_data[[col]] - min_val) / (max_val - min_val))
      }
    } else {
      warning(paste("Column", col, "not found in normalization parameters"))
      return(rep(0, nrow(new_data)))
    }
  }))
  names(normalized_new_data) <- names(new_data)
  
  prediction_result <- compute(mlp_model, normalized_new_data)
  prediction_norm <- prediction_result$net.result
  
  prediction_actual <- denormalize(prediction_norm, "close", norm_params)
  
  return(prediction_actual)
}

prepare_stock_encoding <- function(stock_name, all_stock_names) {
  stock_encoding <- data.frame(matrix(0, nrow = 1, ncol = length(all_stock_names)))
  colnames(stock_encoding) <- paste0("Name_", all_stock_names)
  
  col_name <- paste0("Name_", stock_name)
  if (col_name %in% colnames(stock_encoding)) {
    stock_encoding[[col_name]] <- 1
  }
  
  return(stock_encoding)
}

predict_future_prices <- function(model_result, latest_data, stock_name, days_ahead = 5) {
  all_stocks <- model_result$stock_names
  
  if (!stock_name %in% all_stocks) {
    stop(paste("Stock", stock_name, "not found in the training data"))
  }
  
  current_data <- latest_data
  
  future_prices <- numeric(days_ahead)
  
  volatility <- .015
  
  for (i in 1:days_ahead) {
    next_price <- predict_stock_price(model_result, current_data, stock_name)
    
    future_prices[i] <- next_price
    
    if("open" %in% names(current_data)){
      current_data$open <- next_price * (1 + rnorm(1, 0, .005))
    }
    
    if("high" %in% names(current_data)){
      current_data$high <- next_price * (1 + volatility/2)
    }
    
    if("low" %in% names(current_data)){
      current_data$low <- next_price * (1 - volatility/2)
    }
    
    current_data$close <- next_price
    
    if("volume" %in% names(current_data)){
      current_data$volume <- current_data$volume * (1 + rnorm(1, 0, .1))
    }
  }
  
  return(future_prices)
}

predict_random_stock <- function(model_result, file_path, days_ahead = 5){
  data <- load_data(file_path)
  
  trained_stocks <- model_result$stock_names
  
  #effectively random
  set.seed(Sys.time())
  random_stock <- sample(trained_stocks, 1)
  cat("Random Stock:", random_stock, "\n")
  
  stock_data <- data[data$Name == random_stock, ]
  
  if("date" %in% colnames(stock_data)) {
    stock_data <- stock_data[order(stock_data$date), ]
  }
  
  latest_data <- stock_data[nrow(stock_data), ]
  
  cat("Latest date available:", as.character(latest_data$date), "\n")
  cat("Latest closing price: $", round(latest_data$close, 2), "\n")
  
  future_prices <- predict_future_prices(model_result, latest_data, random_stock, days_ahead)
  
  prediction_dates <- seq.Date(from = latest_data$date + 1, by = "day", length.out = days_ahead)
  prediction_df <- data.frame(
    Date = prediction_dates,
    Predicted_Close = future_prices
  )
  
  cat("\nPredicted closing prices for the next", days_ahead, "days:\n")
  for(i in 1:nrow(prediction_df)) {
    cat(format(prediction_df$Date[i], "%Y-%m-%d"), ": $", round(prediction_df$Predicted_Close[i], 2), "\n")
  }
  
  historical_days <- min(30, nrow(stock_data))
  plot_data <- data.frame(
    Date = c(tail(stock_data$date, historical_days), prediction_df$Date),
    Close = c(tail(stock_data$close, historical_days), rep(NA, days_ahead)),
    Predicted = c(rep(NA, historical_days), prediction_df$Predicted_Close)
  )
  
  price_plot <- ggplot(plot_data, aes(x = Date)) +
    geom_line(aes(y = Close, color = "Historical"), linewidth = 1) +
    geom_line(aes(y = Predicted, color = "Predicted"), linewidth = 1, linetype = "dashed") +
    geom_point(aes(y = Predicted, color = "Predicted"), size = 3) +
    labs(
      title = paste("Stock Price Prediction for", random_stock),
      subtitle = paste("MLP model with", 
                       paste(model_result$model$hidden, collapse = ","), 
                       "hidden layers"),
      x = "Date",
      y = "Closing Price ($)",
      color = "Type"
    ) +
    scale_color_manual(values = c("Historical" = "blue", "Predicted" = "red")) +
    theme_minimal() +
    theme(legend.position = "bottom")
  
  print(price_plot)
  
  return(list(
    stock = random_stock,
    latest_data = latest_data,
    predictions = prediction_df,
    plot = price_plot
  ))
}

main <- function() {
  file_path <- "all_stocks_5yr.csv"
  
  cat("Training multi-stock MLP model...\n")
  model_result <- train_multi_stock_mlp(
    file_path = file_path,
    test_size = 0.2,
    hidden_layers = c(64, 32, 16),
    threshold = 0.01,
    stepmax = 10000,
    max_stocks = 50
  )
  
  save(model_result, file = "stock_mlp_model.RData")
  cat("Model saved to stock_mlp_model.RData\n\n")
  
  cat("Predicting future prices for a random stock...\n")
  prediction_result <- predict_random_stock(model_result, file_path, days_ahead = 5)
}

load_and_predict <- function() {

  load("stock_mlp_model.RData")
  
  file_path <- "all_stocks_5yr.csv"
  prediction_result <- predict_random_stock(model_result, file_path, days_ahead = 5)
}

main()