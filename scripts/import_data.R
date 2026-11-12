# Import daily OHLCV data from Yahoo Finance into the project's CSV format.
#
# Usage in R / RStudio:
#   1. Edit `symbol`, `start_date` below.
#   2. source("scripts/import_data.R")
#   3. The file data/<SYMBOL>.csv is created with the schema expected by the
#      Python package: Date, Open, High, Low, Close, Volume, Adj.Close.

library(quantmod)

symbol <- "amzn"          # any Yahoo Finance ticker, e.g. "msft", "EURUSD=X"
start_date <- "2021-01-01"
out_file <- sprintf("data/%s.csv", toupper(symbol))

# Download
data <- getSymbols(
  symbol, src = "yahoo",
  from = start_date, to = Sys.Date() + 1,
  auto.assign = FALSE
)

# Normalize column names to the project schema
colnames(data) <- c("Open", "High", "Low", "Close", "Volume", "Adj.Close")

# Data frame with the date kept as a column
df <- data.frame(Date = index(data), coredata(data))
df <- na.omit(df)

dir.create("data", showWarnings = FALSE)
write.csv(df, file = out_file, row.names = FALSE)
cat(sprintf("Wrote %d rows to %s\n", nrow(df), out_file))
