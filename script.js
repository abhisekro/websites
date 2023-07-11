function predictBitcoinPrice() {
    // Historical Bitcoin prices (replace with your own data)
    var bitcoinPrices = [5000, 5500, 6000, 5800, 6200, 6500, 7000, 7200, 7500, 8000];

    // Get the latest Bitcoin price
    var latestPrice = bitcoinPrices[bitcoinPrices.length - 1];

    // Generate a random percentage change
    var randomPercentage = Math.random() * 0.2 - 0.1;  // Adjust the range as needed

    // Calculate the predicted price
    var predictedPrice = latestPrice + (latestPrice * randomPercentage);

    return predictedPrice.toFixed(2);
}

function displayPredictedPrice() {
    var predictedPrice = predictBitcoinPrice();
    document.getElementById('predicted-price').value = '$' + predictedPrice;
}

document.getElementById('predict-button').addEventListener('click', displayPredictedPrice);
