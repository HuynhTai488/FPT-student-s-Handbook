var featureButton = document.getElementById("feature-button");
var featureList = document.getElementById("feature-list");

featureButton.addEventListener("mouseover", function() {
    featureList.style.display = "block";
});

featureButton.addEventListener("mouseout", function() {
    featureList.style.display = "none";
});
