// Simulate content loading with a timeout
function loadContent() {
    // Simulating content load delay
    setTimeout(() => {
        // Redirect to recommendation.html after loading
        window.location.href = "recommendation.html";
    }, 3000); // 3000 milliseconds (3 seconds) delay
}

// When the DOM is fully loaded, execute loadContent
document.addEventListener('DOMContentLoaded', loadContent);
