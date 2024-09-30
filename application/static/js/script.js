function submitForm(event) {
    event.preventDefault(); // Prevent default form submission
    
    const form = document.getElementById('surveyForm');
    const formData = new FormData(form);
    var button = document.getElementById('submitButton');
    
    // Disable the button
    button.disabled = true;
    
    // Change the text to "Please wait..."
    button.innerHTML = 'Please wait...';
    


    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(Object.fromEntries(formData))
    })
    .then(response => response.json())
    .then(data => {
        // Hide the button
        button.style.display = 'none';
        
        var message = document.getElementById('msg');
        message.innerHTML = 'Thanks you for taking the survey!!';

        // const resultDiv = document.getElementById('result');
        resultDiv.innerHTML = `<p>Prediction: ${data.prediction}</p><p>Profile: ${data.profile}</p>`;
        if (data.mail_status === 'success') {
            resultDiv.innerHTML += '<p>A Email is sent to you successfully!</p>';
        } else {
            resultDiv.innerHTML += '<p>Failed to send email.</p>';
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

