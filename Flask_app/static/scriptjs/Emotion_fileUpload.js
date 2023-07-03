//Function to check if file/image has been uloaded by the user
function validateForm() {
    const fileInput = document.getElementById('file-upload');
    if (fileInput.files.length === 0) {
        alert('Please upload an image');
        return false; // Prevent form submission
    }
    return true; // Proceed with form submission
}

//Function to click the file type tag in form when function is called
function selectFile() {
    const fileInput = document.getElementById('file-upload');
    fileInput.click();
}

//Function to ensure user can upload only image type files
function handleFileUpload(event) {
    const file = event.target.files[0]; // Get the first file selected by the user

    // Check if the file format is valid
    const allowedFormats = ['image/png', 'image/jpeg', 'image/jpg'];
    if (!allowedFormats.includes(file.type)) {
        alert('Invalid file format. Please select a PNG, JPG, or JPEG image.');
        return;
    }

    //function to display the image once the file has been uploaded
    const reader = new FileReader();
    reader.onload = function(e) {
        const previewImage = document.getElementById('preview-image');
        previewImage.src = e.target.result;
        previewImage.style.display = 'block';

        const imageNameElement = document.getElementById('image-name');
        imageNameElement.textContent = 'Image: ' + file.name;
    };
    reader.readAsDataURL(file);
}